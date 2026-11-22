"""Fine-tune a simulated-data classifier on real, labelled particles (needs PyTorch).

Each label comes from an existing cryoSPARC particle set: for example, the
particles of 2D classes you know are bo3, or your own target.

1. **Load.** Real particles are prepared exactly as in ``classify`` (resampled,
   noise-normalised, cropped or noise-padded to the model box). A random
   10% per label is held out for validation.
2. **Train.** Each step mixes real particles (random rotations and flips; a
   mirrored projection is the projection from the opposite direction, so
   flips are valid) with freshly simulated particles of every class. The
   simulated half stops the network forgetting the contaminants that have no
   real examples.
3. **Report.** Accuracy on the held-out real particles before and after, and
   what they are mistaken for. That is the honest measure of whether the model
   transfers to your data.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .apply import iter_stacks, prepare_particles


def load_real(cs: np.ndarray, project_dir, ckpt: dict, max_n: int | None = None,
              invert: bool = False, seed: int = 0) -> np.ndarray:
    """Prepared real particles [n, box, box] from a particles .cs (random subset if max_n)."""
    rng = np.random.default_rng(seed)
    if max_n and len(cs) > max_n:
        cs = cs[np.sort(rng.choice(len(cs), max_n, replace=False))]
    apix_in = float(cs["blob/psize_A"][0])
    out = np.empty((len(cs), ckpt["box"], ckpt["box"]), np.float32)
    for rows, imgs in iter_stacks(cs, project_dir):
        out[rows] = prepare_particles(imgs, apix_in, ckpt["apix"], ckpt["box"], invert, rng)
    return out


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    k = int(torch.randint(4, (1,), generator=gen))
    x = torch.rot90(x, k, dims=(-2, -1))
    if torch.rand(1, generator=gen) < 0.5:
        x = torch.flip(x, dims=(-1,))
    return x


@torch.no_grad()
def real_confusion(model, x: np.ndarray, y: np.ndarray, n_cls: int, device="cpu", batch=512) -> np.ndarray:
    model.eval()
    conf = np.zeros((n_cls, n_cls), np.int64)
    for i in range(0, len(x), batch):
        pred = model(torch.from_numpy(x[i:i + batch])[:, None].to(device)).argmax(1).cpu().numpy()
        np.add.at(conf, (y[i:i + batch], pred), 1)
    return conf


def finetune(model, sim, x_real: np.ndarray, y_real: np.ndarray, steps: int = 2000, batch: int = 128,
             frac_real: float = 0.5, lr: float = 3e-4, seed: int = 0, log_every: int = 100, log=print):
    """Mixed real + simulated training; returns the model (in eval mode)."""
    device = sim.device
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    xr = torch.from_numpy(x_real)
    yr = torch.from_numpy(y_real.astype(np.int64))
    n_real = max(1, int(round(batch * frac_real)))
    # Balance labels: sample each real label equally often.
    by_label = [torch.nonzero(yr == c).flatten() for c in torch.unique(yr)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    t0, run = time.time(), [0.0, 0.0, 0]
    for step in range(1, steps + 1):
        picks = torch.cat([g[torch.randint(len(g), (n_real // len(by_label) + 1,), generator=cpu_gen)]
                           for g in by_label])[:n_real]
        xb_r = _augment(xr[picks][:, None], cpu_gen).to(device)
        yb_r = yr[picks].to(device)
        xb_s, yb_s = sim.sample(batch - n_real, gen)
        xb, yb = torch.cat([xb_r, xb_s]), torch.cat([yb_r, yb_s])
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run[0] += loss.item()
        run[1] += (out[:n_real].argmax(1) == yb_r).float().mean().item()
        run[2] += 1
        if step % log_every == 0:
            log(f"  step {step:5d}  loss {run[0] / run[2]:.3f}  real-batch acc {run[1] / run[2]:.3f}  "
                f"({time.time() - t0:.0f} s)", flush=True)
            run = [0.0, 0.0, 0]
    model.eval()
    return model


def confusion_report(conf: np.ndarray, keys: list[str], labels: list[int]) -> str:
    lines = []
    for c in labels:
        n = conf[c].sum()
        if not n:
            continue
        row = conf[c].copy()
        acc = row[c] / n
        row[c] = 0
        top = ", ".join(f"{keys[j]} {row[j] / n:.0%}" for j in np.argsort(-row)[:3] if row[j])
        lines.append(f"  {keys[c]:<22s} {acc:6.1%} correct of {n}" + (f"; mistaken for {top}" if top else ""))
    return "\n".join(lines)
