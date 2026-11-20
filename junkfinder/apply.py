"""Apply a trained particle classifier to real cryoSPARC particles (needs PyTorch).

Input is a particles ``.cs`` file, e.g. from an Extract, 2D Classification or
Select 2D job, plus the cryoSPARC project directory its ``blob/path`` entries
are relative to. Each particle is brought onto the model's grid, then
classified:

1. **Pixel size.** Fourier-crop to the model's pixel size.
2. **Box.** Crop to the model's box. If the particle box is smaller, pad with
   noise, not zeros, because the network was trained on boxes full of noise.
3. **Normalisation.** Subtract the mean and divide by the standard deviation
   of the outer annulus (mostly noise). The simulator normalises by the noise
   level in the same way.
4. **Contrast.** Training images have dark protein, like raw cryoSPARC
   particles. Use ``invert=True`` for particles extracted with flipped
   contrast.

Results: one row per particle (predicted class, probability, runner-up). If
the ``.cs`` carries ``alignments2D/class``, there is also a breakdown of each
2D class by predicted label. A vote over a 2D class is much more reliable
than any single particle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import mrcfile
import numpy as np
import torch

from .classifier import ParticleNet


def load_model(path: str | Path, device: str = "cpu") -> tuple[ParticleNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ParticleNet(len(ckpt["keys"]), ckpt["units"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _fourier_resample_stack(imgs: np.ndarray, src_apix: float, dst_apix: float) -> np.ndarray:
    n = imgs.shape[-1]
    m = int(round(n * src_apix / dst_apix))
    m += m % 2
    if m == n:
        return imgs.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(imgs), axes=(-2, -1))
    k = min(n, m)
    s_in, s_out = (n - k) // 2, (m - k) // 2
    out = np.zeros((len(imgs), m, m), dtype=complex)
    out[:, s_out:s_out + k, s_out:s_out + k] = F[:, s_in:s_in + k, s_in:s_in + k]
    res = np.fft.ifft2(np.fft.ifftshift(out, axes=(-2, -1))).real
    return (res * (m * m) / (n * n)).astype(np.float32)


def prepare_particles(imgs: np.ndarray, apix_in: float, apix: float, box: int,
                      invert: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    """Raw particle images [n, b, b] -> model input [n, box, box] (noise-normalised)."""
    rng = rng or np.random.default_rng(0)
    x = _fourier_resample_stack(np.asarray(imgs, dtype=np.float32), apix_in, apix)
    n = x.shape[-1]
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.5
    ring = np.hypot(xx, yy) > 0.4 * n
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    sd = x[:, ring].std(axis=1)[:, None, None]
    x = x / np.maximum(sd, 1e-6)
    if invert:
        x = -x
    if n >= box:
        o = (n - box) // 2
        return x[:, o:o + box, o:o + box]
    out = rng.normal(size=(len(x), box, box)).astype(np.float32)
    o = (box - n) // 2
    out[:, o:o + n, o:o + n] = x
    return out


def iter_stacks(cs: np.ndarray, project_dir: Path):
    """Yield (row indices, images) per particle stack file referenced by the .cs."""
    paths = cs["blob/path"]
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[p.decode() if isinstance(p, bytes) else str(p)].append(i)
    for rel, rows in groups.items():
        rows = np.asarray(rows)
        rel = rel.lstrip(">")  # cryoSPARC marks some paths with a leading '>'
        f = Path(rel) if Path(rel).is_absolute() else project_dir / rel
        with mrcfile.mmap(f, mode="r", permissive=True) as m:
            data = m.data if m.data.ndim == 3 else m.data[None]
            yield rows, np.asarray(data[cs["blob/idx"][rows]], dtype=np.float32)


@torch.no_grad()
def classify_cs(cs: np.ndarray, project_dir: Path, model: ParticleNet, ckpt: dict,
                apix_in: float | None = None, invert: bool = False, batch: int = 512,
                device: str = "cpu", log=print) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities [n_particles, n_classes] and the predicted index per particle."""
    apix_in = apix_in or float(cs["blob/psize_A"][0])
    probs = np.zeros((len(cs), len(ckpt["keys"])), dtype=np.float32)
    rng = np.random.default_rng(0)
    done = 0
    for rows, imgs in iter_stacks(cs, project_dir):
        x = prepare_particles(imgs, apix_in, ckpt["apix"], ckpt["box"], invert, rng)
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch])[:, None].to(device)
            probs[rows[i:i + batch]] = torch.softmax(model(t), 1).cpu().numpy()
        done += len(rows)
        log(f"  {done}/{len(cs)} particles", flush=True)
    return probs, probs.argmax(1)


def write_particle_table(path: Path, cs: np.ndarray, probs: np.ndarray, keys: list[str]) -> None:
    order = np.argsort(-probs, axis=1)
    has_cls = "alignments2D/class" in cs.dtype.names
    with open(path, "w") as f:
        f.write("uid\tclass2D\tprediction\tprob\tsecond\tprob2\n" if has_cls else
                "uid\tprediction\tprob\tsecond\tprob2\n")
        for i in range(len(cs)):
            a, b = order[i, 0], order[i, 1]
            c2d = f"\t{int(cs['alignments2D/class'][i])}" if has_cls else ""
            f.write(f"{int(cs['uid'][i])}{c2d}\t{keys[a]}\t{probs[i, a]:.3f}\t{keys[b]}\t{probs[i, b]:.3f}\n")


def summarise_predictions(cs: np.ndarray, pred: np.ndarray, probs: np.ndarray, keys: list[str],
                          top: int = 3) -> str:
    n = len(pred)
    overall = Counter(pred.tolist())
    lines = [f"{n} particles. Predicted labels:"]
    for k, c in overall.most_common():
        lines.append(f"  {keys[k]:<20s} {c:7d}  {c / n:6.1%}")
    if "alignments2D/class" in cs.dtype.names:
        cls = cs["alignments2D/class"].astype(int)
        lines += ["", "Per 2D class (vote over its particles; mean probability in brackets):",
                  f"  {'class':>5s} {'n':>6s}  top labels"]
        for c in np.unique(cls):
            sel = cls == c
            votes = Counter(pred[sel].tolist()).most_common(top)
            mean_p = probs[sel].mean(0)
            txt = ", ".join(f"{keys[k]} {v / sel.sum():.0%} ({mean_p[k]:.2f})" for k, v in votes)
            lines.append(f"  {c:5d} {int(sel.sum()):6d}  {txt}")
    return "\n".join(lines)
