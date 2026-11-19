"""Per-particle contaminant classifier (needs PyTorch).

The backbone follows the ResNet8 design described for Topaz (Bepler et al.
2019, doi:10.1038/s41592-019-0575-8): a 7x7 convolution, three dilated
residual blocks (two of them strided) and a 5x5 convolution, widths
32-64-128. It is re-implemented here from that description, not copied (Topaz
is GPL-3). Topaz ends in a single 1x1 'particle or not' output; here the
features are pooled and fed to one output per library entry plus background.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .particles import ParticleSimulator


class ResBlock(nn.Module):
    def __init__(self, nin: int, nout: int, dilation: int = 2, stride: int = 1):
        super().__init__()
        self.conv0 = nn.Conv2d(nin, nin, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(nin)
        self.conv1 = nn.Conv2d(nin, nout, 3, padding=dilation, dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(nout)
        self.proj = None if nin == nout and stride == 1 else nn.Conv2d(nin, nout, 1, stride=stride, bias=False)

    def forward(self, x):
        y = F.relu(self.bn0(self.conv0(x)))
        y = self.bn1(self.conv1(y))
        return F.relu(y + (x if self.proj is None else self.proj(x)))


class ParticleNet(nn.Module):
    """ResNet8-style feature extractor + global pooling + linear head."""

    def __init__(self, n_classes: int, units: int = 32):
        super().__init__()
        u = [units, 2 * units, 4 * units]
        self.features = nn.Sequential(
            nn.Conv2d(1, u[0], 7, padding=3, bias=False), nn.BatchNorm2d(u[0]), nn.ReLU(inplace=True),
            ResBlock(u[0], u[0]),
            ResBlock(u[0], u[1], stride=2),
            ResBlock(u[1], u[1]),
            ResBlock(u[1], u[2], stride=2),
            nn.Conv2d(u[2], u[2], 5, padding=2, bias=False), nn.BatchNorm2d(u[2]), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(u[2], n_classes)

    def forward(self, x):
        f = self.features(x)
        return self.head(f.mean(dim=(-2, -1)))


@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    acc: list[float] = field(default_factory=list)


def train(
    sim: ParticleSimulator, steps: int = 3000, batch: int = 128, lr: float = 2e-3,
    units: int = 32, seed: int = 0, log_every: int = 100, verbose: bool = True,
) -> tuple[ParticleNet, TrainLog]:
    """Train on freshly simulated particles every step (no fixed training set)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=sim.device).manual_seed(seed)
    model = ParticleNet(len(sim.keys), units).to(sim.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    log, t0, run_loss, run_acc = TrainLog(), time.time(), 0.0, 0.0
    model.train()
    for step in range(1, steps + 1):
        x, y = sim.sample(batch, gen)
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item()
        run_acc += (out.argmax(1) == y).float().mean().item()
        if step % log_every == 0:
            log.steps.append(step)
            log.loss.append(run_loss / log_every)
            log.acc.append(run_acc / log_every)
            if verbose:
                print(f"  step {step:5d}  loss {log.loss[-1]:.3f}  train acc {log.acc[-1]:.3f}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
            run_loss = run_acc = 0.0
    model.eval()
    return model, log


@torch.no_grad()
def evaluate(model: ParticleNet, sim: ParticleSimulator, per_class: int = 400, seed: int = 1,
             batch: int = 500) -> np.ndarray:
    """Confusion matrix [true, predicted] on freshly simulated particles."""
    gen = torch.Generator(device=sim.device).manual_seed(seed)
    n_cls = len(sim.keys)
    labels = torch.arange(n_cls, device=sim.device).repeat_interleave(per_class)
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    for i in range(0, len(labels), batch):
        y = labels[i:i + batch]
        x, _ = sim.sample(len(y), gen, labels=y)
        pred = model(x).argmax(1)
        np.add.at(conf, (y.cpu().numpy(), pred.cpu().numpy()), 1)
    return conf


def report(conf: np.ndarray, keys: list[str]) -> str:
    acc = conf.diagonal() / conf.sum(1)
    lines = [f"overall accuracy {conf.diagonal().sum() / conf.sum():.3f} "
             f"(chance {1 / len(keys):.3f})", "",
             f"{'class':<14s} {'acc':>5s}  most often mistaken for"]
    for i in np.argsort(-acc):
        row = conf[i].copy()
        row[i] = 0
        top = [f"{keys[j]} {row[j] / conf[i].sum():.0%}" for j in np.argsort(-row)[:2] if row[j]]
        lines.append(f"{keys[i]:<14s} {acc[i]:5.2f}  {', '.join(top)}")
    return "\n".join(lines)
