"""Scan whole micrographs with a trained particle classifier (needs PyTorch).

Topaz-style sliding window: each micrograph is brought onto the model's pixel
size, normalised to unit noise, and cut into overlapping model-sized windows
on a regular grid (``stride`` px). Every window is classified. That gives one
probability map per class at the grid resolution.

Picks are local maxima of ``1 - P(background)`` above ``threshold``,
separated by at least ``min_dist`` px. Each pick is labelled with its most
likely non-background class.

Caveats:
- The network was trained on simulated particles in idealised ice. Carbon
  edges, ice contamination, aggregates and empty micelles were never shown to
  it, so expect false picks on them.
- Treat per-micrograph and per-class counts as a screen, not a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from scipy import ndimage

from .apply import _fourier_resample_stack


@dataclass
class Pick:
    micrograph: str
    x: float  # original micrograph pixels
    y: float
    label: str
    prob: float  # P(label)
    score: float  # 1 - P(background)


def read_micrograph(path: str | Path) -> tuple[np.ndarray, float]:
    with mrcfile.open(path, permissive=True) as m:
        img = np.asarray(m.data, dtype=np.float32)
        apix = float(m.voxel_size.x)
    if img.ndim == 3:  # a stack of frames: sum them
        img = img.sum(0)
    return img, apix


def prepare_micrograph(img: np.ndarray, apix_in: float, apix: float, highpass_A: float = 600.0,
                       invert: bool = False) -> np.ndarray:
    """Downsample to the model grid, remove slow gradients, and normalise to unit noise."""
    h, w = img.shape
    n = min(h, w)
    # Square crop for the Fourier resampler; keep the offset so coordinates map back.
    sq = img[:n, :n] if h == w else img[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n]
    x = _fourier_resample_stack(sq[None], apix_in, apix)[0]
    x = x - ndimage.gaussian_filter(x, highpass_A / apix / 2.355)
    mad = np.median(np.abs(x - np.median(x))) * 1.4826  # robust noise level
    x = (x - np.median(x)) / max(mad, 1e-6)
    return -x if invert else x


@torch.no_grad()
def probability_maps(x: np.ndarray, model, box: int, stride: int, batch: int = 256,
                     device: str = "cpu") -> np.ndarray:
    """Class probabilities for windows on a grid: [n_classes, ny, nx]."""
    n = x.shape[0]
    starts = np.arange(0, n - box + 1, stride)
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    out = []
    coords = [(i, j) for i in starts for j in starts]
    for k in range(0, len(coords), batch):
        chunk = coords[k:k + batch]
        win = torch.stack([t[i:i + box, j:j + box] for i, j in chunk])[:, None]
        out.append(torch.softmax(model(win), 1).cpu().numpy())
    p = np.concatenate(out)
    return p.reshape(len(starts), len(starts), -1).transpose(2, 0, 1)


def find_picks(probs: np.ndarray, keys: list[str], threshold: float, min_dist_px: int):
    """Local maxima of 1 - P(background): list of (row, col, label_idx, p_label, score)."""
    bg = keys.index("background")
    score = 1.0 - probs[bg]
    size = max(1, 2 * min_dist_px + 1)
    peaks = (score == ndimage.maximum_filter(score, size=size, mode="constant")) & (score >= threshold)
    fg = probs.copy()
    fg[bg] = -1
    out = []
    for r, c in zip(*np.nonzero(peaks)):
        lab = int(fg[:, r, c].argmax())
        out.append((int(r), int(c), lab, float(probs[lab, r, c]), float(score[r, c])))
    return out


def scan_micrograph(path: str | Path, model, ckpt: dict, stride_A: float = 24.0,
                    threshold: float = 0.7, min_dist_A: float | None = None, apix_in: float | None = None,
                    invert: bool = False, device: str = "cpu"):
    """Returns (picks, probability maps, prepared image, (y offset, x offset, scale))."""
    img, header_apix = read_micrograph(path)
    apix_in = apix_in or header_apix
    if not apix_in or apix_in <= 0:
        raise SystemExit(f"{path}: pixel size unknown; pass --apix")
    apix, box = ckpt["apix"], ckpt["box"]
    x = prepare_micrograph(img, apix_in, apix, invert=invert)
    stride = max(1, int(round(stride_A / apix)))
    probs = probability_maps(x, model, box, stride, device=device)
    min_dist_A = min_dist_A or box * apix / 4
    picks = []
    h, w = img.shape
    n = min(h, w)
    oy, ox = ((h - n) // 2, (w - n) // 2) if h != w else (0, 0)
    scale = n / x.shape[0]  # original px per model px (exact after Fourier cropping)
    for r, c, lab, p, sc in find_picks(probs, ckpt["keys"], threshold,
                                       max(1, int(round(min_dist_A / apix / stride)))):
        cy, cx = r * stride + box / 2, c * stride + box / 2  # window centre, model px
        picks.append(Pick(str(path), ox + cx * scale, oy + cy * scale, ckpt["keys"][lab], p, sc))
    return picks, probs, x, (oy, ox, scale)


def save_overlay(path: Path, x: np.ndarray, picks: list[Pick], ckpt: dict,
                 geom: tuple[int, int, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oy, ox, scale = geom
    labels = sorted({p.label for p in picks})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))
    lo, hi = np.percentile(x, [1, 99])
    ax.imshow(ndimage.gaussian_filter(x, 1.0), cmap="gray", vmin=lo, vmax=hi)
    r = ckpt["box"] / 4
    for p in picks:
        c = cmap(labels.index(p.label) % 20)
        cx, cy = (p.x - ox) / scale, (p.y - oy) / scale
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=c, lw=1))
    for i, lab in enumerate(labels):
        ax.plot([], [], "o", mfc="none", color=cmap(i % 20), label=lab)
    if labels:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
