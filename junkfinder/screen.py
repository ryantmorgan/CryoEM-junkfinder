"""Screen class averages against a reference library and summarise the result."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate, shift as nd_shift

from .library import Library
from .match import (
    MatchResult, PolarGrid, best_match, image_spectra, prepare_class_average, reference_spectra,
    shift_grid,
)


@dataclass
class ClassReport:
    index: int
    n_particles: int | None
    scores: dict[str, MatchResult]  # entry key -> best match
    verdict: str
    top_key: str
    margin: float


def ranked(scores: dict[str, MatchResult]) -> list[tuple[str, MatchResult]]:
    return sorted(scores.items(), key=lambda kv: kv[1].score, reverse=True)


def decide(
    scores: dict[str, MatchResult], kinds: dict[str, str], min_score: float, min_margin: float,
) -> tuple[str, str, float]:
    """Verdict for one class.

    Flag a contaminant only if its score clears ``min_score`` and beats every
    other entry (including the user's target) by ``min_margin``.
    """
    order = ranked(scores)
    top_key, top = order[0]
    margin = top.score - order[1][1].score if len(order) > 1 else top.score
    if top.score < min_score:
        return "no_match", top_key, margin
    if margin < min_margin:
        return "ambiguous", top_key, margin
    return ("target" if kinds[top_key] == "target" else "contaminant"), top_key, margin


def screen(
    library: Library,
    class_averages: np.ndarray,
    apix: float,
    counts: np.ndarray | None = None,
    invert: bool | None = None,
    max_shift_A: float = 12.0,
    mask_radius_A: float | None = None,
    min_score: float = 0.5,
    min_margin: float = 0.03,
) -> tuple[list[ClassReport], np.ndarray]:
    radius = None if mask_radius_A is None else min(mask_radius_A / library.apix, 0.5 * library.box - 1)
    grid = PolarGrid.for_box(library.box, radius)
    shifts = shift_grid(max_shift_A / library.apix)
    spectra = {e.key: reference_spectra(grid, e.projections) for e in library.entries}
    kinds = {e.key: e.kind for e in library.entries}

    prepared = np.stack([
        prepare_class_average(img, apix, library.apix, library.box, library.resolution, invert)
        for img in class_averages
    ])
    reports = []
    for i, img in enumerate(prepared):
        C = image_spectra(grid, img, shifts)
        scores = {k: best_match(grid, None, s, shifts, C) for k, s in spectra.items()}
        verdict, top_key, margin = decide(scores, kinds, min_score, min_margin)
        n = None if counts is None else int(counts[i])
        reports.append(ClassReport(i, n, scores, verdict, top_key, margin))
    return reports, prepared


def aligned_reference(library: Library, key: str, m: MatchResult) -> np.ndarray:
    """The best-matching projection, rotated and shifted onto the class average."""
    entry = next(e for e in library.entries if e.key == key)
    ref = entry.projections[m.view].astype(np.float64)
    ref = rotate(ref, -m.angle_deg, reshape=False, order=1)
    return nd_shift(ref, (m.shift[1], m.shift[0]), order=1)


def summarise(reports: list[ClassReport], library: Library) -> str:
    names = {e.key: e.name for e in library.entries}
    have_counts = all(r.n_particles is not None for r in reports)
    weight = (lambda r: r.n_particles) if have_counts else (lambda r: 1)
    unit = "particles" if have_counts else "classes"
    total = sum(weight(r) for r in reports) or 1

    lines = [f"Screened {len(reports)} class averages against {len(library.entries)} references "
             f"({library.apix:.1f} Å/px, {library.resolution:.0f} Å low-pass)."]
    by_key: dict[str, int] = {}
    for r in reports:
        if r.verdict == "contaminant":
            by_key[r.top_key] = by_key.get(r.top_key, 0) + weight(r)
    if by_key:
        lines.append("Possible contaminants:")
        for k, w in sorted(by_key.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {names[k]:<48s} {w:>8d} {unit} ({100 * w / total:.1f}%)")
    else:
        lines.append("No class averages confidently matched a known contaminant.")
    for label in ("target", "ambiguous", "no_match"):
        w = sum(weight(r) for r in reports if r.verdict == label)
        if w:
            lines.append(f"  {label:<48s} {w:>8d} {unit} ({100 * w / total:.1f}%)")
    return "\n".join(lines)


def write_tsv(reports: list[ClassReport], library: Library, path: str | Path, top_n: int = 3) -> None:
    keys = [e.key for e in library.entries]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        header = ["class", "n_particles", "verdict", "top_hit", "margin"]
        for j in range(1, top_n + 1):
            header += [f"hit{j}", f"score{j}"]
        header += [f"score:{k}" for k in keys]
        w.writerow(header)
        for r in reports:
            row = [r.index, "" if r.n_particles is None else r.n_particles, r.verdict,
                   r.top_key, f"{r.margin:.3f}"]
            for k, m in ranked(r.scores)[:top_n]:
                row += [k, f"{m.score:.3f}"]
            row += [f"{r.scores[k].score:.3f}" for k in keys]
            w.writerow(row)


def write_montage(
    reports: list[ClassReport], prepared: np.ndarray, library: Library, path: str | Path,
    max_classes: int = 60,
) -> None:
    """Side-by-side panels: class average | best match for each class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(reports, key=lambda r: -(r.n_particles or 0))[:max_classes]
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.3, rows * 1.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    colours = {"contaminant": "#c0392b", "target": "#1e8449", "ambiguous": "#b9770e", "no_match": "#555"}
    for i, r in enumerate(order):
        ax_c, ax_r = axes[i // cols, (i % cols) * 2], axes[i // cols, (i % cols) * 2 + 1]
        m = r.scores[r.top_key]
        ax_c.imshow(prepared[r.index], cmap="gray")
        ax_r.imshow(aligned_reference(library, r.top_key, m), cmap="gray")
        n = f" n={r.n_particles}" if r.n_particles is not None else ""
        ax_c.set_title(f"#{r.index}{n}", fontsize=6)
        ax_r.set_title(f"{r.top_key} {m.score:.2f}", fontsize=6, color=colours[r.verdict])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
