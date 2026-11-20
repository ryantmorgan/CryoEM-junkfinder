"""Fit micelle / nanodisc / SMA parameters to real 2D class averages.

Each candidate environment is wrapped around the membrane protein, and the
model is projected over the full sphere. Every class average is matched
against those projections with the same rotation- and shift-tolerant NCC used
for screening. The score of a candidate is the mean best NCC over the classes.
A random search over the ranges in :func:`membrane.random_params` is followed
by a local refinement around the best candidate.

The fit only makes sense on classes of the protein itself (e.g. a bo3
dataset), not on a mixture.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .match import PolarGrid, best_match, image_spectra, prepare_class_average, reference_spectra, shift_grid
from .membrane import EnvParams, MembraneModel, random_params, typical_params
from .project import directions_for_step, project_all

# Parameters refined locally, with step sizes (Å or contrast units).
_STEPS = {"half_height": 2.0, "diameter": 6.0, "head": 1.5, "head_contrast": 0.1,
          "core_contrast": 0.1, "ellipticity": 0.05, "msp_contrast": 0.1, "wall": 1.0}
_FIELDS = {"micelle": ["half_height", "diameter", "head", "head_contrast", "core_contrast", "ellipticity"],
           "nanodisc": ["half_height", "head", "head_contrast", "core_contrast", "msp_contrast"],
           "sma": ["half_height", "diameter", "head", "head_contrast", "core_contrast", "msp_contrast", "wall"]}

_STATE: dict = {}


def _init(structure, classes, apix_in, apix, box, resolution, angular_step, max_shift_A, invert):
    model = MembraneModel(structure)
    grid = PolarGrid.for_box(box)
    shifts = shift_grid(max_shift_A / apix)
    prepared = [prepare_class_average(c, apix_in, apix, box, resolution, invert) for c in classes]
    _STATE.update(model=model, grid=grid, shifts=shifts, prepared=prepared,
                  spectra=[image_spectra(grid, p, shifts) for p in prepared],
                  dirs=directions_for_step(angular_step), apix=apix, box=box, resolution=resolution)


def _score(args) -> tuple[float, list[float]]:
    kind, params = args
    s = _STATE
    xyz, w = s["model"].variant(kind, params=params)
    proj = project_all(xyz, w, s["dirs"], s["box"], s["apix"], s["resolution"])
    ref = reference_spectra(s["grid"], proj)
    scores = [best_match(s["grid"], None, ref, s["shifts"], C).score for C in s["spectra"]]
    return float(np.mean(scores)), scores


def _jitter(p: EnvParams, kind: str, rng: np.random.Generator, scale: float) -> EnvParams:
    changes = {}
    for f in _FIELDS[kind]:
        v = getattr(p, f)
        if v is None:
            continue
        changes[f] = v + rng.normal(0, _STEPS[f] * scale)
    changes["seed"] = int(rng.integers(1 << 31))
    q = dataclasses.replace(p, **changes)
    q.half_height = max(q.half_height, 10.0)
    q.head = float(np.clip(q.head, 2.0, q.half_height))
    q.ellipticity = float(np.clip(q.ellipticity, 0.0, 0.4))
    q.wall = max(q.wall, 1.0)
    if q.diameter is not None:
        q.diameter = max(q.diameter, 40.0)
    return q


def fit_environment(
    structure, classes: np.ndarray, apix_in: float, kind: str = "micelle",
    n_random: int = 48, n_refine: int = 4, per_round: int = 16, apix: float = 4.0,
    box: int | None = None, resolution: float = 12.0, angular_step: float = 7.5,
    max_shift_A: float = 12.0, invert: bool | None = None, workers: int = 4, seed: int = 0,
    log=print,
) -> dict:
    """Random search plus local refinement; returns the best params and the score table."""
    box = box or int(round(classes.shape[-1] * apix_in / apix / 2)) * 2
    rng = np.random.default_rng(seed)
    init = (structure, classes, apix_in, apix, box, resolution, angular_step, max_shift_A, invert)
    results: list[tuple[float, list[float], str, EnvParams | None]] = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=init) as pool:
        def run(cands, label):
            out = list(pool.map(_score, cands))
            for (k, p), (m, sc) in zip(cands, out):
                results.append((m, sc, k, p))
            best = max(results, key=lambda r: r[0] if r[2] == kind else -1)
            log(f"  {label}: best so far {best[0]:.4f}")

        run([("none", None), (kind, typical_params(kind))], "bare protein and typical environment")
        log(f"  bare protein {results[0][0]:.4f}, typical {kind} {results[1][0]:.4f}")
        run([(kind, random_params(kind, rng)) for _ in range(n_random)], f"{n_random} random")
        scale = 1.0
        for r in range(n_refine):
            top = sorted((x for x in results if x[2] == kind), key=lambda x: -x[0])[:4]
            cands = [(kind, _jitter(t[3], kind, rng, scale)) for t in top for _ in range(per_round // 4)]
            run(cands, f"refine round {r + 1}")
            scale *= 0.7
    env = sorted((x for x in results if x[2] == kind), key=lambda x: -x[0])
    return {"best": env[0], "bare": results[0], "typical": results[1], "all": results, "box": box}


def describe(p: EnvParams, kind: str) -> str:
    return ", ".join(f"{f}={getattr(p, f):.2f}" for f in _FIELDS[kind] if getattr(p, f) is not None)
