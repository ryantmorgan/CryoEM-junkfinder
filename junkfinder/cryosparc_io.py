"""Read cryoSPARC 2D classification outputs.

cryoSPARC writes class averages as an MRC stack (``*_class_averages.mrc``)
alongside a ``.cs`` file (a NumPy structured array) that records the pixel size.
Per-class particle counts come from the job's particles ``.cs`` file
(``alignments2D/class`` field).
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np


def read_stack(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (images[n, box, box], pixel size from header in Å; 0 if unset)."""
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32)
        apix = float(m.voxel_size.x)
    if data.ndim == 2:
        data = data[None]
    return data, apix


def read_cs(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def pixel_size_from_cs(cs: np.ndarray) -> float | None:
    if "blob/psize_A" in cs.dtype.names:
        return float(cs["blob/psize_A"][0])
    return None


def class_counts(particles_cs: np.ndarray, n_classes: int) -> np.ndarray | None:
    """Particles per class from a particles .cs file, or None if not present."""
    field = "alignments2D/class"
    if field not in particles_cs.dtype.names:
        return None
    return np.bincount(particles_cs[field].astype(int), minlength=n_classes)[:n_classes]


def load_class_averages(
    mrc_path: str | Path,
    cs_path: str | Path | None = None,
    particles_path: str | Path | None = None,
    apix: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Load class averages, their pixel size and (optionally) particle counts.

    Pixel size priority: explicit ``apix`` > ``.cs`` file > MRC header.
    """
    imgs, header_apix = read_stack(mrc_path)
    if apix is None and cs_path is not None:
        apix = pixel_size_from_cs(read_cs(cs_path))
    if apix is None:
        apix = header_apix
    if not apix or apix <= 0:
        raise SystemExit("Pixel size unknown: pass --apix or the class averages .cs file.")
    counts = class_counts(read_cs(particles_path), len(imgs)) if particles_path else None
    return imgs, apix, counts


def exposures_from_cs(cs_paths: list[str | Path], project_dir: Path) -> list[tuple[Path, float]]:
    """(micrograph path, pixel size) per exposure from cryoSPARC exposures .cs files.

    A job's own .cs and its passthrough .cs are merged by uid. The
    ``micrograph_blob`` (the dose-weighted, motion-corrected image) usually
    sits in the passthrough file.
    """
    fields: dict[int, dict] = {}
    for p in cs_paths:
        cs = read_cs(p)
        names = cs.dtype.names
        for row in cs:
            d = fields.setdefault(int(row["uid"]), {})
            for f in ("micrograph_blob/path", "micrograph_blob/psize_A"):
                if f in names:
                    d[f] = row[f]
    out = []
    for d in fields.values():
        if "micrograph_blob/path" not in d:
            continue
        rel = d["micrograph_blob/path"]
        rel = (rel.decode() if isinstance(rel, bytes) else str(rel)).lstrip(">")
        out.append((project_dir / rel, float(d.get("micrograph_blob/psize_A", 0)) or None))
    if not out:
        raise SystemExit("No micrograph_blob/path in the given .cs files; include the passthrough file.")
    return sorted(out)
