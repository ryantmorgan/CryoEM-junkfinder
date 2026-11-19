"""Load atomic models (local files or RCSB downloads) as weighted point clouds."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

RCSB_ASSEMBLY = "https://files.rcsb.org/download/{pdb}-assembly{n}.cif.gz"
RCSB_ENTRY = "https://files.rcsb.org/download/{pdb}.cif.gz"

DEFAULT_CACHE = Path.home() / ".cache" / "junkfinder" / "structures"


def fetch_structure(pdb_id: str, assembly: int = 1, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download a biological assembly from RCSB (falls back to the deposited entry)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = pdb_id.upper()
    candidates = [
        (cache_dir / f"{pdb}-assembly{assembly}.cif", RCSB_ASSEMBLY.format(pdb=pdb, n=assembly)),
        (cache_dir / f"{pdb}.cif", RCSB_ENTRY.format(pdb=pdb)),
    ]
    for path, _ in candidates:
        if path.exists():
            return path
    errors = []
    for path, url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(path, "wb") as out:
                with gzip.GzipFile(fileobj=resp) as gz:
                    shutil.copyfileobj(gz, out)
            return path
        except Exception as exc:  # noqa: BLE001 - report all attempts
            path.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {pdb}:\n  " + "\n  ".join(errors))


def load_atoms(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords[N,3] in Å, centred on centre of mass; weights[N] = atomic number).

    Waters and hydrogens are dropped; ligands, lipids and cofactors are kept
    because they contribute to the projected density.
    """
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_waters()
    if len(st) == 0:
        raise ValueError(f"No models in {path}")
    coords, weights = [], []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                p = atom.pos
                coords.append((p.x, p.y, p.z))
                weights.append(atom.element.atomic_number)
    if not coords:
        raise ValueError(f"No atoms in {path}")
    xyz = np.asarray(coords, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    xyz -= (xyz * w[:, None]).sum(0) / w.sum()
    return xyz, w
