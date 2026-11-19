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


def cached_structure(pdb_id: str, assembly: int = 1, cache_dir: Path = DEFAULT_CACHE) -> Path | None:
    """Path to an already-downloaded structure (assembly preferred), or None."""
    pdb = pdb_id.upper()
    for name in (f"{pdb}-assembly{assembly}.cif", f"{pdb}.cif"):
        for suffix in (".gz", ""):
            path = Path(cache_dir) / (name + suffix)
            if path.exists():
                return path
    return None


def fetch_structure(pdb_id: str, assembly: int = 1, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download a biological assembly from RCSB (falls back to the deposited entry).

    Files are kept gzipped (gemmi reads them directly), so a ``structures/``
    folder can be committed to git without bloating it.
    """
    cache_dir = Path(cache_dir)
    found = cached_structure(pdb_id, assembly, cache_dir)
    if found:
        return found
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = pdb_id.upper()
    errors = []
    for name, url in (
        (f"{pdb}-assembly{assembly}.cif.gz", RCSB_ASSEMBLY.format(pdb=pdb, n=assembly)),
        (f"{pdb}.cif.gz", RCSB_ENTRY.format(pdb=pdb)),
    ):
        path = cache_dir / name
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(path, "wb") as out:
                shutil.copyfileobj(resp, out)
            gzip.open(path).read(1)  # sanity check: is it really gzip?
            return path
        except Exception as exc:  # noqa: BLE001 - report all attempts
            path.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {pdb}:\n  " + "\n  ".join(errors))


def describe_structure(path: str | Path) -> dict:
    """Metadata used to check a file is the protein we think it is."""
    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()

    def values(tag: str) -> list[str]:
        col = block.find_values(tag)
        return [gemmi.cif.as_string(v) for v in col] if col else []

    organisms = set()
    for tag in ("_entity_src_gen.pdbx_gene_src_scientific_name",
                "_entity_src_nat.pdbx_organism_scientific",
                "_pdbx_entity_src_syn.organism_scientific"):
        organisms.update(v for v in values(tag) if v not in ("", "?", "."))
    st = gemmi.read_structure(str(path))
    st.remove_ligands_and_waters()
    n_res = sum(len(ch) for ch in st[0]) if len(st) else 0
    return {
        "id": block.name,
        "title": (values("_struct.title") or [""])[0],
        "entities": [v for v in values("_entity.pdbx_description") if v not in ("?", ".")],
        "organisms": sorted(organisms),
        "chains": len(st[0]) if len(st) else 0,
        "approx_kda": round(n_res * 0.110),
    }


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
