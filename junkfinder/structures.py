"""Load atomic models (local files or RCSB downloads) as weighted point clouds."""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

RCSB_ASSEMBLY = "https://files.rcsb.org/download/{pdb}-assembly{n}.cif.gz"
RCSB_ENTRY = "https://files.rcsb.org/download/{pdb}.cif.gz"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"

# Assembly files from the RCSB omit the source-organism records, so 'fetch'
# also saves a small JSON sidecar (<ID>.json) with entry metadata.
_META_QUERY = """query($id: String!) { entry(entry_id: $id) {
  struct { title }
  rcsb_entry_info { experimental_method resolution_combined }
  polymer_entities { rcsb_entity_source_organism { scientific_name } }
} }"""

# Approximate residue masses in kDa, for the mass sanity check.
_KDA_PER_AA = 0.110
_KDA_PER_NT = 0.330

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


def fetch_metadata(pdb_id: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Save RCSB entry metadata (title, method, source organisms) as <ID>.json."""
    pdb = pdb_id.upper()
    path = Path(cache_dir) / f"{pdb}.json"
    if path.exists():
        return path
    body = json.dumps({"query": _META_QUERY, "variables": {"id": pdb}}).encode()
    req = urllib.request.Request(RCSB_GRAPHQL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        entry = json.load(resp)["data"]["entry"]
    if entry is None:
        raise RuntimeError(f"RCSB has no entry {pdb}")
    organisms = sorted({
        o["scientific_name"].strip()
        for ent in entry.get("polymer_entities") or []
        for o in ent.get("rcsb_entity_source_organism") or []
        if o.get("scientific_name")
    })
    info = entry.get("rcsb_entry_info") or {}
    meta = {
        "id": pdb,
        "title": (entry.get("struct") or {}).get("title", ""),
        "method": info.get("experimental_method"),
        "resolution": (info.get("resolution_combined") or [None])[0],
        "organisms": organisms,
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1) + "\n")
    return path


def _sidecar(path: Path) -> dict:
    """Metadata saved by fetch_metadata next to a structure file, if any."""
    pdb = path.name.split("-assembly")[0].split(".")[0].upper()
    side = path.parent / f"{pdb}.json"
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return {}


def describe_structure(path: str | Path) -> dict:
    """Metadata used to check a file is the protein we think it is."""
    try:
        block = gemmi.cif.read(str(path)).sole_block()
    except (ValueError, RuntimeError):  # not mmCIF (e.g. a local PDB file)
        block = None

    def values(tag: str) -> list[str]:
        col = block.find_values(tag) if block is not None else None
        return [gemmi.cif.as_string(v) for v in col] if col else []

    organisms = set()
    for tag in ("_entity_src_gen.pdbx_gene_src_scientific_name",
                "_entity_src_nat.pdbx_organism_scientific",
                "_pdbx_entity_src_syn.organism_scientific"):
        organisms.update(v for v in values(tag) if v not in ("", "?", "."))
    meta = _sidecar(Path(path))
    if not organisms:
        organisms.update(meta.get("organisms", []))
    st = gemmi.read_structure(str(path))
    st.setup_entities()  # PDB-format files have no entity records
    st.remove_ligands_and_waters()
    kda = 0.0
    if len(st):
        for ch in st[0]:
            for res in ch:
                info = gemmi.find_tabulated_residue(res.name)
                is_na = info is not None and info.is_nucleic_acid()
                kda += _KDA_PER_NT if is_na else _KDA_PER_AA
    return {
        "id": meta.get("id") or (block.name if block is not None else Path(path).stem),
        "title": (values("_struct.title") or [meta.get("title", "")])[0],
        "entities": [v for v in values("_entity.pdbx_description") if v not in ("?", ".")],
        "organisms": sorted(organisms),
        "chains": len(st[0]) if len(st) else 0,
        "approx_kda": round(kda),
        "method": meta.get("method"),
        "resolution": meta.get("resolution"),
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
