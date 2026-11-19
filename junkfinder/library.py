"""Build, save and load a library of simulated reference projections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .catalog import Contaminant
from .project import directions_for_step, max_dimension, project_all
from .structures import fetch_structure, load_atoms


@dataclass
class LibraryEntry:
    key: str
    name: str
    source: str  # PDB ID or file path
    kind: str  # "contaminant" or "target"
    max_dim_A: float
    projections: np.ndarray = field(repr=False)  # [n_views, box, box]


@dataclass
class Library:
    apix: float
    box: int
    resolution: float
    angular_step: float
    directions: np.ndarray
    entries: list[LibraryEntry]

    def save(self, path: str | Path) -> None:
        meta = {
            "apix": self.apix,
            "box": self.box,
            "resolution": self.resolution,
            "angular_step": self.angular_step,
            "entries": [
                {k: v for k, v in asdict(e).items() if k != "projections"} for e in self.entries
            ],
        }
        arrays = {f"proj_{i}": e.projections for i, e in enumerate(self.entries)}
        np.savez_compressed(path, meta=json.dumps(meta), directions=self.directions, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Library":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            entries = [
                LibraryEntry(projections=z[f"proj_{i}"], **e) for i, e in enumerate(meta["entries"])
            ]
            directions = z["directions"]
        return cls(meta["apix"], meta["box"], meta["resolution"], meta["angular_step"],
                   directions, entries)


def make_entry(
    key: str, name: str, source: str, kind: str, structure_path: Path,
    directions: np.ndarray, box: int, apix: float, resolution: float,
) -> LibraryEntry:
    coords, weights = load_atoms(structure_path)
    dim = max_dimension(coords)
    if dim > box * apix * 0.95:
        print(f"  warning: {key} is {dim:.0f} Å across but the box is {box * apix:.0f} Å; "
              "projections will be truncated. Increase --box.")
    proj = project_all(coords, weights, directions, box, apix, resolution)
    return LibraryEntry(key, name, source, kind, dim, proj)


def build_library(
    contaminants: list[Contaminant],
    targets: list[Path] = (),
    local_structures: dict[str, Path] | None = None,
    apix: float = 4.0,
    box: int = 96,
    resolution: float = 15.0,
    angular_step: float = 7.5,
    cache_dir: Path | None = None,
    skip_failed: bool = True,
) -> Library:
    """Render projections for each contaminant (downloaded unless given locally) and target."""
    directions = directions_for_step(angular_step)
    local_structures = local_structures or {}
    entries: list[LibraryEntry] = []
    fetch_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    for c in contaminants:
        print(f"[{c.key}] {c.name} ({c.pdb_id})", flush=True)
        try:
            path = local_structures.get(c.key) or fetch_structure(c.pdb_id, c.assembly, **fetch_kwargs)
            entries.append(make_entry(c.key, c.name, c.pdb_id, "contaminant", Path(path),
                                      directions, box, apix, resolution))
        except Exception as exc:  # noqa: BLE001
            if not skip_failed:
                raise
            print(f"  skipped: {exc}")

    for t in targets:
        t = Path(t)
        print(f"[target] {t.name}", flush=True)
        entries.append(make_entry(f"target:{t.stem}", f"Your target ({t.name})", str(t),
                                  "target", t, directions, box, apix, resolution))

    if not entries:
        raise RuntimeError("Library is empty: no structures could be loaded.")
    return Library(apix, box, resolution, angular_step, directions, entries)
