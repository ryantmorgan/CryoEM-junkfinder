"""Synthetic end-to-end tests (no network needed).

Toy "proteins" are clouds of carbon atoms arranged in distinct shapes. Fake
class averages are made by projecting a toy from directions NOT on the library
grid, then rotating, shifting, rescaling, contrast-inverting and adding noise.
"""

from pathlib import Path

import mrcfile
import numpy as np
import pytest
from scipy.ndimage import rotate, shift as nd_shift

from junkfinder.catalog import Contaminant
from junkfinder.cryosparc_io import load_class_averages
from junkfinder.library import Library, build_library
from junkfinder.match import PolarGrid, best_match, reference_spectra, shift_grid
from junkfinder.project import fibonacci_directions, project
from junkfinder.screen import aligned_reference, screen
from junkfinder.structures import describe_structure, load_atoms

RNG = np.random.default_rng(0)


def blob_cloud(centres, spread=6.0, per=400, rng=RNG):
    return np.concatenate([rng.normal(c, spread, size=(per, 3)) for c in centres])


SHAPES = {
    # elongated, asymmetric
    "rod": [(0, 0, z) for z in (-40, -20, 0, 20, 40)] + [(15, 0, 40)],
    # L-shape
    "ell": [(x, 0, 0) for x in (-30, -10, 10, 30)] + [(30, 20, 0), (30, 40, 0)],
    # tetrahedral-ish compact
    "tet": [(25, 25, 25), (-25, -25, 25), (-25, 25, -25), (25, -25, -25), (0, 0, 0)],
    # flat triangle (trimer-like)
    "tri": [(40 * np.cos(a), 40 * np.sin(a), 0) for a in (0, 2.1, 4.2)] + [(0, 0, 10)],
}


def write_pdb(path: Path, xyz: np.ndarray) -> None:
    lines = []
    for i, (x, y, z) in enumerate(xyz, 1):
        lines.append(
            f"ATOM  {i % 100000:5d}  CA  ALA A{(i // 10) % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")


@pytest.fixture(scope="module")
def toy_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("toys")
    files = {}
    for name, centres in SHAPES.items():
        files[name] = d / f"{name}.pdb"
        write_pdb(files[name], blob_cloud(np.array(centres, float)))
    return files


@pytest.fixture(scope="module")
def library(toy_files):
    contaminants = [Contaminant(k, k, "TOY", 100, False) for k in ("rod", "ell", "tet")]
    local = {k: toy_files[k] for k in ("rod", "ell", "tet")}
    return build_library(contaminants, targets=[toy_files["tri"]], local_structures=local,
                         apix=4.0, box=48, resolution=15.0, angular_step=10.0)


def fake_class_average(path, rng, apix=1.6, box=128, noise=0.3):
    xyz, w = load_atoms(path)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    img = project(xyz, w, d, box, apix, 12.0)
    img = rotate(img, rng.uniform(0, 360), reshape=False, order=1)
    img = nd_shift(img, rng.uniform(-4, 4, size=2), order=1)
    img = img / img.std()
    img += rng.normal(0, noise, img.shape)
    return -img  # dark particle on light background, as in raw micrographs


def test_recovers_rotation_and_shift(toy_files):
    xyz, w = load_atoms(toy_files["ell"])
    dirs = fibonacci_directions(40)
    box, apix = 48, 4.0
    refs = np.stack([project(xyz, w, d, box, apix, 15.0) for d in dirs])
    grid = PolarGrid.for_box(box)
    spectra = reference_spectra(grid, refs)

    true_view, angle, sh = 7, 63.0, (2.0, -1.0)
    img = rotate(refs[true_view], -angle, reshape=False, order=1)
    img = nd_shift(img, (sh[1], sh[0]), order=1)

    m = best_match(grid, img, spectra, shift_grid(3))
    assert m.view == true_view
    assert m.score > 0.95
    assert abs((m.angle_deg - angle + 180) % 360 - 180) < 360 / grid.n_theta + 1e-6
    # scipy's rotate pivots on (n-1)/2, our grid on n//2: allow 1 px
    assert np.hypot(m.shift[0] - sh[0], m.shift[1] - sh[1]) <= 1.0


def test_aligned_reference_overlays(library, toy_files):
    rng = np.random.default_rng(1)
    img = fake_class_average(toy_files["rod"], rng, apix=4.0, box=48, noise=0.0)
    reports, prepared = screen(library, img[None], 4.0)
    r = reports[0]
    ref = aligned_reference(library, r.top_key, r.scores[r.top_key])
    cc = np.corrcoef(ref.ravel(), prepared[0].ravel())[0, 1]
    assert cc > 0.85


def test_end_to_end_identification(library, toy_files, tmp_path):
    rng = np.random.default_rng(2)
    truth, imgs = [], []
    for name in ("rod", "ell", "tet", "tri"):
        for _ in range(4):
            truth.append(name)
            imgs.append(fake_class_average(toy_files[name], rng))
    stack = np.stack(imgs).astype(np.float32)

    # Write as cryoSPARC would: MRC stack + .cs with pixel size, particles .cs with classes
    mrc = tmp_path / "J1_class_averages.mrc"
    with mrcfile.new(mrc) as m:
        m.set_data(stack)
        m.voxel_size = 1.0  # deliberately wrong; .cs must win
    cs = np.zeros(len(stack), dtype=[("uid", "<u8"), ("blob/psize_A", "<f4")])
    cs["blob/psize_A"] = 1.6
    np.save(tmp_path / "cs.npy", cs)
    parts = np.zeros(1000, dtype=[("alignments2D/class", "<u4")])
    parts["alignments2D/class"] = rng.integers(0, len(stack), 1000)
    np.save(tmp_path / "parts.npy", parts)

    imgs2, apix, counts = load_class_averages(mrc, tmp_path / "cs.npy", tmp_path / "parts.npy")
    assert apix == pytest.approx(1.6)
    assert counts.sum() == 1000

    reports, _ = screen(library, imgs2, apix, counts)
    predicted = [r.top_key.removeprefix("target:") for r in reports]
    accuracy = np.mean([p == t for p, t in zip(predicted, truth)])
    assert accuracy >= 0.85, list(zip(truth, predicted, [round(r.margin, 3) for r in reports]))
    for r, t in zip(reports, truth):
        if r.verdict == "contaminant":
            assert r.top_key == t
        if t == "tri" and r.verdict not in ("ambiguous", "no_match"):
            assert r.verdict == "target"


def test_library_roundtrip(library, tmp_path):
    path = tmp_path / "lib.npz"
    library.save(path)
    lib2 = Library.load(path)
    assert [e.key for e in lib2.entries] == [e.key for e in library.entries]
    assert lib2.entries[3].kind == "target"
    np.testing.assert_allclose(lib2.entries[0].projections, library.entries[0].projections)


def test_catalogs_combine_and_validate():
    from junkfinder.catalog import get_catalog

    both = get_catalog("ecoli, lab")
    keys = [c.key for c in both]
    assert "cyo_bo3" in keys and "apoferritin" in keys
    assert len(keys) == len(set(keys))
    assert get_catalog("ecoli,ecoli") == get_catalog("ecoli")
    with pytest.raises(SystemExit):
        get_catalog("plants")
    # Eukaryotic hosts share keys for the same complex; the first catalog listed wins.
    mixed = get_catalog("yeast,mammalian")
    keys = [c.key for c in mixed]
    assert len(keys) == len(set(keys))
    assert next(c for c in mixed if c.key == "ribosome80S").pdb_id == "4V88"


def test_describe_uses_sidecar_metadata_and_counts_nucleic_acids(tmp_path):
    """Assembly files lack organism records; 'verify' falls back to <ID>.json."""
    import json
    lines = []
    for i in range(10):  # 10 amino acids + 10 nucleotides
        lines.append(f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {i:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C")
    for i in range(10):
        lines.append(f"ATOM  {i + 11:5d}  P     A B{i + 1:4d}    {i:8.3f}{9:8.3f}{0:8.3f}  1.00  0.00           P")
    pdb = tmp_path / "9XYZ-assembly1.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    (tmp_path / "9XYZ.json").write_text(json.dumps(
        {"id": "9XYZ", "title": "Toy", "organisms": ["Escherichia coli"], "method": "EM", "resolution": 3.0}))
    d = describe_structure(pdb)
    assert d["organisms"] == ["Escherichia coli"]
    assert d["id"] == "9XYZ" and d["method"] == "EM"
    assert d["approx_kda"] == round(10 * 0.110 + 10 * 0.330)
