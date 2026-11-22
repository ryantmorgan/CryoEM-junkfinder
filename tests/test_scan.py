"""scan: a synthetic micrograph with toy particles at known places is picked and labelled."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mrcfile = pytest.importorskip("mrcfile")

from junkfinder.classifier import train  # noqa: E402
from junkfinder.particles import SIGMA_E_300KV, Microscope, ParticleSimulator, build_views  # noqa: E402
from junkfinder.scan import scan_micrograph  # noqa: E402

from test_particles import APIX, BOX, toy  # noqa: E402


def make_micrograph(views, sim, n=320, spacing=64, dose=200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, n), np.float32)
    truth = []
    keys = list(views)
    for i, cy in enumerate(range(spacing, n - spacing // 2, spacing)):
        for j, cx in enumerate(range(spacing, n - spacing // 2, spacing)):
            key = keys[(i + j) % len(keys)]
            v = views[key][rng.integers(len(views[key]))]
            t[cy - BOX // 2:cy + BOX // 2, cx - BOX // 2:cx + BOX // 2] += v
            truth.append((cy, cx, key))
    phase = torch.from_numpy(SIGMA_E_300KV * sim.scope.delta_v * t)[None]
    k2 = sim._k2(n)
    contrast = torch.fft.ifft2(torch.fft.fft2(phase) * sim.ctf(torch.tensor([1.5e4]), k2)).real[0]
    img = -2 * contrast.numpy() * np.sqrt(dose * APIX**2) + rng.normal(size=(n, n))
    return img.astype(np.float32), truth


def test_scan_finds_and_labels_toy_particles(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    views = build_views({"rod": toy("rod", rng), "ball": toy("ball", rng)}, 30, BOX, APIX)
    # Neighbour particles at the box edge (labelled background) teach it to centre picks.
    sim = ParticleSimulator(views, Microscope(apix=APIX, dose=200.0, p_neighbour=0.3))
    model, _ = train(sim, steps=120, batch=32, units=8, lr=3e-3, log_every=40, verbose=False)
    ckpt = {"keys": sim.keys, "units": 8, "box": BOX, "apix": APIX}

    img, truth = make_micrograph(views, sim)
    path = tmp_path / "mic.mrc"
    with mrcfile.new(path) as m:
        m.set_data(img)
        m.voxel_size = APIX
    picks, probs, _, _ = scan_micrograph(path, model, ckpt, stride_A=2 * APIX, threshold=0.5)
    assert probs.shape[0] == 3
    hits = 0
    for cy, cx, key in truth:
        near = [p for p in picks if np.hypot(p.x - cx, p.y - cy) <= 6]
        hits += any(p.label == key for p in near)
    assert hits / len(truth) >= 0.6, (hits, len(truth), len(picks))
    assert len(picks) <= 1.5 * len(truth)  # no picks between particles
