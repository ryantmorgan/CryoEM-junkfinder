"""Particle simulator and classifier (skipped when PyTorch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from junkfinder.classifier import ParticleNet, evaluate, train  # noqa: E402
from junkfinder.particles import Microscope, ParticleSimulator, build_views  # noqa: E402

BOX, APIX = 32, 6.0


def toy(shape: str, rng) -> tuple[np.ndarray, np.ndarray]:
    if shape == "rod":
        centres = [(0, 0, z) for z in np.linspace(-50, 50, 6)]
    else:  # compact ball
        centres = [(0, 0, 0)]
    xyz = np.concatenate([rng.normal(c, 12.0, size=(3000, 3)) for c in centres])
    return xyz - xyz.mean(0), np.full(len(xyz), 6.0)


@pytest.fixture(scope="module")
def views():
    rng = np.random.default_rng(0)
    structs = {"rod": toy("rod", rng), "ball": toy("ball", rng)}
    return build_views(structs, 30, BOX, APIX)


def test_thickness_integral_matches_volume(views):
    # 3000 carbon-like atoms (Z=6) at 2.61 Å^3 per unit Z
    integral = views["ball"][0].sum() * APIX**2
    assert integral == pytest.approx(3000 * 6 * 2.0 * 1.08 * 1.21, rel=0.05)


def test_ctf_at_zero_frequency_is_amplitude_contrast(views):
    sim = ParticleSimulator(views, Microscope(apix=APIX))
    c = sim.ctf(torch.tensor([1.5e4]), sim.k2)
    assert float(c[0, 0, 0]) == pytest.approx(0.07, abs=1e-6)


def test_background_is_unit_noise_and_signal_scales_with_dose(views):
    gen = torch.Generator().manual_seed(0)
    quiet = Microscope(apix=APIX, p_neighbour=0.0, ice_rms_A=0.0, dose=20.0)
    sim = ParticleSimulator(views, quiet)
    bg = sim.keys.index("background")
    x, y = sim.sample(64, gen, labels=torch.full((64,), bg))
    # background picks may still get a neighbour (p >= 0.5), so check the median sd
    assert float(x.flatten(1).std(1).median()) == pytest.approx(1.0, abs=0.15)

    def signal_sd(dose):
        s = ParticleSimulator(views, Microscope(apix=APIX, p_neighbour=0.0, ice_rms_A=0.0, dose=dose))
        g = torch.Generator().manual_seed(1)
        xs, _ = s.sample(256, g, labels=torch.zeros(256, dtype=torch.long))
        return float((xs.flatten(1).var(1) - 1).clamp_min(0).mean().sqrt())

    assert signal_sd(40.0) / signal_sd(10.0) == pytest.approx(2.0, rel=0.25)


def test_classifier_learns_toy_shapes(views):
    torch.set_num_threads(2)
    sim = ParticleSimulator(views, Microscope(apix=APIX, dose=200.0))
    model, log = train(sim, steps=60, batch=32, units=8, lr=3e-3, log_every=20, verbose=False)
    conf = evaluate(model, sim, per_class=40, batch=120)
    assert conf.diagonal().sum() / conf.sum() > 0.8
    assert isinstance(model, ParticleNet)
