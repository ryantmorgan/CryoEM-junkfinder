"""finetune: real particles with a domain shift are learned while simulated classes are kept."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from junkfinder.classifier import evaluate, train  # noqa: E402
from junkfinder.finetune import finetune, real_confusion  # noqa: E402
from junkfinder.particles import Microscope, ParticleSimulator, build_views  # noqa: E402

from test_particles import APIX, BOX, toy  # noqa: E402


def test_finetune_fixes_a_domain_shift():
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    views = build_views({"rod": toy("rod", rng), "ball": toy("ball", rng)}, 30, BOX, APIX)
    sim = ParticleSimulator(views, Microscope(apix=APIX, dose=200.0))
    model, _ = train(sim, steps=80, batch=32, units=8, lr=3e-3, log_every=40, verbose=False)

    # "Real" rods with inverted contrast: a shift the simulator never produced.
    y = torch.zeros(200, dtype=torch.long)
    x, _ = sim.sample(200, torch.Generator().manual_seed(9), labels=y)
    x_real, y_real = (-x[:, 0]).numpy(), y.numpy()
    before = real_confusion(model, x_real[150:], y_real[150:], 3)
    model = finetune(model, sim, x_real[:150], y_real[:150], steps=120, batch=32, lr=2e-3,
                     log=lambda *a, **k: None)
    after = real_confusion(model, x_real[150:], y_real[150:], 3)
    acc_before, acc_after = before[0, 0] / before[0].sum(), after[0, 0] / after[0].sum()
    assert acc_after >= 0.9 and acc_after > acc_before
    # The simulated classes are still recognised (replay prevents forgetting).
    conf = evaluate(model, sim, per_class=40, batch=120)
    assert conf.diagonal().sum() / conf.sum() > 0.7
