"""classify: simulated particles written as a fake cryoSPARC job come back labelled."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mrcfile = pytest.importorskip("mrcfile")

from junkfinder.apply import (_fourier_resample_stack, classify_cs, prepare_particles,  # noqa: E402
                              summarise_predictions)
from junkfinder.classifier import train  # noqa: E402
from junkfinder.particles import Microscope, ParticleSimulator  # noqa: E402

from test_particles import APIX, BOX, toy  # noqa: E402
from junkfinder.particles import build_views  # noqa: E402


def test_prepare_pads_with_noise_and_normalises():
    rng = np.random.default_rng(0)
    imgs = rng.normal(5.0, 3.0, size=(4, 20, 20)).astype(np.float32)
    x = prepare_particles(imgs, 6.0, 6.0, 32, rng=rng)
    assert x.shape == (4, 32, 32)
    assert abs(float(x.std()) - 1.0) < 0.15  # noise-normalised, noise padding


def test_classify_fake_cryosparc_job(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    views = build_views({"rod": toy("rod", rng), "ball": toy("ball", rng)}, 30, BOX, APIX)
    sim = ParticleSimulator(views, Microscope(apix=APIX, dose=200.0))
    model, _ = train(sim, steps=60, batch=32, units=8, lr=3e-3, log_every=20, verbose=False)
    ckpt = {"keys": sim.keys, "units": 8, "box": BOX, "apix": APIX}

    labels = torch.arange(3).repeat_interleave(30)
    x, _ = sim.sample(len(labels), torch.Generator().manual_seed(5), labels=labels)
    # Write them as a cryoSPARC-like stack at half the pixel size (box 2x).
    fine = _fourier_resample_stack(x[:, 0].numpy(), APIX, APIX / 2)
    (tmp_path / "J1").mkdir()
    with mrcfile.new(tmp_path / "J1" / "particles.mrcs") as m:
        m.set_data(fine.astype(np.float32))
    cs = np.zeros(len(labels), dtype=[("uid", "<u8"), ("blob/path", "S32"), ("blob/idx", "<u4"),
                                      ("blob/psize_A", "<f4"), ("blob/shape", "<u4", (2,)),
                                      ("alignments2D/class", "<u4")])
    cs["uid"] = np.arange(len(cs)) + 1000
    cs["blob/path"] = b"J1/particles.mrcs"
    cs["blob/idx"] = np.arange(len(cs))
    cs["blob/psize_A"] = APIX / 2
    cs["blob/shape"] = fine.shape[-1]
    cs["alignments2D/class"] = labels.numpy()
    probs, pred = classify_cs(cs, tmp_path, model, ckpt, log=lambda *a, **k: None)
    assert probs.shape == (len(cs), 3)
    assert np.allclose(probs.sum(1), 1, atol=1e-4)
    assert (pred == labels.numpy()).mean() > 0.6
    assert "Per 2D class" in summarise_predictions(cs, pred, probs, sim.keys)
