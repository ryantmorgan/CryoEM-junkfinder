"""Membrane-plane detection and micelle/nanodisc pseudo-atoms (no PyTorch needed)."""

from pathlib import Path

import numpy as np
import pytest

from junkfinder.membrane import EnvParams, MembraneModel, _frame, build_environment
from junkfinder.project import project

OMPF = Path(__file__).resolve().parents[1] / "structures" / "2OMF-assembly1.cif.gz"
pytestmark = pytest.mark.skipif(not OMPF.exists(), reason="structures/ not present")


@pytest.fixture(scope="module")
def ompf():
    return MembraneModel(OMPF)


def test_normal_is_the_trimer_axis(ompf):
    c = ompf.xyz - ompf.xyz.mean(0)
    ev, vec = np.linalg.eigh(c.T @ c)
    axis = vec[:, 0]  # flat trimer: the unique (smallest-spread) axis is the 3-fold
    assert abs(axis @ ompf.membrane.normal) > np.cos(np.radians(10))


@pytest.mark.parametrize("kind", ["micelle", "nanodisc"])
def test_environment_is_a_belt_in_the_membrane_plane(ompf, kind):
    p = EnvParams(kind=kind)
    c, w = build_environment(ompf.xyz, ompf.membrane, p)
    assert len(c) > 1000
    z = (c - ompf.membrane.centre) @ ompf.membrane.normal
    assert np.abs(z).max() <= p.half_height + p.grid
    # Seen down the normal, the environment surrounds the protein rather than covering it.
    R = _frame(ompf.membrane.normal)
    img = project(c, w, R[2], 96, 2.0, resolution=4.0)
    prot = project(ompf.xyz, ompf.z, R[2], 96, 2.0, resolution=4.0)
    ring = img[prot > 0.2 * prot.max()].mean()
    outside = img[(prot < 0.01 * prot.max()) & (img > 0)].mean()
    assert ring < 0.5 * outside


def test_variant_is_centred(ompf):
    xyz, w = ompf.variant("micelle", np.random.default_rng(0))
    com = (xyz * w[:, None]).sum(0) / w.sum()
    assert np.linalg.norm(com) < 1e-6
    assert len(xyz) > len(ompf.xyz)
