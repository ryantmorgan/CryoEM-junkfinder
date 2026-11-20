"""Simulate single-particle images of library structures (needs PyTorch).

Physics (deliberately simple, every step is a knob):

1. Each structure is projected along many directions to a map of protein
   *thickness* in Å (atoms splatted with weight = atomic number, scaled so the
   integral equals the protein volume, ~1.21 Å^3/Da; hydrogens are added via
   a fixed 8% mass correction).
2. Thickness -> phase shift: ``phi = sigma_e * delta_V * t`` where ``delta_V``
   is the mean-inner-potential difference between protein and vitreous ice.
3. Weak-phase image with a CTF at random defocus:
   ``I = dose * (1 - 2 * F^-1[(sqrt(1-A^2) sin chi + A cos chi) * F[phi]])``.
4. Shot noise with variance ``dose`` per pixel, then the image is normalised
   by the noise level (like per-micrograph normalisation), so big particles
   stay brighter than small ones.

Augmentations: random in-plane rotation, centring error, slow ice-thickness
variation, and a neighbouring particle partly inside the box.

``delta_V`` and ``dose`` only enter as ``dose * delta_V^2`` (the SNR), so a
sweep over dose also covers uncertainty in ``delta_V``, detector DQE and
inelastic losses. Absolute accuracies should be read with that in mind; which
contaminants get confused with which is the more robust output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .project import VOLUME_PER_Z, fibonacci_directions, project

SIGMA_E_300KV = 6.53e-4  # interaction constant, rad / (V Å)
WAVELENGTH_300KV = 0.01969  # Å


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def thickness_views(
    coords: np.ndarray, weights: np.ndarray, directions: np.ndarray, box: int, apix: float,
) -> np.ndarray:
    """Projected protein thickness (Å) per direction: [n, box, box] float32.

    A light Gaussian (sigma ~0.45 px) anti-aliases the splat.
    """
    out = np.empty((len(directions), box, box), dtype=np.float32)
    for i, d in enumerate(directions):
        out[i] = project(coords, weights, d, box, apix, resolution=2 * apix)
    return out * np.float32(VOLUME_PER_Z / apix**2)


@dataclass
class Microscope:
    apix: float = 6.0
    dose: float | tuple[float, float] = 20.0  # effective e/Å^2 (after DQE etc.); range = per-particle
    delta_v: float = 2.0  # V, protein vs vitreous ice
    cs_mm: float = 2.7
    amp_contrast: float = 0.07
    defocus_um: tuple[float, float] = (0.8, 2.5)
    max_shift_A: float = 12.0
    ice_rms_A: float = 15.0  # rms of slow ice-thickness variation (Å)
    p_neighbour: float = 0.3


class ParticleSimulator:
    """Turns precomputed thickness views into noisy CTF-modulated particles."""

    def __init__(self, views: dict[str, np.ndarray], scope: Microscope, background: bool = True,
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.keys = list(views) + (["background"] if background else [])
        self.views = [torch.from_numpy(v).to(self.device) for v in views.values()]
        self.scope = scope
        self.box = next(iter(views.values())).shape[-1]
        self.background = background
        self.k2 = self._k2(self.box)  # [n, n] Å^-2
        self.k2_pad = self._k2(2 * self.box)  # CTF is applied on a 2x padded box (no wrap-around)

    def _k2(self, n: int) -> torch.Tensor:
        k = torch.fft.fftfreq(n, d=self.scope.apix, device=self.device)
        return (k[:, None] ** 2 + k[None, :] ** 2).float()

    def ctf(self, defocus_A: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        s, lam = self.scope, WAVELENGTH_300KV
        chi = (np.pi * lam * defocus_A[:, None, None] * k2
               - 0.5 * np.pi * s.cs_mm * 1e7 * lam**3 * k2**2)
        a = s.amp_contrast
        return np.sqrt(1 - a * a) * torch.sin(chi) + a * torch.cos(chi)

    def _pick_views(self, labels: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        imgs = torch.zeros(len(labels), self.box, self.box, device=self.device)
        for c, v in enumerate(self.views):
            idx = (labels == c).nonzero().flatten()
            if len(idx):
                j = torch.randint(len(v), (len(idx),), generator=gen, device=self.device)
                imgs[idx] = v[j]
        return imgs

    def _rotate_shift(self, imgs: torch.Tensor, shift_px: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        b = len(imgs)
        ang = torch.rand(b, generator=gen, device=self.device) * 2 * np.pi
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.zeros(b, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
        theta[:, :, 2] = shift_px * (2.0 / self.box)
        grid = F.affine_grid(theta, (b, 1, self.box, self.box), align_corners=False)
        return F.grid_sample(imgs[:, None], grid, align_corners=False, padding_mode="zeros")[:, 0]

    def sample(self, n: int, gen: torch.Generator, labels: torch.Tensor | None = None):
        """Return (images [n,1,box,box] float32, labels [n] long)."""
        s, box, dev = self.scope, self.box, self.device
        n_cls = len(self.keys)
        if labels is None:
            labels = torch.randint(n_cls, (n,), generator=gen, device=dev)
        n_struct = len(self.views)
        struct_labels = labels.to(dev).clone()
        is_bg = labels >= n_struct
        struct_labels[is_bg] = 0

        shift = (torch.rand(n, 2, generator=gen, device=dev) * 2 - 1) * (s.max_shift_A / s.apix)
        t = self._rotate_shift(self._pick_views(struct_labels, gen), shift, gen)
        t[is_bg] = 0.0

        # Neighbouring particle, centred 0.55-0.8 box away (background picks get one more often).
        p = torch.where(is_bg, torch.full((n,), max(0.5, s.p_neighbour), device=dev),
                        torch.full((n,), s.p_neighbour, device=dev))
        has_nb = torch.rand(n, generator=gen, device=dev) < p
        if has_nb.any():
            m = int(has_nb.sum())
            nb_lab = torch.randint(n_struct, (m,), generator=gen, device=dev)
            ang = torch.rand(m, generator=gen, device=dev) * 2 * np.pi
            r = (0.55 + 0.25 * torch.rand(m, generator=gen, device=dev)) * box
            nb_shift = torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 1)
            t[has_nb] += self._rotate_shift(self._pick_views(nb_lab, gen), nb_shift, gen)

        # Slow ice-thickness variation (random field low-passed to ~1/4 box).
        ice = torch.randn(n, box, box, generator=gen, device=dev)
        lp = torch.exp(-self.k2 * (box * s.apix / 4) ** 2 * 2)
        ice = torch.fft.ifft2(torch.fft.fft2(ice) * lp).real
        ice = ice / ice.flatten(1).std(1)[:, None, None].clamp_min(1e-6) * s.ice_rms_A
        # Ice is weaker than protein by the ice potential (~3.5 V vs delta_V); treat as extra
        # thickness of 'protein-equivalent' scaled down by 0.5 as a rough stand-in.
        phase = SIGMA_E_300KV * s.delta_v * (t - 0.5 * ice)

        lo, hi = s.defocus_um
        defocus = (lo + (hi - lo) * torch.rand(n, generator=gen, device=dev)) * 1e4
        q = box // 2
        padded = F.pad(phase[:, None], (q, q, q, q), mode="reflect")[:, 0]  # no step at the edge
        contrast = torch.fft.ifft2(torch.fft.fft2(padded) * self.ctf(defocus, self.k2_pad)).real
        contrast = contrast[:, q:q + box, q:q + box]
        if isinstance(s.dose, (tuple, list)):
            dose = s.dose[0] + (s.dose[1] - s.dose[0]) * torch.rand(n, generator=gen, device=dev)
        else:
            dose = torch.full((n,), float(s.dose), device=dev)
        dose_px = (dose * s.apix**2)[:, None, None]
        # Image in electrons is dose_px * (1 - 2 contrast) + noise(var = dose_px);
        # normalise by the noise sd: signal = -2 * contrast * sqrt(dose_px).
        img = -2.0 * contrast * dose_px.sqrt() + torch.randn(n, box, box, generator=gen, device=dev)
        return img[:, None].float(), labels.to(dev)


def build_views(
    structures: dict[str, tuple[np.ndarray, np.ndarray]],
    n_views: int, box: int, apix: float, rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Thickness views per structure: Fibonacci directions, or random ones if ``rng`` given."""
    out = {}
    for key, (xyz, w) in structures.items():
        dirs = fibonacci_directions(n_views) if rng is None else random_directions(n_views, rng)
        out[key] = thickness_views(xyz, w, dirs, box, apix)
    return out
