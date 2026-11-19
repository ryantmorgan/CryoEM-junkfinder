"""Rotation- and shift-tolerant matching of 2D class averages to reference projections.

Both images are resampled onto a polar grid about the box centre. For a fixed
centre, the normalised cross-correlation over every in-plane rotation is then a
1D circular correlation along theta, which we evaluate for all references at
once with FFTs. Small translations are handled by re-sampling the class average
about a grid of shifted centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .project import SIGMA_FACTOR


# ---------------------------------------------------------------- image prep

def fourier_resample(img: np.ndarray, src_apix: float, dst_apix: float) -> np.ndarray:
    """Change pixel size by Fourier cropping/padding (square images)."""
    n = img.shape[-1]
    m = int(round(n * src_apix / dst_apix))
    m += m % 2  # keep even
    if m == n:
        return img.astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(img))
    out = np.zeros((m, m), dtype=complex)
    k = min(n, m)
    s_in, s_out = (n - k) // 2, (m - k) // 2
    out[s_out:s_out + k, s_out:s_out + k] = F[s_in:s_in + k, s_in:s_in + k]
    res = np.fft.ifft2(np.fft.ifftshift(out)).real
    return res * (m * m) / (n * n)


def crop_or_pad(img: np.ndarray, box: int) -> np.ndarray:
    n = img.shape[-1]
    if n == box:
        return img
    if n > box:
        s = (n - box) // 2
        return img[s:s + box, s:s + box]
    out = np.zeros((box, box), dtype=img.dtype)
    s = (box - n) // 2
    out[s:s + n, s:s + n] = img
    return out


def detect_inverted(img: np.ndarray) -> bool:
    """True if the particle appears darker than background (needs inverting)."""
    n = img.shape[-1]
    yy, xx = np.mgrid[:n, :n] - n // 2
    r = np.hypot(xx, yy)
    centre = img[r < n * 0.2].mean()
    ring = img[(r > n * 0.35) & (r < n * 0.45)].mean()
    return bool(centre < ring)


def prepare_class_average(
    img: np.ndarray,
    src_apix: float,
    dst_apix: float,
    box: int,
    resolution: float,
    invert: bool | None = None,
) -> np.ndarray:
    """Bring a class average onto the library's pixel size, box and resolution."""
    out = fourier_resample(np.asarray(img, dtype=np.float64), src_apix, dst_apix)
    out = crop_or_pad(out, box)
    if invert is None:
        invert = detect_inverted(out)
    if invert:
        out = -out
    return gaussian_filter(out, SIGMA_FACTOR * resolution / dst_apix, mode="constant")


# --------------------------------------------------------------- polar maths

@dataclass(frozen=True)
class PolarGrid:
    box: int
    radius: float  # px
    n_r: int
    n_theta: int

    @classmethod
    def for_box(cls, box: int, radius: float | None = None, n_theta: int = 128) -> "PolarGrid":
        radius = radius if radius is not None else 0.45 * box
        return cls(box, float(radius), int(np.ceil(radius)), n_theta)

    @property
    def radii(self) -> np.ndarray:
        return (np.arange(self.n_r) + 0.5) * self.radius / self.n_r

    @property
    def thetas(self) -> np.ndarray:
        return np.arange(self.n_theta) * 2 * np.pi / self.n_theta

    @property
    def weights(self) -> np.ndarray:
        return self.radii  # area element r dr dtheta

    def sample(self, imgs: np.ndarray, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Polar resampling of one image or a stack -> [..., n_r, n_theta]."""
        imgs = np.asarray(imgs)
        single = imgs.ndim == 2
        if single:
            imgs = imgs[None]
        c = self.box // 2
        r, t = np.meshgrid(self.radii, self.thetas, indexing="ij")
        cols = c + shift[0] + r * np.cos(t)
        rows = c + shift[1] + r * np.sin(t)
        out = np.stack([
            map_coordinates(im, [rows, cols], order=1, mode="constant") for im in imgs
        ])
        return out[0] if single else out

    def normalise(self, p: np.ndarray) -> np.ndarray:
        """Zero weighted mean, unit weighted norm over the disc."""
        w = self.weights[:, None]
        wsum = w.sum() * self.n_theta
        mean = (p * w).sum(axis=(-2, -1), keepdims=True) / wsum
        p = p - mean
        norm = np.sqrt((p * p * w).sum(axis=(-2, -1), keepdims=True))
        return p / np.maximum(norm, 1e-12)


def reference_spectra(grid: PolarGrid, projections: np.ndarray) -> np.ndarray:
    """Pre-compute conj(FFT_theta) of normalised polar references: [n_ref, n_r, n_k]."""
    polar = grid.normalise(grid.sample(projections))
    return np.conj(np.fft.rfft(polar, axis=-1)).astype(np.complex64)


def shift_grid(max_shift: float, step: float = 1.0) -> list[tuple[float, float]]:
    s = np.arange(-max_shift, max_shift + 1e-9, step)
    return [(float(x), float(y)) for y in s for x in s if np.hypot(x, y) <= max_shift + 1e-9]


@dataclass
class MatchResult:
    score: float
    view: int
    angle_deg: float
    shift: tuple[float, float]


def image_spectra(grid: PolarGrid, image: np.ndarray, shifts: list[tuple[float, float]]) -> np.ndarray:
    """Weighted FFT_theta of the normalised polar image at each shift: [n_shift, n_r, n_k]."""
    w = grid.weights.astype(np.float32)[None, :, None]
    polar = grid.normalise(np.stack([grid.sample(image, s) for s in shifts]))
    return np.fft.rfft(polar, axis=-1).astype(np.complex64) * w


def best_match(
    grid: PolarGrid,
    image: np.ndarray | None,
    ref_spectra: np.ndarray,
    shifts: list[tuple[float, float]],
    img_spectra: np.ndarray | None = None,
) -> MatchResult:
    """Max NCC of ``image`` against all references over rotations and shifts.

    To overlay the reference on ``image``: ``scipy.ndimage.rotate(ref, -angle_deg)``
    then shift by ``shift`` (x, y) px (see ``screen.aligned_reference``).
    Pass ``img_spectra`` (from ``image_spectra``) to reuse work across libraries.
    """
    C = img_spectra if img_spectra is not None else image_spectra(grid, image, shifts)
    # cross[s, n, k] = sum_r C[s,r,k] * conj(P[n,r,k])
    cross = np.einsum("srk,nrk->snk", C, ref_spectra, optimize=True)
    cc = np.fft.irfft(cross, n=grid.n_theta, axis=-1)  # [S, N, theta]
    s, n, t = np.unravel_index(int(np.argmax(cc)), cc.shape)
    return MatchResult(float(cc[s, n, t]), int(n), float(np.rad2deg(grid.thetas[t])), shifts[s])
