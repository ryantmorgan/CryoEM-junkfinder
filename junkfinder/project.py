"""Simulate 2D projections of atomic models.

Each atom is treated as a Gaussian, so the projection of the 3D density along
the beam is a sum of 2D Gaussians. We splat projected atom positions onto a
grid (bilinear) and blur once with the resolution-dependent sigma, which gives
the same result as projecting a molmap-style density map, at a fraction of the
cost.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Same sigma/resolution relation as UCSF Chimera's molmap (sigma = 0.225 * res).
SIGMA_FACTOR = 1.0 / (np.pi * np.sqrt(2.0))


VOLUME_PER_Z = 2.0 * 1.08 * 1.21  # Å^3 per unit atomic number (mass ≈ 2Z Da, +8% H)

def fibonacci_directions(n: int, hemisphere: bool = False) -> np.ndarray:
    """Approximately uniform unit vectors on the sphere (or upper hemisphere)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n if hemisphere else 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + 5**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def directions_for_step(step_deg: float) -> np.ndarray:
    """Directions on the full sphere spaced roughly ``step_deg`` apart.

    The full sphere is required: projections from v and -v are mirror images,
    and we do not allow mirroring during matching (class averages are not
    mirrored, and mirroring would match the wrong-handed object).
    """
    step = np.deg2rad(step_deg)
    n = max(8, int(np.ceil(4.0 * np.pi / (step * step))))
    return fibonacci_directions(n)


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ v = +z (in-plane angle is arbitrary)."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def splat(xy: np.ndarray, w: np.ndarray, box: int, apix: float) -> np.ndarray:
    """Bilinear accumulate weighted 2D points (Å, origin at box centre)."""
    img = np.zeros((box, box), dtype=np.float64)
    # Array index (row=y, col=x); centre at box//2 to match FFT conventions.
    px = xy[:, 0] / apix + box // 2
    py = xy[:, 1] / apix + box // 2
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    fx = px - x0
    fy = py - y0
    for dx, dy, f in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < box) & (yi >= 0) & (yi < box)
        np.add.at(img, (yi[ok], xi[ok]), (w * f)[ok])
    return img


def project(
    coords: np.ndarray,
    weights: np.ndarray,
    direction: np.ndarray,
    box: int,
    apix: float,
    resolution: float,
) -> np.ndarray:
    """Projection of the model viewed along ``direction``, low-passed to ``resolution`` Å."""
    R = rotation_to_z(direction)
    xyz = coords @ R.T
    img = splat(xyz[:, :2], weights, box, apix)
    return gaussian_filter(img, SIGMA_FACTOR * resolution / apix, mode="constant")


def project_all(
    coords: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray,
    box: int,
    apix: float,
    resolution: float,
) -> np.ndarray:
    return np.stack(
        [project(coords, weights, d, box, apix, resolution) for d in directions]
    ).astype(np.float32)


def max_dimension(coords: np.ndarray) -> float:
    """Longest extent of the model (Å), via its principal axes."""
    c = coords - coords.mean(0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    proj = c @ vt.T
    return float((proj.max(0) - proj.min(0)).max())
