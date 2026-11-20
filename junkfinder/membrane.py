"""Detergent micelles and nanodiscs around membrane proteins.

The environment is added as pseudo-atoms, so it goes through the same
projection code as the protein. A pseudo-atom stands for a small cube of
volume ``g^3`` whose contrast is a fraction ``f`` of protein contrast (1.0 =
protein, 0 = vitreous ice). Its weight is ``f * g^3 / VOLUME_PER_Z``, the
weight an atom of that volume would carry.

1. **Membrane plane.** Like OPM in spirit but much cruder: scan directions
   and slab positions for the ~30 Å slab that holds the most exposed
   hydrophobic residues. :func:`find_membrane` returns its normal and centre.
2. **Micelle.** A belt around the hydrophobic region with a rounded profile
   and lumpy, slightly elliptical width. It has a dense outer headgroup
   shell and a weak tail core. Enclosed lumens, such as porin channels, stay
   empty.
3. **Nanodisc.** A lipid bilayer disc (headgroup layers plus a tail core)
   inside a two-helix MSP belt with protein contrast. The disc is the
   smallest of MSP1D1, MSP1E3D1 and MSP2N2 that fits around the protein.

Contrast defaults are rough estimates from electron densities relative to
water: maltoside or phosphocholine heads are denser than protein, alkyl
tails slightly less dense than water. They are drawn per variant from
ranges, so the network does not learn one exact micelle.
"""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .project import VOLUME_PER_Z

# Crude hydrophobicity for membrane-slab detection (exposed residues only).
_HYDRO = {
    "ALA": 0.5, "ILE": 1.0, "LEU": 1.0, "MET": 0.8, "PHE": 1.0, "VAL": 1.0, "TRP": 0.6,
    "CYS": 0.3, "TYR": 0.2, "GLY": 0.0, "PRO": 0.0, "THR": -0.2, "SER": -0.3, "HIS": -0.5,
    "ASN": -0.8, "GLN": -0.8, "ASP": -1.5, "GLU": -1.5, "LYS": -1.5, "ARG": -1.5,
}

# Inner lipid radius (Å) of common MSP nanodiscs, from outer diameters of
# about 9.7, 12.9 and 16.5 nm minus the ~10 Å belt.
MSP_INNER_RADIUS = {"MSP1D1": 38.0, "MSP1E3D1": 54.0, "MSP2N2": 72.0}


@dataclass
class Membrane:
    normal: np.ndarray  # unit vector
    centre: np.ndarray  # point on the mid-plane (Å, same frame as the atoms)
    score: float


def read_atoms_and_residues(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(atom xyz, atom Z, CA xyz, residue names) in the file's frame (not centred)."""
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_waters()
    xyz, z, ca, names = [], [], [], []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
                z.append(atom.element.atomic_number)
            a = res.find_atom("CA", "*")
            if a is not None and res.name in _HYDRO:
                ca.append((a.pos.x, a.pos.y, a.pos.z))
                names.append(res.name)
    return np.asarray(xyz, float), np.asarray(z, float), np.asarray(ca, float), names


def _hemisphere(n: int) -> np.ndarray:
    i = np.arange(2 * n) + 0.5
    phi = np.arccos(1 - 2 * i / (2 * n))
    theta = np.pi * (1 + 5**0.5) * i
    d = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return d[d[:, 2] >= 0]


def find_membrane(ca: np.ndarray, names: list[str], thickness: float = 30.0,
                  n_dirs: int = 2000) -> Membrane:
    """Slab of the given thickness holding the most exposed hydrophobic residues."""
    tree = cKDTree(ca)
    counts = np.array([len(x) for x in tree.query_ball_point(ca, 10.0)])
    exposed = counts <= np.percentile(counts, 50)  # outer half of residues
    h = np.array([_HYDRO[n] for n in names])[exposed]
    p = ca[exposed]
    c0 = p.mean(0)
    p = p - c0

    def scan(dirs):
        best = (-np.inf, None, 0.0)
        for d in dirs:
            t = p @ d
            lo = np.floor(t.min()) - 1
            hist = np.bincount(np.floor(t - lo).astype(int), weights=h)
            win = np.convolve(hist, np.ones(int(thickness)), mode="valid")
            k = int(np.argmax(win))
            if win[k] > best[0]:
                best = (win[k], d, lo + k + thickness / 2)
        return best

    score, d, off = scan(_hemisphere(n_dirs))
    # Refine around the best direction.
    rng = np.random.default_rng(0)
    local = d + 0.05 * rng.normal(size=(400, 3))
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    s2, d2, off2 = scan(local)
    if s2 > score:
        score, d, off = s2, d2, off2
    return Membrane(normal=d / np.linalg.norm(d), centre=c0 + off * d, score=float(score))


def _frame(normal: np.ndarray) -> np.ndarray:
    """Rows: u, v, n (orthonormal, n = membrane normal)."""
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u), n])


@dataclass
class EnvParams:
    kind: str = "micelle"  # 'micelle' or 'nanodisc'
    grid: float = 2.0  # pseudo-atom spacing (Å)
    half_height: float = 21.0  # micelle / bilayer half-thickness (Å)
    width: float = 18.0  # micelle belt width from the protein surface (Å)
    head: float = 7.0  # headgroup shell / layer thickness (Å)
    head_contrast: float = 0.6  # fraction of protein contrast
    core_contrast: float = 0.1
    lumpiness: float = 0.15  # relative angular variation of the belt width
    ellipticity: float = 0.1
    msp: str | None = None  # nanodisc size; None = smallest that fits
    msp_contrast: float = 1.0
    seed: int = 0


def random_params(kind: str, rng: np.random.Generator) -> EnvParams:
    """Draw a plausible environment (ranges are deliberately wide)."""
    return EnvParams(
        kind=kind,
        half_height=rng.uniform(18, 24) if kind == "micelle" else rng.uniform(19, 22),
        width=rng.uniform(13, 24),
        head=rng.uniform(5, 9),
        head_contrast=rng.uniform(0.4, 0.9),
        core_contrast=rng.uniform(-0.1, 0.25),
        lumpiness=rng.uniform(0.05, 0.25),
        ellipticity=rng.uniform(0.0, 0.25),
        seed=int(rng.integers(1 << 31)),
    )


def build_environment(xyz: np.ndarray, mem: Membrane, p: EnvParams) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-atoms (coords in the protein frame, weights) for a micelle or nanodisc."""
    rng = np.random.default_rng(p.seed)
    R = _frame(mem.normal)
    local = (xyz - mem.centre) @ R.T  # protein in membrane frame: z along the normal
    g = p.grid
    zmax = p.half_height
    slab = local[np.abs(local[:, 2]) < zmax + 4]
    if len(slab) == 0:
        raise ValueError("no protein atoms near the membrane plane")
    axis_xy = slab[:, :2].mean(0)
    r_prot = np.linalg.norm(slab[:, :2] - axis_xy, axis=1).max()

    if p.kind == "nanodisc":
        sizes = sorted(MSP_INNER_RADIUS.items(), key=lambda kv: kv[1])
        if p.msp:
            r_in = MSP_INNER_RADIUS[p.msp]
        else:
            r_in = next((r for _, r in sizes if r >= r_prot + 6), sizes[-1][1])
        r_in *= 1 + rng.uniform(-0.04, 0.04)
        extent = r_in + 14
    else:
        extent = r_prot + p.width * (1 + p.lumpiness + p.ellipticity) + 6

    nxy = int(np.ceil(2 * extent / g)) + 1
    nz = int(np.ceil(2 * zmax / g)) + 1
    xs = axis_xy[0] - extent + g * np.arange(nxy)
    ys = axis_xy[1] - extent + g * np.arange(nxy)
    zs = -zmax + g * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)

    # Protein occupancy (within 3 Å of a heavy atom), with enclosed lumens filled per layer.
    tree = cKDTree(slab)
    d_prot, _ = tree.query(pts, distance_upper_bound=3.0)
    prot = np.isfinite(d_prot).reshape(nxy, nxy, nz)
    filled = np.stack([ndimage.binary_fill_holes(prot[:, :, k]) for k in range(nz)], -1)

    # Lateral distance (Å) from the protein in each layer.
    lat = np.stack([ndimage.distance_transform_edt(~filled[:, :, k]) * g if filled[:, :, k].any()
                    else np.full((nxy, nxy), np.inf) for k in range(nz)], -1)
    dx, dy = X - axis_xy[0], Y - axis_xy[1]
    ang = np.arctan2(dy, dx)
    rad = np.hypot(dx, dy)

    if p.kind == "micelle":
        phi1, phi2 = rng.uniform(0, 2 * np.pi, 2)
        mod = (1 + p.ellipticity * np.cos(2 * (ang - phi1))
               + p.lumpiness * np.cos(3 * (ang - phi2)) * rng.uniform(0.5, 1))
        profile = np.sqrt(np.clip(1 - (Z / zmax) ** 2, 0, 1))
        env = (~filled) & (lat <= p.width * mod * profile) & (profile > 0)
        # Headgroup shell: within `head` Å of the solvent (not of the protein).
        solvent = ~(env | filled)
        d_solv = ndimage.distance_transform_edt(~solvent) * g
        is_head = env & (d_solv <= p.head)
        contrast = np.where(is_head, p.head_contrast, p.core_contrast)
        extra = []
    else:
        disc = (~filled) & (rad <= r_in)
        env = disc
        is_head = disc & (np.abs(Z) > zmax - p.head)
        contrast = np.where(is_head, p.head_contrast, p.core_contrast)
        # MSP belt: two stacked amphipathic helices, ~10 Å thick, ~24 Å tall.
        belt = (rad > r_in) & (rad <= r_in + 10) & (np.abs(Z) <= 12)
        extra = [(belt, p.msp_contrast)]

    coords, weights = [pts[env.reshape(-1)]], [contrast[env].reshape(-1)]
    for mask, c in extra:
        coords.append(pts[mask.reshape(-1)])
        weights.append(np.full(int(mask.sum()), c))
    c = np.concatenate(coords)
    w = np.concatenate(weights) * g**3 / VOLUME_PER_Z
    # Small jitter so the regular grid does not alias in projections.
    c = c + rng.uniform(-g / 2, g / 2, size=c.shape)
    keep = w != 0
    return c[keep] @ R + mem.centre, w[keep]


class MembraneModel:
    """A membrane protein with its detected membrane plane; makes environment variants."""

    def __init__(self, path):
        self.xyz, self.z, ca, names = read_atoms_and_residues(path)
        self.membrane = find_membrane(ca, names)

    def variant(self, kind: str, rng: np.random.Generator | None = None,
                params: EnvParams | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(coords, weights) of protein + environment, centred on the total density.

        Pickers and 2D classification centre on everything they see, so the
        environment is included in the centre of mass.
        """
        if kind == "none":
            xyz, w = self.xyz, self.z
        else:
            p = params or random_params(kind, rng or np.random.default_rng())
            ec, ew = build_environment(self.xyz, self.membrane, p)
            xyz, w = np.concatenate([self.xyz, ec]), np.concatenate([self.z, ew])
        com = (xyz * w[:, None]).sum(0) / w.sum()
        return xyz - com, w
