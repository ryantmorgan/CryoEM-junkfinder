"""Detergent micelles and nanodiscs around membrane proteins.

The environment is added as pseudo-atoms, so it goes through the same
projection code as the protein. A pseudo-atom stands for a small cube of
volume ``g^3`` whose contrast is a fraction ``f`` of protein contrast (1.0 =
protein, 0 = vitreous ice). Its weight is ``f * g^3 / VOLUME_PER_Z``, the
weight an atom of that volume would carry.

1. **Membrane plane.** Like OPM in spirit but much cruder: scan directions
   and slab positions for the ~30 Å slab that holds the most exposed
   hydrophobic residues. :func:`find_membrane` returns its normal and centre.
2. **Micelle.** A smooth oblate ellipsoid centred on the hydrophobic region,
   fitted to 2D classes of bo3 in DDM (``junkfinder fit-env``). It is
   100-135 Å across (always at least 8 Å beyond large transmembrane domains),
   54-74 Å thick, slightly elliptical
   in-plane and only faintly lumpy. A headgroup shell slightly denser than
   the core keeps it evenly filled. Enclosed lumens, such as porin channels, stay
   empty.
3. **Nanodisc.** A lipid bilayer disc (headgroup layers plus a tail core)
   with a rounded rim, wrapped by a 10 Å MSP scaffold. Outer diameters are
   100, 130 and 170 Å for MSP1D1, MSP1E3D1 and MSP2N2. Half the time the
   smallest that fits is used, otherwise a random size that fits. Small proteins may sit off-centre, as
   cs-create-nanodisc-templates (O. Clarke) also allows.

4. **SMA lipid particle.** Like a nanodisc, but sized to the protein (95-125
   Å across) with a thin, diffuse polymer rim as bright as the lipid heads.
   It looks evenly filled with no ring, as in 2D classes of bo3 in SMA.

Contrasts are rough estimates from electron densities relative to vitreous
ice (protein = 1).
- DDM: maltose heads (~0.5 e/Å^3) are denser than protein (~0.43), and
  dodecyl tails (~0.27) are slightly less dense than ice (~0.31). The two
  occupy roughly equal volumes, so a micelle averages ~0.6x protein. Real
  2D classes of bo3 in DDM show it nearly as bright as the protein and
  evenly filled, with no bright rim. Heads and tails also intermix, and at
  ~10 Å the classes cannot separate them. The fit to bo3/DDM gave heads at
  1.16x and the core at 0.48x protein contrast. Micelles draw heads from
  0.85-1.4x and cores from 0.3-0.65x.
- Lipid bilayers have a deeper methyl trough (cs-create-nanodisc-templates
  uses heads 1.0, core -0.5). Real 2D classes of bo3 in MSP1E3D1 discs look
  evenly filled, though, with no bright scaffold ring. So nanodiscs draw
  heads from 0.6-1.0x, cores from 0.15-0.5x and the scaffold from 0.3-0.7x.
Drawing them per variant keeps the network from learning one exact
environment.
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

# Outer diameters (Å) of common MSP nanodiscs, as used by
# github.com/olibclarke/cs-create-nanodisc-templates. The scaffold wall is
# ~10 Å, so the lipid radius is D/2 - 10.
MSP_OUTER_DIAMETER = {"MSP1D1": 100.0, "MSP1E3D1": 130.0, "MSP2N2": 170.0}
MSP_WALL = 10.0


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
    kind: str = "micelle"  # 'micelle', 'nanodisc' or 'sma'
    grid: float = 2.0  # pseudo-atom spacing (Å)
    half_height: float = 21.0  # micelle / bilayer half-thickness (Å)
    width: float = 18.0  # micelle reach beyond the TM domain if no diameter is given (Å)
    diameter: float | None = None  # overall micelle diameter (Å)
    min_width: float = 8.0  # micelle extends at least this far beyond the TM domain (Å)
    head: float = 7.0  # headgroup shell / layer thickness (Å)
    head_contrast: float = 0.6  # fraction of protein contrast
    core_contrast: float = 0.1
    lumpiness: float = 0.15  # relative angular variation of the micelle radius
    ellipticity: float = 0.1
    msp: str | None = None  # nanodisc size; None = smallest that fits
    msp_contrast: float = 1.0  # scaffold (MSP belt, or SMA polymer) contrast
    wall: float = MSP_WALL  # scaffold thickness (Å)
    msp_half_height: float = 15.0  # scaffold belt height / 2 (Å)
    offcentre: float = 0.0  # 0-1: fraction of the free space the protein is shifted by
    seed: int = 0


def typical_params(kind: str) -> EnvParams:
    """Middle of the random ranges (used for class-average libraries)."""
    if kind == "micelle":
        # Fitted to 33 class averages of E. coli bo3 in DDM ('junkfinder fit-env').
        return EnvParams(kind=kind, half_height=32.5, diameter=120, head=9.5, head_contrast=1.16,
                         core_contrast=0.48, lumpiness=0.03, ellipticity=0.13)
    if kind == "sma":
        return EnvParams(kind=kind, half_height=22, diameter=110, head=8, head_contrast=0.8,
                         core_contrast=0.45, msp_contrast=0.8, wall=5.0)
    return EnvParams(kind=kind, half_height=21, head=8, head_contrast=0.8, core_contrast=0.3,
                     msp_contrast=0.5)


def random_params(kind: str, rng: np.random.Generator) -> EnvParams:
    """Draw a plausible environment (ranges are deliberately wide)."""
    sizes = list(MSP_OUTER_DIAMETER)
    if kind == "sma":
        # SMA lipid particles: native-lipid disc sized to the protein, with a thin,
        # diffuse polymer rim as bright as the lipid heads (no distinct ring).
        head_c = rng.uniform(0.6, 1.0)
        return EnvParams(kind=kind, half_height=rng.uniform(20, 25), diameter=rng.uniform(95, 125),
                         head=rng.uniform(6, 10), head_contrast=head_c,
                         core_contrast=rng.uniform(0.3, 0.6),
                         msp_contrast=head_c * rng.uniform(0.8, 1.1), wall=rng.uniform(3, 7),
                         offcentre=rng.uniform(0, 0.3), seed=int(rng.integers(1 << 31)))
    return EnvParams(
        kind=kind,
        **(dict(half_height=rng.uniform(27, 37), diameter=rng.uniform(100, 135),
                head=rng.uniform(7, 12), head_contrast=rng.uniform(0.85, 1.4),
                core_contrast=rng.uniform(0.3, 0.65), lumpiness=rng.uniform(0.0, 0.06),
                ellipticity=rng.uniform(0.0, 0.2))
           if kind == "micelle" else
           dict(half_height=rng.uniform(19, 24), head=rng.uniform(6, 10),
                head_contrast=rng.uniform(0.6, 1.0), core_contrast=rng.uniform(0.15, 0.5),
                msp_contrast=rng.uniform(0.3, 0.7))),
        # Usually the smallest disc that fits; sometimes a bigger one (then None -> fit).
        msp=None if rng.random() < 0.5 else sizes[int(rng.integers(len(sizes)))],
        offcentre=rng.uniform(0, 1) ** 2 if kind == "nanodisc" else 0.0,
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
    core = local[np.abs(local[:, 2]) < 15]
    core = core if len(core) else slab
    r_tm = np.percentile(np.linalg.norm(core[:, :2] - axis_xy, axis=1), 98)
    # Micelle: in-plane radius from the overall diameter, but never thinner than min_width.
    r_mic = max((p.diameter / 2) if p.diameter else r_tm + p.width, r_tm + p.min_width)
    disc_xy = axis_xy

    if p.kind in ("nanodisc", "sma"):
        if p.kind == "sma":
            r_out = p.diameter / 2 if p.diameter else r_tm + p.width
            r_in = max(r_out - p.wall, r_tm + p.min_width)
        else:
            radii = sorted(d / 2 - p.wall for d in MSP_OUTER_DIAMETER.values())
            fit = next((r for r in radii if r >= r_tm), radii[-1])
            r_in = max(MSP_OUTER_DIAMETER[p.msp] / 2 - p.wall, fit) if p.msp else fit
        r_in *= 1 + rng.uniform(-0.04, 0.04)
        # Small proteins in big discs often sit off-centre.
        slack = max(r_in - r_tm - 4, 0.0)
        a = rng.uniform(0, 2 * np.pi)
        disc_xy = axis_xy + p.offcentre * slack * np.array([np.cos(a), np.sin(a)])
        extent = np.linalg.norm(disc_xy - axis_xy) + r_in + p.wall + 4
    else:
        extent = max(r_prot, r_mic * (1 + p.ellipticity + p.lumpiness)) + 6

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
    dx, dy = X - axis_xy[0], Y - axis_xy[1]
    ang = np.arctan2(dy, dx)
    rad = np.hypot(X - disc_xy[0], Y - disc_xy[1])

    if p.kind == "micelle":
        # Smooth oblate ellipsoid centred on the TM region (as in 2D classes of
        # e.g. bo3 in DDM), with slight in-plane ellipticity and faint lumpiness.
        phi1, phi2 = rng.uniform(0, 2 * np.pi, 2)
        r_xy = r_mic * (1 + p.ellipticity * np.cos(2 * (ang - phi1))
                        + p.lumpiness * np.cos(3 * (ang - phi2)))
        env = (~filled) & ((rad / r_xy) ** 2 + (Z / zmax) ** 2 <= 1)
        # Headgroup shell: within `head` Å of the solvent (not of the protein).
        solvent = ~(env | filled)
        d_solv = ndimage.distance_transform_edt(~solvent) * g
        is_head = env & (d_solv <= p.head)
        contrast = np.where(is_head, p.head_contrast, p.core_contrast)
        extra = []
    else:
        # Rounded rim (as in cs-create-nanodisc-templates): the bilayer edge is a
        # half-circle of radius zmax in the r-z plane, and the MSP scaffold wraps it.
        shoulder = max(r_in - zmax, 0.0)
        d_rim = np.where(rad > shoulder, np.hypot(rad - shoulder, Z), np.abs(Z))
        disc = (~filled) & (d_rim <= zmax)
        env = disc
        is_head = disc & (d_rim > zmax - p.head)
        contrast = np.where(is_head, p.head_contrast, p.core_contrast)
        belt = (~filled) & (rad > shoulder) & (d_rim > zmax) & (d_rim <= zmax + p.wall)
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
