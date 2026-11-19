"""Curated catalog of common cryo-EM contaminants, grouped by expression host.

Each entry points at a deposited structure. ``assembly`` selects the biological
assembly to download (RCSB ``<ID>-assembly<N>.cif``) so that oligomers such as
the AcrB trimer or the GroEL 14-mer are rendered correctly instead of the
crystallographic asymmetric unit.

Entries with ``verify=True`` have a PDB ID that should be double-checked against
RCSB before relying on results for that contaminant.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contaminant:
    key: str
    name: str
    pdb_id: str
    approx_kda: int
    membrane: bool
    notes: str = ""
    assembly: int = 1
    alt_ids: tuple[str, ...] = field(default_factory=tuple)
    verify: bool = False


ECOLI: tuple[Contaminant, ...] = (
    Contaminant(
        "cyo_bo3", "Cytochrome bo3 ubiquinol oxidase (CyoABCD)", "1FFT", 144, True,
        "Abundant in aerobic E. coli membranes; co-purifies with many His-tagged "
        "membrane proteins.", alt_ids=("6WTI",),
    ),
    Contaminant(
        "acrB", "AcrB multidrug efflux pump (trimer)", "1IWG", 340, True,
        "Classic Ni-NTA contaminant of E. coli membrane preps (His-rich C-terminus).",
    ),
    Contaminant(
        "cyt_bd", "Cytochrome bd-I oxidase (CydABX)", "6RKO", 110, True,
        "Upregulated under low aeration.", verify=True,
    ),
    Contaminant(
        "sdh", "Succinate dehydrogenase (SdhCDAB, trimer)", "1NEK", 360, True,
        verify=True,
    ),
    Contaminant(
        "atp_synthase", "F1Fo ATP synthase", "6OQR", 550, True, verify=True,
    ),
    Contaminant(
        "ompF", "OmpF porin (trimer)", "2OMF", 110, True,
        "Outer-membrane porin; survives most detergent extractions.",
    ),
    Contaminant(
        "arnA", "ArnA bifunctional enzyme (hexamer)", "1Z7E", 450, False,
        "Very common Ni-NTA contaminant (surface histidine clusters).", verify=True,
    ),
    Contaminant(
        "groEL", "GroEL chaperonin (with GroES cap)", "1AON", 870, False,
        "Often co-purifies with misfolded targets. Entry includes GroES.",
    ),
    Contaminant(
        "katE", "Catalase HPII (KatE, tetramer)", "1GGE", 340, False, verify=True,
    ),
    Contaminant(
        "dps", "Dps DNA-protection protein (12-mer)", "1DPS", 220, False,
        "Hollow sphere ~9 nm; stationary-phase cultures.", verify=True,
    ),
    Contaminant(
        "bfr", "Bacterioferritin (24-mer)", "1BCF", 450, False,
        "Hollow sphere ~12 nm, easy to confuse with apoferritin.", verify=True,
    ),
    Contaminant(
        "ribosome70S", "70S ribosome", "4YBB", 2500, False,
        "Large and dense; contaminates soluble preps and lysate-based grids.",
    ),
)

CATALOGS: dict[str, tuple[Contaminant, ...]] = {"ecoli": ECOLI}


def get_catalog(organism: str) -> tuple[Contaminant, ...]:
    try:
        return CATALOGS[organism.lower()]
    except KeyError:
        raise SystemExit(
            f"Unknown organism '{organism}'. Available: {', '.join(sorted(CATALOGS))}"
        ) from None
