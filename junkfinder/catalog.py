"""Curated catalog of common cryo-EM contaminants, grouped by expression host.

Each entry points at a deposited structure. ``assembly`` selects the biological
assembly to download (RCSB ``<ID>-assembly<N>.cif``) so that oligomers such as
the AcrB trimer or the GroEL 14-mer are rendered correctly instead of the
crystallographic asymmetric unit.

Entries with ``verify=True`` have a PDB ID that has not yet been checked
against the RCSB (run ``junkfinder verify`` on downloaded files). An empty
``pdb_id`` means no structure has been chosen yet; supply one with
``--structure KEY=PATH``. ``refs`` holds DOIs reporting the protein as a
co-purifying contaminant.

Proteins reported as IMAC contaminants but too small (<~100 kDa) to give
useful 2D classes in cryo-EM are deliberately left out: SlyD, Fur, Crp, ArgE,
YadF, GlgA, ribosomal S15, BCCP (doi:10.1016/j.bbagen.2006.03.027).
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
    refs: tuple[str, ...] = field(default_factory=tuple)
    source_organism: str = "Escherichia coli"  # expected in the file, checked by 'verify'


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
        "Very common Ni-NTA contaminant (surface histidine clusters); co-elutes on SEC "
        "with FolE and GldA.", verify=True, refs=("10.1002/prot.24364", "10.1107/S2059798326010302"),
    ),
    Contaminant(
        "glmS", "GlmS / GFAT glucosamine-6-phosphate synthase (dimer)", "2J6H", 135, False,
        "Surface histidine clusters; persists even in LOBSTR strains.", verify=True,
        refs=("10.1016/j.bbagen.2006.03.027", "10.1002/bit.70244"),
    ),
    Contaminant(
        "pdh_E1", "Pyruvate dehydrogenase E1 (AceE, dimer)", "1L8A", 200, False,
        "Persists even in IMACulate strains.", verify=True, refs=("10.1002/bit.70244",),
    ),
    Contaminant(
        "ogdh_E2", "2-Oxoglutarate dehydrogenase E2 core (SucB, 24-mer cube)", "1E2O", 700, False,
        "Listed as ODO2; the cubic E2 core gives very distinctive classes.", verify=True,
        refs=("10.1016/j.bbagen.2006.03.027",),
    ),
    Contaminant(
        "folE", "GTP cyclohydrolase I (FolE, decamer)", "1GTP", 250, False,
        "Seen as off-target particles in an E. coli-expressed cryo-EM sample.", verify=True,
        refs=("10.1107/S2059798326010302",),
    ),
    Contaminant(
        "gldA", "Glycerol dehydrogenase (GldA, octamer)", "", 330, False,
        "Seen as off-target particles in an E. coli-expressed cryo-EM sample. "
        "Structure ID still to be chosen.", refs=("10.1107/S2059798326010302",),
    ),
    Contaminant(
        "groEL", "GroEL chaperonin (with GroES cap)", "1AON", 870, False,
        "Often co-purifies with misfolded targets. Entry includes GroES.",
    ),
    Contaminant(
        "katE", "Catalase HPII (KatE, tetramer)", "1GGE", 340, False,
        "Co-purifies with His-tagged proteins; has been crystallised in place of targets.",
        verify=True, refs=("10.1007/s00253-026-13820-2",),
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

# Standard test and calibration samples. They are not from the expression
# host: they get onto grids by carry-over from shared plunge-freezers,
# tweezers, glow-dischargers, autoloader cassettes and stock tubes in
# facilities where they are used routinely. No published reports of
# carry-over were found, so these entries carry no refs.
LAB: tuple[Contaminant, ...] = (
    Contaminant(
        "apoferritin", "Apoferritin (human heavy chain, 24-mer)", "7A4M", 480, False,
        "The standard resolution test sample. Hollow sphere ~12 nm; looks like "
        "bacterioferritin at low resolution.", source_organism="Homo sapiens",
    ),
    Contaminant(
        "beta_gal", "beta-Galactosidase (LacZ, tetramer)", "5A1A", 465, False,
        "Common test specimen; D2 tetramer gives distinctive rectangular views.",
    ),
    Contaminant(
        "aldolase", "Rabbit muscle aldolase (tetramer)", "5VY5", 160, False,
        "Small-particle benchmark.", source_organism="Oryctolagus cuniculus", verify=True,
    ),
    Contaminant(
        "t20s", "20S proteasome (T. acidophilum)", "1PMA", 700, False,
        "Barrel-shaped test specimen; side views look like four stacked rings.",
        source_organism="Thermoplasma acidophilum", verify=True,
    ),
)

CATALOGS: dict[str, tuple[Contaminant, ...]] = {"ecoli": ECOLI, "lab": LAB}
DEFAULT_CATALOGS = "ecoli,lab"


def get_catalog(names: str) -> tuple[Contaminant, ...]:
    """One catalog or several, comma-separated (e.g. 'ecoli,lab'); keys de-duplicated."""
    out: dict[str, Contaminant] = {}
    for name in (n.strip().lower() for n in names.split(",") if n.strip()):
        if name not in CATALOGS:
            raise SystemExit(f"Unknown catalog '{name}'. Available: {', '.join(sorted(CATALOGS))}")
        for c in CATALOGS[name]:
            out.setdefault(c.key, c)
    return tuple(out.values())
