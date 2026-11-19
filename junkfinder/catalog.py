"""Curated catalog of common cryo-EM contaminants, grouped by expression host.

Each entry points at a deposited structure. ``assembly`` selects the biological
assembly to download (RCSB ``<ID>-assembly<N>.cif``) so that oligomers such as
the AcrB trimer or the GroEL 14-mer are rendered correctly instead of the
crystallographic asymmetric unit.

All current entries have been checked against the RCSB with ``junkfinder
verify``. Set ``verify=True`` on new entries until they have been. An empty
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
        "cyo_bo3", "Cytochrome bo3 ubiquinol oxidase (CyoABCD)", "6WTI", 144, True,
        "Abundant in aerobic E. coli membranes; the most abundant particle in an impure "
        "His-tag membrane prep. 6WTI is the complete cryo-EM model solved from that "
        "contaminant; the older X-ray structure 1FFT lacks ~25% of the complex.",
        alt_ids=("1FFT",), refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "acrB", "AcrB multidrug efflux pump (trimer)", "1IWG", 340, True,
        "Classic Ni-NTA contaminant of E. coli membrane preps (His-rich C-terminus).",
    ),
    Contaminant(
        "cyt_bd", "Cytochrome bd-I oxidase (CydABX)", "6RKO", 110, True,
        "Upregulated under low aeration.",
    ),
    Contaminant(
        "sdh", "Succinate dehydrogenase (SdhCDAB, trimer)", "7JZ2", 360, True,
        "Second most abundant particle in detergent-solubilised K12 membranes. 7JZ2 is "
        "the cryo-EM trimer from that sample; 1NEK (X-ray, assembly 2) is an alternative.",
        alt_ids=("1NEK",), refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "atp_synthase", "F1Fo ATP synthase", "6OQR", 550, True,
    ),
    Contaminant(
        "ompF", "OmpF porin (trimer)", "2OMF", 110, True,
        "Outer-membrane porin; survives most detergent extractions.",
        alt_ids=("6WTZ",), refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "ompC", "OmpC osmoporin (trimer)", "7JZ3", 110, True,
        "Most abundant particle in detergent-solubilised K12 membranes; looks very "
        "like OmpF (60% identical).", alt_ids=("2J1N",), refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "arnA", "ArnA bifunctional enzyme (hexamer)", "1Z7E", 450, False,
        "Very common Ni-NTA contaminant (surface histidine clusters); co-elutes on SEC "
        "with FolE and GldA.", refs=("10.1002/prot.24364", "10.1107/S2059798326010302"),
    ),
    Contaminant(
        "glmS", "GlmS / GFAT glucosamine-6-phosphate synthase (dimer)", "2J6H", 135, False,
        "Surface histidine clusters; persists even in LOBSTR strains.",
        refs=("10.1016/j.bbagen.2006.03.027", "10.1002/bit.70244", "10.1038/s41592-020-01021-2"),
    ),
    Contaminant(
        "pdh_E1", "Pyruvate dehydrogenase E1 (AceE, dimer)", "1L8A", 200, False,
        "Persists even in IMACulate strains.", refs=("10.1002/bit.70244",),
    ),
    Contaminant(
        "ogdh_E2", "2-Oxoglutarate dehydrogenase E2 core (SucB, 24-mer cube)", "1E2O", 700, False,
        "Listed as ODO2; the cubic E2 core gives very distinctive classes.",
        refs=("10.1016/j.bbagen.2006.03.027",),
    ),
    Contaminant(
        "folE", "GTP cyclohydrolase I (FolE, decamer)", "1GTP", 250, False,
        "Seen as off-target particles in an E. coli-expressed cryo-EM sample.",
        refs=("10.1107/S2059798326010302",),
    ),
    Contaminant(
        "gldA", "Glycerol dehydrogenase (GldA, octamer)", "43WP", 330, False,
        "Seen as off-target particles in an E. coli-expressed cryo-EM sample. "
        "43WP is a 3.2 Å cryo-EM octamer; 8GOA is an X-ray alternative.",
        alt_ids=("8GOA",), refs=("10.1107/S2059798326010302",),
    ),
    Contaminant(
        "groEL", "GroEL chaperonin (with GroES cap)", "1AON", 870, False,
        "Often co-purifies with misfolded targets. Entry includes GroES.",
    ),
    Contaminant(
        "katE", "Catalase HPII (KatE, tetramer)", "1GGE", 340, False,
        "Co-purifies with His-tagged proteins; has been crystallised in place of targets.",
        refs=("10.1007/s00253-026-13820-2",),
    ),
    Contaminant(
        "katG", "Catalase-peroxidase HPI (KatG, tetramer)", "7JZ6", 320, False,
        "Most abundant particle in size-selected (80-800 kDa) K12 lysate.",
        refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "gadB", "Glutamate decarboxylase (GadB, hexamer)", "7JZH", 320, False,
        "Second most abundant particle in size-selected K12 lysate; strongly induced "
        "in acidic or stationary-phase cultures.", alt_ids=("1PMM",), refs=("10.1038/s41592-020-01021-2",),
    ),
    Contaminant(
        "dps", "Dps DNA-protection protein (12-mer)", "1DPS", 220, False,
        "Hollow sphere ~9 nm; stationary-phase cultures.",
    ),
    Contaminant(
        "bfr", "Bacterioferritin (24-mer)", "1BCF", 450, False,
        "Hollow sphere ~12 nm, easy to confuse with apoferritin.",
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
        "apoferritin", "Apoferritin (mouse heavy chain, 24-mer)", "7A4M", 480, False,
        "The standard resolution test sample (mouse and human heavy chains look the "
        "same at this resolution). Hollow sphere ~12 nm; looks like bacterioferritin "
        "at low resolution.", source_organism="Mus musculus",
    ),
    Contaminant(
        "beta_gal", "beta-Galactosidase (LacZ, tetramer)", "5A1A", 465, False,
        "Common test specimen; D2 tetramer gives distinctive rectangular views.",
    ),
    Contaminant(
        "aldolase", "Rabbit muscle aldolase (tetramer)", "5VY5", 160, False,
        "Small-particle benchmark.", source_organism="Oryctolagus cuniculus",
    ),
    Contaminant(
        "t20s", "20S proteasome (T. acidophilum)", "1PMA", 700, False,
        "Barrel-shaped test specimen; side views look like four stacked rings.",
        source_organism="Thermoplasma acidophilum",
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
