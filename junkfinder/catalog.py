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
        "Listed as ODO2; the cubic E2 core gives very distinctive classes. Identified "
        "by cryo-EM (as DLST) co-purifying with designed proteins.",
        refs=("10.1016/j.bbagen.2006.03.027", "10.1016/j.yjsbx.2025.100120"),
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

# Mammalian (HEK293 / Expi293) expression. Three groups:
# 1. Reported by cryo-EM + mass spectrometry (refs given).
# 2. The other endogenously biotinylated carboxylases. They should bind
#    Strep-Tactin / streptavidin for the same reason PCC does, but no cryo-EM
#    report was found.
# 3. Large, abundant complexes that are frequent off-target particles in
#    lysate-derived samples. No specific reports were found; they are
#    included because they are big and common.
_HS = "Homo sapiens"
_BELCHER = "10.1042/BSR20260232"
MAMMALIAN: tuple[Contaminant, ...] = (
    Contaminant(
        "pcc", "Propionyl-CoA carboxylase (PCCA/PCCB, a6b6)", "7YBU", 750, False,
        "Endogenously biotinylated, so it binds Strep-Tactin; identified by cryo-EM + MS "
        "in twin-Strep purifications from HEK293.",
        refs=(_BELCHER,), source_organism=_HS,
    ),
    Contaminant(
        "prmt5_mep50", "PRMT5:MEP50 methylosome (hetero-octamer)", "4GQB", 450, False,
        "Identified by cryo-EM + MS in FLAG purifications from HEK293.",
        refs=(_BELCHER,), source_organism=_HS,
    ),
    Contaminant(
        "mcc", "3-Methylcrotonyl-CoA carboxylase (MCCC1/MCCC2, a6b6)", "8XL6", 750, False,
        "Biotinylated like PCC; expected Strep-Tactin/streptavidin binder (no cryo-EM "
        "report found). Same a6b6 architecture as PCC, so expect confusion between them.",
        source_organism=_HS,
    ),
    Contaminant(
        "pyr_carboxylase", "Pyruvate carboxylase (PC, tetramer)", "8XL9", 520, False,
        "Biotinylated; expected Strep-Tactin/streptavidin binder (no cryo-EM report found). "
        "Only ~300 of ~520 kDa is modelled (flexible biotin-carboxylase domains), so real "
        "particles look larger than the reference.",
        source_organism=_HS,
    ),
    Contaminant(
        "ribosome80S", "80S ribosome (human)", "6QZP", 3300, False,
        "Most abundant large particle in any mammalian lysate.", source_organism=_HS,
    ),
    Contaminant(
        "proteasome26S", "26S proteasome (human)", "5GJR", 2500, False,
        "Abundant; elongated with one or two 19S caps. This singly capped model is ~450 Å "
        "long, longer than a 384 Å box, so references are truncated (as particles would be).",
        source_organism=_HS,
    ),
    Contaminant(
        "proteasome20S", "20S proteasome core (human)", "5LE5", 720, False,
        "Abundant; barrel of four stacked rings (cf. archaeal T20S in 'lab').",
        source_organism=_HS,
    ),
    Contaminant(
        "tric", "TRiC/CCT chaperonin (closed)", "7LUM", 950, False,
        "Abundant double-ring chaperonin; resembles GroEL in some views.", source_organism=_HS,
    ),
    Contaminant(
        "vcp", "VCP/p97 (hexamer)", "5FTK", 540, False,
        "Abundant AAA+ hexamer with a distinctive two-ring side view.", source_organism=_HS,
    ),
    Contaminant(
        "acly", "ATP citrate lyase (tetramer)", "6UI9", 490, False,
        "Abundant cytosolic tetramer.", source_organism=_HS,
    ),
    Contaminant(
        "fasn", "Fatty acid synthase (dimer)", "8VLE", 540, False,
        "Abundant in HEK cells; flexible X-shaped dimer (composite model).", source_organism=_HS,
    ),
    Contaminant(
        "hsp60", "HSP60-HSP10 'football' complex", "4PJ1", 1000, False,
        "Mitochondrial chaperonin; also seen as single heptameric rings.", source_organism=_HS,
    ),
    Contaminant(
        "atp_synthase_mito", "Mitochondrial ATP synthase (human, monomer)", "8H9S", 595, True,
        "Abundant in detergent-solubilised membranes; long stalk distinguishes it.",
        source_organism=_HS,
    ),
    Contaminant(
        "complex_I", "Respiratory complex I (human)", "5XTD", 964, True,
        "Abundant in detergent-solubilised mitochondrial membranes; L-shaped.",
        source_organism=_HS,
    ),
)

_M = {c.key: c for c in MAMMALIAN}


def _proxy(key: str, host: str) -> Contaminant:
    """A human structure standing in for the host's own homolog (same 2D appearance)."""
    from dataclasses import replace

    c = _M[key]
    return replace(c, notes=f"{c.notes} Human structure used as a stand-in for the {host} "
                            "homolog (indistinguishable in 2D).")


# Insect cells (Sf9, Sf21 / Spodoptera frugiperda; High Five / Trichoplusia ni).
# Keys match the mammalian catalog where the complex is the same, so combining
# catalogs keeps one entry per complex (the first catalog listed wins).
INSECT: tuple[Contaminant, ...] = (
    Contaminant(
        "acc", "Acetyl-CoA carboxylase, carboxyltransferase core (T. ni)", "9CV6", 170, False,
        "Biotinylated, so a likely Strep-Tactin/streptavidin binder. The endogenous High Five "
        "enzyme was solved by cryo-EM. Only the CT-domain dimer (~175 kDa) is modelled; the "
        "flexible biotin-carboxylase arms add fuzzy density in real classes.",
        refs=("10.1016/j.jbc.2024.107800",), source_organism="Trichoplusia ni",
    ),
    Contaminant(
        "ruvbl", "RUVBL1/RUVBL2 (Pontin/Reptin) heterohexamer (Sf9)", "9WJX", 300, False,
        "Host RUVBL1/2 co-purified with human R2TP/TTT-pathway proteins expressed in Sf9 (PDB "
        "9WJX/9WJM). Expect it with any client of that chaperone system.",
        source_organism="Spodoptera frugiperda",
    ),
    Contaminant(
        "ribosome80S", "80S ribosome (D. melanogaster ovary)", "6XU8", 3300, False,
        "Most abundant large particle in any insect-cell lysate. Drosophila ovary 80S stands in "
        "for the Spodoptera/Trichoplusia ribosome (Sf9 cells are ovary-derived).",
        source_organism="Drosophila melanogaster",
    ),
    *(_proxy(k, "insect") for k in ("pcc", "mcc", "pyr_carboxylase", "proteasome26S",
                                    "proteasome20S", "tric", "vcp", "hsp60",
                                    "atp_synthase_mito", "complex_I")),
)

# Yeast: S. cerevisiae, plus Pichia pastoris (Komagataella phaffii) entries.
_SC = "Saccharomyces cerevisiae"
YEAST: tuple[Contaminant, ...] = (
    Contaminant(
        "aox", "Alcohol oxidase AOX1 (Pichia, octamer)", "5HSA", 600, False,
        "Pichia only, methanol-induced; the major contaminant of Pichia preps (re-tagging was "
        "needed to remove it in a cGMP vaccine process). Its cryo-EM structure was also solved. "
        "Do not use for S. cerevisiae.",
        refs=("10.1002/bit.28181", "10.1371/journal.pone.0159476"),
        source_organism="Komagataella pastoris",
    ),
    Contaminant(
        "fasn", "Fatty acid synthase (fungal a6b6 barrel)", "6TA1", 2600, False,
        "Very abundant 2.6 MDa barrel, unlike the mammalian X-shaped FASN. Fungal FAS was solved "
        "by cryo-EM after co-purifying with tagged Rad3.",
        refs=("10.1093/jmcb/mjz096",), source_organism=_SC,
    ),
    Contaminant(
        "acc", "Acetyl-CoA carboxylase (yeast, dimer)", "5CSL", 500, False,
        "Biotinylated; likely Strep-Tactin/streptavidin binder (no cryo-EM report found).",
        source_organism=_SC,
    ),
    Contaminant(
        "ribosome80S", "80S ribosome (yeast)", "4V88", 3300, False,
        "Most abundant large particle in any yeast lysate.", source_organism=_SC,
    ),
    Contaminant(
        "proteasome26S", "26S proteasome (yeast)", "5WVK", 1700, False,
        "Abundant; singly capped particle (all 47 subunits modelled).", source_organism=_SC,
    ),
    Contaminant(
        "proteasome20S", "20S proteasome core (yeast)", "5CZ4", 730, False,
        "Abundant; barrel of four stacked rings.", source_organism=_SC,
    ),
    Contaminant(
        "tric", "TRiC/CCT chaperonin (yeast)", "5GW4", 950, False,
        "Abundant double-ring chaperonin.", source_organism=_SC,
    ),
    Contaminant(
        "pfk_pichia", "Phosphofructokinase (Pichia, dodecamer)", "3OPY", 1000, False,
        "Abundant glycolytic enzyme; large (~1 MDa) a4b4g4 assembly in Pichia.",
        source_organism="Pichia pastoris",
    ),
    Contaminant(
        "atp_synthase_mito", "Mitochondrial ATP synthase (yeast, monomer)", "6CP6", 570, True,
        "Abundant in detergent-solubilised membranes.", source_organism=_SC,
    ),
    *(_proxy(k, "yeast") for k in ("pcc", "mcc", "pyr_carboxylase", "vcp", "hsp60")),
)

CATALOGS: dict[str, tuple[Contaminant, ...]] = {
    "ecoli": ECOLI, "lab": LAB, "mammalian": MAMMALIAN, "insect": INSECT, "yeast": YEAST,
}
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
