# Resin-by-resin contaminant sweep

Summary of a literature sweep by purification resin. It was run in a separate
session; its catalog entries were re-applied and re-verified here.

## Result: mostly negative

Outside IMAC there is no published, resin-specific contaminant list for
cryo-EM. Only two resins select host proteins by a real mechanism:

- **IMAC (Ni/Co):** proteins with surface histidine clusters, or proteins that
  bind the metal (ArnA, SlyD, GlmS, AcrB, ... and GlnA, below).
- **Strep-Tactin / streptavidin:** endogenously biotinylated carboxylases (PCC,
  MCC, PC, ACC). These are already in the `mammalian`, `insect` and `yeast`
  catalogs.

Every other matrix picks up whatever is abundant in the host. The five host
catalogs already cover that. So the resin axis adds four E. coli entries, not
a new catalog.

## Added (`ecoli`)

| key | PDB | assembly | tier |
|---|---|---|---|
| glnA | 8V22 | dodecamer, ~620 kDa | mechanistic |
| nuo | 7NYR | complex I, ~550 kDa (membrane) | abundance |
| ahpC | 4O5R | decamer, ~205 kDa | abundance |
| rnap | 4YG2 | σ70 holoenzyme, ~450 kDa | abundance |

GlnA is the one mechanistic find. Its N-terminal histidines chelate Ni²⁺, and
nickel stacks the dodecamers into helical filaments (PDB 7W85; Huang et al.
2022, Protein Sci, doi:10.1002/pro.4304). Glutamate or ATP reverses the
stacking. So a nickel prep can show isolated double rings, short stacks, or
both, depending on the buffer.

## Rejected, so they need not be re-checked

- **Under the ~150 kDa threshold for useful 2D classes:** SlyD, Crp, Fur, ArgE,
  YadF, GlgA, ribosomal S15 and BCCP.
- **Glycogen phosphorylase (E. coli):** the only amylose-resin candidate above
  the threshold, but no report was found, and amylose is rare in cryo-EM preps.
- **Leached Protein A/G and IgG (~150 kDa):** flexible and Y-shaped, so they
  align poorly.
- **Antibody and epitope-tag beads (FLAG, HA, Myc, GFP-trap):** checked against
  the quantitative AP-MS background sets of Trinkle-Mulcahy et al. 2008 and the
  CRAPome (Mellacheruvu et al. 2013, doi:10.1038/nmeth.2557). After removing
  the sub-150 kDa majority, what remains is already in the `mammalian` catalog.
- **GST / glutathione:** the main issue is complexes bridged by nucleic acid.
  That is a sample and grid artefact, not a protein class.

## Not swept

Chitin, calmodulin/CBP, lectins, Protein L and plain ion exchange.
