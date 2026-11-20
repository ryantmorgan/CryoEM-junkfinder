# CryoEM-junkfinder

Catch contaminants **before** you collect 18,000 movies of them.

`junkfinder` compares your cryoSPARC 2D class averages against simulated
projections of proteins that commonly contaminate cryo-EM samples (E. coli
cytochrome bo3, AcrB, ArnA, GroEL, ...), and against a model of **your own
target** as a positive control. For each class it reports which reference it
resembles most, and how confidently.

> Status: v0.1, Phase 1 (class-average matching). Validated on synthetic data
> only. Score thresholds are not yet calibrated on real datasets.

## How it works

1. **Library.** For each contaminant, the biological assembly is downloaded from
   the RCSB and projected from ~700 directions over the full sphere. Atoms are
   treated as Gaussians and the projections are low-passed (default 15 Å) at a
   working pixel size (default 4 Å/px). Your target's model (e.g. AlphaFold)
   is rendered the same way.
2. **Matching.** Each class average is resampled to the library's pixel size,
   its contrast is checked automatically, and it is low-passed to the same
   resolution. Every reference is then scored over all in-plane rotations
   (polar FFT) and small shifts. The score is the best normalised
   cross-correlation.
3. **Verdict per class.**
   - `contaminant` or `target`: the top hit clears `--min-score` and beats the
     runner-up by `--min-margin`.
   - `ambiguous`: two references score too close to call.
   - `no_match`: nothing scores high enough.

   If particle counts are supplied, the summary is weighted by them.

## Install

```bash
pip install -e .          # numpy, scipy, gemmi, mrcfile, matplotlib
pip install -e .[ml]      # + PyTorch, for the experimental particle classifier
```

## Getting the structures

`build-library` downloads from the RCSB automatically. If the machine running
it cannot reach the RCSB (e.g. a locked-down cluster), fetch them once
somewhere that can, and commit or copy the folder:

```bash
junkfinder fetch --dir structures     # gzipped assemblies + <ID>.json metadata
junkfinder verify --dir structures    # organism, title, mass vs catalog (offline)
git add structures && git commit -m "Add contaminant structures"
junkfinder build-library --cache structures ...
```

## Use with cryoSPARC

```bash
# 1. Build a library once per target (downloads structures to ~/.cache/junkfinder)
junkfinder list                                   # show the E. coli catalog
junkfinder build-library -o mylib.npz --target my_protein_AF3.cif

# 2. Screen a 2D classification job (export from the job's outputs)
junkfinder screen J42_020_class_averages.mrc \
    --cs J42_020_class_averages.cs \
    --particles J42_020_particles.cs \
    --library mylib.npz --out J42_screen
```

Outputs in `J42_screen/`:
- `summary.txt`: what fraction of particles look like each contaminant.
- `classes.tsv`: per-class verdict, top hits, and the score against every
  reference.
- `montage.png`: each class average next to its best-matching aligned
  projection. **Always look at this.**

Useful options:
- `--box` / `--apix` on `build-library`: the box must be bigger than your
  largest reference. The default is 96 px × 4 Å = 384 Å; the 70S ribosome needs
  roughly this much.
- `--structure KEY=PATH`: use a local file for a catalog entry, or add your own
  lab's usual suspects (any key not in the catalog becomes a custom
  contaminant).
- `--only acrB,cyo_bo3`: build a smaller library.
- `--invert yes|no`: override automatic contrast detection.

## Catalog sources

Every catalog entry has been checked against the RCSB with `junkfinder verify`
(source organism, oligomeric state and mass). Where a contaminant has been
solved by cryo-EM from the contaminated sample itself, that structure is used
(cytochrome bo3 6WTI, SQR 7JZ2, OmpC 7JZ3, KatG 7JZ6, GadB 7JZH, GldA 43WP).

Each catalog entry lists DOIs reporting it as a co-purifying contaminant
(`junkfinder/catalog.py`). Main sources so far:

- Bolanos-Garcia & Davies 2006, native E. coli proteins co-purified by IMAC
  ([doi](https://doi.org/10.1016/j.bbagen.2006.03.027))
- Andersen et al. 2013, LOBSTR strain (ArnA, SlyD)
  ([doi](https://doi.org/10.1002/prot.24364))
- Truong et al. 2026, IMACulate strain (GlmS/GFAT, PDH E1)
  ([doi](https://doi.org/10.1002/bit.70244))
- Scholz et al. 2026, catalase contamination
  ([doi](https://doi.org/10.1007/s00253-026-13820-2))
- Kim et al. 2026, ArnA, FolE and GldA found as off-target cryo-EM particles
  ([doi](https://doi.org/10.1107/S2059798326010302))
- Su et al. 2021, "Build and Retrieve": structures solved directly from an
  impure His-tag membrane prep (bo3, OmpF, SQR), detergent-solubilised K12
  membranes (OmpC, SQR) and size-selected K12 lysate (KatG, GadB)
  ([doi](https://doi.org/10.1038/s41592-020-01021-2))
- Carr et al. 2025, DLST (the OGDH E2 core) identified by cryo-EM as a
  co-purifier ([doi](https://doi.org/10.1016/j.yjsbx.2025.100120))

### Mammalian (HEK293/Expi293): `--organism mammalian`

Fourteen human complexes, in three tiers. Select them with `--organism
mammalian,lab`; the default stays `ecoli,lab`.

1. **Reported by cryo-EM plus mass spectrometry.** Propionyl-CoA
   carboxylase (twin-Strep purifications) and PRMT5:MEP50 (FLAG), both from
   HEK293 (Belcher et al. 2026,
   [doi](https://doi.org/10.1042/BSR20260232)).
2. **The other endogenously biotinylated carboxylases.** MCC and pyruvate
   carboxylase should bind Strep-Tactin/streptavidin for the same reason PCC
   does, but no cryo-EM report was found. MCC has the same α6β6 architecture
   as PCC, so expect the two to be confused.
3. **Large, abundant complexes** that turn up in lysate-derived samples:
   - Soluble: 80S ribosome, 26S and 20S proteasomes, TRiC/CCT, VCP/p97,
     ATP citrate lyase, fatty acid synthase, HSP60–HSP10.
   - Membrane, for detergent preps: mitochondrial ATP synthase, complex I.

   No specific contamination reports were found for these; they are
   included because they are big and common.

### Insect cells (Sf9/Sf21, High Five): `--organism insect`

- **Host proteins seen by cryo-EM:**
  - **Acetyl-CoA carboxylase:** the endogenous T. ni (High Five) enzyme was
    solved by cryo-EM (Wang et al. 2024,
    [doi](https://doi.org/10.1016/j.jbc.2024.107800)). It is biotinylated,
    so it is a likely Strep-Tactin binder. Only its ~170 kDa
    carboxyltransferase core is modelled.
  - **RUVBL1/2:** Sf9 RUVBL1/2 co-purified with human proteins expressed in
    Sf9 (PDB 9WJX/9WJM).
- **80S ribosome:** Drosophila ovary 80S (6XU8) stands in for the
  Spodoptera/Trichoplusia ribosome.
- **Other complexes:** human structures stand in for the insect homologs,
  which look the same in 2D:
  - the biotinylated carboxylases (PCC, MCC, PC);
  - proteasomes, TRiC, VCP and HSP60;
  - mitochondrial ATP synthase and complex I.

### Yeast (S. cerevisiae, Pichia): `--organism yeast`

- **Pichia alcohol oxidase (AOX1, 600 kDa octamer):** described as the major
  contaminant of Pichia-expressed preps (Mukhopadhyay et al. 2022,
  [doi](https://doi.org/10.1002/bit.28181)). Its cryo-EM structure was
  solved by Vonck et al. 2016
  ([doi](https://doi.org/10.1371/journal.pone.0159476)). It is
  methanol-induced and absent from S. cerevisiae.
- **Fungal fatty acid synthase (2.6 MDa α6β6 barrel, 6TA1):** very abundant,
  and quite unlike mammalian FASN. It was solved by cryo-EM after
  co-purifying with tagged Rad3 (Qiu et al. 2019,
  [doi](https://doi.org/10.1093/jmcb/mjz096)).
- **Other yeast structures:** ACC (biotinylated), 80S ribosome, 26S and 20S
  proteasomes, TRiC, Pichia phosphofructokinase (~1 MDa) and mitochondrial
  ATP synthase.
- **Human stand-ins:** PCC, MCC, PC, VCP/Cdc48 and HSP60.

### Native human tissue: `--organism tissue`

Seventeen complexes identified by cryo-EM in brain and liver fractions:

- **Brain microsomes** (Build-and-Retrieve style;
  [doi](https://doi.org/10.26508/lsa.202201724)): GLUL, ferritin, DPYSL2,
  ALDH1A1.
- **Liver microsomes** ([doi](https://doi.org/10.1016/j.celrep.2023.112609)):
  glycogen phosphorylase, H6PD, GANAB, PRDX4, CES1, AOX1, GLUD1.
- **Liver mitochondria**
  ([doi](https://doi.org/10.1016/j.mcpro.2023.100666)): SCAD, MCAD, IVD,
  catalase, ALDH2.
- **Vault:** contaminated a tau-filament prep from brain
  ([doi](https://doi.org/10.1016/j.str.2025.07.014)).

Use `--organism tissue,mammalian` for native mammalian preps.

### Additions from a cryo-EM literature sweep

- **E. coli:**
  - AdhE spirosome segment. Spirosomes were seen on lysate-purified 70S
    ribosomes ([doi](https://doi.org/10.1038/srep19936)); structure
    [doi](https://doi.org/10.1038/s41467-019-12427-8).
  - Glutamine synthetase GlnA. Ni²⁺ turns its dodecamers into filaments, so
    it matters after Ni-NTA ([doi](https://doi.org/10.1002/pro.4304)).
  - Lysine decarboxylase LdcI. Included for abundance; no contamination
    report found.
  - More evidence for AcrB and ArnA as off-target particles in His-tag
    cryo-EM datasets
    ([doi](https://doi.org/10.1107/S2059798325007089),
    [doi](https://doi.org/10.1107/S2059798326002846)).
- **Lab:** bovine GDH and bovine liver catalase.

**Known but not in the libraries** (too small for useful 2D classes, or not
single particles):
- **Small proteins:** SlyD and CAP (IMAC), Hfq, carbonic anhydrase,
  haemoglobin, streptavidin, Fab or IgG fragments from antibody grids, ALFA
  nanobodies.
- **Filaments and lattices:** TMV, F-actin, flagella, phage tail tubes from
  bacterially contaminated buffers, baculovirus nucleocapsids, streptavidin
  2D-crystal grids.

### Combining catalogs

Keys are shared across the eukaryotic catalogs for the same complex
(`ribosome80S`, `proteasome26S`, `tric`, `acc`, …). Combining catalogs keeps
one entry per complex, and the first catalog listed wins. For example,
`--organism yeast,lab` uses the yeast ribosome.

Not included yet:
- **V-ATPase:** every human structure carries a non-native binder (SidK or
  mEAK7).
- **Na⁺/K⁺-ATPase:** no human cryo-EM structure.
- **Vault:** about 67 nm long, too big for typical boxes.

A separate `lab` catalog covers standard test and calibration samples that can
be carried over on shared equipment (plunge-freezers, tweezers, cassettes):
mouse apoferritin, β-galactosidase, aldolase and the T20S proteasome. GroEL is
already in the E. coli catalog. Both catalogs are used by default; choose
with `--organism ecoli`, `--organism lab` or `--organism ecoli,lab`.

Small IMAC contaminants (SlyD, Fur, Crp, ArgE, YadF, GlgA, S15, BCCP) are left
out on purpose, because they are too small to produce recognisable 2D classes.

## How distinguishable are the references?

Simulated class averages (6 random views of each of the 24 references at
15 Å, rotated, shifted, inverted and noisy) are assigned to the right
protein 143/144 times. The one miss is OmpC vs OmpF. But the gap between the
right answer and the runner-up is small, so many correct calls come out
`ambiguous` at the default `--min-margin 0.03`:

| Pair | Typical gap | Comment |
|---|---|---|
| OmpF / OmpC | ~0.002 | 60% identical trimeric porins; treat as one call. |
| Bacterioferritin / apoferritin | ~0.005-0.01 | Both 24-mer hollow spheres ~12 nm. |
| cyt bd, GlmS, PDH E1, bo3, FolE, KatE | ~0.01-0.02 | Compact mid-size blobs. |
| Ribosome, GroEL, OGDH E2, T20S, ATP synthase | >0.06 | Distinctive shapes, easy. |

Setting `--mask-radius` to about the particle radius plus ~10 Å widens the
gaps by about 50% (median 0.026 -> 0.039), because less empty background
enters the correlation.

## Phase 2 (experimental): per-particle classifier

The aim is to classify individual particles (and eventually whole
micrographs, like a Topaz picker that knows which contaminant it is looking
at) rather than only 2D class averages.

- **Network** (`junkfinder/classifier.py`): the ResNet8 design described for
  Topaz (Bepler et al. 2019, [doi](https://doi.org/10.1038/s41592-019-0575-8)),
  re-implemented rather than copied (Topaz is GPL-3), with one output per
  library entry plus "background" instead of Topaz's single particle/not
  output.
- **Training data** (`junkfinder/particles.py`): simulated from the atomic
  models, so no manual labelling is needed. Projected protein thickness is
  turned into phase, a CTF at random defocus (0.8-2.5 µm, 300 kV) is applied,
  and shot noise for the chosen electron dose is added. Augmentations: random
  rotation, centring errors of up to 12 Å, slow ice-thickness variation, and
  neighbouring particles at the box edge. Big particles therefore come out
  brighter than small ones, as on a real grid.
- **Benchmark**: train on views from one set of directions, test on new
  random directions with fresh noise:

```bash
junkfinder particle-benchmark --cache structures --out particle_benchmark
```

It uses a GPU automatically when PyTorch can see one (`--device` overrides
this). On a SLURM cluster, `sbatch scripts/particle_benchmark.slurm` runs a
longer job; edit its header for your partition and environment. Override the
defaults with variables, e.g. `STEPS=40000 APIX=3.0 BOX=128 sbatch ...`.

### Results

Test particles come from view directions not seen in training, with fresh
noise. Chance level is 4% (24 structures plus background).

| Run | Pixel / box | Training | Hardware | Final train acc |
|---|---|---|---|---|
| A | 6 Å, 64 px | 3000 steps × 128, 1000 views | 4 CPU cores, ~1.5 h | 61% |
| B | 6 Å, 64 px | 20000 steps × 256, 3000 views | 1 RTX 6000 Ada (shared), ~2 h | 70% |
| C | 3 Å, 128 px | 40000 steps × 256, 3000 views | 1 RTX 6000 Ada (shared), ~9.5 h | 77% |

| Dose (e/Å²) | All 25 classes (A / B / C) | OmpF/OmpC and Bfr/ferritin merged | Contaminant called "not background" | Empty picks wrongly called a contaminant |
|---|---|---|---|---|
| 10 | 38% / 46% / 50% | 41% / 49% / 53% | 88% / 92% / 90% | 5% / 7% / 6% |
| 20 | 59% / 67% / 75% | 62% / 71% / 78% | 98% / 98.6% / 98.5% | 5% / 7% / 6% |
| 40 | 72% / 81% / 88% | 75% / 85% / 91% | 99.8% / 99.9% / 99.9% | 7% / 7% / 5% |

Run C changes two things at once, the pixel size and twice as many steps, so
the gain from B to C is not purely from the finer pixels.

Per-class accuracy at 40 e/Å² (run C):

- **≥ 90%**: 17 of 24 structures. In order: ribosome, OGDH E2, GroEL, Dps,
  T20S, ATP synthase, ArnA, β-gal, apoferritin, KatE, AcrB, SDH, GldA, KatG,
  FolE, Bfr, GadB. Background is at 95%.
- **77–79%**: aldolase, PDH E1, GlmS (60–67% in run B).
- **53–57%**: OmpC, OmpF, cyo bo3, cyt bd.

At 20 e/Å², the same hard group sits at 31–55%.

The main confusions:

- OmpF and OmpC are confused with each other (as in the class-average
  test).
- Bfr and apoferritin are confused with each other.
- The 150–250 kDa set (cyo bo3, cyt bd, GlmS, PDH E1, aldolase) is confused
  among itself.

How to read these numbers:

- **The absolute numbers depend on the simulated SNR.** Dose and protein
  contrast (ΔV = 2 V) are only known roughly, and they enter as a product. So
  treat the dose axis as a "signal level" knob, not a literal dose. The
  ordering of classes and the confusions are the more robust results.
- **These three runs used bare membrane proteins**, without a micelle or
  nanodisc (added since, see below). No carbon edges are simulated, and the
  background is idealised ice.

What this means for the plan:

- **A per-particle "is this a known contaminant?" flag looks realistic.**
  About 99% of contaminant particles were called something other than
  background at 20 e/Å², with about 5% false alarms on empty picks.
- **Naming which contaminant it is works for about three-quarters of the
  library**, at moderate signal and at 3 Å/px.
- **The four small membrane proteins stay near 55% even at the highest
  signal.** Those will need votes pooled across a 2D class, a micelle in the
  simulation, or real labelled particles from the self-training loop.

### Micelles and nanodiscs

Membrane proteins are seen inside a detergent micelle or nanodisc, and in 2D
that belt is a large part of the particle. `junkfinder/membrane.py` adds one
as pseudo-atoms, which then go through the normal projection pipeline:

1. **Membrane plane.** Scan for the ~30 Å slab with the most exposed
   hydrophobic residues. This is OPM-like in spirit but needs no download.
   - For the trimers (AcrB, SDH, OmpF, OmpC) the normal is within 4° of the
     3-fold axis; for ATP synthase it runs along the rotor.
   - The transmembrane helices of bo3 and cyt bd sit at a median tilt of 24°
     to it.
2. **Micelle.** A smooth oblate ellipsoid centred on the hydrophobic region.
   - Fitted to 33 class averages of E. coli bo3 in DDM with `junkfinder
     fit-env`.
   - The fit gave a micelle 120 Å across and 65 Å thick, with heads at 1.16×
     and the core at 0.48× protein contrast. That is evenly filled and nearly
     as bright as the protein, as in the real classes.
   - On mean best NCC over the classes, the fitted micelle scores 0.979,
     against 0.951 for my earlier hand-set micelle and 0.883 for bare bo3.
   - Detergent scales with the transmembrane perimeter, so the in-plane size
     is set by how far the micelle reaches beyond the TM domain. The bo3 fit
     gives 21.5 Å; training draws 15–28 Å.
   - The micelle is 54–74 Å thick, slightly elliptical and only faintly
     lumpy.
   - Enclosed lumens, such as porin channels, stay empty.
3. **Nanodisc.** A lipid bilayer disc with a rounded rim, wrapped by a 10 Å
   MSP scaffold.
   - Outer diameters are 100, 130 and 170 Å for MSP1D1, MSP1E3D1 and MSP2N2.
   - Half the time the smallest disc that fits is used, otherwise a random
     size that fits. Small proteins may sit off-centre.
   - Contrast is set so the disc looks evenly filled, with no bright scaffold
     ring, as in 2D classes of bo3 in MSP1E3D1: heads 0.6–1.0×, core
     0.15–0.5×, scaffold 0.3–0.7× protein contrast.

   The rounded rim, sizes and off-centre option follow
   [cs-create-nanodisc-templates](https://github.com/olibclarke/cs-create-nanodisc-templates)
   (O. Clarke).

4. **SMA lipid particle (SMALP).** A native-lipid disc sized to the protein
   (95–125 Å across), with a thin, diffuse polymer rim about as bright as the
   lipid heads. It looks evenly filled with no ring, as in 2D classes of bo3
   in SMA. SMA extracts proteins straight from the native membrane, so host
   contaminants are especially likely in these preps.

For the per-particle benchmark, every membrane protein gets
`--env-variants` (default 24) random micelles or discs for training, and
different ones for testing. Use `--membrane-env micelle|nanodisc|sma|both|all|none`
(`both` mixes micelles and nanodiscs, `all` adds SMA);
the default is `micelle`.

Add your own protein as an extra class with `--target NAME=PDBID` (or
`NAME=path/to/model.cif`) and `--target-env micelle|nanodisc|sma|...`. On
real data the classifier then has somewhere to put target particles instead
of forcing them into a contaminant or "background":

```bash
junkfinder particle-benchmark --cache structures --target rodA_pbp2=8TJ3 --target-env micelle \
    --membrane-env all --apix 3.0 --box 128 --steps 40000 --batch 256 --out pb_target
```

### Classifying real particles

Apply a trained model to a cryoSPARC particle set (Extract, 2D
Classification or Select 2D output):

```bash
junkfinder classify /path/to/P274/J123/J123_020_particles.cs --project-dir /path/to/P274 \
    --model pb_target/particle_classifier.pt --out classify_J123
```

- **Preparation.** Each particle is brought onto the model's pixel size and
  box, normalised by its edge noise, and classified.
  - Particles extracted with flipped contrast (protein white) need
    `--invert yes`.
  - It warns if the particles' defocus falls outside the training range.
- **Output.**
  - `particles.tsv`: prediction, probability and runner-up for each
    particle.
  - `summary.txt`: label counts. If the `.cs` comes from 2D classification,
    also a vote per 2D class; that vote is far more reliable than single
    particles.
- **Status.** This is the first contact with real data. Expect a domain gap
  between simulated and real particles until the model is tuned on labelled
  real examples (the self-training step).

For class-average screening, `build-library --membrane-env micelle` adds a
typical micelle to the membrane contaminants. `--target-env micelle` does the
same for your own `--target` model if it is a membrane protein, so that
targets and contaminants are compared on equal terms.

The contrast ranges are rough estimates from electron densities, not fitted
to data. A real dataset with a known micelle (for example bo3 in DDM) is the
way to tune them.

### Fitting the environment to your own data

```bash
junkfinder fit-env bo3_classes.mrc --key cyo_bo3 --cache structures --env micelle --out fit.json
```

It needs class averages of the one membrane protein only, not a mixture. It
tries random micelles, nanodiscs or SMA discs, then refines around the best.
Each candidate is scored by how well its projections match the classes
(mean best NCC). It takes about 20 minutes on 4 CPU cores.

The score is dominated by the overall outline at 12 Å. So it pins down size
and thickness well, and the head/core split less well.

## Known limitations

- **Micelles and nanodiscs are approximate.** Membrane references can be
  surrounded with a micelle or nanodisc (see below). Its shape, size and
  contrast are generic, not fitted to a particular detergent, so absolute
  scores for membrane references stay lower than for soluble proteins.
- **Thresholds are uncalibrated.** Defaults (`--min-score 0.5`,
  `--min-margin 0.03`) are placeholders. Calibrating them on real datasets is
  the next job.
- 2D only. Small proteins (<~100 kDa) and featureless blobs will often come
  out as `ambiguous`, which is the honest answer.

## Roadmap

- [ ] Validate on a real bo3-contaminated dataset; calibrate thresholds.
  Public raw data that could serve: EMPIAR-10737 (pure E. coli bo3 in
  nanodiscs, 3,446 movies, 851 GB) and EMPIAR-13232/13235 (clarified E. coli
  lysate, 7,900-9,700 movies, 2-2.4 TB each).
- [ ] Choose the mask radius automatically from each class average.
- [x] Micelle/nanodisc model around TM regions (built-in membrane-plane detection).
- [ ] Empty micelles and nanodiscs as a "background" variant (they are common false picks).
- [x] Catalog for mammalian (HEK293) expression.
- [x] Catalogs for insect-cell (Sf9/High Five) and yeast (S. cerevisiae, Pichia) hosts.
- [ ] Phase 2 (started, see below): per-particle classifier.
  1. [x] Prototype on simulated particles (`particle-benchmark`).
  2. [ ] Classify real extracted particles (cryoSPARC `.cs` + stack).
  3. [ ] Self-training: particles in classes the class-average screen calls
     confidently become labels; fine-tune; check on held-out micrographs.
  4. [ ] Fully convolutional multi-class picker on whole micrographs, for
     cryoSPARC Live.
- [ ] RELION `.star` input.

## Tests

```bash
pytest
```

The tests build toy "proteins" and check that the matcher recovers rotation,
shift and identity from noisy, contrast-inverted, rescaled class averages
written in cryoSPARC's file layout.
