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

Two runs, both at 64 px and 6 Å/px. Test particles come from view directions
not seen in training, with fresh noise. Chance level is 4% (24 structures plus
background).

- **CPU run**: 3000 steps of 128 particles, 1000 training views per
  structure, about 1.5 h on 4 CPU cores.
- **GPU run**: 20000 steps of 256 particles, 3000 views, about 2 h on one RTX
  6000 Ada that was shared with another job. Training accuracy reached 70%,
  against 61% for the CPU run.

| Dose (e/Å²) | All 25 classes (CPU → GPU) | OmpF/OmpC and Bfr/ferritin merged | Contaminant called "not background" | Empty picks wrongly called a contaminant |
|---|---|---|---|---|
| 10 | 38% → 46% | 41% → 49% | 88% → 92% | 5% → 7% |
| 20 | 59% → 67% | 62% → 71% | 98% → 98.6% | 5% → 7% |
| 40 | 72% → 81% | 75% → 85% | 99.8% → 99.9% | 7% → 7% |

Per-class accuracy at 40 e/Å² (GPU run):

- **≥ 90%**: 70S ribosome, GroEL, OGDH E2 core, T20S, Dps, ArnA, ATP
  synthase, β-gal, apoferritin, AcrB.
- **80–89%**: KatE, GldA, SDH, KatG, FolE, Bfr, GadB. The longer training
  helped this group most: GadB went from 46% to 80% and KatG from 67% to 86%.
- **< 70%**: PDH E1 (67%), GlmS (63%), aldolase (60%), OmpF (53%), cyt bd
  (47%), OmpC (35%), cyo bo3 (33%).

At 20 e/Å² the order of the classes is the same, and the hard group sits at
17–50%.

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
- **No detergent micelle or carbon edges are simulated**, and the background
  is idealised ice.
- **Only 6 Å/px has been tested so far.** A run at 3 Å/px (`--apix 3.0 --box
  128`) is the next test for the hard group.

What this means for the plan:

- **A per-particle "is this a known contaminant?" flag looks realistic.**
  About 99% of contaminant particles were called something other than
  background at 20 e/Å².
- **Naming which contaminant it is works for most of the library at moderate
  signal.**
- **Small membrane proteins and the 150–250 kDa group will need more.** That
  means finer pixels, pooling votes across a 2D class, or real labelled
  particles from the self-training loop.

## Known limitations

- **Detergent micelles and nanodiscs are not modelled yet.** In class averages
  of membrane proteins the micelle is a big feature, which lowers scores for
  membrane references. Comparing against your own target (also without a
  micelle) keeps the comparison fair, but absolute scores will be lower than
  for soluble proteins.
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
- [ ] Micelle/nanodisc model around TM regions (e.g. using OPM orientations).
- [ ] Catalogs for HEK293, Sf9/High Five and yeast expression hosts.
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
