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
(source organism, oligomeric state and mass). GldA uses 43WP, a 3.2 Å cryo-EM
structure of the octamer; succinate dehydrogenase uses assembly 2 of 1NEK (the
trimer).

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

A separate `lab` catalog covers standard test and calibration samples that can
be carried over on shared equipment (plunge-freezers, tweezers, cassettes):
mouse apoferritin, β-galactosidase, aldolase and the T20S proteasome. GroEL is
already in the E. coli catalog. Both catalogs are used by default; choose
with `--organism ecoli`, `--organism lab` or `--organism ecoli,lab`.

Small IMAC contaminants (SlyD, Fur, Crp, ArgE, YadF, GlgA, S15, BCCP) are left
out on purpose, because they are too small to produce recognisable 2D classes.

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
- [ ] Micelle/nanodisc model around TM regions (e.g. using OPM orientations).
- [ ] Catalogs for HEK293, Sf9/High Five and yeast expression hosts.
- [ ] Phase 2: per-particle classifier trained on simulated and EMPIAR
  particles, for live use during collection (cryoSPARC Live).
- [ ] RELION `.star` input.

## Tests

```bash
pytest
```

The tests build toy "proteins" and check that the matcher recovers rotation,
shift and identity from noisy, contrast-inverted, rescaled class averages
written in cryoSPARC's file layout.
