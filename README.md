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

## Known limitations

- **Detergent micelles and nanodiscs are not modelled yet.** In class averages
  of membrane proteins the micelle is a big feature, which lowers scores for
  membrane references. Comparing against your own target (also without a
  micelle) keeps the comparison fair, but absolute scores will be lower than
  for soluble proteins.
- **Thresholds are uncalibrated.** Defaults (`--min-score 0.5`,
  `--min-margin 0.03`) are placeholders. Calibrating them on real datasets is
  the next job.
- **Check catalog IDs marked "(verify ID)"** in `junkfinder list` against the
  RCSB before trusting results for those entries.
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
