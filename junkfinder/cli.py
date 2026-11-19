"""Command-line interface: ``junkfinder build-library`` and ``junkfinder screen``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import CATALOGS, DEFAULT_CATALOGS, Contaminant, get_catalog
from .cryosparc_io import load_class_averages
from .library import Library, build_library
from .structures import cached_structure, describe_structure, fetch_metadata, fetch_structure
from .screen import screen, summarise, write_montage, write_tsv


def cmd_list(args: argparse.Namespace) -> None:
    for c in get_catalog(args.organism):
        flag = "  (verify ID)" if c.verify else ""
        kind = "membrane" if c.membrane else "soluble"
        print(f"{c.key:<14s} {c.pdb_id:<6s} ~{c.approx_kda:>5d} kDa  {kind:<8s} {c.name}{flag}")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download every catalog structure into a folder (run where RCSB is reachable)."""
    out = Path(args.dir)
    failed = []
    for c in get_catalog(args.organism):
        if not c.pdb_id:
            print(f"{c.key:<14s} no structure chosen yet, skipped")
            continue
        try:
            path = fetch_structure(c.pdb_id, c.assembly, out)
            fetch_metadata(c.pdb_id, out)
            print(f"{c.key:<14s} {path}")
        except Exception as exc:  # noqa: BLE001
            failed.append(c.key)
            print(f"{c.key:<14s} FAILED: {exc}")
    print(f"\nDone. {len(failed)} failed." if failed else "\nDone.")


def cmd_verify(args: argparse.Namespace) -> None:
    """Check that each downloaded file is the protein the catalog says it is."""
    for c in get_catalog(args.organism):
        path = cached_structure(c.pdb_id, c.assembly, Path(args.dir)) if c.pdb_id else None
        if path is None:
            print(f"[{c.key}] {c.name}: MISSING\n")
            continue
        d = describe_structure(path)
        problems = []
        if not any(c.source_organism.lower() in o.lower() for o in d["organisms"]):
            problems.append(f"organism is {d['organisms'] or 'unknown'}")
        if c.approx_kda and not 0.5 <= d["approx_kda"] / c.approx_kda <= 2.0:
            problems.append(f"mass {d['approx_kda']} kDa vs expected ~{c.approx_kda}")
        status = "CHECK: " + "; ".join(problems) if problems else "ok"
        print(f"[{c.key}] {c.name}  ->  {path.name}: {status}")
        print(f"    title:    {d['title']}")
        print(f"    entities: {'; '.join(d['entities'][:6])}")
        method = f"; {d['method']}" if d.get("method") else ""
        if d.get("resolution"):
            method += f" {d['resolution']:.1f} Å"
        print(f"    organism: {', '.join(d['organisms'])}; {d['chains']} chains, "
              f"~{d['approx_kda']} kDa polymer{method}\n")


def cmd_build(args: argparse.Namespace) -> None:
    catalog = list(get_catalog(args.organism))
    local = {}
    for item in args.structure or []:
        key, sep, path = item.partition("=")
        if not sep or not key or not path:
            raise SystemExit(f"--structure expects KEY=PATH, got '{item}'")
        local[key] = Path(path)
    # Local structures whose key is not in the catalog become custom contaminants.
    known = {c.key for c in catalog}
    for key, path in local.items():
        if key not in known:
            catalog.append(Contaminant(key, f"Custom: {path.name}", str(path), 0, False))
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.key for c in catalog}
        if unknown:
            raise SystemExit(f"--only: unknown key(s) {', '.join(sorted(unknown))}. "
                             "See 'junkfinder list'.")
        catalog = [c for c in catalog if c.key in wanted]
    lib = build_library(
        catalog, targets=[Path(t) for t in args.target or []], local_structures=local,
        apix=args.apix, box=args.box, resolution=args.resolution,
        angular_step=args.angular_step,
        cache_dir=Path(args.cache) if args.cache else None,
    )
    if not any(e.kind == "contaminant" for e in lib.entries):
        print("warning: library contains no contaminant references")
    if not any(e.kind == "target" for e in lib.entries):
        print("note: no --target given; without your own model as a control, every class "
              "is forced to choose between contaminants (expect more false alarms)")
    lib.save(args.output)
    n_views = len(lib.directions)
    print(f"Saved {len(lib.entries)} references x {n_views} views to {args.output}")


def cmd_screen(args: argparse.Namespace) -> None:
    lib = Library.load(args.library)
    imgs, apix, counts = load_class_averages(args.class_averages, args.cs, args.particles, args.apix)
    invert = {"auto": None, "yes": True, "no": False}[args.invert]
    reports, prepared = screen(
        lib, imgs, apix, counts, invert=invert, max_shift_A=args.max_shift,
        mask_radius_A=args.mask_radius, min_score=args.min_score, min_margin=args.min_margin,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(reports, lib, out / "classes.tsv")
    summary = summarise(reports, lib)
    (out / "summary.txt").write_text(summary + "\n")
    if not args.no_montage:
        write_montage(reports, prepared, lib, out / "montage.png")
    print(summary)
    print(f"\nDetails: {out / 'classes.tsv'}" + ("" if args.no_montage else f", {out / 'montage.png'}"))


ORG_HELP = (f"Catalog(s), comma-separated. Available: {', '.join(sorted(CATALOGS))} "
            f"(default {DEFAULT_CATALOGS})")


def cmd_particle_benchmark(args: argparse.Namespace) -> None:
    """Train a per-particle classifier on simulated particles and report accuracy."""
    import dataclasses
    import json

    import numpy as np
    import torch

    from .classifier import evaluate, report, train
    from .particles import Microscope, ParticleSimulator, build_views
    from .structures import load_atoms

    torch.set_num_threads(args.threads or torch.get_num_threads())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = [c for c in get_catalog(args.organism) if c.pdb_id]
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        catalog = [c for c in catalog if c.key in keep]
    structs = {}
    for c in catalog:
        path = cached_structure(c.pdb_id, c.assembly, Path(args.cache)) if args.cache else None
        structs[c.key] = load_atoms(path or fetch_structure(c.pdb_id, c.assembly))
    print(f"Rendering {args.train_views} training views and {args.test_views} test views "
          f"for {len(structs)} structures...", flush=True)
    train_views = build_views(structs, args.train_views, args.box, args.apix)
    test_views = build_views(structs, args.test_views, args.box, args.apix, rng=np.random.default_rng(123))

    scope = Microscope(apix=args.apix, dose=(args.dose_min, args.dose_max))
    sim = ParticleSimulator(train_views, scope)
    print(f"Training ({args.steps} steps of {args.batch}, dose {args.dose_min}-{args.dose_max} e/Å²)...")
    model, log = train(sim, steps=args.steps, batch=args.batch, units=args.units)
    torch.save({"state_dict": model.state_dict(), "keys": sim.keys, "units": args.units,
                "box": args.box, "apix": args.apix}, out / "particle_classifier.pt")

    summary = {"keys": sim.keys, "train_log": dataclasses.asdict(log), "eval": {}}
    text = []
    for dose in args.eval_doses:
        test_sim = ParticleSimulator(test_views, dataclasses.replace(scope, dose=dose))
        conf = evaluate(model, test_sim, per_class=args.eval_per_class)
        np.savetxt(out / f"confusion_dose{dose:g}.tsv", conf, fmt="%d", delimiter="\t",
                   header="\t".join(sim.keys))
        summary["eval"][f"{dose:g}"] = conf.tolist()
        block = f"== dose {dose:g} e/Å² (unseen view directions, fresh noise)\n" + report(conf, sim.keys)
        text.append(block)
        print("\n" + block, flush=True)
    (out / "report.txt").write_text("\n\n".join(text) + "\n")
    (out / "summary.json").write_text(json.dumps(summary))
    print(f"\nWrote {out}/report.txt, confusion matrices and particle_classifier.pt")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="junkfinder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="Show the contaminant catalog")
    s.add_argument("--organism", default=DEFAULT_CATALOGS, help=ORG_HELP)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("fetch", help="Download catalog structures into a folder")
    s.add_argument("--organism", default=DEFAULT_CATALOGS, help=ORG_HELP)
    s.add_argument("--dir", default="structures")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("verify", help="Check downloaded structures match the catalog")
    s.add_argument("--organism", default=DEFAULT_CATALOGS, help=ORG_HELP)
    s.add_argument("--dir", default="structures")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("build-library", help="Simulate reference projections")
    s.add_argument("-o", "--output", default="library.npz")
    s.add_argument("--organism", default=DEFAULT_CATALOGS, help=ORG_HELP)
    s.add_argument("--only", help="Comma-separated catalog keys to include")
    s.add_argument("--target", action="append",
                   help="Model of YOUR protein (PDB/mmCIF, e.g. AlphaFold). Strongly recommended.")
    s.add_argument("--structure", action="append", metavar="KEY=PATH",
                   help="Use a local file for a catalog entry instead of downloading")
    s.add_argument("--apix", type=float, default=4.0, help="Working pixel size (Å)")
    s.add_argument("--box", type=int, default=96, help="Working box (px)")
    s.add_argument("--resolution", type=float, default=15.0, help="Low-pass resolution (Å)")
    s.add_argument("--angular-step", type=float, default=7.5, help="Projection spacing (deg)")
    s.add_argument("--cache", help="Structure folder/cache (e.g. structures/ from 'fetch')")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("screen", help="Compare cryoSPARC class averages to the library")
    s.add_argument("class_averages", help="cryoSPARC *_class_averages.mrc")
    s.add_argument("--library", default="library.npz")
    s.add_argument("--cs", help="Matching *_class_averages.cs (for pixel size)")
    s.add_argument("--particles", help="2D job particles .cs (for per-class particle counts)")
    s.add_argument("--apix", type=float, help="Override pixel size (Å)")
    s.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                   help="Contrast handling; 'auto' expects particles brighter than background")
    s.add_argument("--max-shift", type=float, default=12.0, help="Translation search (Å)")
    s.add_argument("--mask-radius", type=float, help="Radius compared (Å); default 45%% of box")
    s.add_argument("--min-score", type=float, default=0.5)
    s.add_argument("--min-margin", type=float, default=0.03)
    s.add_argument("--out", default="junkfinder_out")
    s.add_argument("--no-montage", action="store_true")
    s.set_defaults(func=cmd_screen)

    s = sub.add_parser("particle-benchmark",
                       help="Train/evaluate a per-particle classifier on simulated particles (needs torch)")
    s.add_argument("--organism", default=DEFAULT_CATALOGS)
    s.add_argument("--only", help="Comma-separated catalog keys to include")
    s.add_argument("--cache", help="Structure folder (e.g. structures/)")
    s.add_argument("--out", default="particle_benchmark")
    s.add_argument("--box", type=int, default=64)
    s.add_argument("--apix", type=float, default=6.0)
    s.add_argument("--train-views", type=int, default=1000)
    s.add_argument("--test-views", type=int, default=100)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--units", type=int, default=32)
    s.add_argument("--dose-min", type=float, default=10.0)
    s.add_argument("--dose-max", type=float, default=40.0)
    s.add_argument("--eval-doses", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    s.add_argument("--eval-per-class", type=int, default=400)
    s.add_argument("--threads", type=int)
    s.set_defaults(func=cmd_particle_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
