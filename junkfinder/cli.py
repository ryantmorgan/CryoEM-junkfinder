"""Command-line interface: ``junkfinder build-library`` and ``junkfinder screen``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import CATALOGS, Contaminant, get_catalog
from .cryosparc_io import load_class_averages
from .library import Library, build_library
from .screen import screen, summarise, write_montage, write_tsv


def cmd_list(args: argparse.Namespace) -> None:
    for c in get_catalog(args.organism):
        flag = "  (verify ID)" if c.verify else ""
        kind = "membrane" if c.membrane else "soluble"
        print(f"{c.key:<14s} {c.pdb_id:<6s} ~{c.approx_kda:>5d} kDa  {kind:<8s} {c.name}{flag}")


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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="junkfinder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="Show the contaminant catalog")
    s.add_argument("--organism", default="ecoli", choices=sorted(CATALOGS))
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("build-library", help="Simulate reference projections")
    s.add_argument("-o", "--output", default="library.npz")
    s.add_argument("--organism", default="ecoli", choices=sorted(CATALOGS))
    s.add_argument("--only", help="Comma-separated catalog keys to include")
    s.add_argument("--target", action="append",
                   help="Model of YOUR protein (PDB/mmCIF, e.g. AlphaFold). Strongly recommended.")
    s.add_argument("--structure", action="append", metavar="KEY=PATH",
                   help="Use a local file for a catalog entry instead of downloading")
    s.add_argument("--apix", type=float, default=4.0, help="Working pixel size (Å)")
    s.add_argument("--box", type=int, default=96, help="Working box (px)")
    s.add_argument("--resolution", type=float, default=15.0, help="Low-pass resolution (Å)")
    s.add_argument("--angular-step", type=float, default=7.5, help="Projection spacing (deg)")
    s.add_argument("--cache", help="Structure download cache directory")
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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
