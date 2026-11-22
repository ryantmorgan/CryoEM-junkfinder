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
        membrane_env=args.membrane_env, target_env=args.target_env,
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


def cmd_scan(args: argparse.Namespace) -> None:
    """Sliding-window scan of whole micrographs with a trained classifier."""
    import glob
    from collections import Counter

    import torch

    from .apply import load_model
    from .scan import save_overlay, scan_micrograph

    paths = sorted({p for pat in args.micrographs for p in (glob.glob(pat) or [pat])})
    apix_of: dict[str, float] = {}
    defocus_of: dict[str, float | None] = {}
    if args.exposures:
        from .cryosparc_io import exposures_from_cs

        if not args.project_dir:
            raise SystemExit("--exposures needs --project-dir (the cryoSPARC project folder)")
        exp = exposures_from_cs(args.exposures, Path(args.project_dir), with_defocus=True)
        print(f"{len(exp)} accepted exposures from {', '.join(args.exposures)}")
        paths += [str(p) for p, _, _ in exp]
        apix_of = {str(p): a for p, a, _ in exp}
        defocus_of = {str(p): df for p, _, df in exp}
    if args.max_micrographs:
        paths = paths[:args.max_micrographs]
    if not paths:
        raise SystemExit("No micrographs found.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.model, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ckpt["keys"]
    total, rows = Counter(), []
    with open(out / "picks.tsv", "w") as f:
        f.write("micrograph\tx\ty\tlabel\tprob\tscore\n")
        for i, path in enumerate(paths):
            picks, _, x, geom = scan_micrograph(path, model, ckpt, args.stride, args.threshold,
                                                args.min_dist, args.apix or apix_of.get(str(path)),
                                                args.invert == "yes", device)
            counts = Counter(p.label for p in picks)
            total.update(counts)
            rows.append((Path(path).name, len(picks), counts))
            for p in picks:
                f.write(f"{path}\t{p.x:.1f}\t{p.y:.1f}\t{p.label}\t{p.prob:.3f}\t{p.score:.3f}\n")
            if i < args.overlays:
                save_overlay(out / f"{Path(path).stem}_picks.png", x, picks, ckpt, geom)
            df = defocus_of.get(str(path))
            lo, hi = ckpt.get("defocus_um", (0.8, 2.5))
            dtxt = "" if df is None else f" (defocus {df:.2f} µm{'' if lo <= df <= hi else ', OUTSIDE training range'})"
            print(f"  [{i + 1}/{len(paths)}] {Path(path).name}{dtxt}: {len(picks)} picks  " +
                  ", ".join(f"{k} {v}" for k, v in counts.most_common(4)), flush=True)
    n = sum(total.values())
    lines = [f"{len(paths)} micrographs, {n} picks (threshold {args.threshold}, stride {args.stride} Å)",
             "", "Picks by label:"]
    lines += [f"  {k:<22s} {v:7d}  {v / max(n, 1):6.1%}" for k, v in total.most_common()]
    lines += ["", "Micrographs where each label is the most common pick:"]
    tops = Counter(c.most_common(1)[0][0] for _, _, c in rows if c)
    lines += [f"  {k:<22s} {v:5d}" for k, v in tops.most_common()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nWrote {out}/picks.tsv, summary.txt and {min(args.overlays, len(paths))} overlays")


def cmd_fit_env(args: argparse.Namespace) -> None:
    """Fit micelle/nanodisc/SMA parameters to class averages of one membrane protein."""
    import dataclasses
    import json

    from .fit_env import describe, fit_environment

    imgs, apix, _ = load_class_averages(args.class_averages, args.cs, None, args.apix)
    if args.classes:
        keep = [int(x) for x in args.classes.split(",")]
        imgs = imgs[keep]
    structure = args.structure
    if structure is None:
        c = next((c for c in get_catalog(DEFAULT_CATALOGS) if c.key == args.key), None)
        if c is None:
            raise SystemExit(f"Unknown catalog key '{args.key}'; give --structure PATH instead.")
        structure = (cached_structure(c.pdb_id, c.assembly, Path(args.cache)) if args.cache else None) \
            or fetch_structure(c.pdb_id, c.assembly)
    print(f"Fitting a {args.env} to {len(imgs)} class averages at {apix:.2f} Å/px "
          f"({args.random} random candidates, {args.refine} refinement rounds)...", flush=True)
    r = fit_environment(structure, imgs, apix, args.env, n_random=args.random, n_refine=args.refine,
                        workers=args.workers, invert={"auto": None, "yes": True, "no": False}[args.invert])
    m, scores, _, p = r["best"]
    print(f"\nMean best NCC: bare protein {r['bare'][0]:.4f}, typical {args.env} "
          f"{r['typical'][0]:.4f}, fitted {m:.4f}")
    print(f"Fitted {args.env}: {describe(p, args.env)}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "env": args.env, "apix": apix, "score": m, "params": dataclasses.asdict(p),
            "per_class": {"bare": r["bare"][1], "typical": r["typical"][1], "fitted": scores},
        }, indent=1, default=float))
        print(f"Wrote {args.out}")


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify real particles with a model trained by particle-benchmark."""
    import numpy as np
    import torch

    from .apply import classify_cs, load_model, summarise_predictions, write_particle_table
    from .cryosparc_io import read_particles

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.model, device)
    cs = read_particles(args.particles)
    if args.max_particles and len(cs) > args.max_particles:
        cs = cs[np.random.default_rng(0).choice(len(cs), args.max_particles, replace=False)]
    apix_in = args.apix or float(cs["blob/psize_A"][0])
    print(f"{len(cs)} particles at {apix_in:.2f} Å/px, box {int(cs['blob/shape'][0][0])} px -> "
          f"model grid {ckpt['apix']} Å/px, {ckpt['box']} px; {len(ckpt['keys'])} classes; device {device}")
    if "ctf/df1_A" in cs.dtype.names and "defocus_um" in ckpt:
        df = (cs["ctf/df1_A"] + cs["ctf/df2_A"]) / 2e4
        lo, hi = ckpt["defocus_um"]
        frac = float(np.mean((df < lo) | (df > hi)))
        print(f"Defocus {np.percentile(df, 5):.2f}-{np.percentile(df, 95):.2f} µm (5-95%); "
              f"model trained on {lo}-{hi} µm" + (f"; {frac:.0%} of particles outside" if frac > 0.05 else ""))
    probs, pred = classify_cs(cs, Path(args.project_dir), model, ckpt, apix_in, args.invert == "yes",
                              device=device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_particle_table(out / "particles.tsv", cs, probs, ckpt["keys"])
    np.save(out / "probabilities.npy", probs)
    text = summarise_predictions(cs, pred, probs, ckpt["keys"])
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nWrote {out}/particles.tsv, summary.txt and probabilities.npy")


ENV_KINDS = {"none": [], "micelle": ["micelle"], "nanodisc": ["nanodisc"], "sma": ["sma"],
             "both": ["micelle", "nanodisc"], "all": ["micelle", "nanodisc", "sma"]}


def parse_targets(items, target_env: str):
    """--target NAME=PDBID|PATH -> (Contaminant list, local paths, environment per key)."""
    cats, local_paths, env_of = [], {}, {}
    for item in items or []:
        name, sep, src = item.partition("=")
        if not sep or not name or not src:
            raise SystemExit(f"--target expects NAME=PDBID or NAME=PATH, got '{item}'")
        key = f"target_{name}"
        if Path(src).exists():
            local_paths[key] = Path(src)
            cats.append(Contaminant(key, f"Target {name}", "", 0, target_env != "none"))
        else:
            cats.append(Contaminant(key, f"Target {name}", src.upper(), 0, target_env != "none"))
        env_of[key] = ENV_KINDS[target_env]
    return cats, local_paths, env_of


def render_views(catalog, local_paths, env_of, default_kinds, n_train, n_test, box, apix,
                 env_variants=24, cache=None, seed_train=7, seed_test=123):
    """Thickness views per structure; membrane proteins get several random environments."""
    import numpy as np

    from .membrane import MembraneModel
    from .particles import build_views
    from .structures import load_atoms

    rng_train, rng_test = np.random.default_rng(seed_train), np.random.default_rng(seed_test)
    train_views, test_views = {}, {}
    for c in catalog:
        path = local_paths.get(c.key)
        if path is None:
            path = cached_structure(c.pdb_id, c.assembly, Path(cache)) if cache else None
            path = path or fetch_structure(c.pdb_id, c.assembly, **({"cache_dir": Path(cache)} if cache else {}))
        kinds = env_of.get(c.key, default_kinds)
        if c.membrane and kinds:
            # Several random environments per structure; test ones are never seen in training.
            model = MembraneModel(path)
            for views, n, rng in ((train_views, n_train, rng_train), (test_views, n_test, rng_test)):
                if n <= 0:
                    continue
                k = max(1, min(env_variants, n))
                parts = [build_views({c.key: model.variant(kinds[i % len(kinds)], rng)},
                                     n // k + (i < n % k), box, apix, rng=rng)[c.key]
                         for i in range(k)]
                views[c.key] = np.concatenate(parts)
        else:
            atoms = load_atoms(path)
            train_views[c.key] = build_views({c.key: atoms}, n_train, box, apix)[c.key]
            if n_test > 0:
                test_views[c.key] = build_views({c.key: atoms}, n_test, box, apix, rng=rng_test)[c.key]
    return train_views, test_views


def cmd_finetune(args: argparse.Namespace) -> None:
    """Fine-tune a trained classifier on real particles labelled by your 2D classes."""
    import numpy as np
    import torch

    from .apply import load_model
    from .catalog import CATALOGS
    from .cryosparc_io import read_particles
    from .finetune import confusion_report, finetune, load_real, real_confusion
    from .particles import Microscope, ParticleSimulator

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.model, device)
    keys = ckpt["keys"]
    # Real particles per label.
    xs, ys = [], []
    for item in args.real:
        label, sep, path = item.partition("=")
        if not sep or label not in keys:
            raise SystemExit(f"--real expects LABEL=particles.cs with LABEL one of: {', '.join(keys)}")
        x = load_real(read_particles(path), Path(args.project_dir), ckpt, args.max_per_label,
                      args.invert == "yes")
        print(f"  {label}: {len(x)} real particles from {path}", flush=True)
        xs.append(x)
        ys.append(np.full(len(x), keys.index(label)))
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    rng = np.random.default_rng(0)
    val = np.zeros(len(x_all), bool)
    for c in np.unique(y_all):
        idx = np.nonzero(y_all == c)[0]
        val[rng.choice(idx, max(1, int(len(idx) * args.val_frac)), replace=False)] = True
    labels = sorted(set(y_all.tolist()))
    before = real_confusion(model, x_all[val], y_all[val], len(keys), device)
    print("\nHeld-out real particles BEFORE fine-tuning:\n" + confusion_report(before, keys, labels), flush=True)

    # Simulated replay for every class the model knows (same keys, same order).
    known = {}
    for cat in CATALOGS.values():
        for c in cat:
            known.setdefault(c.key, c)
    tcats, local_paths, env_of = parse_targets(args.target, args.target_env)
    for c in tcats:
        known[c.key] = c
    missing = [k for k in keys if k != "background" and k not in known]
    if missing:
        raise SystemExit(f"No structure for {', '.join(missing)}; pass --target NAME=PDBID for targets.")
    catalog = [known[k] for k in keys if k != "background"]
    print(f"\nRendering {args.sim_views} simulated views per class for replay...", flush=True)
    views, _ = render_views(catalog, local_paths, env_of, ENV_KINDS[ckpt.get("membrane_env", "micelle")],
                            args.sim_views, 0, ckpt["box"], ckpt["apix"], args.env_variants, args.cache)
    lo, hi = ckpt.get("dose", [10.0, 40.0])
    sim = ParticleSimulator(views, Microscope(apix=ckpt["apix"], dose=(lo, hi)), device=device)
    assert sim.keys == keys, "simulated class order must match the model"
    print(f"Fine-tuning ({args.steps} steps of {args.batch}, {args.frac_real:.0%} real)...", flush=True)
    model = finetune(model, sim, x_all[~val], y_all[~val], args.steps, args.batch, args.frac_real, args.lr)
    after = real_confusion(model, x_all[val], y_all[val], len(keys), device)
    text = ("Held-out real particles BEFORE fine-tuning:\n" + confusion_report(before, keys, labels) +
            "\n\nAFTER fine-tuning:\n" + confusion_report(after, keys, labels))
    print("\n" + text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = dict(ckpt, state_dict=model.state_dict(), finetuned_on=args.real)
    torch.save(ckpt, out / "particle_classifier.pt")
    (out / "report.txt").write_text(text + "\n")
    print(f"\nWrote {out}/particle_classifier.pt and report.txt")


def cmd_particle_benchmark(args: argparse.Namespace) -> None:
    """Train a per-particle classifier on simulated particles and report accuracy."""
    import dataclasses
    import json

    import numpy as np
    import torch

    from .classifier import evaluate, report, train
    from .particles import Microscope, ParticleSimulator

    torch.set_num_threads(args.threads or torch.get_num_threads())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.startswith("cuda") else
                                 f" ({torch.get_num_threads()} threads)"), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = [c for c in get_catalog(args.organism) if c.pdb_id]
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        catalog = [c for c in catalog if c.key in keep]
    env_kinds = ENV_KINDS
    # Your own protein(s) as extra classes: NAME=PDBID or NAME=path/to/model.
    tcats, local_paths, env_of = parse_targets(args.target, args.target_env)
    catalog += tcats
    print(f"Rendering {args.train_views} training views and {args.test_views} test views "
          f"for {len(catalog)} structures (membrane environment: {args.membrane_env})...", flush=True)
    train_views, test_views = render_views(
        catalog, local_paths, env_of, env_kinds[args.membrane_env], args.train_views, args.test_views,
        args.box, args.apix, args.env_variants, args.cache)

    scope = Microscope(apix=args.apix, dose=(args.dose_min, args.dose_max))
    sim = ParticleSimulator(train_views, scope, device=device)
    print(f"Training ({args.steps} steps of {args.batch}, dose {args.dose_min}-{args.dose_max} e/Å²)...")
    model, log = train(sim, steps=args.steps, batch=args.batch, units=args.units)
    torch.save({"state_dict": model.state_dict(), "keys": sim.keys, "units": args.units,
                "box": args.box, "apix": args.apix, "membrane_env": args.membrane_env,
                "defocus_um": list(scope.defocus_um), "dose": [args.dose_min, args.dose_max]}, out / "particle_classifier.pt")

    summary = {"keys": sim.keys, "membrane_env": args.membrane_env,
               "train_log": dataclasses.asdict(log), "eval": {}}
    text = []
    for dose in args.eval_doses:
        test_sim = ParticleSimulator(test_views, dataclasses.replace(scope, dose=dose), device=device)
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
    s.add_argument("--membrane-env", choices=["none", "micelle", "nanodisc", "sma"], default="none",
                   help="Surround membrane contaminants with a detergent micelle, nanodisc or SMA disc")
    s.add_argument("--target-env", choices=["none", "micelle", "nanodisc", "sma"], default="none",
                   help="Same for --target models (use it if your target is a membrane protein)")
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

    s = sub.add_parser("classify", help="Classify real cryoSPARC particles with a trained "
                       "particle_classifier.pt (needs torch)")
    s.add_argument("particles", help="Particles .cs (Extract / 2D Classification / Select 2D job); "
                   "add the passthrough file comma-separated if blob/path lives there")
    s.add_argument("--model", required=True, help="particle_classifier.pt from particle-benchmark")
    s.add_argument("--project-dir", required=True,
                   help="cryoSPARC project directory (blob/path entries are relative to it)")
    s.add_argument("--apix", type=float, help="Override particle pixel size (Å)")
    s.add_argument("--invert", choices=["no", "yes"], default="no",
                   help="'yes' if particles were extracted with flipped contrast (protein white)")
    s.add_argument("--max-particles", type=int, help="Classify a random subset of this size")
    s.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda if available)")
    s.add_argument("--out", default="classify_out")
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("finetune", help="Fine-tune a trained classifier on real particles labelled "
                       "by your 2D classes (needs torch)")
    s.add_argument("--model", required=True, help="particle_classifier.pt to start from")
    s.add_argument("--real", nargs="+", required=True, metavar="LABEL=CS",
                   help="Real particles per label; for Select 2D give the job's file and its "
                        "passthrough, comma-separated: cyo_bo3=J60/particles_selected.cs,"
                        "J60/J60_passthrough_particles_selected.cs")
    s.add_argument("--project-dir", required=True, help="cryoSPARC project folder")
    s.add_argument("--target", action="append", metavar="NAME=PDBID|PATH",
                   help="Same --target as used for training (needed to simulate it)")
    s.add_argument("--target-env", choices=list(ENV_KINDS), default="micelle")
    s.add_argument("--cache", default="structures")
    s.add_argument("--max-per-label", type=int, default=20000)
    s.add_argument("--val-frac", type=float, default=0.1, help="Held-out real particles per label")
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--frac-real", type=float, default=0.5, help="Fraction of each batch from real data")
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--sim-views", type=int, default=500, help="Simulated views per class for replay")
    s.add_argument("--env-variants", type=int, default=12)
    s.add_argument("--invert", choices=["no", "yes"], default="no")
    s.add_argument("--device")
    s.add_argument("--out", default="finetuned")
    s.set_defaults(func=cmd_finetune)

    s = sub.add_parser("scan", help="Scan whole micrographs with a trained particle_classifier.pt "
                       "(sliding window, Topaz-style; needs torch)")
    s.add_argument("micrographs", nargs="*", help="Micrograph .mrc files or glob patterns (quote globs)")
    s.add_argument("--exposures", nargs="+", metavar="CS",
                   help="cryoSPARC exposures .cs file(s), e.g. a Curate Exposures job's "
                        "J5_exposures_accepted.cs and J5_passthrough_exposures_accepted.cs")
    s.add_argument("--project-dir", help="cryoSPARC project folder (paths in --exposures are relative to it)")
    s.add_argument("--model", required=True, help="particle_classifier.pt from particle-benchmark")
    s.add_argument("--apix", type=float, help="Micrograph pixel size (Å) if the header lacks it")
    s.add_argument("--stride", type=float, default=24.0, help="Window spacing (Å)")
    s.add_argument("--threshold", type=float, default=0.7, help="Min 1 - P(background) for a pick")
    s.add_argument("--min-dist", type=float, help="Min distance between picks (Å); default box/4")
    s.add_argument("--invert", choices=["no", "yes"], default="no",
                   help="'yes' if protein is white in the micrographs")
    s.add_argument("--max-micrographs", type=int, help="Only scan the first N (sorted)")
    s.add_argument("--overlays", type=int, default=5, help="Save pick overlays for the first N")
    s.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda if available)")
    s.add_argument("--out", default="scan_out")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("fit-env", help="Fit a micelle/nanodisc/SMA model to class averages of a "
                       "membrane protein (e.g. bo3 in DDM)")
    s.add_argument("class_averages", help="Class averages .mrc (classes of one protein only)")
    s.add_argument("--key", default="cyo_bo3", help="Catalog key of the protein (default cyo_bo3)")
    s.add_argument("--structure", help="Model file instead of a catalog key")
    s.add_argument("--cache", help="Structure folder (e.g. structures/)")
    s.add_argument("--env", choices=["micelle", "nanodisc", "sma"], default="micelle")
    s.add_argument("--cs", help="Matching *_class_averages.cs (for pixel size)")
    s.add_argument("--apix", type=float, help="Override pixel size (Å)")
    s.add_argument("--classes", help="Comma-separated class indices to use (default: all)")
    s.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    s.add_argument("--random", type=int, default=48, help="Random candidates")
    s.add_argument("--refine", type=int, default=4, help="Local refinement rounds")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out", help="Write the fit (JSON)")
    s.set_defaults(func=cmd_fit_env)

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
    s.add_argument("--membrane-env", choices=["none", "micelle", "nanodisc", "sma", "both", "all"],
                   default="micelle",
                   help="Environment around membrane proteins ('both' = micelle + nanodisc, 'all' adds SMA)")
    s.add_argument("--target", action="append", metavar="NAME=PDBID|PATH",
                   help="Your own protein as an extra class (repeatable), e.g. rodA_pbp2=8TJ3")
    s.add_argument("--target-env", choices=["none", "micelle", "nanodisc", "sma", "both", "all"],
                   default="micelle", help="Environment around --target proteins (default micelle)")
    s.add_argument("--env-variants", type=int, default=24,
                   help="Random micelles/nanodiscs per membrane protein (train and test separately)")
    s.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda if available)")
    s.set_defaults(func=cmd_particle_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
