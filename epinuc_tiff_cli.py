#!/usr/bin/env python3
"""Command-line runner for the EPINUC TIF pipeline, driven by a GUI-tuned config.

Intended workflow
-----------------
1. Tune interactively, then save the parameters::

       streamlit run epinuc_tiff_gui.py
       # ... move the sliders, watch the overlays ... then press "💾 Save"
       # -> writes every tunable (spot SNR, coloc radius, bead + artifact knobs) to
       #    epinuc_config.json

2. Run the full analysis headlessly with exactly those parameters::

       python epinuc_tiff_cli.py --config epinuc_config.json --output per_run_output

The GUI and the CLI share one source of truth: ``epinuc_config.json`` is written by
``tl.save_config`` (all of ``ep.CONFIG_KEYS``, incl. the ``TIF_CHANNEL_MAP`` laser assignment) and
read back by ``ep.load_config``, so what you tuned is exactly what runs. The same file also serves
the ND2 path, which owns the separate ``CHANNEL_MAP`` key. Fields of view are spread across
``ncores-1`` worker processes, so a single sample still uses the whole machine.

Examples
--------
    python epinuc_tiff_cli.py --list                       # inventory only, no analysis
    python epinuc_tiff_cli.py                              # every run x lane, tuned config
    python epinuc_tiff_cli.py --runs NUC388 --lanes ch1 ch2
    python epinuc_tiff_cli.py --scenes 0-3 --n-jobs 4      # quick subset on 4 workers
    python epinuc_tiff_cli.py --no-artifact-masking        # override the saved flag
    python epinuc_tiff_cli.py --channel-map nucleosome=G,R_PTM=B,B_PTM=R   # red/blue swapped
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

import epinuc_colocalization as ep
import epinuc_tiff_loader as tl

DEFAULT_IMAGES_ROOT = "/Volumes/scBC/EpiVision/Images"
DEFAULT_CONFIG = "epinuc_config.json"
DEFAULT_OUTPUT = "per_run_output"
DEFAULT_RUN_INFO = "/Volumes/scBC/EpiVision/Images/Runinformation.xlsx"


def parse_scenes(spec: str | None):
    """``None`` -> all FOVs. Accepts ``"0-3"``, ``"0,2,5"`` or a mix (``"0-2,7"``)."""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_channel_map(spec: str | None):
    """``"nucleosome=G,R_PTM=B,B_PTM=R"`` -> ``{"nucleosome": "G", ...}``; ``None`` -> ``None``.

    Which laser carries which role. Validated by ``tl.validate_tif_channel_map`` (all three roles
    distinct, nucleosome present); errors surface as a clean ``SystemExit``, not a traceback.
    """
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"error: --channel-map entry {part!r} is not ROLE=LASER")
        role, laser = (s.strip() for s in part.split("=", 1))
        if role not in tl.ROLE_ORDER:
            raise SystemExit(f"error: --channel-map has unknown role {role!r} "
                             f"(expected one of {', '.join(tl.ROLE_ORDER)})")
        out[role] = laser.upper()
    try:
        return tl.validate_tif_channel_map(out)
    except ValueError as e:
        raise SystemExit(f"error: {e}")


def wait_for_mount(root: str, attempts: int = 5, delay: float = 1.0) -> None:
    """The /Volumes/scBC SMB share drops out; retry before failing with a clear message."""
    for _ in range(attempts):
        if os.path.isdir(root):
            return
        time.sleep(delay)
    raise SystemExit(f"error: {root} is not reachable — the network share appears unmounted. "
                     f"Remount it and re-run.")


def discover(images_root: str, only_runs=None, only_lanes=None):
    """Return ``({run: index_df}, [(run, lane), ...])`` restricted to the requested runs/lanes."""
    run_dirs = sorted(d for d in (os.path.join(images_root, n) for n in os.listdir(images_root))
                      if os.path.isdir(d))
    run_index, jobs = {}, []
    for d in run_dirs:
        run = os.path.basename(d)
        if only_runs and run not in only_runs:
            continue
        idx = tl.index_run(d)
        lanes = tl.lanes_in_run(idx)
        if only_lanes:
            lanes = [l for l in lanes if l in only_lanes]
        if not lanes:
            continue
        run_index[run] = idx
        jobs.extend((run, lane) for lane in lanes)
    return run_index, jobs


def build_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the EPINUC TIF pipeline with GUI-tuned parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tune with:  streamlit run epinuc_tiff_gui.py  (press Save)  then run this CLI.")
    p.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT,
                   help=f"folder containing the NUC### run folders (default: {DEFAULT_IMAGES_ROOT})")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"JSON written by the GUI's Save button (default: {DEFAULT_CONFIG}). "
                        f"Missing file -> module defaults.")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"output root; per-lane CSVs go to <output>/<run>/<lane>/ (default: {DEFAULT_OUTPUT})")
    p.add_argument("--runs", nargs="*", metavar="NUC###", help="restrict to these run folders")
    p.add_argument("--lanes", nargs="*", metavar="chN", help="restrict to these flowcell lanes")
    p.add_argument("--scenes", metavar="SPEC", help='FOV subset, e.g. "0-3" or "0,2,5" (default: all)')
    p.add_argument("--n-jobs", type=int, default=None,
                   help="worker processes; -2 = all cores but one (default, from ep.N_JOBS), "
                        "-1 = all cores, 1 = serial")
    p.add_argument("--run-info", default=DEFAULT_RUN_INFO, metavar="XLSX",
                   help="Run information spreadsheet; used to label results with the plasma sample "
                        "and red/blue antibody names (default: %(default)s). Pass '' to skip.")
    p.add_argument("--pixel-size", type=float, default=None, help="override PIXEL_SIZE_UM")
    p.add_argument("--channel-map", metavar="ROLE=LASER,...", default=None,
                   help="which laser letter carries which role, e.g. "
                        "'nucleosome=G,R_PTM=B,B_PTM=R'. Overrides TIF_CHANNEL_MAP from the config; "
                        f"default {tl.DEFAULT_TIF_CHANNEL_MAP}. Use --list to see a run's lasers.")
    p.add_argument("--qc", action="store_true", help="also write QC figures (slow)")
    p.add_argument("--list", action="store_true", help="print the run/lane inventory and exit")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--artifact-masking", dest="artifact_masking", action="store_true", default=None,
                   help="force blob/streak artifact masking ON (overrides the config)")
    g.add_argument("--no-artifact-masking", dest="artifact_masking", action="store_false",
                   help="force artifact masking OFF (overrides the config)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)

    # 1. Apply the GUI-tuned parameters, then the TIF-specific structural setup. Order matters:
    #    apply_config restores every CONFIG_KEY (incl. ARTIFACT_* and TIF_CHANNEL_MAP), and
    #    configure_pipeline then pins the identity channel map + TIME_COARSE_ALIGN=False, preserving
    #    the saved masking flag and channel map unless the user overrode them on the command line.
    #    Both must precede discover(), which stamps each file's role from the active channel map.
    channel_map = parse_channel_map(args.channel_map)
    if os.path.isfile(args.config):
        ep.apply_config(ep.load_config(args.config))
    else:
        print(f"note: {args.config} not found — using module defaults "
              f"(tune with: streamlit run epinuc_tiff_gui.py)", file=sys.stderr)
    masking = ep.ARTIFACT_MASKING if args.artifact_masking is None else args.artifact_masking
    tl.configure_pipeline(pixel_size_um=args.pixel_size, artifact_masking=masking,
                          tif_channel_map=channel_map)  # None -> keep the config's / the default
    ep.SAVE_QC_FIGURES = bool(args.qc)

    wait_for_mount(args.images_root)
    run_index, jobs = discover(args.images_root, args.runs, args.lanes)
    if not jobs:
        raise SystemExit("error: no matching run/lane found (check --images-root / --runs / --lanes)")

    if args.list:
        rows = []
        for r, l in jobs:
            li = run_index[r]
            li = li[li["lane"] == l]
            rows.append({"run": r, "lane": l, "sample": ",".join(sorted(li["sample"].unique())),
                         "n_FOVs": li["pos"].nunique(), "cycles": len(tl.lane_cycles(li, l)),
                         "lasers": ",".join(tl.lasers_in_run(li)), "n_files": len(li)})
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nchannel map: {tl.current_tif_channel_map()}   "
              f"(override with --channel-map ROLE=LASER,...)")
        return 0

    scenes = parse_scenes(args.scenes)
    n_jobs = ep.N_JOBS if args.n_jobs is None else args.n_jobs
    print(f"config           : {args.config if os.path.isfile(args.config) else '(defaults)'}")
    print(f"channel map      : {tl.current_tif_channel_map()}"
          f"{'  (--channel-map)' if channel_map else ''}")
    print(f"artifact masking : {ep.ARTIFACT_MASKING} "
          f"(pct={ep.ARTIFACT_BRIGHT_PCT}, min_area={ep.ARTIFACT_MIN_AREA_PX}, "
          f"dilation={ep.ARTIFACT_DILATION_PX})")
    print(f"spot SNR (k)     : {ep.SPOT_DETECTION_SNR}")
    print(f"coloc radius px  : {ep.COLOCALIZATION_RADIUS_PX}")
    print(f"lanes            : {len(jobs)}   FOVs: {'all' if scenes is None else scenes}")
    print(f"workers          : {tl.resolve_n_jobs(n_jobs)} of {os.cpu_count()} cores")

    t0 = time.time()
    all_summaries = tl.run_channels_parallel(
        run_index, jobs, scenes=scenes, output_root=args.output,
        n_jobs=n_jobs, save_qc=ep.SAVE_QC_FIGURES)

    os.makedirs(args.output, exist_ok=True)
    written = []

    summary_path = os.path.join(args.output, "all_lanes_summary.csv")
    all_summaries.to_csv(summary_path, index=False)
    written.append(summary_path)

    results_path = os.path.join(args.output, "results_format.csv")
    results = tl.results_format_table(all_summaries)
    results.to_csv(results_path, index=False)
    written.append(results_path)

    # Label the lanes with the plasma sample + red/blue antibody names from the run sheet.
    final = results
    if args.run_info and os.path.isfile(args.run_info):
        annotated = tl.annotate_summaries(all_summaries, args.run_info)
        ann_path = os.path.join(args.output, "results_annotated.csv")
        annotated.to_csv(ann_path, index=False)
        written.append(ann_path)

        long_path = os.path.join(args.output, "results_by_antibody.csv")
        tl.annotated_long_table(annotated).to_csv(long_path, index=False)
        written.append(long_path)
        final = annotated
    elif args.run_info:
        print(f"note: run info not found at {args.run_info} — results left unannotated.",
              file=sys.stderr)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(final.to_string(index=False))
    print("\nWrote " + "\n      ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
