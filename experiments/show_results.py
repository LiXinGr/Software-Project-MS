#!/usr/bin/env python3
"""
Show experiment results as a formatted comparison table.

Usage:
    python3 experiments/show_results.py
    python3 experiments/show_results.py --method dinov3
    python3 experiments/show_results.py --run_id phase0_real_dinov3
    python3 experiments/show_results.py --scene sacre_coeur
    python3 experiments/show_results.py --csv
    python3 experiments/show_results.py --compare run1 run2

Output columns: Method | Run ID | Scene | Cal mAA10 | SF mAA10 | VF mAA10 | Inliers%
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
KNOWN_SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]
EXP_TYPES = ["calibrated", "shared_focal", "varying_focal"]

# Primary solvers used in the thesis — used to extract summary mAA values
PRIMARY_SOLVERS = {
    "calibrated":    "3p_ours_shift_scale+12",
    "shared_focal":  "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}


def load_experiments(filters=None):
    """Load all experiment JSON files, applying optional filters."""
    filters = filters or {}
    records = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(path) as f:
            rec = json.load(f)
        if filters.get("run_id") and rec.get("run_id") not in filters["run_id"]:
            continue
        if filters.get("method") and rec.get("method") not in filters["method"]:
            continue
        records.append(rec)
    return records


def extract_rows(records, scene_filter=None):
    """Extract per-scene rows from records. Returns list of dicts."""
    rows = []
    for rec in records:
        scenes = rec.get("scenes", {})
        for scene, scene_data in scenes.items():
            if scene_filter and scene not in scene_filter:
                continue
            row = {
                "method":     rec.get("method", "?"),
                "run_id":     rec.get("run_id", "?"),
                "config_key": rec.get("config_key", "?"),
                "scene":      scene,
            }
            # Prefer primary-solver values from eval JSONs when available
            solver_data = load_solver_data(rec, scene)
            for et in EXP_TYPES:
                primary = PRIMARY_SOLVERS[et]
                if et in solver_data and primary in solver_data[et]:
                    m = compute_solver_metrics({primary: solver_data[et][primary]})[primary]
                    row[et + "_mAA10"]  = round(m["mAA10"], 2)
                    row[et + "_inlier"] = round(m["mean_inlier"] * 100, 2)
                else:
                    # Fall back to stored JSON values (e.g. semestral reference files)
                    stored = scene_data.get(et, {})
                    row[et + "_mAA10"]  = stored.get("mAA10")
                    row[et + "_inlier"] = stored.get("inlier_pct")
            rows.append(row)
    return rows


def avg_rows(rows):
    """Compute the average across all scenes for each (method, run_id) combo."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["run_id"], row["config_key"])
        groups[key].append(row)

    avg = []
    for (method, run_id, config_key), group in groups.items():
        a = {"method": method, "run_id": run_id, "config_key": config_key,
             "scene": f"AVERAGE ({len(group)} scene{'s' if len(group) != 1 else ''})"}
        for et in EXP_TYPES:
            vals = [r[et + "_mAA10"] for r in group if r[et + "_mAA10"] is not None]
            a[et + "_mAA10"] = round(sum(vals) / len(vals), 2) if vals else None
            ivals = [r[et + "_inlier"] for r in group if r[et + "_inlier"] is not None]
            a[et + "_inlier"] = round(sum(ivals) / len(ivals), 2) if ivals else None
        avg.append(a)
    return avg


def fmt(v, digits=2):
    if v is None:
        return "  -  "
    return f"{v:.{digits}f}"


def print_table(rows, include_avg=True):
    """Print a formatted ASCII table."""
    all_rows = list(rows)
    avgs = avg_rows(all_rows)

    # Interleave averages after each group
    from collections import defaultdict
    groups = defaultdict(list)
    for row in all_rows:
        groups[(row["method"], row["run_id"])].append(row)

    # Column widths
    w_method  = max(6, max((len(r["method"])  for r in all_rows), default=6))
    w_run_id  = max(6, max((len(r["run_id"])  for r in all_rows), default=6))
    w_scene   = max(5, max((len(r["scene"])   for r in all_rows), default=5))
    # Also account for AVERAGE scene label
    for a in avgs:
        w_scene = max(w_scene, len(a["scene"]))
    w_maa     = 9
    w_inlier  = 8

    sep = (
        f"{'─' * w_method}─"
        f"{'─' * w_run_id}─"
        f"{'─' * w_scene}─"
        f"{'─' * w_maa}──"
        f"{'─' * w_maa}──"
        f"{'─' * w_maa}──"
        f"{'─' * w_inlier}"
    )

    hdr = (
        f"{'Method':<{w_method}} "
        f"{'Run ID':<{w_run_id}} "
        f"{'Scene':<{w_scene}} "
        f"{'Cal mAA10':>{w_maa}}  "
        f"{'SF mAA10':>{w_maa}}  "
        f"{'VF mAA10':>{w_maa}}  "
        f"{'Inliers%':>{w_inlier}}"
    )

    print(sep)
    print(hdr)
    print(sep)

    first_group = True
    for (method, run_id), group_rows in groups.items():
        if not first_group:
            print()
        first_group = False
        for i, row in enumerate(group_rows):
            m  = row["method"]  if i == 0 else ""
            ri = row["run_id"]  if i == 0 else ""
            print(
                f"{m:<{w_method}} "
                f"{ri:<{w_run_id}} "
                f"{row['scene']:<{w_scene}} "
                f"{fmt(row['calibrated_mAA10']):>{w_maa}}  "
                f"{fmt(row['shared_focal_mAA10']):>{w_maa}}  "
                f"{fmt(row['varying_focal_mAA10']):>{w_maa}}  "
                f"{fmt(row['calibrated_inlier']):>{w_inlier}}"
            )
        if include_avg:
            avg_row = next(a for a in avgs if a["run_id"] == run_id and a["method"] == method)
            print(
                f"{'':>{w_method}} "
                f"{'AVERAGE':<{w_run_id}} "
                f"{avg_row['scene']:<{w_scene}} "
                f"{fmt(avg_row['calibrated_mAA10']):>{w_maa}}  "
                f"{fmt(avg_row['shared_focal_mAA10']):>{w_maa}}  "
                f"{fmt(avg_row['varying_focal_mAA10']):>{w_maa}}  "
                f"{fmt(avg_row['calibrated_inlier']):>{w_inlier}}"
            )

    print(sep)


def write_csv(rows, out=None):
    """Write results to CSV (stdout or file)."""
    avgs = avg_rows(rows)
    writer = csv.writer(out or sys.stdout)
    writer.writerow(["method", "run_id", "config_key", "scene",
                     "cal_mAA10", "sf_mAA10", "vf_mAA10", "inliers_pct"])

    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["run_id"])].append(row)

    for (method, run_id), group_rows in groups.items():
        for row in group_rows:
            writer.writerow([
                row["method"], row["run_id"], row["config_key"], row["scene"],
                row["calibrated_mAA10"], row["shared_focal_mAA10"],
                row["varying_focal_mAA10"], row["calibrated_inlier"],
            ])
        avg_row = next(a for a in avgs if a["run_id"] == run_id and a["method"] == method)
        writer.writerow([
            avg_row["method"], avg_row["run_id"], avg_row["config_key"], avg_row["scene"],
            avg_row["calibrated_mAA10"], avg_row["shared_focal_mAA10"],
            avg_row["varying_focal_mAA10"], avg_row["calibrated_inlier"],
        ])


def print_compare(records_a, records_b, run_id_a, run_id_b, scene_filter=None):
    """Print side-by-side comparison of two runs with delta."""
    rows_a = {r["scene"]: r for r in extract_rows(records_a, scene_filter)}
    rows_b = {r["scene"]: r for r in extract_rows(records_b, scene_filter)}

    scenes = sorted(set(rows_a) | set(rows_b), key=lambda s: KNOWN_SCENES.index(s) if s in KNOWN_SCENES else 99)

    method_a = records_a[0].get("method", "?") if records_a else "?"
    method_b = records_b[0].get("method", "?") if records_b else "?"

    label_a = f"{method_a}/{run_id_a}"
    label_b = f"{method_b}/{run_id_b}"

    w_scene = max(5, max((len(s) for s in scenes), default=5))
    w_maa = 8

    col_a = max(len(label_a), 22)
    col_b = max(len(label_b), 22)

    # Header
    hdr1 = f"{'Scene':<{w_scene}}  {'':^{col_a}}  {'':^{col_b}}  {'Delta':^24}"
    hdr2 = (
        f"{'Scene':<{w_scene}}  "
        f"{'Cal':>{w_maa}} {'SF':>{w_maa}} {'VF':>{w_maa}}  "
        f"{'Cal':>{w_maa}} {'SF':>{w_maa}} {'VF':>{w_maa}}  "
        f"{'ΔCal':>7} {'ΔSF':>7} {'ΔVF':>7}"
    )
    sep_len = len(hdr2)
    sep = "─" * sep_len

    print(f"\n  {label_a:^{col_a}}  {label_b:^{col_b}}")
    print(sep)
    print(hdr2)
    print(sep)

    cal_deltas, sf_deltas, vf_deltas = [], [], []
    for scene in scenes:
        ra = rows_a.get(scene)
        rb = rows_b.get(scene)
        ca  = ra["calibrated_mAA10"]   if ra else None
        sfa = ra["shared_focal_mAA10"] if ra else None
        vfa = ra["varying_focal_mAA10"] if ra else None
        cb  = rb["calibrated_mAA10"]   if rb else None
        sfb = rb["shared_focal_mAA10"] if rb else None
        vfb = rb["varying_focal_mAA10"] if rb else None

        def delta(a, b):
            if a is None or b is None:
                return None
            return b - a

        dc  = delta(ca, cb)
        ds  = delta(sfa, sfb)
        dv  = delta(vfa, vfb)

        def dfmt(v):
            if v is None:
                return "   -   "
            s = f"{v:+.2f}"
            # ANSI colors: green for positive, red for negative
            if v > 0:
                return f"\033[32m{s:>7}\033[0m"
            elif v < 0:
                return f"\033[31m{s:>7}\033[0m"
            return f"{s:>7}"

        print(
            f"{scene:<{w_scene}}  "
            f"{fmt(ca):>{w_maa}} {fmt(sfa):>{w_maa}} {fmt(vfa):>{w_maa}}  "
            f"{fmt(cb):>{w_maa}} {fmt(sfb):>{w_maa}} {fmt(vfb):>{w_maa}}  "
            f"{dfmt(dc)} {dfmt(ds)} {dfmt(dv)}"
        )
        if dc is not None: cal_deltas.append(dc)
        if ds is not None: sf_deltas.append(ds)
        if dv is not None: vf_deltas.append(dv)

    # Average delta row
    if cal_deltas or sf_deltas or vf_deltas:
        print(sep)
        avg_dc = sum(cal_deltas) / len(cal_deltas) if cal_deltas else None
        avg_ds = sum(sf_deltas)  / len(sf_deltas)  if sf_deltas  else None
        avg_dv = sum(vf_deltas)  / len(vf_deltas)  if vf_deltas  else None

        def dfmt_avg(v):
            if v is None:
                return "   -   "
            return f"{v:+.2f}".rjust(7)

        print(
            f"{'AVERAGE':<{w_scene}}  "
            f"{'':>{w_maa}} {'':>{w_maa}} {'':>{w_maa}}  "
            f"{'':>{w_maa}} {'':>{w_maa}} {'':>{w_maa}}  "
            f"{dfmt_avg(avg_dc)} {dfmt_avg(avg_ds)} {dfmt_avg(avg_dv)}"
        )
    print(sep)


EXP_TYPE_LABELS = {
    "calibrated":    "Calibrated (known focal)",
    "shared_focal":  "Shared Focal",
    "varying_focal": "Varying Focal",
}


def load_solver_data(record, scene):
    """
    Load per-solver metrics from raw eval JSONs for a given record + scene.

    Returns dict: {exp_type: {solver_name: {pose_errs, runtimes, inlier_ratios}}}
    """
    results_path = record.get("paths", {}).get("results")
    config_key   = record.get("config_key", "")

    if not results_path:
        return {}

    results_dir = Path(results_path)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    scene_dir = results_dir / scene

    data = {}
    for et in EXP_TYPES:
        json_path = scene_dir / f"{et}-{config_key}_{scene}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            entries = json.load(f)
        solvers = {}
        for e in entries:
            solver = e.get("experiment", "unknown")
            if solver not in solvers:
                solvers[solver] = {"pose_errs": [], "runtimes": [], "inlier_ratios": []}
            r_err = e.get("R_err", float("nan"))
            t_err = e.get("t_err", float("nan"))
            pose_err = max(r_err, t_err) if not (r_err != r_err or t_err != t_err) else float("nan")
            solvers[solver]["pose_errs"].append(pose_err)
            info = e.get("info", {})
            solvers[solver]["runtimes"].append(info.get("runtime", float("nan")))
            solvers[solver]["inlier_ratios"].append(info.get("inlier_ratio", float("nan")))
        data[et] = solvers
    return data


def compute_solver_metrics(solver_data):
    """Convert raw arrays to summary metrics per solver."""
    metrics = {}
    for solver, raw in solver_data.items():
        pose_errs     = np.array(raw["pose_errs"],     dtype=float)
        runtimes      = np.array(raw["runtimes"],      dtype=float)
        inlier_ratios = np.array(raw["inlier_ratios"], dtype=float)

        pose_errs_clamped = np.where(np.isnan(pose_errs), 180.0, pose_errs)
        maa = float(np.mean([np.sum(pose_errs_clamped < t) / len(pose_errs_clamped)
                             for t in range(1, 11)]) * 100)
        metrics[solver] = {
            "median_pose_err": float(np.nanmedian(pose_errs)),
            "mAA10":           round(maa, 4),
            "mean_time_ms":    float(np.nanmean(runtimes)),
            "mean_inlier":     float(np.nanmean(inlier_ratios)),
            "n_pairs":         len(pose_errs),
        }
    return metrics


def print_solver_tables(record, scenes):
    """Print per-solver breakdown tables for each scene and experiment type."""
    run_id     = record.get("run_id", "?")
    config_key = record.get("config_key", "?")
    method     = record.get("method", "?")

    for scene in scenes:
        solver_data = load_solver_data(record, scene)
        if not solver_data:
            print(f"  (no eval JSONs found for {run_id} / {scene})")
            continue

        print(f"\n  {method}  |  {run_id}  |  {scene}  |  {config_key}")

        for et in EXP_TYPES:
            if et not in solver_data:
                continue
            metrics = compute_solver_metrics(solver_data[et])
            if not metrics:
                continue

            label = EXP_TYPE_LABELS.get(et, et)
            print(f"\n  {label}:")

            # Column widths
            w_solver  = max(len("Solver"), max(len(s) for s in metrics))
            w_pose    = max(len("Med. Pose Err"), 13)
            w_maa     = max(len("mAA@10"), 8)
            w_time    = max(len("Mean Time(ms)"), 13)
            w_inlier  = max(len("Mean Inliers"), 12)

            # Table border
            border = (f"+{'-' * (w_solver + 2)}"
                      f"+{'-' * (w_pose + 2)}"
                      f"+{'-' * (w_maa + 2)}"
                      f"+{'-' * (w_time + 2)}"
                      f"+{'-' * (w_inlier + 2)}+")
            hdr = (f"| {'Solver':<{w_solver}} "
                   f"| {'Med. Pose Err':>{w_pose}} "
                   f"| {'mAA@10':>{w_maa}} "
                   f"| {'Mean Time(ms)':>{w_time}} "
                   f"| {'Mean Inliers':>{w_inlier}} |")

            print(f"  {border}")
            print(f"  {hdr}")
            print(f"  {border}")
            for solver, m in sorted(metrics.items()):
                row = (f"| {solver:<{w_solver}} "
                       f"| {m['median_pose_err']:>{w_pose}.4f} "
                       f"| {m['mAA10']:>{w_maa}.4f} "
                       f"| {m['mean_time_ms']:>{w_time}.4f} "
                       f"| {m['mean_inlier']:>{w_inlier}.4f} |")
                print(f"  {row}")
            print(f"  {border}")


def main():
    parser = argparse.ArgumentParser(description="Show experiment results comparison table.")
    parser.add_argument("--method",  nargs="+", help="Filter by method name(s)")
    parser.add_argument("--run_id",  nargs="+", help="Filter by run ID(s)")
    parser.add_argument("--scene",   nargs="+", help="Filter by scene name(s)")
    parser.add_argument("--csv",     action="store_true", help="Output as CSV instead of table")
    parser.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"),
                        help="Side-by-side comparison of two run IDs with delta")
    parser.add_argument("--no-avg",  action="store_true", help="Omit AVERAGE rows")
    parser.add_argument("--solvers", action="store_true",
                        help="Show per-solver breakdown tables (reads raw eval JSONs)")
    args = parser.parse_args()

    if args.compare:
        run_id_a, run_id_b = args.compare
        recs_a = load_experiments({"run_id": [run_id_a]})
        recs_b = load_experiments({"run_id": [run_id_b]})
        if not recs_a:
            print(f"Error: no experiment found with run_id={run_id_a!r}", file=sys.stderr)
            sys.exit(1)
        if not recs_b:
            print(f"Error: no experiment found with run_id={run_id_b!r}", file=sys.stderr)
            sys.exit(1)
        print_compare(recs_a, recs_b, run_id_a, run_id_b, scene_filter=args.scene)
        return

    filters = {}
    if args.method: filters["method"] = args.method
    if args.run_id: filters["run_id"] = args.run_id

    records = load_experiments(filters)
    if not records:
        print("No experiments found.", file=sys.stderr)
        sys.exit(1)

    rows = extract_rows(records, scene_filter=args.scene)
    if not rows:
        print("No results match the given filters.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        write_csv(rows)
    else:
        print_table(rows, include_avg=not args.no_avg)

    if args.solvers and not args.csv:
        print()
        from collections import defaultdict
        # Group records by (method, run_id) to avoid duplicates
        seen = {}
        for rec in records:
            key = (rec.get("method"), rec.get("run_id"))
            seen[key] = rec
        for rec in seen.values():
            scenes_in_rec = [
                s for s in rec.get("scenes", {})
                if not args.scene or s in args.scene
            ]
            if scenes_in_rec:
                print_solver_tables(rec, scenes_in_rec)


if __name__ == "__main__":
    main()
