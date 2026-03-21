#!/usr/bin/env python3
"""
Reconstruct benchmark CSV summaries from raw per-pair JSON results.

This is useful when only the calibrated/shared/varying JSON outputs are
available in a results directory but the timestamped CSV summaries are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def mAA_from_pose_errors(pose_err: np.ndarray) -> float:
    return float(np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100)


def mAA_f_from_errors(f_err: np.ndarray) -> float:
    return float(np.mean([np.sum(f_err < t / 100) / len(f_err) for t in range(1, 11)]) * 100)


def reconstruct_rows(results: list[dict], exp_type_override: str | None = None) -> list[dict[str, str]]:
    experiments: dict[str, dict[str, object]] = {}

    for record in results:
        if not isinstance(record, dict):
            continue
        exp = record.get("experiment", "unknown")
        exp_type = exp_type_override or record.get("exp_type", "calibrated")
        key = f"{exp}|{exp_type}"
        if key not in experiments:
            experiments[key] = {
                "R_err": [],
                "t_err": [],
                "runtime": [],
                "inlier_ratio": [],
                "f_err": [],
                "exp": exp,
                "exp_type": exp_type,
            }

        experiments[key]["R_err"].append(record.get("R_err", float("nan")))
        experiments[key]["t_err"].append(record.get("t_err", float("nan")))
        if "f_err" in record:
            experiments[key]["f_err"].append(record.get("f_err", float("nan")))
        info = record.get("info", {}) or {}
        experiments[key]["runtime"].append(info.get("runtime", float("nan")))
        experiments[key]["inlier_ratio"].append(info.get("inlier_ratio", float("nan")))

    rows = []
    for _, data in sorted(experiments.items()):
        exp = str(data["exp"])
        exp_type = str(data["exp_type"])
        r_err = np.array(data["R_err"], dtype=float)
        t_err = np.array(data["t_err"], dtype=float)
        runtimes = np.array(data["runtime"], dtype=float)
        inliers = np.array(data["inlier_ratio"], dtype=float)
        f_err = np.array(data["f_err"], dtype=float) if data["f_err"] else np.array([], dtype=float)

        pose_err = np.maximum(r_err, t_err)
        pose_err[np.isnan(pose_err)] = 180.0

        med_r = float(np.nanmedian(r_err))
        med_t = float(np.nanmedian(t_err))
        mAA_10 = mAA_from_pose_errors(pose_err)

        if f_err.size:
            f_err[np.isnan(f_err)] = 1.0
            mAA_f_10 = f"{mAA_f_from_errors(f_err):.1f}"
        else:
            mAA_f_10 = "N/A"

        opt_type = "H" if "hybrid" in exp.lower() else "S"
        mean_time = float(np.nanmean(runtimes))
        mean_inliers = float(np.nanmean(inliers) * 100.0)

        rows.append(
            {
                "Solver": exp,
                "Exp.Type": exp_type,
                "Opt.": opt_type,
                "εr(°)": f"{med_r:.2f}",
                "εt(°)": f"{med_t:.2f}",
                "mAA@10": f"{mAA_10:.1f}",
                "mAA_f@10": mAA_f_10,
                "τ(ms)": f"{mean_time:.1f}",
                "Inliers": f"{mean_inliers:.1f}",
                "Num_Pairs": str(len(r_err)),
            }
        )

    return rows


def write_csv(
    output_csv: Path,
    rows: list[dict[str, str]],
    matcher_name: str,
    depth_method: str,
    max_points: str,
    img_size: str,
    feat_level: str,
    up_ft_index: str,
    dift_t: str,
    ratio_thresh: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Matches",
        "Depth",
        "Solver",
        "Exp.Type",
        "Opt.",
        "εr(°)",
        "εt(°)",
        "mAA@10",
        "mAA_f@10",
        "τ(ms)",
        "Inliers",
        "Num_Pairs",
        "max_points",
        "img_size",
        "feat_level",
        "up_ft_index",
        "dift_t",
        "ratio_thresh",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    matcher_name,
                    depth_method,
                    row["Solver"],
                    row["Exp.Type"],
                    row["Opt."],
                    row["εr(°)"],
                    row["εt(°)"],
                    row["mAA@10"],
                    row["mAA_f@10"],
                    row["τ(ms)"],
                    row["Inliers"],
                    row["Num_Pairs"],
                    max_points,
                    img_size,
                    feat_level,
                    up_ft_index,
                    dift_t,
                    ratio_thresh,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct benchmark CSV from raw results JSON")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--matcher", default="projection")
    parser.add_argument("--depth", default="UniDepth")
    parser.add_argument("--exp-type", default=None)
    parser.add_argument("--max-points", default="2000")
    parser.add_argument("--img-size", default="768")
    parser.add_argument("--feat-level", default="-8")
    parser.add_argument("--up-ft-index", default="2")
    parser.add_argument("--dift-t", default="0")
    parser.add_argument("--ratio-thresh", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = json.loads(args.input_json.read_text())
    rows = reconstruct_rows(results, exp_type_override=args.exp_type)
    write_csv(
        output_csv=args.output_csv,
        rows=rows,
        matcher_name=args.matcher,
        depth_method=args.depth,
        max_points=args.max_points,
        img_size=args.img_size,
        feat_level=args.feat_level,
        up_ft_index=args.up_ft_index,
        dift_t=args.dift_t,
        ratio_thresh=args.ratio_thresh,
    )
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
