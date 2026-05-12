import json
import os

import numpy as np


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def summarize_results(results):
    grouped = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        experiment = result.get("experiment")
        if not experiment:
            continue
        grouped.setdefault(experiment, []).append(result)

    rows = []
    for experiment, group in sorted(grouped.items()):
        r_err = np.array([_safe_float(item.get("R_err")) for item in group], dtype=float)
        t_err = np.array([_safe_float(item.get("t_err")) for item in group], dtype=float)
        pose_err = np.maximum(r_err, t_err)
        pose_err_for_auc = pose_err.copy()
        pose_err_for_auc[np.isnan(pose_err_for_auc)] = 180.0

        runtimes = np.array(
            [_safe_float(item.get("info", {}).get("runtime")) for item in group],
            dtype=float,
        )
        inlier_ratios = np.array(
            [_safe_float(item.get("info", {}).get("inlier_ratio")) for item in group],
            dtype=float,
        )

        row = {
            "solver": experiment,
            "mAA@10": float(np.mean([(pose_err_for_auc < t).mean() for t in range(1, 11)]) * 100.0),
            "median_pose_error": float(np.nanmedian(pose_err)) if pose_err.size else np.nan,
            "median_R_error": float(np.nanmedian(r_err)) if r_err.size else np.nan,
            "median_t_error": float(np.nanmedian(t_err)) if t_err.size else np.nan,
            "mean_inlier_ratio": float(np.nanmean(inlier_ratios)) if np.isfinite(inlier_ratios).any() else np.nan,
            "solver_runtime_ms_per_pair": float(np.nanmean(runtimes)) if np.isfinite(runtimes).any() else np.nan,
            "num_evaluated_pairs": int(len(group)),
        }

        f_err = np.array([_safe_float(item.get("f_err")) for item in group], dtype=float)
        if np.isfinite(f_err).any():
            f_err_for_auc = f_err.copy()
            f_err_for_auc[np.isnan(f_err_for_auc)] = 1.0
            row["mAA_f@10"] = float(
                np.mean([(f_err_for_auc < (t / 100.0)).mean() for t in range(1, 11)]) * 100.0
            )
            row["median_f_error"] = float(np.nanmedian(f_err))

        rows.append(row)

    return rows


def write_summary_json(results, output_dir, filename, solver_mode, dataset_path):
    summary_path = os.path.join(output_dir, filename)
    payload = {
        "solver_mode": solver_mode,
        "dataset_path": str(dataset_path),
        "experiments": summarize_results(results),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Summary saved to: {summary_path}")
    return summary_path
