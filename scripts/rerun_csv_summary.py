#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def first_summary(results_root: Path, config_key: str, scene: str, mode: str):
    pattern = results_root / config_key / scene / f"{mode}-{config_key}_{scene}*_summary.json"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        return None
    return Path(matches[-1])


def pick_experiment(summary, target_solver):
    experiments = summary.get("experiments", [])
    for experiment in experiments:
        if experiment.get("solver") == target_solver:
            return experiment
    if not experiments:
        return {}
    return max(experiments, key=lambda item: float(item.get("mAA@10", float("-inf"))))


def scene_list(config, scene_spec):
    if isinstance(scene_spec, list):
        return scene_spec
    if scene_spec == "validation":
        return config["validation_scenes"]
    if scene_spec == "test":
        return config["test_scenes"]
    return [scene_spec]


def timing_stats(timing_root: Path, config_key: str, scene: str):
    path = timing_root / f"{config_key}_{scene}_timing.json"
    if not path.exists():
        return {
            "matching_runtime_ms_per_pair": float("nan"),
            "feature_extract_ms_per_image": float("nan"),
            "peak_gpu_memory_mb": float("nan"),
            "num_correspondences_mean": float("nan"),
        }
    payload = load_json(path)
    pair_timings = payload.get("pair_timings", [])
    feature_timings = payload.get("feature_timings", [])
    return {
        "matching_runtime_ms_per_pair": mean([item.get("time_ms") for item in pair_timings]),
        "feature_extract_ms_per_image": mean([item.get("extract_ms") for item in feature_timings]),
        "peak_gpu_memory_mb": float(payload.get("peak_gpu_memory_mb", float("nan"))),
        "num_correspondences_mean": mean([item.get("num_matches") for item in pair_timings]),
    }


def collect_rows(config):
    output_root = PROJECT_ROOT / config["output_root"]
    results_root = output_root / "results_v2"
    timing_root = output_root / "timing"
    target_solvers = config["target_solvers"]
    rows = []

    for method in config["method_configs"]:
        for scene in scene_list(config, method["scenes"]):
            for mode in method["solver_modes"]:
                summary_path = first_summary(results_root, method["config_key"], scene, mode)
                if summary_path is None:
                    continue
                summary = load_json(summary_path)
                experiment = pick_experiment(summary, target_solvers.get(mode))
                timing = timing_stats(timing_root, method["config_key"], scene)
                row = {
                    "chapter": method["chapter"],
                    "config_key": method["config_key"],
                    "method_name": method["method_name"],
                    "scene": scene,
                    "solver_mode": mode,
                    "solver": experiment.get("solver", ""),
                    "mAA@10": experiment.get("mAA@10", float("nan")),
                    "median_pose_error": experiment.get("median_pose_error", float("nan")),
                    "median_R_error": experiment.get("median_R_error", float("nan")),
                    "median_t_error": experiment.get("median_t_error", float("nan")),
                    "mean_inlier_ratio": experiment.get("mean_inlier_ratio", float("nan")),
                    "num_pairs": experiment.get("num_evaluated_pairs", 0),
                    "solver_runtime_ms": experiment.get("solver_runtime_ms_per_pair", float("nan")),
                    "matching_runtime_ms_per_pair": timing["matching_runtime_ms_per_pair"],
                    "feature_extract_ms_per_image": timing["feature_extract_ms_per_image"],
                    "peak_gpu_memory_mb": timing["peak_gpu_memory_mb"],
                    "num_correspondences_mean": timing["num_correspondences_mean"],
                    "summary_json": str(summary_path.relative_to(PROJECT_ROOT)),
                }
                rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chapter",
        "config_key",
        "method_name",
        "scene",
        "solver_mode",
        "solver",
        "mAA@10",
        "median_pose_error",
        "median_R_error",
        "median_t_error",
        "mean_inlier_ratio",
        "num_pairs",
        "solver_runtime_ms",
        "matching_runtime_ms_per_pair",
        "feature_extract_ms_per_image",
        "peak_gpu_memory_mb",
        "num_correspondences_mean",
        "summary_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate v2 rerun summary CSVs")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rerun_config.json"))
    args = parser.parse_args()

    config = load_json(Path(args.config))
    output_root = PROJECT_ROOT / config["output_root"]
    csv_root = output_root / "csv"
    rows = collect_rows(config)

    write_csv(csv_root / "master_results.csv", rows)
    write_csv(csv_root / "chapter4_baselines.csv", [r for r in rows if r["chapter"] == "chapter4"])
    write_csv(csv_root / "chapter5_training_free.csv", [r for r in rows if r["chapter"] == "chapter5"])
    write_csv(csv_root / "chapter6_supervised.csv", [r for r in rows if r["chapter"] == "chapter6"])
    write_csv(csv_root / "chapter7_learned_matcher.csv", [r for r in rows if r["chapter"] == "chapter7"])
    write_csv(csv_root / "chapter9_test_eval.csv", [r for r in rows if r["chapter"] == "chapter9"])
    write_csv(csv_root / "appendix_focal_results.csv", [r for r in rows if r["solver_mode"] in {"shared_focal", "varying_focal"}])
    write_csv(
        csv_root / "timing_comparison.csv",
        sorted(rows, key=lambda r: (r["chapter"], r["config_key"], r["scene"], r["solver_mode"])),
    )

    missing_path = csv_root / "unidentified_or_not_run.txt"
    missing_path.write_text("\n".join(config.get("unidentified_or_not_run", [])) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {csv_root}")


if __name__ == "__main__":
    main()
