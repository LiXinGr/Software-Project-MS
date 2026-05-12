#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_sizes(image_dir: Path):
    sizes = {}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as img:
            sizes[path.stem] = (img.width, img.height)
    return sizes


def passfail(ok, label, detail=""):
    return {"check": label, "status": "PASS" if ok else "FAIL", "detail": detail}


def finite_positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except Exception:
        return False


def first_summary(results_root: Path, config_key: str, scene: str, mode: str):
    matches = sorted((results_root / config_key / scene).glob(f"{mode}-{config_key}_{scene}*_summary.json"))
    return matches[-1] if matches else None


def check_depth(output_root: Path, scene: str, sizes):
    depth_dir = output_root / "depth_raw" / scene
    files = sorted(depth_dir.glob("*_depth.npy"))
    rows = [passfail(len(files) >= 10, "Depth estimation on 10 raw images", f"{len(files)} depth maps")]
    for depth_path in files[:10]:
        stem = depth_path.name.removesuffix("_depth.npy")
        if stem not in sizes:
            rows.append(passfail(False, "Depth filename matches raw image", depth_path.name))
            continue
        depth = np.load(depth_path)
        width, height = sizes[stem]
        rows.append(passfail(depth.shape == (height, width), "Depth map dimensions match raw image", f"{depth_path.name}: {depth.shape} vs {(height, width)}"))
    return rows


def check_method(output_root: Path, config, method, scene, sizes, limit):
    config_key = method["config_key"]
    label = method["method_name"]
    matches_dir = output_root / "matches_v2_smoke" / config_key / scene
    timing_path = output_root / "timing_smoke" / f"{config_key}_{scene}_timing.json"
    benchmark_path = output_root / "benchmarks_v2_smoke" / f"{config_key}_{scene}.h5"
    results_root = output_root / "results_v2_smoke"
    rows = []

    match_files = sorted(matches_dir.glob("*.npz"))
    rows.append(passfail(bool(match_files), f"{label}: matches saved", f"{len(match_files)} files"))
    coord_ok = True
    count_ok = True
    coord_detail = ""
    for match_path in match_files[: min(len(match_files), limit, 20)]:
        stem0, stem1 = match_path.stem.split("__", 1)
        data = np.load(match_path)
        mkpts0 = data["mkpts0"]
        mkpts1 = data["mkpts1"]
        count_ok = count_ok and len(mkpts0) <= config["max_points"]
        if stem0 not in sizes or stem1 not in sizes:
            coord_ok = False
            coord_detail = f"missing image size for {match_path.name}"
            continue
        w0, h0 = sizes[stem0]
        w1, h1 = sizes[stem1]
        if len(mkpts0):
            ok0 = (mkpts0[:, 0].min() >= -1e-3 and mkpts0[:, 0].max() < w0 + 1e-3 and mkpts0[:, 1].min() >= -1e-3 and mkpts0[:, 1].max() < h0 + 1e-3)
            ok1 = (mkpts1[:, 0].min() >= -1e-3 and mkpts1[:, 0].max() < w1 + 1e-3 and mkpts1[:, 1].min() >= -1e-3 and mkpts1[:, 1].max() < h1 + 1e-3)
            if not (ok0 and ok1):
                coord_ok = False
                coord_detail = match_path.name
                break
    rows.append(passfail(coord_ok, f"{label}: match coordinates in raw frame", coord_detail))
    rows.append(passfail(count_ok, f"{label}: matches per pair <= {config['max_points']}"))

    if timing_path.exists():
        timing = load_json(timing_path)
        pair_times = [item.get("time_ms", 0) for item in timing.get("pair_timings", [])]
        rows.append(passfail(any(finite_positive(v) for v in pair_times), f"{label}: timing logged", str(timing_path)))
    else:
        rows.append(passfail(False, f"{label}: timing logged", "missing timing JSON"))

    if benchmark_path.exists():
        with h5py.File(benchmark_path, "r") as handle:
            corr_keys = [key for key in handle.keys() if key.startswith("corr_")]
            k_keys = [key for key in handle.keys() if key.startswith("K_")]
            k_ok = False
            if k_keys:
                K = np.array(handle[k_keys[0]])
                k_ok = np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
        rows.append(passfail(bool(corr_keys), f"{label}: packed HDF5 correspondences", f"{len(corr_keys)} pairs"))
        rows.append(passfail(k_ok, f"{label}: K matrix finite at raw scale"))
    else:
        rows.append(passfail(False, f"{label}: packed HDF5 exists", str(benchmark_path)))

    for mode in ["calibrated", "shared_focal", "varying_focal"]:
        summary_path = first_summary(results_root, config_key, scene, mode)
        if not summary_path:
            rows.append(passfail(False, f"{label}: eval {mode}", "missing summary"))
            continue
        summary = load_json(summary_path)
        experiments = summary.get("experiments", [])
        maa_values = [float(item.get("mAA@10", float("nan"))) for item in experiments]
        ok = any(math.isfinite(v) and v > 0 for v in maa_values)
        rows.append(passfail(ok, f"{label}: eval {mode}", str(summary_path)))
    return rows


def check_csv(output_root: Path):
    csv_path = output_root / "csv" / "master_results.csv"
    if not csv_path.exists():
        return [passfail(False, "CSV generation", "missing master_results.csv")]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "chapter",
        "config_key",
        "method_name",
        "scene",
        "solver_mode",
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
    }
    columns_ok = required.issubset(set(reader.fieldnames or []))
    nan_ok = True
    for row in rows:
        for key in ["mAA@10", "median_pose_error", "median_R_error", "median_t_error"]:
            try:
                if not math.isfinite(float(row[key])):
                    nan_ok = False
            except Exception:
                nan_ok = False
    return [
        passfail(columns_ok, "CSV generation: all columns present"),
        passfail(bool(rows), "CSV generation: rows present", f"{len(rows)} rows"),
        passfail(nan_ok, "CSV generation: no NaN in core metric columns"),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate v2 smoke-test outputs")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rerun_config.json"))
    parser.add_argument("--scene", default="sacre_coeur")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    output_root = PROJECT_ROOT / config["output_root"]
    scene_info = config["scenes"][args.scene]
    sizes = image_sizes(PROJECT_ROOT / scene_info["image_dir"])
    methods = [m for m in config["method_configs"] if m["chapter"] == "chapter4" and m["scenes"] == "validation"]

    rows = []
    rows.extend(check_depth(output_root, args.scene, sizes))
    for method in methods:
        rows.extend(check_method(output_root, config, method, args.scene, sizes, args.limit))
    rows.extend(check_csv(output_root))

    ok = all(row["status"] == "PASS" for row in rows)
    report = {"status": "PASS" if ok else "FAIL", "checks": rows}
    report_path = Path(args.output) if args.output else output_root / "logs" / "smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("SMOKE TEST REPORT")
    for row in rows:
        detail = f" - {row['detail']}" if row.get("detail") else ""
        print(f"[{row['status']}] {row['check']}{detail}")
    print(f"Saved report to {report_path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
