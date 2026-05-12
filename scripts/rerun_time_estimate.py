#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_images(path: Path):
    return sum(1 for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"})


def count_pairs(path: Path, limit: int):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        n = sum(1 for _ in handle)
    return min(n, limit) if limit > 0 else n


def mean_pair_ms(timing_root: Path, config_key: str):
    matches = sorted(timing_root.glob(f"{config_key}_*_timing.json"))
    values = []
    for path in matches:
        payload = load_json(path)
        values.extend(float(item.get("time_ms", 0.0)) for item in payload.get("pair_timings", []) if item.get("time_ms", 0.0) > 0)
    return sum(values) / len(values) if values else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Print v2 rerun time estimate")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rerun_config.json"))
    parser.add_argument("--pair-limit", type=int, default=15000)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    output_root = PROJECT_ROOT / config["output_root"]
    timing_root = output_root / "timing"

    total_images = sum(count_images(PROJECT_ROOT / scene["image_dir"]) for scene in config["scenes"].values())
    depth_hours = total_images * 1.5 / 3600.0

    total_gpu_ms = 0.0
    missing_timing = []
    for method in config["method_configs"]:
        ms = mean_pair_ms(timing_root, method["config_key"])
        if ms is None:
            missing_timing.append(method["config_key"])
            ms = 1000.0
        if method["scenes"] == "validation":
            scenes = config["validation_scenes"]
        elif method["scenes"] == "test":
            scenes = config["test_scenes"]
        elif isinstance(method["scenes"], list):
            scenes = method["scenes"]
        else:
            scenes = [method["scenes"]]
        for scene in scenes:
            pair_file = PROJECT_ROOT / config["scenes"][scene]["pairs_file"]
            total_gpu_ms += ms * count_pairs(pair_file, args.pair_limit)

    gpu_hours_serial = total_gpu_ms / 1000.0 / 3600.0
    gpu_hours_4way_wall = gpu_hours_serial / 4.0
    cpu_hours = 0.25 * len(config["method_configs"])

    print("TIME ESTIMATE")
    print(f"Images for depth: {total_images}")
    print(f"Depth estimate: {depth_hours:.2f} GPU-hours on one GPU")
    print(f"Matching estimate: {gpu_hours_serial:.2f} GPU-hours serial, {gpu_hours_4way_wall:.2f} wall-hours with 4 GPUs")
    print(f"Packing+eval rough estimate: {cpu_hours:.2f} CPU-hours")
    print(f"Expected wall time after smoke: {depth_hours + gpu_hours_4way_wall + cpu_hours:.2f} hours")
    if missing_timing:
        print("Missing smoke timing; used 1000 ms/pair fallback for:")
        for key in sorted(set(missing_timing)):
            print(f"  {key}")


if __name__ == "__main__":
    main()
