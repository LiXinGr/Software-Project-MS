#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ROOT = PROJECT_ROOT / "output_v2"
RUNTIME_ROOT = OUTPUT_ROOT / "runtime" / "final_runtime_benchmark"
CSV_ROOT = OUTPUT_ROOT / "csv"
REPORT_ROOT = OUTPUT_ROOT / "reports"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
LOG_ROOT = OUTPUT_ROOT / "logs" / "final_runtime_benchmark"

SUBSET_CSV = CSV_ROOT / "final_runtime_subset_pairs.csv"
COMPARISON_CSV = CSV_ROOT / "final_runtime_comparison.csv"
REPORT_PATH = REPORT_ROOT / "final_runtime_comparison_report.md"
TRADEOFF_FIG = FIGURE_ROOT / "final_runtime_accuracy_tradeoff_calibrated.png"
BREAKDOWN_FIG = FIGURE_ROOT / "final_runtime_breakdown.png"

REPOSED_PY = Path(os.environ.get("REPOSED_PY", "/home.stud/gorbuden/.conda/envs/reposed/bin/python"))
REPOSED_DIR = PROJECT_ROOT / "external" / "RePoseD"

SCENES = [
    "british_museum",
    "florence_cathedral_side",
    "lincoln_memorial_statue",
    "milan_cathedral",
    "mount_rushmore",
    "piazza_san_marco",
    "sagrada_familia",
    "st_pauls_cathedral",
    "taj_mahal",
    "temple_nara_japan",
]

FINAL_METHOD = "final_selected_expanded151_lg_proj_dinov3_dift_ft002_mp2048"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str
    runner: str
    accuracy_key: str
    notes: str


METHODS = {
    "superpoint_lg_mp2048": MethodSpec(
        method="superpoint_lg_mp2048",
        label="SP+LG",
        runner="splg",
        accuracy_key="test_splg",
        notes="SP+LG sparse baseline; feature and matching components separated.",
    ),
    FINAL_METHOD: MethodSpec(
        method=FINAL_METHOD,
        label="Final selected",
        runner="final",
        accuracy_key=FINAL_METHOD,
        notes="DINOv3+DIFT+projection descriptors with fine-tuned LightGlue; feature and matching components separated.",
    ),
    "roma_outdoor_mp2048": MethodSpec(
        method="roma_outdoor_mp2048",
        label="RoMa",
        runner="roma",
        accuracy_key="test_roma",
        notes="Dense matcher; feature extraction and matching are recorded as a combined component.",
    ),
    "romav2_precise_mp2048": MethodSpec(
        method="romav2_precise_mp2048",
        label="RoMaV2",
        runner="romav2",
        accuracy_key="test_romav2",
        notes="Dense matcher; feature extraction and matching are recorded as a combined component.",
    ),
}

BASELINE_BENCHMARK_KEYS = {
    "superpoint_lg_mp2048": "test_splg",
    FINAL_METHOD: FINAL_METHOD,
    "roma_outdoor_mp2048": "test_roma",
    "romav2_precise_mp2048": "test_romav2",
}

CSV_FIELDS = [
    "method",
    "scene",
    "num_pairs",
    "feature_time_ms_mean",
    "feature_time_ms_median",
    "matching_time_ms_mean",
    "matching_time_ms_median",
    "solver_time_ms_mean",
    "solver_time_ms_median",
    "total_online_ms_mean",
    "total_online_ms_median",
    "total_cached_ms_mean",
    "total_cached_ms_median",
    "peak_gpu_memory_mb",
    "notes",
]


def ensure_dirs() -> None:
    for path in [RUNTIME_ROOT, CSV_ROOT, REPORT_ROOT, FIGURE_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def read_pairs_file(scene: str) -> list[tuple[int, str, str]]:
    path = PROJECT_ROOT / "output" / f"pairs_{scene}.txt"
    pairs: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((idx, parts[0], parts[1]))
    return pairs


def pair_key(img0: str, img1: str) -> str:
    return f"{Path(img0).stem}_o_{Path(img1).stem}"


def h5_pair_keys(path: Path) -> set[str]:
    import h5py

    keys: set[str] = set()
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if key.startswith("corr_"):
                keys.add(key[len("corr_") :])
    return keys


def benchmark_path(benchmark_key: str, scene: str) -> Path:
    return OUTPUT_ROOT / "benchmarks_v2" / f"{benchmark_key}_{scene}.h5"


def prepare_subset(seed: int, pairs_per_scene: int, force: bool = False) -> None:
    ensure_dirs()
    if SUBSET_CSV.exists() and not force:
        print(f"[subset] exists: {SUBSET_CSV}")
        return

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    pair_file_dir = RUNTIME_ROOT / "pairs"
    pair_file_dir.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        all_pairs = read_pairs_file(scene)
        benchmark_paths = [benchmark_path(key, scene) for key in BASELINE_BENCHMARK_KEYS.values()]
        missing = [str(path) for path in benchmark_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing evaluated benchmark H5 files for {scene}: {missing}")

        import h5py

        shuffled = list(all_pairs)
        rng.shuffle(shuffled)
        sampled: list[tuple[int, str, str]] = []
        probed = 0
        handles = [h5py.File(path, "r") for path in benchmark_paths]
        try:
            for source_idx, img0, img1 in shuffled:
                probed += 1
                key = f"corr_{pair_key(img0, img1)}"
                if all(key in handle for handle in handles):
                    sampled.append((source_idx, img0, img1))
                    if len(sampled) >= pairs_per_scene:
                        break
        finally:
            for handle in handles:
                handle.close()

        if not sampled:
            raise RuntimeError(f"No valid evaluated pairs available for {scene}.")

        sample_size = len(sampled)
        available_valid_pairs = f">={sample_size}"

        pairs_txt = pair_file_dir / f"{scene}.txt"
        with pairs_txt.open("w", encoding="utf-8") as handle:
            for _, img0, img1 in sampled:
                handle.write(f"{img0} {img1}\n")

        for local_idx, (source_idx, img0, img1) in enumerate(sampled):
            rows.append(
                {
                    "scene": scene,
                    "local_pair_index": local_idx,
                    "source_pair_index": source_idx,
                    "img0": img0,
                    "img1": img1,
                    "pair_key": pair_key(img0, img1),
                    "available_valid_pairs": available_valid_pairs,
                    "sampled_pairs": sample_size,
                    "seed": seed,
                    "pairs_file": str(pairs_txt.relative_to(PROJECT_ROOT)),
                }
            )
        print(f"[subset] {scene}: sampled {sample_size} valid evaluated pairs after {probed} random probes", flush=True)

    with SUBSET_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "local_pair_index",
                "source_pair_index",
                "img0",
                "img1",
                "pair_key",
                "available_valid_pairs",
                "sampled_pairs",
                "seed",
                "pairs_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[subset] wrote {SUBSET_CSV}")


def load_subset() -> dict[str, list[dict[str, str]]]:
    if not SUBSET_CSV.exists():
        raise FileNotFoundError(f"Missing subset CSV: {SUBSET_CSV}. Run --prepare-subset first.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SUBSET_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["scene"]].append(row)
    return {scene: grouped.get(scene, []) for scene in SCENES}


def synchronize(device: Any | None) -> None:
    try:
        import torch

        if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    except Exception:
        return


def timed_ms(device: Any | None, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def image_dir(scene: str) -> Path:
    return PROJECT_ROOT / "datasets" / "phototourism" / scene / "images_preprocessed"


def scene_pairs_txt(scene: str) -> Path:
    return RUNTIME_ROOT / "pairs" / f"{scene}.txt"


def matches_dir(method: str, scene: str) -> Path:
    return RUNTIME_ROOT / "matches" / method / scene


def benchmark_out(method: str, scene: str) -> Path:
    return RUNTIME_ROOT / "benchmarks" / f"final_runtime_{method}_{scene}.h5"


def results_dir(method: str, scene: str) -> Path:
    return RUNTIME_ROOT / "results" / method / scene


def detail_json(method: str, scene: str) -> Path:
    return RUNTIME_ROOT / "timing" / method / f"{scene}.json"


def method_root(method: str) -> Path:
    return RUNTIME_ROOT / "method_runs" / method


def scoped_clean_method(method: str) -> None:
    for path in [
        RUNTIME_ROOT / "matches" / method,
        RUNTIME_ROOT / "results" / method,
        RUNTIME_ROOT / "timing" / method,
        method_root(method),
    ]:
        if path.exists():
            shutil.rmtree(path)
    bench_dir = RUNTIME_ROOT / "benchmarks"
    if bench_dir.exists():
        for path in bench_dir.glob(f"final_runtime_{method}_*.h5"):
            path.unlink()


def save_npz_atomic(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}_{os.getpid()}_{time.time_ns()}.npz")
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_pack(method: str, scene: str, num_pairs: int) -> float:
    output = benchmark_out(method, scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPOSED_PY),
        str(PROJECT_ROOT / "scripts" / "pack_benchmark.py"),
        "--matches_dir",
        str(matches_dir(method, scene)),
        "--depth_dir",
        str(PROJECT_ROOT / "datasets" / "phototourism" / scene / "depth_unidepth"),
        "--sparse_dir",
        str(PROJECT_ROOT / "datasets" / "phototourism" / scene / "dense" / "sparse"),
        "--pairs_file",
        str(scene_pairs_txt(scene)),
        "--output",
        str(output),
    ]
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return ((time.perf_counter() - start) * 1000.0) / max(1, num_pairs)


def run_solver(method: str, scene: str) -> tuple[float | None, float | None, int, list[float]]:
    out_dir = results_dir(method, scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = benchmark_out(method, scene)
    cmd = [
        str(REPOSED_PY),
        str(REPOSED_DIR / "eval.py"),
        str(bench),
        "-nw",
        os.environ.get("REPOSED_NUM_WORKERS", "8"),
        "--thesis",
        "--output_dir",
        str(out_dir),
        "--preprocess_info",
        str(PROJECT_ROOT / "datasets" / "phototourism" / scene / "images_preprocessed" / "preprocess_info.json"),
        "--max_epipolar_error",
        "2.0",
        "--reproj_threshold",
        "16.0",
    ]
    subprocess.run(cmd, cwd=str(REPOSED_DIR), check=True)

    stem = bench.stem
    raw = out_dir / f"calibrated-{stem}-2.0t.json"
    summary = out_dir / f"calibrated-{stem}-2.0t_summary.json"
    runtimes: list[float] = []
    evaluated = 0
    if raw.exists():
        payload = json.loads(raw.read_text(encoding="utf-8"))
        for item in payload:
            if item.get("experiment") == "3p_ours_shift_scale+12":
                runtime = item.get("info", {}).get("runtime")
                if runtime is not None:
                    runtimes.append(float(runtime))
                evaluated += 1
    if runtimes:
        return mean(runtimes), median(runtimes), evaluated, runtimes

    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for exp in payload.get("experiments", []):
            if exp.get("solver") == "3p_ours_shift_scale+12":
                return float(exp.get("solver_runtime_ms_per_pair", "nan")), None, int(exp.get("num_evaluated_pairs", 0)), []
    return None, None, evaluated, []


def run_superpoint(method: str, subset: dict[str, list[dict[str, str]]], warmup_pairs: int, force: bool) -> None:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image

    sys.path.insert(0, str(SCRIPT_DIR))
    from util import current_peak_memory_mb, reset_peak_memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    def process(img0_path: Path, img1_path: Path, save_path: Path | None):
        image0, load0_ms = timed_ms(device, lambda: load_image(img0_path).to(device))
        feats0, feat0_ms = timed_ms(device, lambda: extractor.extract(image0.unsqueeze(0)))
        image1, load1_ms = timed_ms(device, lambda: load_image(img1_path).to(device))
        feats1, feat1_ms = timed_ms(device, lambda: extractor.extract(image1.unsqueeze(0)))

        def do_match():
            with torch.no_grad():
                matches01 = matcher({"image0": feats0, "image1": feats1})
                matches = matches01["matches"][0]
                kpts0 = feats0["keypoints"][0]
                kpts1 = feats1["keypoints"][0]
                if matches.numel() == 0:
                    mkpts0 = np.zeros((0, 2), dtype=np.float32)
                    mkpts1 = np.zeros((0, 2), dtype=np.float32)
                else:
                    mkpts0 = kpts0[matches[:, 0]].detach().cpu().numpy().astype(np.float32, copy=False)
                    mkpts1 = kpts1[matches[:, 1]].detach().cpu().numpy().astype(np.float32, copy=False)
                return mkpts0, mkpts1

        (mkpts0, mkpts1), match_ms = timed_ms(device, do_match)
        if save_path is not None:
            save_npz_atomic(save_path, mkpts0=mkpts0, mkpts1=mkpts1)
        return {
            "load_ms": load0_ms + load1_ms,
            "feature_ms_total": feat0_ms + feat1_ms,
            "feature_ms_per_image": [feat0_ms, feat1_ms],
            "matching_ms": match_ms,
            "num_matches": int(len(mkpts0)),
            "combined_feature_matching": False,
        }

    run_recorded_method(method, subset, process, warmup_pairs, force, current_peak_memory_mb, device)


def run_final_selected(method: str, subset: dict[str, list[dict[str, str]]], warmup_pairs: int, force: bool) -> None:
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sys.path.insert(0, str(SCRIPT_DIR))
    import lightglue_projection_matches as lgp
    from util import current_peak_memory_mb, reset_peak_memory

    args = argparse.Namespace(
        config_key=method,
        checkpoint=str(PROJECT_ROOT / "experiments" / "phase2_projection_wide" / "best.pt"),
        lightglue_checkpoint=str(
            PROJECT_ROOT
            / "external"
            / "glue-factory"
            / "outputs"
            / "training"
            / "stage2_dinov3_lg_151scenes_v1"
            / "checkpoint_best.tar"
        ),
        max_points=2048,
        feat_level=-8,
        img_size=[768, 768],
        t=0,
        up_ft_index=2,
        ensemble_size=2,
        alpha=0.5,
        filter_threshold=0.02,
        depth_confidence=0.95,
        width_confidence=0.99,
        flash=True,
        mp=False,
        save_scores=True,
        online_extraction=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    extractor = lgp.build_online_extractor(args, device)
    matcher = lgp.build_matcher(args, device)

    def extract_bundle(path: Path):
        image, load_ms = timed_ms(device, lambda: lgp.load_image_tensor(path, device))
        image_tensor, (height, width) = image

        def do_extract():
            with torch.no_grad():
                pred = extractor(
                    {
                        "image": image_tensor.unsqueeze(0),
                        "image_size": torch.tensor([[width, height]], device=device, dtype=torch.float32),
                        "scales": torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32),
                    }
                )
            return {
                "kpts": pred["keypoints"][0].detach().cpu().to(dtype=torch.float32),
                "desc": pred["descriptors"][0].detach().cpu().to(dtype=torch.float32),
            }

        bundle, feat_ms = timed_ms(device, do_extract)
        return bundle, load_ms, feat_ms

    def process(img0_path: Path, img1_path: Path, save_path: Path | None):
        bundle0, load0_ms, feat0_ms = extract_bundle(img0_path)
        bundle1, load1_ms, feat1_ms = extract_bundle(img1_path)

        (mkpts0, mkpts1, scores, stop_layer), match_ms = timed_ms(
            device,
            lambda: lgp.match_with_lightglue(bundle0, bundle1, img0_path, img1_path, matcher, device),
        )
        if save_path is not None:
            lgp.save_match_archive(save_path, mkpts0, mkpts1, args, match_scores=scores, stop_layer=stop_layer)
        return {
            "load_ms": load0_ms + load1_ms,
            "feature_ms_total": feat0_ms + feat1_ms,
            "feature_ms_per_image": [feat0_ms, feat1_ms],
            "matching_ms": match_ms,
            "num_matches": int(len(mkpts0)),
            "combined_feature_matching": False,
            "stop_layer": int(stop_layer),
        }

    run_recorded_method(method, subset, process, warmup_pairs, force, current_peak_memory_mb, device)


def run_roma(method: str, subset: dict[str, list[dict[str, str]]], warmup_pairs: int, force: bool) -> None:
    import torch

    sys.path.insert(0, str(SCRIPT_DIR))
    import roma_matches
    from util import current_peak_memory_mb, reset_peak_memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    model = roma_matches.roma_outdoor(device=device)
    args = argparse.Namespace(max_points=2048)

    def process(img0_path: Path, img1_path: Path, save_path: Path | None):
        def do_match():
            return roma_matches.process_pair(img0_path, img1_path, model, args, device)

        (mkpts0, mkpts1, _, _), combined_ms = timed_ms(device, do_match)
        if save_path is not None:
            save_npz_atomic(save_path, mkpts0=mkpts0, mkpts1=mkpts1)
        return {
            "load_ms": 0.0,
            "feature_ms_total": 0.0,
            "feature_ms_per_image": [],
            "matching_ms": combined_ms,
            "num_matches": int(len(mkpts0)),
            "combined_feature_matching": True,
        }

    run_recorded_method(method, subset, process, warmup_pairs, force, current_peak_memory_mb, device)


def run_romav2(method: str, subset: dict[str, list[dict[str, str]]], warmup_pairs: int, force: bool) -> None:
    import torch

    sys.path.insert(0, str(SCRIPT_DIR))
    import romav2_matches
    from util import current_peak_memory_mb, reset_peak_memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    model = romav2_matches.RoMaV2()
    model.apply_setting("precise")
    model = model.to(device)
    args = argparse.Namespace(max_points=2048, setting="precise")

    def process(img0_path: Path, img1_path: Path, save_path: Path | None):
        def do_match():
            return romav2_matches.process_pair(img0_path, img1_path, model, args, device)

        (mkpts0, mkpts1, _, _), combined_ms = timed_ms(device, do_match)
        if save_path is not None:
            save_npz_atomic(save_path, mkpts0=mkpts0, mkpts1=mkpts1)
        return {
            "load_ms": 0.0,
            "feature_ms_total": 0.0,
            "feature_ms_per_image": [],
            "matching_ms": combined_ms,
            "num_matches": int(len(mkpts0)),
            "combined_feature_matching": True,
        }

    run_recorded_method(method, subset, process, warmup_pairs, force, current_peak_memory_mb, device)


def iter_global_pairs(subset: dict[str, list[dict[str, str]]]):
    for scene in SCENES:
        for row in subset[scene]:
            yield scene, row


def run_recorded_method(
    method: str,
    subset: dict[str, list[dict[str, str]]],
    process,
    warmup_pairs: int,
    force: bool,
    peak_mem_fn,
    device: Any,
) -> None:
    if force:
        scoped_clean_method(method)

    warmups = list(iter_global_pairs(subset))[:warmup_pairs]
    print(f"[{method}] warm-up pairs: {len(warmups)}")
    for scene, row in warmups:
        img0_path = image_dir(scene) / row["img0"]
        img1_path = image_dir(scene) / row["img1"]
        process(img0_path, img1_path, None)

    for scene in SCENES:
        rows = subset[scene]
        if not rows:
            continue
        out_json = detail_json(method, scene)
        if out_json.exists() and not force:
            print(f"[{method}][{scene}] timing exists, skipping")
            continue

        scene_match_dir = matches_dir(method, scene)
        scene_match_dir.mkdir(parents=True, exist_ok=True)
        pair_timings = []
        feature_image_times = []

        print(f"[{method}][{scene}] recording {len(rows)} pairs")
        for idx, row in enumerate(rows, start=1):
            img0 = row["img0"]
            img1 = row["img1"]
            img0_path = image_dir(scene) / img0
            img1_path = image_dir(scene) / img1
            save_path = scene_match_dir / f"{Path(img0).stem}__{Path(img1).stem}.npz"
            timing = process(img0_path, img1_path, save_path)
            for value in timing.get("feature_ms_per_image", []):
                feature_image_times.append(float(value))
            pair_timings.append(
                {
                    "scene": scene,
                    "img0": img0,
                    "img1": img1,
                    "pair_key": row["pair_key"],
                    "load_ms": float(timing["load_ms"]),
                    "feature_ms_total": float(timing["feature_ms_total"]),
                    "matching_ms": float(timing["matching_ms"]),
                    "num_matches": int(timing["num_matches"]),
                    "combined_feature_matching": bool(timing["combined_feature_matching"]),
                }
            )
            if idx == len(rows) or idx % 25 == 0:
                print(f"[{method}][{scene}] pair {idx}/{len(rows)}")

        pack_ms_per_pair = run_pack(method, scene, len(rows))
        solver_mean, solver_median, evaluated_pairs, solver_runtimes = run_solver(method, scene)
        peak_mem = float(peak_mem_fn(device))

        payload = {
            "method": method,
            "scene": scene,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_pairs": len(rows),
            "num_evaluated_pairs": evaluated_pairs,
            "pair_timings": pair_timings,
            "feature_image_times_ms": feature_image_times,
            "pack_depth_sampling_ms_per_pair": pack_ms_per_pair,
            "solver_time_ms_mean": solver_mean,
            "solver_time_ms_median": solver_median,
            "solver_runtimes_ms": solver_runtimes,
            "peak_gpu_memory_mb": peak_mem,
            "notes": METHODS[method].notes,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out_json)
        print(
            f"[{method}][{scene}] done: pack={pack_ms_per_pair:.3f} ms/pair, "
            f"solver_mean={solver_mean}, peak={peak_mem:.1f} MiB"
        )


def list_detail_files() -> list[Path]:
    files: list[Path] = []
    for method in METHODS:
        files.extend(sorted((RUNTIME_ROOT / "timing" / method).glob("*.json")))
    return files


def summarize_detail(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload["method"]
    pair_timings = payload["pair_timings"]
    feature_times = [float(x) for x in payload.get("feature_image_times_ms", [])]
    matching_times = [float(x["matching_ms"]) for x in pair_timings]
    load_times = [float(x.get("load_ms", 0.0)) for x in pair_timings]
    feature_total_times = [float(x.get("feature_ms_total", 0.0)) for x in pair_timings]
    pack_ms = float(payload.get("pack_depth_sampling_ms_per_pair", 0.0))
    solver_mean = payload.get("solver_time_ms_mean")
    solver_median = payload.get("solver_time_ms_median")
    solver_mean_f = None if solver_mean is None else float(solver_mean)
    solver_median_f = None if solver_median is None else float(solver_median)
    solver_runtimes = [float(x) for x in payload.get("solver_runtimes_ms", [])]

    online_totals = []
    cached_totals = []
    for idx, row in enumerate(pair_timings):
        match_ms = float(row["matching_ms"])
        load_ms = float(row.get("load_ms", 0.0))
        feature_ms = float(row.get("feature_ms_total", 0.0))
        solver_for_pair = solver_runtimes[idx] if idx < len(solver_runtimes) else (solver_mean_f or 0.0)
        online_totals.append(load_ms + feature_ms + match_ms + pack_ms + solver_for_pair)
        if METHODS[method].runner in {"splg", "final"}:
            cached_totals.append(match_ms + pack_ms + solver_for_pair)

    notes = payload.get("notes", "")
    if pack_ms:
        notes = f"{notes} pack/depth_sampling_mean_ms_per_pair={pack_ms:.3f}; "
    load_mean = mean(load_times) if load_times else 0.0
    if load_mean:
        notes = f"{notes} image_load_preprocess_mean_ms_per_pair={load_mean:.3f}; "
    if METHODS[method].runner in {"roma", "romav2"}:
        notes = f"{notes} total_cached not applicable for dense combined matcher."

    return {
        "method": method,
        "scene": payload["scene"],
        "num_pairs": int(payload.get("num_pairs", len(pair_timings))),
        "feature_time_ms_mean": mean(feature_times) if feature_times else None,
        "feature_time_ms_median": median(feature_times) if feature_times else None,
        "matching_time_ms_mean": mean(matching_times) if matching_times else None,
        "matching_time_ms_median": median(matching_times) if matching_times else None,
        "solver_time_ms_mean": mean(solver_runtimes) if solver_runtimes else solver_mean_f,
        "solver_time_ms_median": median(solver_runtimes) if solver_runtimes else solver_median_f,
        "total_online_ms_mean": mean(online_totals) if online_totals else None,
        "total_online_ms_median": median(online_totals) if online_totals else None,
        "total_cached_ms_mean": mean(cached_totals) if cached_totals else None,
        "total_cached_ms_median": median(cached_totals) if cached_totals else None,
        "peak_gpu_memory_mb": float(payload.get("peak_gpu_memory_mb", 0.0)),
        "notes": notes.strip(),
        "_load_ms_mean": load_mean,
        "_feature_pair_ms_mean": mean(feature_total_times) if feature_total_times else 0.0,
        "_pack_ms_mean": pack_ms,
    }


def weighted_average(rows: list[dict[str, Any]], field: str) -> float | None:
    total = 0.0
    count = 0
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        n = int(row.get("num_pairs", 0))
        total += float(value) * n
        count += n
    if count == 0:
        return None
    return total / count


def read_accuracy_values() -> dict[str, float]:
    values: dict[str, float] = {}
    final_summary = CSV_ROOT / "final_selected_expanded151_lg_test_summary.csv"
    if final_summary.exists():
        with final_summary.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("config") == FINAL_METHOD and row.get("solver_mode") == "calibrated":
                    values[FINAL_METHOD] = float(row["average"])

    chapter9 = CSV_ROOT / "chapter9_test_eval.csv"
    if chapter9.exists():
        grouped: dict[str, list[float]] = defaultdict(list)
        with chapter9.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("solver_mode") != "calibrated":
                    continue
                key = row.get("config_key", "")
                for method, spec in METHODS.items():
                    if spec.accuracy_key == key:
                        grouped[method].append(float(row["mAA@10"]))
        for method, vals in grouped.items():
            if vals and method not in values:
                values[method] = mean(vals)
    return values


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def hardware_string() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return output or "nvidia-smi returned no GPU rows"
    except Exception as exc:
        return f"GPU query unavailable: {exc}"


def aggregate(command: str) -> None:
    ensure_dirs()
    rows = [summarize_detail(path) for path in list_detail_files()]
    if not rows:
        raise RuntimeError("No runtime detail JSON files found. Run methods first.")
    rows.sort(key=lambda row: (list(METHODS).index(row["method"]), SCENES.index(row["scene"])))

    with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        row[field]
                        if field in {"method", "scene", "notes"}
                        else str(int(row[field]))
                        if field == "num_pairs"
                        else fmt(row[field])
                    )
                    for field in CSV_FIELDS
                }
            )

    write_report(rows, command)
    write_figures(rows)
    print(f"[aggregate] wrote {COMPARISON_CSV}")
    print(f"[aggregate] wrote {REPORT_PATH}")
    print(f"[aggregate] wrote {TRADEOFF_FIG}")
    print(f"[aggregate] wrote {BREAKDOWN_FIG}")


def subset_stats_markdown() -> list[str]:
    grouped = load_subset()
    lines = [
        "| scene | sampled pairs | available valid evaluated pairs |",
        "|---|---:|---:|",
    ]
    for scene in SCENES:
        rows = grouped[scene]
        available = rows[0]["available_valid_pairs"] if rows else "0"
        lines.append(f"| {scene} | {len(rows)} | {available} |")
    return lines


def method_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    accuracy = read_accuracy_values()
    out = []
    for method in METHODS:
        method_rows = by_method.get(method, [])
        if not method_rows:
            continue
        out.append(
            {
                "method": method,
                "label": METHODS[method].label,
                "num_pairs": sum(int(row["num_pairs"]) for row in method_rows),
                "feature_pair_ms": weighted_average(method_rows, "_feature_pair_ms_mean"),
                "matching_ms": weighted_average(method_rows, "matching_time_ms_mean"),
                "solver_ms": weighted_average(method_rows, "solver_time_ms_mean"),
                "online_ms": weighted_average(method_rows, "total_online_ms_mean"),
                "cached_ms": weighted_average(method_rows, "total_cached_ms_mean"),
                "load_ms": weighted_average(method_rows, "_load_ms_mean"),
                "pack_ms": weighted_average(method_rows, "_pack_ms_mean"),
                "peak_gpu_memory_mb": max(float(row["peak_gpu_memory_mb"]) for row in method_rows),
                "maa10": accuracy.get(method),
            }
        )
    return out


def write_report(rows: list[dict[str, Any]], command: str) -> None:
    avg_rows = method_average_rows(rows)
    lines = [
        "# Final Runtime Comparison",
        "",
        "## Protocol",
        "",
        "- Dataset: 10 held-out PhotoTourism final-test scenes.",
        "- Subset: fixed seed 42; 100 valid evaluated pairs per scene when available; identical sampled pair list for all methods.",
        "- Matching cap: 2048 correspondences.",
        "- Pose protocol: calibrated RePoseD final protocol with Sampson threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, and 25 LO iterations.",
        "- Final selected method: DINOv3 ViT-L/16 block 16 (`feat_level=-8`), DIFT SD v1.5 (`t=0`, `up_ft_index=2`, ensemble 2), selected projection checkpoint, selected expanded-151 LightGlue checkpoint, `filter_threshold=0.02`.",
        "- Warm-up: 10 pairs per method before recording.",
        "- CUDA timing: `torch.cuda.synchronize()` is called before and after timed GPU blocks.",
        "- Online totals include image loading/preprocessing, feature extraction, matching, benchmark packing/depth sampling, and the calibrated RePoseD solver. Cached totals, where available, use the measured matcher plus packing and solver time with feature extraction excluded.",
        "",
        "## Hardware / GPU",
        "",
        "```text",
        hardware_string(),
        "```",
        "",
        "## Exact Command",
        "",
        "```bash",
        command or "unknown",
        "```",
        "",
        "## Git Commit",
        "",
        f"`{git_commit()}`",
        "",
        "## Sampled-Pair Statistics",
        "",
        *subset_stats_markdown(),
        "",
        "## Per-Method Average Runtime",
        "",
        "| method | pairs | feature extraction / pair (ms) | matching (ms) | solver (ms) | load/preprocess (ms) | pack/depth (ms) | total online (ms) | total cached (ms) | held-out calibrated mAA@10 | peak GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in avg_rows:
        lines.append(
            "| {label} | {pairs} | {feature} | {matching} | {solver} | {load} | {pack} | {online} | {cached} | {maa} | {peak} |".format(
                label=row["label"],
                pairs=row["num_pairs"],
                feature=fmt(row["feature_pair_ms"]),
                matching=fmt(row["matching_ms"]),
                solver=fmt(row["solver_ms"]),
                load=fmt(row["load_ms"]),
                pack=fmt(row["pack_ms"]),
                online=fmt(row["online_ms"]),
                cached=fmt(row["cached_ms"]),
                maa=fmt(row["maa10"]),
                peak=fmt(row["peak_gpu_memory_mb"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SP+LG is the lightweight sparse baseline.",
            "- The final selected method is sparse at the matcher stage but more expensive because it extracts DINOv3 and DIFT descriptors.",
            "- RoMa and RoMaV2 are dense baselines and have a different cost profile; their feature extraction and matching are timed as a combined block.",
            "- Runtime numbers are diagnostic and are not used for model selection.",
            "",
            "## Outputs",
            "",
            f"- `{SUBSET_CSV.relative_to(PROJECT_ROOT)}`",
            f"- `{COMPARISON_CSV.relative_to(PROJECT_ROOT)}`",
            f"- `{TRADEOFF_FIG.relative_to(PROJECT_ROOT)}`",
            f"- `{BREAKDOWN_FIG.relative_to(PROJECT_ROOT)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_rows = method_average_rows(rows)
    labels = [row["label"] for row in avg_rows]
    x = [row["online_ms"] for row in avg_rows]
    y = [row["maa10"] for row in avg_rows]

    plt.figure(figsize=(7.0, 4.8))
    for row in avg_rows:
        if row["online_ms"] is None or row["maa10"] is None:
            continue
        plt.scatter(row["online_ms"], row["maa10"], s=80)
        plt.annotate(row["label"], (row["online_ms"], row["maa10"]), xytext=(6, 5), textcoords="offset points")
    plt.xlabel("Total online time per pair (ms)")
    plt.ylabel("Held-out calibrated mAA@10")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRADEOFF_FIG, dpi=220)
    plt.close()

    feature = [row["feature_pair_ms"] or 0.0 for row in avg_rows]
    matching_sparse = [
        (row["matching_ms"] or 0.0) if METHODS[row["method"]].runner in {"splg", "final"} else 0.0
        for row in avg_rows
    ]
    matching_dense = [
        (row["matching_ms"] or 0.0) if METHODS[row["method"]].runner in {"roma", "romav2"} else 0.0
        for row in avg_rows
    ]
    solver = [row["solver_ms"] or 0.0 for row in avg_rows]
    positions = np.arange(len(labels))

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(positions, feature, label="feature extraction", color="#4C78A8")
    plt.bar(positions, matching_sparse, bottom=feature, label="matching", color="#F58518")
    plt.bar(
        positions,
        matching_dense,
        bottom=feature,
        label="feature extraction + matching",
        color="#F58518",
        hatch="//",
        edgecolor="#A65B00",
        linewidth=0.6,
    )
    bottoms = [a + b + c for a, b, c in zip(feature, matching_sparse, matching_dense)]
    plt.bar(positions, solver, bottom=bottoms, label="solver", color="#54A24B")
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylabel("Runtime per pair (ms)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(BREAKDOWN_FIG, dpi=220)
    plt.close()


def run_method(method: str, warmup_pairs: int, pairs_per_scene: int, seed: int, force: bool) -> None:
    if method not in METHODS:
        raise SystemExit(f"Unknown method: {method}. Choices: {', '.join(METHODS)}")
    if not SUBSET_CSV.exists():
        prepare_subset(seed=seed, pairs_per_scene=pairs_per_scene, force=False)
    subset = load_subset()
    runner = METHODS[method].runner
    if runner == "splg":
        run_superpoint(method, subset, warmup_pairs, force)
    elif runner == "final":
        run_final_selected(method, subset, warmup_pairs, force)
    elif runner == "roma":
        run_roma(method, subset, warmup_pairs, force)
    elif runner == "romav2":
        run_romav2(method, subset, warmup_pairs, force)
    else:
        raise AssertionError(runner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled final runtime benchmark on held-out PhotoTourism subset.")
    parser.add_argument("--prepare-subset", action="store_true")
    parser.add_argument("--run-method", choices=list(METHODS), default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-scene", type=int, default=100)
    parser.add_argument("--warmup-pairs", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--command", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.prepare_subset:
        prepare_subset(args.seed, args.pairs_per_scene, force=args.force)
    if args.run_method:
        run_method(args.run_method, args.warmup_pairs, args.pairs_per_scene, args.seed, force=args.force)
    if args.aggregate:
        aggregate(args.command)
    if not (args.prepare_subset or args.run_method or args.aggregate):
        raise SystemExit("Nothing to do. Use --prepare-subset, --run-method, or --aggregate.")


if __name__ == "__main__":
    main()
