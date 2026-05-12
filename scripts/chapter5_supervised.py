#!/usr/bin/env python3
"""Chapter 5 supervised-adaptation training/evaluation runner.

This script intentionally uses the final raw-image output_v2 protocol instead
of the older run_thesis_benchmark.sh path, which writes to output/ and applies
the shared preprocessed-image directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]
TRAIN_SCENES = ["0080", "0042", "0380", "0000", "0366", "0001", "0005", "0237", "0011", "0148"]

DINOV3_PY = Path("/home.stud/gorbuden/.conda/envs/dinov3/bin/python")
DIFT_PY = Path("/home.stud/gorbuden/.conda/envs/dift/bin/python")
REPOSED_PY = Path("/home.stud/gorbuden/.conda/envs/reposed/bin/python")
LIGHTGLUE_PY = Path("/home.stud/gorbuden/.conda/envs/lightglue/bin/python")
ROMA_PY = Path("/home.stud/gorbuden/.conda/envs/roma/bin/python")
ROMAV2_PY = Path("/home.stud/gorbuden/.conda/envs/romav2/bin/python")

REPORT_DIR = ROOT / "output_v2" / "reports" / "chapter5_supervised"
CSV_DIR = ROOT / "output_v2" / "csv"
LOG_DIR = ROOT / "output_v2" / "logs" / "chapter5_supervised"
CKPT_DIR = ROOT / "output_v2" / "checkpoints" / "chapter5_supervised"
MATCHES_ROOT = ROOT / "output_v2" / "matches_v2"
BENCH_ROOT = ROOT / "output_v2" / "benchmarks_v2"
RESULTS_ROOT = ROOT / "output_v2" / "results_v2"
FEATURE_ROOT = ROOT / "output_v2" / "feature_cache_raw"
SP_ROOT = ROOT / "output_v2" / "sp_cache_raw"
TIMING_ROOT = ROOT / "output_v2" / "timing"
FAIL_CSV = CSV_DIR / "chapter5_missing_or_failed_eval.csv"

TARGET_SOLVERS = {
    "calibrated": "3p_ours_shift_scale+12",
    "shared_focal": "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}


PROJECTION_SPECS: list[dict[str, Any]] = [
    {
        "config_key": "ch5_eval_proj_temp003_h512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_temp003/best.pt",
        "architecture": "1664 -> 512 -> 256",
        "hidden_dims": [512],
        "output_dim": 256,
        "temperature": 0.03,
        "sweep_type": "temperature",
    },
    {
        "config_key": "ch5_eval_proj_temp005_h512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_temp005/best.pt",
        "architecture": "1664 -> 512 -> 256",
        "hidden_dims": [512],
        "output_dim": 256,
        "temperature": 0.05,
        "sweep_type": "temperature",
    },
    {
        "config_key": "ch5_eval_proj_temp007_h512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_v1/best.pt",
        "architecture": "1664 -> 512 -> 256",
        "hidden_dims": [512],
        "output_dim": 256,
        "temperature": 0.07,
        "sweep_type": "temperature",
    },
    {
        "config_key": "ch5_eval_proj_temp010_h512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_temp010/best.pt",
        "architecture": "1664 -> 512 -> 256",
        "hidden_dims": [512],
        "output_dim": 256,
        "temperature": 0.10,
        "sweep_type": "temperature",
    },
    {
        "config_key": "ch5_eval_proj_temp015_h512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_temp015/best.pt",
        "architecture": "1664 -> 512 -> 256",
        "hidden_dims": [512],
        "output_dim": 256,
        "temperature": 0.15,
        "sweep_type": "temperature",
    },
    {
        "config_key": "ch5_eval_proj_wide_h1024_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_wide/best.pt",
        "architecture": "1664 -> 1024 -> 256",
        "hidden_dims": [1024],
        "output_dim": 256,
        "temperature": 0.07,
        "sweep_type": "architecture",
    },
    {
        "config_key": "ch5_eval_proj_deep_h512x512_d256_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_deep/best.pt",
        "architecture": "1664 -> 512 -> 512 -> 256",
        "hidden_dims": [512, 512],
        "output_dim": 256,
        "temperature": 0.07,
        "sweep_type": "architecture",
    },
    {
        "config_key": "ch5_eval_proj_wide_h1024_d128_sp_mnn_mp2048",
        "checkpoint": "output_v2/checkpoints/chapter5_supervised/ch5_train_proj_wide_t007_h1024_d128/best.pt",
        "architecture": "1664 -> 1024 -> 128",
        "hidden_dims": [1024],
        "output_dim": 128,
        "temperature": 0.07,
        "sweep_type": "output_dim",
    },
    {
        "config_key": "ch5_eval_proj_wide_h1024_d512_sp_mnn_mp2048",
        "checkpoint": "output_v2/checkpoints/chapter5_supervised/ch5_train_proj_wide_t007_h1024_d512/best.pt",
        "architecture": "1664 -> 1024 -> 512",
        "hidden_dims": [1024],
        "output_dim": 512,
        "temperature": 0.07,
        "sweep_type": "output_dim",
    },
]

TRAIN_SPECS = [
    {
        "config_key": "ch5_train_proj_wide_t007_h1024_d128",
        "output_dim": 128,
        "architecture": "1664 -> 1024 -> 128",
    },
    {
        "config_key": "ch5_train_proj_wide_t007_h1024_d512",
        "output_dim": 512,
        "architecture": "1664 -> 1024 -> 512",
    },
]

LORA_SPECS: list[dict[str, Any]] = [
    {
        "config_key": "ch5_eval_lora_raw_dinov3_sp_mnn_mp2048",
        "method_name": "Raw LoRA-DINOv3",
        "checkpoint": "experiments/phase2_lora_dinov3only/best.pt",
        "script": "lora_raw_matches.py",
        "uses_dift": False,
        "uses_projection": False,
        "joint_training": False,
        "architecture": "1024",
    },
    {
        "config_key": "ch5_eval_lora_fusion_noproj_sp_mnn_mp2048",
        "method_name": "LoRA-DINOv3 + DIFT fusion",
        "checkpoint": "experiments/phase2_lora_dinov3only/best.pt",
        "script": "lora_fusion_noproj_matches.py",
        "uses_dift": True,
        "uses_projection": False,
        "joint_training": False,
        "architecture": "1664",
    },
    {
        "config_key": "ch5_eval_lora_proj_dinov3only_sp_mnn_mp2048",
        "method_name": "LoRA-DINOv3 + projection",
        "checkpoint": "experiments/phase2_lora_proj_dinov3only_v2/best.pt",
        "script": "lora_matches.py",
        "uses_dift": False,
        "uses_projection": True,
        "joint_training": False,
        "architecture": "1024 -> 512 -> 256",
    },
    {
        "config_key": "ch5_eval_lora_proj_fusion_sp_mnn_mp2048",
        "method_name": "LoRA-DINOv3 + DIFT + projection",
        "checkpoint": "experiments/phase2_lora_proj_fusion_wide512clean_20260327_214806/best.pt",
        "script": "lora_matches.py",
        "uses_dift": True,
        "uses_projection": True,
        "joint_training": False,
        "architecture": "1664 -> 1024 -> 256",
    },
    {
        "config_key": "ch5_eval_lora_joint_fusion_projwarm_sp_mnn_mp2048",
        "method_name": "Joint warm-start LoRA + projection",
        "checkpoint": "experiments/phase2_lora_joint_fusion_projwarm_jointwarm10k_20260328_184458/best.pt",
        "script": "lora_matches.py",
        "uses_dift": True,
        "uses_projection": True,
        "joint_training": True,
        "architecture": "1664 -> 1024 -> 256",
    },
]

REFERENCE_SPECS = [
    {
        "method_name": "DINOv3 block 12 + MNN",
        "config_key": "dinov3_l-12_sp_mnn_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
    {
        "method_name": "DIFT t0 up2 ens2 + MNN",
        "config_key": "dift_t0_up2_ens2_sp_mnn_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
    {
        "method_name": "DINOv3+DIFT fusion + MNN",
        "config_key": "fusion_dinov3b12_dift_t0up2_sp_mnn_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
    {
        "method_name": "SuperPoint+LightGlue",
        "config_key": "superpoint_lg_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
    {
        "method_name": "RoMa",
        "config_key": "roma_outdoor_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
    {
        "method_name": "RoMaV2",
        "config_key": "romav2_precise_mp2048",
        "checkpoint": "",
        "kind": "reference",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def ensure_dirs() -> None:
    for path in [REPORT_DIR, CSV_DIR, LOG_DIR, CKPT_DIR, MATCHES_ROOT, BENCH_ROOT, RESULTS_ROOT, FEATURE_ROOT, SP_ROOT, TIMING_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def env_for_gpu(gpu_id: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("DIFFUSERS_OFFLINE", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_chapter5_supervised")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = (
        "/home.stud/gorbuden/.conda/pkgs/mkl-2023.1.0-h6d00ec8_46342/lib:"
        "/home.stud/gorbuden/.conda/pkgs/intel-openmp-2023.1.0-hdb19cb5_46305/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    return env


def append_failure(experiment: str, expected_config: str, status: str, reason: str, next_action: str, log_path: Path | str) -> None:
    ensure_dirs()
    write_header = not FAIL_CSV.exists()
    with FAIL_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "expected_config", "status", "reason", "needed_for_thesis", "next_action", "log_path"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment": experiment,
                "expected_config": expected_config,
                "status": status,
                "reason": reason,
                "needed_for_thesis": "yes",
                "next_action": next_action,
                "log_path": rel(log_path),
            }
        )


def run_cmd(cmd: list[str], log_path: Path, gpu_id: str | None = None, cwd: Path = ROOT) -> None:
    ensure_dirs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = env_for_gpu(gpu_id)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] RUN {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f"[{now()}] EXIT {code}\n")
        log.flush()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def scene_paths(scene: str) -> dict[str, Path]:
    return {
        "images": ROOT / "datasets" / "phototourism" / scene / "dense" / "images",
        "sparse": ROOT / "datasets" / "phototourism" / scene / "dense" / "sparse",
        "depth": ROOT / "output_v2" / "depth_raw" / scene,
        "pairs": ROOT / "output" / f"pairs_{scene}.txt",
    }


def benchmark_path(config_key: str, scene: str) -> Path:
    return BENCH_ROOT / f"{config_key}_{scene}.h5"


def result_dir(config_key: str, scene: str) -> Path:
    return RESULTS_ROOT / config_key / scene


def summary_glob(config_key: str, scene: str, mode: str = "calibrated") -> list[str]:
    return glob.glob(str(result_dir(config_key, scene) / f"{mode}-*{scene}-2.0t_summary.json"))


def summary_exists(config_key: str, scene: str, mode: str = "calibrated") -> bool:
    return bool(summary_glob(config_key, scene, mode))


def read_summary(config_key: str, scene: str, mode: str = "calibrated") -> dict[str, Any] | None:
    matches = sorted(summary_glob(config_key, scene, mode))
    if not matches:
        return None
    with open(matches[-1], encoding="utf-8") as handle:
        return json.load(handle)


def target_metrics(config_key: str, scene: str, mode: str = "calibrated") -> dict[str, Any] | None:
    summary = read_summary(config_key, scene, mode)
    if not summary:
        return None
    target = TARGET_SOLVERS[mode]
    experiments = summary.get("experiments", [])
    for entry in experiments:
        if entry.get("solver") == target:
            out = dict(entry)
            out["summary_json"] = sorted(summary_glob(config_key, scene, mode))[-1]
            return out
    if experiments:
        out = dict(experiments[0])
        out["summary_json"] = sorted(summary_glob(config_key, scene, mode))[-1]
        out["warning"] = f"target solver {target} missing; used first solver"
        return out
    return None


_AVG_MATCH_CACHE: dict[tuple[str, str], float | None] = {}
AVG_MATCH_CACHE_JSON = REPORT_DIR / "chapter5_avg_match_counts.json"


def _load_avg_match_cache() -> dict[str, Any]:
    if not AVG_MATCH_CACHE_JSON.exists():
        return {}
    try:
        return json.loads(AVG_MATCH_CACHE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_avg_match_cache(payload: dict[str, Any]) -> None:
    AVG_MATCH_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = AVG_MATCH_CACHE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(AVG_MATCH_CACHE_JSON)


def avg_match_count_value(config_key: str, scene: str) -> float | None:
    """Compute the average correspondence count from packed HDF5 or raw NPZ matches."""
    cache_key = (config_key, scene)
    if cache_key in _AVG_MATCH_CACHE:
        return _AVG_MATCH_CACHE[cache_key]

    counts: list[int] = []
    bench = benchmark_path(config_key, scene)
    disk_cache_key = f"{config_key}:{scene}"
    disk_cache = _load_avg_match_cache()
    if bench.exists():
        bench_stat = bench.stat()
        cached = disk_cache.get(disk_cache_key)
        if (
            isinstance(cached, dict)
            and cached.get("source") == rel(bench)
            and cached.get("size") == bench_stat.st_size
            and abs(float(cached.get("mtime", -1.0)) - bench_stat.st_mtime) < 1e-6
        ):
            value = cached.get("avg_matches")
            _AVG_MATCH_CACHE[cache_key] = None if value is None else float(value)
            return _AVG_MATCH_CACHE[cache_key]
        try:
            import h5py  # type: ignore

            with h5py.File(bench, "r") as handle:
                counts = [int(handle[key].shape[0]) for key in handle.keys() if key.startswith("corr_")]
        except Exception:
            counts = []

    if not counts:
        matches_dir = MATCHES_ROOT / config_key / scene
        if matches_dir.exists():
            try:
                import numpy as np  # type: ignore

                for path in sorted(matches_dir.glob("*.npz")):
                    with np.load(path) as data:
                        if "mkpts0" in data:
                            counts.append(int(data["mkpts0"].shape[0]))
                        elif "matches" in data:
                            counts.append(int(data["matches"].shape[0]))
            except Exception:
                counts = []

    value = (sum(counts) / len(counts)) if counts else None
    if bench.exists() and value is not None:
        bench_stat = bench.stat()
        disk_cache[disk_cache_key] = {
            "source": rel(bench),
            "size": bench_stat.st_size,
            "mtime": bench_stat.st_mtime,
            "avg_matches": value,
            "num_pairs": len(counts),
            "updated": now(),
        }
        _save_avg_match_cache(disk_cache)
    _AVG_MATCH_CACHE[cache_key] = value
    return value


def avg_match_count(config_key: str, scene: str) -> str:
    value = avg_match_count_value(config_key, scene)
    return "" if value is None else f"{value:.6f}"


def run_pack_and_eval(config_key: str, scene: str, modes: list[str], gpu_id: str | None, log_path: Path) -> None:
    paths = scene_paths(scene)
    bench = benchmark_path(config_key, scene)
    matches = MATCHES_ROOT / config_key / scene
    out_dir = result_dir(config_key, scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bench.exists():
        run_cmd(
            [
                str(REPOSED_PY),
                str(ROOT / "scripts" / "pack_benchmark.py"),
                "--matches_dir",
                str(matches),
                "--depth_dir",
                str(paths["depth"]),
                "--sparse_dir",
                str(paths["sparse"]),
                "--pairs_file",
                str(paths["pairs"]),
                "--output",
                str(bench),
                "--limit",
                "15000",
            ],
            log_path,
            gpu_id=None,
        )
    for mode in modes:
        if summary_exists(config_key, scene, mode):
            continue
        eval_script = {"calibrated": "eval.py", "shared_focal": "eval_shared_f.py", "varying_focal": "eval_varying_f.py"}[mode]
        run_cmd(
            [
                str(REPOSED_PY),
                eval_script,
                str(bench),
                "-nw",
                "16",
                "--thesis",
                "--output_dir",
                str(out_dir),
                "--max_epipolar_error",
                "2.0",
                "--reproj_threshold",
                "16.0",
            ],
            log_path,
            gpu_id=None,
            cwd=ROOT / "external" / "RePoseD",
        )


def write_eval_config(spec: dict[str, Any], scene: str, command: list[str], gpu_id: str | None) -> None:
    payload = {
        "config_key": spec["config_key"],
        "checkpoint_path": spec.get("checkpoint", ""),
        "command_used": command,
        "timestamp": now(),
        "gpu_id": gpu_id,
        "scene": scene,
        "dino_eval": {
            "backbone": "vit_large_patch16_dinov3.lvd1689m",
            "feat_level": -12,
            "block": 12,
            "internal_long_edge": 1120,
            "divisibility": 16,
        },
        "dift_eval": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "timestep": 0,
            "up_ft_index": 2,
            "ensemble_size": 2,
            "internal_resolution": [768, 768],
        },
        "fusion": {
            "normalization": "independent branch L2 normalization",
            "dift_weight": 0.5,
            "dinov3_weight": 0.5,
            "dimension": 1664,
        },
        "projection_architecture": spec.get("architecture", "none"),
        "lora": {
            "uses_lora": spec.get("kind") == "lora",
            "rank": 4 if spec.get("kind") == "lora" else "unknown",
            "alpha": 8.0 if spec.get("kind") == "lora" else "unknown",
            "uses_dift": spec.get("uses_dift", False),
            "uses_projection": spec.get("uses_projection", spec.get("kind") == "projection"),
            "joint_training": spec.get("joint_training", False),
        },
        "protocol": {
            "raw_dataset_images": True,
            "shared_1120_preprocessing_step": False,
            "original_image_coordinates": True,
            "keypoints": "SuperPoint",
            "matching": "cosine mutual nearest neighbor",
            "max_correspondences": 2048,
            "pair_limit": 15000,
            "min_shared_colmap_points": 100,
            "depth": "UniDepthV2 depth values only",
            "calibrated_intrinsics": "COLMAP intrinsics",
            "max_epipolar_error_px": 2.0,
            "reprojection_threshold_px": 16.0,
            "ransac_iterations": 1000,
            "lo_iterations": 25,
        },
        "warning": "Checkpoint metadata may come from old block16/DIFT-ensemble1 training; evaluation uses Chapter 4 block12/DIFT-ensemble2 settings.",
    }
    out = REPORT_DIR / f"{spec['config_key']}_{scene}_config.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_dift_reference(scene: str, gpu_id: str) -> None:
    config_key = "dift_t0_up2_ens2_sp_mnn_mp2048"
    log_path = LOG_DIR / f"{config_key}_{scene}.log"
    if not summary_exists(config_key, scene, "calibrated"):
        paths = scene_paths(scene)
        matches = MATCHES_ROOT / config_key / scene
        matches.mkdir(parents=True, exist_ok=True)
        command = [
            str(DIFT_PY),
            str(ROOT / "scripts" / "dift_matches.py"),
            "--pairs_file",
            str(paths["pairs"]),
            "--images_dir",
            str(paths["images"]),
            "--output_dir",
            str(matches),
            "--scene",
            scene,
            "--max_points",
            "2048",
            "--use_sp_keypoints",
            "--t",
            "0",
            "--up_ft_index",
            "2",
            "--ensemble_size",
            "2",
            "--img_size",
            "768",
            "768",
            "--feature_cache",
            str(FEATURE_ROOT / config_key / scene),
            "--sp_cache_dir",
            str(SP_ROOT / scene),
            "--device",
            "cuda",
            "--limit",
            "15000",
            "--raw_images",
            "--timing_output",
            str(TIMING_ROOT / f"{config_key}_{scene}_timing.json"),
        ]
        write_eval_config({"config_key": config_key, "checkpoint": "", "architecture": "DIFT reference"}, scene, command, gpu_id)
        run_cmd(command, log_path, gpu_id=gpu_id)
    run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def run_projection(spec: dict[str, Any], scene: str, gpu_id: str) -> None:
    config_key = spec["config_key"]
    log_path = LOG_DIR / f"{config_key}_{scene}.log"
    ckpt = ROOT / spec["checkpoint"]
    if not ckpt.exists():
        append_failure("projection", config_key, "checkpoint_missing", f"Missing checkpoint {ckpt}", "train or locate checkpoint", log_path)
        return
    if not summary_exists(config_key, scene, "calibrated"):
        paths = scene_paths(scene)
        matches = MATCHES_ROOT / config_key / scene
        matches.mkdir(parents=True, exist_ok=True)
        command = [
            str(DINOV3_PY),
            str(ROOT / "scripts" / "projection_matches.py"),
            "--pairs_file",
            str(paths["pairs"]),
            "--images_dir",
            str(paths["images"]),
            "--output_dir",
            str(matches),
            "--scene",
            scene,
            "--checkpoint",
            str(ckpt),
            "--projection_tag",
            config_key.removesuffix("_sp_mnn_mp2048"),
            "--max_points",
            "2048",
            "--feat_level",
            "-12",
            "--img_size",
            "768",
            "768",
            "--t",
            "0",
            "--up_ft_index",
            "2",
            "--ensemble_size",
            "2",
            "--alpha",
            "0.5",
            "--feature_cache",
            str(FEATURE_ROOT / config_key / scene),
            "--cache_root",
            str(FEATURE_ROOT),
            "--sp_cache_dir",
            str(SP_ROOT / scene),
            "--device",
            "cuda",
            "--limit",
            "15000",
            "--timing_output",
            str(TIMING_ROOT / f"{config_key}_{scene}_timing.json"),
        ]
        write_eval_config({**spec, "kind": "projection"}, scene, command, gpu_id)
        run_cmd(command, log_path, gpu_id=gpu_id)
    run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def run_lora(spec: dict[str, Any], scene: str, gpu_id: str) -> None:
    config_key = spec["config_key"]
    log_path = LOG_DIR / f"{config_key}_{scene}.log"
    ckpt = ROOT / spec["checkpoint"]
    if not ckpt.exists():
        append_failure("lora", config_key, "checkpoint_missing", f"Missing checkpoint {ckpt}", "locate checkpoint", log_path)
        return
    if not summary_exists(config_key, scene, "calibrated"):
        paths = scene_paths(scene)
        matches = MATCHES_ROOT / config_key / scene
        matches.mkdir(parents=True, exist_ok=True)
        command = [
            str(DINOV3_PY),
            str(ROOT / "scripts" / spec["script"]),
            "--lora_checkpoint",
            str(ckpt),
            "--pairs_file",
            str(paths["pairs"]),
            "--images_dir",
            str(paths["images"]),
            "--output_dir",
            str(matches),
            "--scene",
            scene,
            "--max_points",
            "2048",
            "--feat_level",
            "-12",
            "--dino_img_size",
            "1120",
            "--device",
            "cuda",
            "--cache_root",
            str(FEATURE_ROOT),
            "--lora_cache",
            str(FEATURE_ROOT / config_key),
            "--limit",
            "15000",
        ]
        if spec.get("uses_dift"):
            command += [
                "--img_size",
                "768",
                "768",
                "--t",
                "0",
                "--up_ft_index",
                "2",
                "--ensemble_size",
                "2",
                "--alpha",
                "0.5",
            ]
        write_eval_config({**spec, "kind": "lora"}, scene, command, gpu_id)
        run_cmd(command, log_path, gpu_id=gpu_id)
    run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def train_is_complete(out_dir: Path) -> bool:
    if not (out_dir / "best.pt").exists():
        return False
    log_path = out_dir / "train_log.json"
    if not log_path.exists():
        return False
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(log) and int(round(float(log[-1].get("epoch", -1)))) >= 9


def train_missing(gpu_id: str) -> None:
    ensure_dirs()
    for spec in TRAIN_SPECS:
        out_dir = CKPT_DIR / spec["config_key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{spec['config_key']}.log"
        if train_is_complete(out_dir):
            if (out_dir / "latest.pt").exists() and not (out_dir / "last.pt").exists():
                shutil.copy2(out_dir / "latest.pt", out_dir / "last.pt")
            continue
        command = [
            str(DINOV3_PY),
            "-u",
            str(ROOT / "scripts" / "train_projection_head.py"),
            "--sparse_dir",
            str(ROOT / "data" / "sparse_train"),
            "--scenes",
            *TRAIN_SCENES,
            "--input_dim",
            "1664",
            "--hidden_dims",
            "1024",
            "--output_dim",
            str(spec["output_dim"]),
            "--epochs",
            "10",
            "--pairs_per_epoch",
            "50000",
            "--val_pairs_per_epoch",
            "1000",
            "--sparse_scene_cache_size",
            "1",
            "--lr",
            "1e-3",
            "--weight_decay",
            "1e-4",
            "--temperature",
            "0.07",
            "--num_correspondences",
            "512",
            "--min_correspondences",
            "50",
            "--seed",
            "42",
            "--device",
            "cuda:0",
            "--num_workers",
            "0",
            "--log_interval",
            "500",
            "--output_dir",
            str(out_dir),
        ]
        (out_dir / "command_used.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        try:
            run_cmd(command, log_path, gpu_id=gpu_id)
            if (out_dir / "latest.pt").exists():
                shutil.copy2(out_dir / "latest.pt", out_dir / "last.pt")
        except subprocess.CalledProcessError as exc:
            append_failure(
                "projection_output_dim_training",
                spec["config_key"],
                "failed",
                f"training command exited {exc.returncode}",
                "inspect log and rerun only this missing output-dim model",
                log_path,
            )


def wait_for_training_and_dift() -> None:
    while True:
        train_done = all(train_is_complete(CKPT_DIR / spec["config_key"]) for spec in TRAIN_SPECS)
        dift_done = all(summary_exists("dift_t0_up2_ens2_sp_mnn_mp2048", scene, "calibrated") for scene in SCENES)
        if train_done and dift_done:
            return
        time.sleep(300)


def eval_projection_existing(gpu_id: str) -> None:
    ensure_dirs()
    for scene in SCENES:
        run_dift_reference(scene, gpu_id)
    for spec in PROJECTION_SPECS[:7]:
        for scene in SCENES:
            try:
                run_projection(spec, scene, gpu_id)
            except subprocess.CalledProcessError as exc:
                append_failure("projection_eval", spec["config_key"], "failed", f"exit {exc.returncode} on {scene}", "inspect log and rerun scene", LOG_DIR / f"{spec['config_key']}_{scene}.log")


def eval_lora_and_new(gpu_id: str) -> None:
    ensure_dirs()
    wait_for_training_and_dift()
    for spec in PROJECTION_SPECS[7:]:
        for scene in SCENES:
            try:
                run_projection(spec, scene, gpu_id)
            except subprocess.CalledProcessError as exc:
                append_failure("projection_eval", spec["config_key"], "failed", f"exit {exc.returncode} on {scene}", "inspect log and rerun scene", LOG_DIR / f"{spec['config_key']}_{scene}.log")
    for spec in LORA_SPECS:
        for scene in SCENES:
            try:
                run_lora(spec, scene, gpu_id)
            except subprocess.CalledProcessError as exc:
                append_failure("lora_eval", spec["config_key"], "failed", f"exit {exc.returncode} on {scene}", "inspect log and rerun scene", LOG_DIR / f"{spec['config_key']}_{scene}.log")


def avg_calibrated(config_key: str) -> float | None:
    vals = []
    for scene in SCENES:
        m = target_metrics(config_key, scene, "calibrated")
        if not m:
            return None
        vals.append(float(m["mAA@10"]))
    return sum(vals) / len(vals)


def run_best_modes() -> None:
    complete_projection = [(spec, avg_calibrated(spec["config_key"])) for spec in PROJECTION_SPECS]
    complete_projection = [(s, v) for s, v in complete_projection if v is not None]
    complete_lora = [(spec, avg_calibrated(spec["config_key"])) for spec in LORA_SPECS]
    complete_lora = [(s, v) for s, v in complete_lora if v is not None]
    best_specs = []
    if complete_projection:
        best_specs.append(max(complete_projection, key=lambda item: item[1])[0])
    if complete_lora:
        best_specs.append(max(complete_lora, key=lambda item: item[1])[0])
    for spec in best_specs:
        for scene in SCENES:
            log_path = LOG_DIR / f"{spec['config_key']}_{scene}.log"
            try:
                run_pack_and_eval(spec["config_key"], scene, ["shared_focal", "varying_focal"], gpu_id=None, log_path=log_path)
            except subprocess.CalledProcessError as exc:
                append_failure("best_solver_modes", spec["config_key"], "failed", f"exit {exc.returncode} on {scene}", "rerun shared/varying focal modes", log_path)


def all_calibrated_done() -> bool:
    expected = [s["config_key"] for s in PROJECTION_SPECS] + [s["config_key"] for s in LORA_SPECS]
    return all(summary_exists(key, scene, "calibrated") for key in expected for scene in SCENES)


def finalize(wait: bool) -> None:
    ensure_dirs()
    if wait:
        while not all_calibrated_done():
            aggregate()
            time.sleep(600)
    run_best_modes()
    aggregate()


def row_from_metrics(config_key: str, scene: str, mode: str) -> dict[str, Any]:
    metrics = target_metrics(config_key, scene, mode)
    if not metrics:
        return {
            "scene": scene,
            "solver_mode": mode,
            "mAA10": "",
            "inlier_ratio": "",
            "avg_matches": avg_match_count(config_key, scene),
            "median_pose_error": "",
            "median_rotation_error": "",
            "median_translation_error": "",
            "summary_json": "",
            "notes": "missing",
        }
    return {
        "scene": scene,
        "solver_mode": mode,
        "mAA10": metrics.get("mAA@10", ""),
        "inlier_ratio": metrics.get("mean_inlier_ratio", ""),
        "avg_matches": avg_match_count(config_key, scene),
        "median_pose_error": metrics.get("median_pose_error", ""),
        "median_rotation_error": metrics.get("median_R_error", ""),
        "median_translation_error": metrics.get("median_t_error", ""),
        "summary_json": rel(metrics.get("summary_json", "")),
        "notes": metrics.get("warning", ""),
    }


def write_config_result_json(spec: dict[str, Any]) -> None:
    key = spec["config_key"]
    scene_metrics = {}
    for scene in SCENES:
        scene_metrics[scene] = {}
        scene_avg_matches = avg_match_count_value(key, scene)
        for mode in ["calibrated", "shared_focal", "varying_focal"]:
            metrics = target_metrics(key, scene, mode)
            if metrics is not None:
                metrics = dict(metrics)
                metrics["avg_matches"] = scene_avg_matches
            scene_metrics[scene][mode] = metrics
    payload = {
        "config_key": key,
        "checkpoint_path": spec.get("checkpoint", ""),
        "scene_metrics": scene_metrics,
        "avg_match_count_by_scene": {scene: avg_match_count_value(key, scene) for scene in SCENES},
        "summary_json_paths": {
            scene: {
                mode: [rel(Path(p)) for p in summary_glob(key, scene, mode)]
                for mode in ["calibrated", "shared_focal", "varying_focal"]
            }
            for scene in SCENES
        },
        "benchmark_paths": {scene: rel(benchmark_path(key, scene)) for scene in SCENES},
        "match_paths": {scene: rel(MATCHES_ROOT / key / scene) for scene in SCENES},
        "warnings": [
            "Evaluation uses Chapter 4 raw-image block12/DIFT-ensemble2 protocol.",
            "Checkpoint training metadata may be old block16/DIFT-ensemble1.",
        ],
    }
    (REPORT_DIR / f"{key}_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def aggregate() -> None:
    ensure_dirs()
    for spec in PROJECTION_SPECS + LORA_SPECS + REFERENCE_SPECS:
        write_config_result_json(spec)

    with (CSV_DIR / "chapter5_supervised_projection_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "config_key", "checkpoint_path", "sweep_type", "temperature", "architecture", "hidden_dims", "output_dim",
            "scene", "solver_mode", "mAA10", "inlier_ratio", "avg_matches", "median_pose_error",
            "median_rotation_error", "median_translation_error", "summary_json", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in PROJECTION_SPECS:
            modes = ["calibrated", "shared_focal", "varying_focal"]
            for scene in SCENES:
                for mode in modes:
                    base = row_from_metrics(spec["config_key"], scene, mode)
                    writer.writerow({
                        "config_key": spec["config_key"],
                        "checkpoint_path": spec["checkpoint"],
                        "sweep_type": spec["sweep_type"],
                        "temperature": spec["temperature"],
                        "architecture": spec["architecture"],
                        "hidden_dims": json.dumps(spec["hidden_dims"]),
                        "output_dim": spec["output_dim"],
                        **base,
                    })

    with (CSV_DIR / "chapter5_supervised_lora_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "config_key", "method_name", "checkpoint_path", "lora_rank", "lora_alpha", "adapted_blocks", "target_modules",
            "uses_dift", "uses_projection", "joint_training", "scene", "solver_mode", "mAA10", "inlier_ratio",
            "avg_matches", "median_pose_error", "median_rotation_error", "median_translation_error", "summary_json", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in LORA_SPECS:
            for scene in SCENES:
                for mode in ["calibrated", "shared_focal", "varying_focal"]:
                    base = row_from_metrics(spec["config_key"], scene, mode)
                    writer.writerow({
                        "config_key": spec["config_key"],
                        "method_name": spec["method_name"],
                        "checkpoint_path": spec["checkpoint"],
                        "lora_rank": 4,
                        "lora_alpha": 8.0,
                        "adapted_blocks": "0-16",
                        "target_modules": "attn.qkv",
                        "uses_dift": spec["uses_dift"],
                        "uses_projection": spec["uses_projection"],
                        "joint_training": spec["joint_training"],
                        **base,
                    })

    write_summary_csv()
    write_missing_csv_snapshot()
    write_markdown_report()


def average_for_mode(config_key: str, mode: str) -> str:
    vals = []
    for scene in SCENES:
        m = target_metrics(config_key, scene, mode)
        if not m:
            return ""
        vals.append(float(m.get("mAA@10", 0.0)))
    return f"{sum(vals) / len(vals):.6f}"


def average_field(config_key: str, field: str, mode: str = "calibrated") -> str:
    vals = []
    for scene in SCENES:
        m = target_metrics(config_key, scene, mode)
        if not m or field not in m:
            return ""
        vals.append(float(m[field]))
    return f"{sum(vals) / len(vals):.6f}"


def average_matches_for_config(config_key: str) -> str:
    vals = []
    for scene in SCENES:
        value = avg_match_count_value(config_key, scene)
        if value is None:
            return ""
        vals.append(value)
    return f"{sum(vals) / len(vals):.6f}"


def write_summary_csv() -> None:
    candidates: list[dict[str, Any]] = []
    fusion_ref = next(s for s in REFERENCE_SPECS if s["config_key"] == "fusion_dinov3b12_dift_t0up2_sp_mnn_mp2048")
    candidates.append({"method_name": "selected training-free fusion", **fusion_ref})
    best_proj = best_spec(PROJECTION_SPECS)
    if best_proj:
        candidates.append({"method_name": "best projection head", **best_proj})
    method_map = [
        "ch5_eval_lora_raw_dinov3_sp_mnn_mp2048",
        "ch5_eval_lora_fusion_noproj_sp_mnn_mp2048",
        "ch5_eval_lora_proj_dinov3only_sp_mnn_mp2048",
        "ch5_eval_lora_proj_fusion_sp_mnn_mp2048",
        "ch5_eval_lora_joint_fusion_projwarm_sp_mnn_mp2048",
    ]
    for key in method_map:
        spec = next(s for s in LORA_SPECS if s["config_key"] == key)
        candidates.append(spec)
    for key in ["superpoint_lg_mp2048", "roma_outdoor_mp2048", "romav2_precise_mp2048"]:
        candidates.append(next(s for s in REFERENCE_SPECS if s["config_key"] == key))

    with (CSV_DIR / "chapter5_supervised_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method_name", "config_key", "calibrated_avg", "shared_focal_avg", "varying_focal_avg",
            "avg_inlier_ratio", "avg_matches", "checkpoint_path", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in candidates:
            key = spec["config_key"]
            writer.writerow(
                {
                    "method_name": spec["method_name"],
                    "config_key": key,
                    "calibrated_avg": average_for_mode(key, "calibrated"),
                    "shared_focal_avg": average_for_mode(key, "shared_focal"),
                    "varying_focal_avg": average_for_mode(key, "varying_focal"),
                    "avg_inlier_ratio": average_field(key, "mean_inlier_ratio", "calibrated"),
                    "avg_matches": average_matches_for_config(key),
                    "checkpoint_path": spec.get("checkpoint", ""),
                    "notes": "pending" if not average_for_mode(key, "calibrated") else "",
                }
            )


def best_spec(specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [(spec, avg_calibrated(spec["config_key"])) for spec in specs]
    complete = [(spec, val) for spec, val in complete if val is not None]
    if not complete:
        return None
    return max(complete, key=lambda item: item[1])[0]


def write_missing_csv_snapshot() -> None:
    rows = []
    for spec in PROJECTION_SPECS + LORA_SPECS:
        for scene in SCENES:
            if not summary_exists(spec["config_key"], scene, "calibrated"):
                rows.append(
                    {
                        "experiment": spec.get("method_name", spec.get("sweep_type", "projection")),
                        "expected_config": f"{spec['config_key']}:{scene}:calibrated",
                        "status": "missing_or_pending",
                        "reason": "calibrated summary not present yet",
                        "needed_for_thesis": "yes",
                        "next_action": "wait for running screen job or inspect log",
                        "log_path": rel(LOG_DIR / f"{spec['config_key']}_{scene}.log"),
                    }
                )
    existing = []
    if FAIL_CSV.exists():
        with FAIL_CSV.open(newline="", encoding="utf-8") as handle:
            existing = [
                row for row in csv.DictReader(handle)
                if row.get("status") != "missing_or_pending"
            ]
    with FAIL_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "expected_config", "status", "reason", "needed_for_thesis", "next_action", "log_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)


def write_markdown_report() -> None:
    best_proj = best_spec(PROJECTION_SPECS)
    best_lora = best_spec(LORA_SPECS)
    lines = [
        "# Chapter 5 Supervised Adaptation Evaluation Report",
        "",
        f"Generated: `{now()}`",
        "",
        "## 1. Runtime estimate and execution plan",
        "",
        "- Existing projection-head training logs show about `1.3-2.5 h` per 10-epoch projection run.",
        "- The two output-dimension models are expected to take about `3-5 h` sequentially.",
        "- Evaluation wall-clock is dominated by DIFT ens2 cache completion and per-method matching; with GPUs 1 and 3 free, expected wall time is roughly overnight if caches are warm, longer if DIFT ens2 caches must be built for all scenes.",
        "- With one free GPU, expect approximately the sum of training plus all matching stages.",
        "- Started/planned screens: `ch5_proj_train_gpu1`, `ch5_eval_gpu3`, `ch5_eval_gpu1`, `ch5_finalize`.",
        "",
        "## 2. Executive summary",
        "",
    ]
    if best_proj:
        lines.append(f"- Best projection so far: `{best_proj['config_key']}` calibrated avg `{average_for_mode(best_proj['config_key'], 'calibrated')}`.")
    else:
        lines.append("- Best projection so far: pending.")
    if best_lora:
        lines.append(f"- Best LoRA so far: `{best_lora['config_key']}` calibrated avg `{average_for_mode(best_lora['config_key'], 'calibrated')}`.")
    else:
        lines.append("- Best LoRA so far: pending.")
    lines += [
        "- Whether LoRA improves over projection head: pending until all calibrated rows are complete.",
        "- Selected method to carry into Chapter 6: pending final comparison.",
        "",
        "## 3. Protocol verification",
        "",
        "- Raw dataset images: yes.",
        "- Shared 1120 px preprocessing directory: not used for Chapter 5 output_v2 runs.",
        "- DINOv3 eval: ViT-L/16, `feat_level=-12`, block 12, internal long edge 1120, div-16 padding.",
        "- DIFT eval: Stable Diffusion v1.5, `t=0`, `up_ft_index=2`, ensemble 2, `768 x 768`.",
        "- Matching: SuperPoint keypoints, cosine MNN, `mp2048`.",
        "- Pose eval: threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, 25 LO iterations.",
        "- Learned matcher training results: excluded.",
        "- Deviations: output-dim projection training uses the existing `data/sparse_train` bundle; inventory evidence indicates that bundle was produced by the older block16/DIFT-ensemble1 training feature path. Evaluation still uses Chapter 4 settings.",
        "",
        "## 4. Projection-head results",
        "",
    ]
    lines.extend(markdown_table(PROJECTION_SPECS, "projection"))
    lines += [
        "",
        "## 5. LoRA results",
        "",
    ]
    lines.extend(markdown_table(LORA_SPECS, "lora"))
    lines += [
        "",
        "## 6. Final supervised-adaptation comparison",
        "",
        "See `output_v2/csv/chapter5_supervised_summary.csv`.",
        "",
        "## 7. Missing or failed evaluations",
        "",
        "See `output_v2/csv/chapter5_missing_or_failed_eval.csv`.",
        "",
        "## 8. Suggested Chapter 5 structure",
        "",
        "1. Supervised projection-head objective and training data.",
        "2. Projection temperature, architecture, and output-dimension ablations.",
        "3. LoRA adaptation of DINOv3 descriptors.",
        "4. Projection-head versus LoRA comparison under the final raw-image protocol.",
        "5. Selection of the Chapter 6 descriptor frontend.",
        "",
        "## 9. Warnings",
        "",
        "- Existing checkpoints were trained under older metadata: DINO block16/`feat_level=-8` and DIFT ensemble1/old caches appear in provenance.",
        "- This evaluation uses block12/`feat_level=-12` and DIFT ensemble2, as requested.",
        "- Smoke/debug/sanity runs and old single fused-QKV LoRA are ignored.",
        "- `mp2000`, 1.0 px threshold, old preprocessing summaries, and learned LightGlue fine-tuning results are not used as Chapter 5 main rows.",
    ]
    (ROOT / "output_v2" / "reports" / "chapter5_supervised_adaptation_eval_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def markdown_table(specs: list[dict[str, Any]], kind: str) -> list[str]:
    lines = ["| Config | Calibrated avg | Sacre | Reichstag | St Peters | Notes |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for spec in specs:
        key = spec["config_key"]
        vals = []
        for scene in SCENES:
            m = target_metrics(key, scene, "calibrated")
            vals.append("" if not m else f"{float(m['mAA@10']):.3f}")
        avg = average_for_mode(key, "calibrated")
        note = "pending" if not avg else ""
        lines.append(f"| `{key}` | {avg} | {vals[0]} | {vals[1]} | {vals[2]} | {note} |")
    return lines


def init_report() -> None:
    ensure_dirs()
    aggregate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["init", "train_missing", "eval_projection_existing", "eval_lora_and_new", "finalize", "aggregate"],
    )
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.mode == "init":
        init_report()
    elif args.mode == "train_missing":
        if args.gpu is None:
            raise SystemExit("--gpu is required for train_missing")
        train_missing(args.gpu)
    elif args.mode == "eval_projection_existing":
        if args.gpu is None:
            raise SystemExit("--gpu is required for eval_projection_existing")
        eval_projection_existing(args.gpu)
    elif args.mode == "eval_lora_and_new":
        if args.gpu is None:
            raise SystemExit("--gpu is required for eval_lora_and_new")
        eval_lora_and_new(args.gpu)
    elif args.mode == "finalize":
        finalize(wait=args.wait)
    elif args.mode == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
