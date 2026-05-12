#!/usr/bin/env python3
"""Chapter 5 supervised adaptation reruns with DINOv3 block 16.

This runner mirrors the block-12 Chapter 5 evaluation, but changes only the
DINOv3 feature level from block 12 (`feat_level=-12`) to block 16
(`feat_level=-8`).  DIFT settings, matching settings, pose evaluation settings,
and checkpoints are intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import chapter5_supervised as ch5


FEAT_LEVEL = -8
DINO_BLOCK = 16
DIFT_ENSEMBLE = 2


def _b16_projection_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["config_key"] = spec["config_key"].replace("ch5_eval_", "ch5_b16_eval_", 1)
    out["dino_block"] = DINO_BLOCK
    out["feat_level"] = FEAT_LEVEL
    out["dift_ensemble"] = DIFT_ENSEMBLE
    return out


def _b16_lora_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["config_key"] = spec["config_key"].replace("ch5_eval_", "ch5_b16_eval_", 1)
    out["dino_block"] = DINO_BLOCK
    out["feat_level"] = FEAT_LEVEL
    out["dift_ensemble"] = DIFT_ENSEMBLE if spec.get("uses_dift") else "not used"
    return out


PROJECTION_SPECS = [_b16_projection_spec(spec) for spec in ch5.PROJECTION_SPECS]
LORA_SPECS = [_b16_lora_spec(spec) for spec in ch5.LORA_SPECS]
ALL_SPECS = PROJECTION_SPECS + LORA_SPECS


def write_eval_config(spec: dict[str, Any], scene: str, command: list[str], gpu_id: str) -> None:
    uses_dift = spec.get("uses_dift", True)
    payload = {
        "config_key": spec["config_key"],
        "checkpoint_path": spec.get("checkpoint", ""),
        "command_used": command,
        "timestamp": ch5.now(),
        "gpu_id": gpu_id,
        "scene": scene,
        "dino_eval": {
            "backbone": "vit_large_patch16_dinov3.lvd1689m",
            "feat_level": FEAT_LEVEL,
            "block": DINO_BLOCK,
            "internal_long_edge": 1120,
            "divisibility": 16,
        },
        "dift_eval": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5" if uses_dift else "not used",
            "timestep": 0 if uses_dift else "not used",
            "up_ft_index": 2 if uses_dift else "not used",
            "ensemble_size": DIFT_ENSEMBLE if uses_dift else "not used",
            "internal_resolution": [768, 768] if uses_dift else "not used",
        },
        "fusion": {
            "normalization": "independent branch L2 normalization" if uses_dift else "single DINOv3 branch",
            "dift_weight": 0.5 if uses_dift else "not used",
            "dinov3_weight": 0.5 if uses_dift else 1.0,
            "dimension": 1664 if uses_dift else 1024,
        },
        "projection_architecture": spec.get("architecture", "none"),
        "lora": {
            "uses_lora": spec.get("kind") == "lora",
            "rank": 4 if spec.get("kind") == "lora" else "unknown",
            "alpha": 8.0 if spec.get("kind") == "lora" else "unknown",
            "uses_dift": spec.get("uses_dift", spec.get("kind") == "projection"),
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
        "warning": "Block-16 rerun: only DINOv3 block changed from the Chapter 5 block-12 run; DIFT ensemble remains 2.",
    }
    out = ch5.REPORT_DIR / f"{spec['config_key']}_{scene}_config.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_projection(spec: dict[str, Any], scene: str, gpu_id: str) -> None:
    config_key = spec["config_key"]
    log_path = ch5.LOG_DIR / f"{config_key}_{scene}.log"
    ckpt = ch5.ROOT / spec["checkpoint"]
    if not ckpt.exists():
        ch5.append_failure("projection_block16", config_key, "checkpoint_missing", f"Missing checkpoint {ckpt}", "locate checkpoint", log_path)
        return
    if not ch5.summary_exists(config_key, scene, "calibrated"):
        paths = ch5.scene_paths(scene)
        matches = ch5.MATCHES_ROOT / config_key / scene
        matches.mkdir(parents=True, exist_ok=True)
        command = [
            str(ch5.DINOV3_PY),
            str(ch5.ROOT / "scripts" / "projection_matches.py"),
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
            str(FEAT_LEVEL),
            "--img_size",
            "768",
            "768",
            "--t",
            "0",
            "--up_ft_index",
            "2",
            "--ensemble_size",
            str(DIFT_ENSEMBLE),
            "--alpha",
            "0.5",
            "--feature_cache",
            str(ch5.FEATURE_ROOT / config_key / scene),
            "--cache_root",
            str(ch5.FEATURE_ROOT),
            "--sp_cache_dir",
            str(ch5.SP_ROOT / scene),
            "--device",
            "cuda",
            "--limit",
            "15000",
            "--timing_output",
            str(ch5.TIMING_ROOT / f"{config_key}_{scene}_timing.json"),
        ]
        write_eval_config({**spec, "kind": "projection", "uses_dift": True}, scene, command, gpu_id)
        ch5.run_cmd(command, log_path, gpu_id=gpu_id)
    ch5.run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def run_lora(spec: dict[str, Any], scene: str, gpu_id: str) -> None:
    config_key = spec["config_key"]
    log_path = ch5.LOG_DIR / f"{config_key}_{scene}.log"
    ckpt = ch5.ROOT / spec["checkpoint"]
    if not ckpt.exists():
        ch5.append_failure("lora_block16", config_key, "checkpoint_missing", f"Missing checkpoint {ckpt}", "locate checkpoint", log_path)
        return
    if not ch5.summary_exists(config_key, scene, "calibrated"):
        paths = ch5.scene_paths(scene)
        matches = ch5.MATCHES_ROOT / config_key / scene
        matches.mkdir(parents=True, exist_ok=True)
        command = [
            str(ch5.DINOV3_PY),
            str(ch5.ROOT / "scripts" / spec["script"]),
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
            str(FEAT_LEVEL),
            "--dino_img_size",
            "1120",
            "--device",
            "cuda",
            "--cache_root",
            str(ch5.FEATURE_ROOT),
            "--lora_cache",
            str(ch5.FEATURE_ROOT / config_key),
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
                str(DIFT_ENSEMBLE),
                "--alpha",
                "0.5",
            ]
        write_eval_config({**spec, "kind": "lora"}, scene, command, gpu_id)
        ch5.run_cmd(command, log_path, gpu_id=gpu_id)
    ch5.run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def task_list() -> list[tuple[str, dict[str, Any], str]]:
    tasks: list[tuple[str, dict[str, Any], str]] = []
    for spec in PROJECTION_SPECS:
        for scene in ch5.SCENES:
            tasks.append(("projection", spec, scene))
    for spec in LORA_SPECS:
        for scene in ch5.SCENES:
            tasks.append(("lora", spec, scene))
    return tasks


def run_shard(gpu_id: str, shard: int, num_shards: int) -> None:
    ch5.ensure_dirs()
    tasks = [task for index, task in enumerate(task_list()) if index % num_shards == shard]
    for kind, spec, scene in tasks:
        try:
            if kind == "projection":
                run_projection(spec, scene, gpu_id)
            else:
                run_lora(spec, scene, gpu_id)
        except subprocess.CalledProcessError as exc:
            ch5.append_failure(
                f"{kind}_block16",
                spec["config_key"],
                "failed",
                f"exit {exc.returncode} on {scene}",
                "inspect log and rerun scene",
                ch5.LOG_DIR / f"{spec['config_key']}_{scene}.log",
            )
    aggregate()


def avg_calibrated(config_key: str) -> float | None:
    vals = []
    for scene in ch5.SCENES:
        metrics = ch5.target_metrics(config_key, scene, "calibrated")
        if not metrics:
            return None
        vals.append(float(metrics["mAA@10"]))
    return sum(vals) / len(vals)


def all_calibrated_done() -> bool:
    return all(ch5.summary_exists(spec["config_key"], scene, "calibrated") for spec in ALL_SPECS for scene in ch5.SCENES)


def run_best_modes() -> None:
    complete_projection = [(spec, avg_calibrated(spec["config_key"])) for spec in PROJECTION_SPECS]
    complete_projection = [(spec, value) for spec, value in complete_projection if value is not None]
    complete_lora = [(spec, avg_calibrated(spec["config_key"])) for spec in LORA_SPECS]
    complete_lora = [(spec, value) for spec, value in complete_lora if value is not None]
    best_specs: list[dict[str, Any]] = []
    if complete_projection:
        best_specs.append(max(complete_projection, key=lambda item: item[1])[0])
    if complete_lora:
        best_specs.append(max(complete_lora, key=lambda item: item[1])[0])
    for spec in best_specs:
        for scene in ch5.SCENES:
            log_path = ch5.LOG_DIR / f"{spec['config_key']}_{scene}.log"
            try:
                ch5.run_pack_and_eval(spec["config_key"], scene, ["shared_focal", "varying_focal"], gpu_id=None, log_path=log_path)
            except subprocess.CalledProcessError as exc:
                ch5.append_failure(
                    "best_solver_modes_block16",
                    spec["config_key"],
                    "failed",
                    f"exit {exc.returncode} on {scene}",
                    "rerun shared/varying focal modes",
                    log_path,
                )


def row_for(spec: dict[str, Any], scene: str, mode: str) -> dict[str, Any]:
    row = ch5.row_from_metrics(spec["config_key"], scene, mode)
    return {
        "config_key": spec["config_key"],
        "checkpoint_path": spec.get("checkpoint", ""),
        "dino_block": DINO_BLOCK,
        "feat_level": FEAT_LEVEL,
        "dift_ensemble": spec.get("dift_ensemble", DIFT_ENSEMBLE),
        **row,
    }


def average_for_mode(config_key: str, mode: str) -> str:
    vals = []
    for scene in ch5.SCENES:
        metrics = ch5.target_metrics(config_key, scene, mode)
        if not metrics:
            return ""
        vals.append(float(metrics["mAA@10"]))
    return f"{sum(vals) / len(vals):.6f}"


def average_field(config_key: str, field: str, mode: str = "calibrated") -> str:
    vals = []
    for scene in ch5.SCENES:
        metrics = ch5.target_metrics(config_key, scene, mode)
        if not metrics or field not in metrics:
            return ""
        vals.append(float(metrics[field]))
    return f"{sum(vals) / len(vals):.6f}"


def average_matches(config_key: str) -> str:
    vals = []
    for scene in ch5.SCENES:
        value = ch5.avg_match_count_value(config_key, scene)
        if value is None:
            return ""
        vals.append(value)
    return f"{sum(vals) / len(vals):.6f}"


def write_result_json(spec: dict[str, Any]) -> None:
    key = spec["config_key"]
    payload = {
        "config_key": key,
        "checkpoint_path": spec.get("checkpoint", ""),
        "dino_block": DINO_BLOCK,
        "feat_level": FEAT_LEVEL,
        "dift_ensemble": spec.get("dift_ensemble", DIFT_ENSEMBLE),
        "scene_metrics": {
            scene: {
                mode: ch5.target_metrics(key, scene, mode)
                for mode in ["calibrated", "shared_focal", "varying_focal"]
            }
            for scene in ch5.SCENES
        },
        "avg_match_count_by_scene": {scene: ch5.avg_match_count_value(key, scene) for scene in ch5.SCENES},
        "summary_json_paths": {
            scene: {
                mode: [ch5.rel(Path(path)) for path in ch5.summary_glob(key, scene, mode)]
                for mode in ["calibrated", "shared_focal", "varying_focal"]
            }
            for scene in ch5.SCENES
        },
        "benchmark_paths": {scene: ch5.rel(ch5.benchmark_path(key, scene)) for scene in ch5.SCENES},
        "match_paths": {scene: ch5.rel(ch5.MATCHES_ROOT / key / scene) for scene in ch5.SCENES},
        "warnings": ["Block-16 rerun: DINO block changed to 16; DIFT ensemble left at 2."],
    }
    (ch5.REPORT_DIR / f"{key}_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def aggregate() -> None:
    ch5.ensure_dirs()
    for spec in ALL_SPECS:
        write_result_json(spec)

    projection_csv = ch5.CSV_DIR / "chapter5_supervised_projection_sweep_block16.csv"
    projection_fields = [
        "config_key",
        "checkpoint_path",
        "sweep_type",
        "temperature",
        "architecture",
        "hidden_dims",
        "output_dim",
        "dino_block",
        "feat_level",
        "dift_ensemble",
        "scene",
        "solver_mode",
        "mAA10",
        "inlier_ratio",
        "avg_matches",
        "median_pose_error",
        "median_rotation_error",
        "median_translation_error",
        "summary_json",
        "notes",
    ]
    with projection_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=projection_fields)
        writer.writeheader()
        for spec in PROJECTION_SPECS:
            for scene in ch5.SCENES:
                for mode in ["calibrated", "shared_focal", "varying_focal"]:
                    writer.writerow(
                        {
                            **row_for(spec, scene, mode),
                            "sweep_type": spec.get("sweep_type", ""),
                            "temperature": spec.get("temperature", ""),
                            "architecture": spec.get("architecture", ""),
                            "hidden_dims": spec.get("hidden_dims", ""),
                            "output_dim": spec.get("output_dim", ""),
                        }
                    )

    lora_csv = ch5.CSV_DIR / "chapter5_supervised_lora_comparison_block16.csv"
    lora_fields = [
        "config_key",
        "method_name",
        "checkpoint_path",
        "lora_rank",
        "lora_alpha",
        "adapted_blocks",
        "target_modules",
        "uses_dift",
        "uses_projection",
        "joint_training",
        "dino_block",
        "feat_level",
        "dift_ensemble",
        "scene",
        "solver_mode",
        "mAA10",
        "inlier_ratio",
        "avg_matches",
        "median_pose_error",
        "median_rotation_error",
        "median_translation_error",
        "summary_json",
        "notes",
    ]
    with lora_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=lora_fields)
        writer.writeheader()
        for spec in LORA_SPECS:
            for scene in ch5.SCENES:
                for mode in ["calibrated", "shared_focal", "varying_focal"]:
                    writer.writerow(
                        {
                            **row_for(spec, scene, mode),
                            "method_name": spec.get("method_name", ""),
                            "lora_rank": 4,
                            "lora_alpha": 8.0,
                            "adapted_blocks": "0-16",
                            "target_modules": "attn.qkv",
                            "uses_dift": spec.get("uses_dift", False),
                            "uses_projection": spec.get("uses_projection", False),
                            "joint_training": spec.get("joint_training", False),
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    best_proj = max(
        ((spec, avg_calibrated(spec["config_key"])) for spec in PROJECTION_SPECS if avg_calibrated(spec["config_key"]) is not None),
        key=lambda item: item[1],
        default=(None, None),
    )[0]
    if best_proj:
        summary_rows.append({"method_name": "best projection head block16", **best_proj})
    for spec in LORA_SPECS:
        summary_rows.append(spec)

    summary_csv = ch5.CSV_DIR / "chapter5_supervised_summary_block16.csv"
    summary_fields = [
        "method_name",
        "config_key",
        "calibrated_avg",
        "shared_focal_avg",
        "varying_focal_avg",
        "avg_inlier_ratio",
        "avg_matches",
        "checkpoint_path",
        "notes",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for spec in summary_rows:
            writer.writerow(
                {
                    "method_name": spec.get("method_name", "best projection head block16"),
                    "config_key": spec["config_key"],
                    "calibrated_avg": average_for_mode(spec["config_key"], "calibrated"),
                    "shared_focal_avg": average_for_mode(spec["config_key"], "shared_focal"),
                    "varying_focal_avg": average_for_mode(spec["config_key"], "varying_focal"),
                    "avg_inlier_ratio": average_field(spec["config_key"], "mean_inlier_ratio"),
                    "avg_matches": average_matches(spec["config_key"]),
                    "checkpoint_path": spec.get("checkpoint", ""),
                    "notes": "DINO block16 rerun; DIFT ensemble remains 2",
                }
            )

    write_markdown_report()


def markdown_table(specs: list[dict[str, Any]]) -> list[str]:
    lines = ["| Config | Calibrated avg | Sacre | Reichstag | St Peters |", "| --- | ---: | ---: | ---: | ---: |"]
    for spec in specs:
        key = spec["config_key"]
        values = []
        for scene in ch5.SCENES:
            metrics = ch5.target_metrics(key, scene, "calibrated")
            values.append("" if not metrics else f"{float(metrics['mAA@10']):.3f}")
        lines.append(f"| `{key}` | {average_for_mode(key, 'calibrated')} | {values[0]} | {values[1]} | {values[2]} |")
    return lines


def write_markdown_report() -> None:
    best_proj = max(
        ((spec, avg_calibrated(spec["config_key"])) for spec in PROJECTION_SPECS if avg_calibrated(spec["config_key"]) is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    best_lora = max(
        ((spec, avg_calibrated(spec["config_key"])) for spec in LORA_SPECS if avg_calibrated(spec["config_key"]) is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    lines = [
        "# Chapter 5 Supervised Adaptation Block-16 Rerun",
        "",
        f"Generated: `{ch5.now()}`",
        "",
        "This report reruns the successful block-12 Chapter 5 projection-head and LoRA evaluations with DINOv3 block 16 (`feat_level=-8`). DIFT remains `t=0`, `up_ft_index=2`, ensemble 2.",
        "",
        "## Status",
        "",
        f"- Calibrated complete: `{all_calibrated_done()}`",
        f"- Best projection: `{best_proj[0]['config_key'] if best_proj[0] else ''}` `{'' if best_proj[1] is None else f'{best_proj[1]:.6f}'}`",
        f"- Best LoRA: `{best_lora[0]['config_key'] if best_lora[0] else ''}` `{'' if best_lora[1] is None else f'{best_lora[1]:.6f}'}`",
        "",
        "## Projection Results",
        "",
        *markdown_table(PROJECTION_SPECS),
        "",
        "## LoRA Results",
        "",
        *markdown_table(LORA_SPECS),
        "",
        "## Output Files",
        "",
        "- `output_v2/csv/chapter5_supervised_projection_sweep_block16.csv`",
        "- `output_v2/csv/chapter5_supervised_lora_comparison_block16.csv`",
        "- `output_v2/csv/chapter5_supervised_summary_block16.csv`",
        "",
        "## Warning",
        "",
        "These are block-16 reruns for comparison. They do not replace the block-12 Chapter 5 selected-protocol results.",
    ]
    out = ch5.ROOT / "output_v2" / "reports" / "chapter5_supervised_block16_eval_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def watch_and_finalize(interval_seconds: int, max_wait_hours: float, gpu_id: str | None) -> None:
    start = time.time()
    max_wait = max_wait_hours * 3600.0
    while True:
        aggregate()
        if all_calibrated_done():
            run_best_modes()
            aggregate()
            return
        if time.time() - start > max_wait:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--gpu", required=True)
    run_parser.add_argument("--shard", type=int, required=True)
    run_parser.add_argument("--num-shards", type=int, required=True)

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--watch", action="store_true")
    final_parser.add_argument("--interval-seconds", type=int, default=600)
    final_parser.add_argument("--max-wait-hours", type=float, default=8.0)
    final_parser.add_argument("--gpu", default=None, help="Unused; retained for log symmetry.")

    args = parser.parse_args()
    if args.mode == "run-shard":
        run_shard(args.gpu, args.shard, args.num_shards)
    elif args.mode == "finalize":
        if args.watch:
            watch_and_finalize(args.interval_seconds, args.max_wait_hours, args.gpu)
        else:
            run_best_modes()
            aggregate()


if __name__ == "__main__":
    main()
