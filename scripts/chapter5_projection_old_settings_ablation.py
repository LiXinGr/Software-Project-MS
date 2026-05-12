#!/usr/bin/env python3
"""Diagnostic projection-head reruns with old DINO/DIFT settings.

This is intentionally kept out of the main Chapter 5 aggregate tables.  It
reuses the final raw-image match/pack/eval path, but changes only the source
descriptor settings suspected to explain the older wide-projection score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import chapter5_supervised as ch5


DIAG_SPECS: list[dict[str, Any]] = [
    {
        "config_key": "ch5_diag_proj_wide_b16_ens2_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_wide/best.pt",
        "architecture": "1664 -> 1024 -> 256",
        "feat_level": -8,
        "dino_block": 16,
        "ensemble_size": 2,
        "diagnostic": "old DINO layer only",
    },
    {
        "config_key": "ch5_diag_proj_wide_b12_ens8_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_wide/best.pt",
        "architecture": "1664 -> 1024 -> 256",
        "feat_level": -12,
        "dino_block": 12,
        "ensemble_size": 8,
        "diagnostic": "old DIFT ensemble only",
    },
    {
        "config_key": "ch5_diag_proj_wide_b16_ens8_sp_mnn_mp2048",
        "checkpoint": "experiments/phase2_projection_wide/best.pt",
        "architecture": "1664 -> 1024 -> 256",
        "feat_level": -8,
        "dino_block": 16,
        "ensemble_size": 8,
        "diagnostic": "old DINO layer and old DIFT ensemble",
    },
]


def write_diag_config(spec: dict[str, Any], scene: str, command: list[str], gpu_id: str) -> None:
    payload = {
        "config_key": spec["config_key"],
        "checkpoint_path": spec["checkpoint"],
        "command_used": command,
        "timestamp": ch5.now(),
        "gpu_id": gpu_id,
        "scene": scene,
        "diagnostic_only": True,
        "diagnostic_question": "Does the older wide-projection score come from DINO block16 and/or DIFT ensemble8?",
        "dino_eval": {
            "backbone": "vit_large_patch16_dinov3.lvd1689m",
            "feat_level": spec["feat_level"],
            "block": spec["dino_block"],
            "internal_long_edge": 1120,
            "divisibility": 16,
        },
        "dift_eval": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "timestep": 0,
            "up_ft_index": 2,
            "ensemble_size": spec["ensemble_size"],
            "internal_resolution": [768, 768],
        },
        "fusion": {
            "normalization": "independent branch L2 normalization",
            "dift_weight": 0.5,
            "dinov3_weight": 0.5,
            "dimension": 1664,
        },
        "projection_architecture": spec["architecture"],
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
        "warning": "Diagnostic rerun only; not part of Chapter 5 main selected-protocol tables.",
    }
    out = ch5.REPORT_DIR / f"{spec['config_key']}_{scene}_config.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_diag_projection(spec: dict[str, Any], scene: str, gpu_id: str) -> None:
    config_key = spec["config_key"]
    log_path = ch5.LOG_DIR / f"{config_key}_{scene}.log"
    ckpt = ch5.ROOT / spec["checkpoint"]
    if not ckpt.exists():
        ch5.append_failure(
            "projection_old_settings_diagnostic",
            config_key,
            "checkpoint_missing",
            f"Missing checkpoint {ckpt}",
            "locate checkpoint",
            log_path,
        )
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
            str(spec["feat_level"]),
            "--img_size",
            "768",
            "768",
            "--t",
            "0",
            "--up_ft_index",
            "2",
            "--ensemble_size",
            str(spec["ensemble_size"]),
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
        write_diag_config(spec, scene, command, gpu_id)
        ch5.run_cmd(command, log_path, gpu_id=gpu_id)

    ch5.run_pack_and_eval(config_key, scene, ["calibrated"], gpu_id, log_path)


def row_for(spec: dict[str, Any], scene: str) -> dict[str, Any]:
    base = {
        "config_key": spec["config_key"],
        "checkpoint_path": spec["checkpoint"],
        "diagnostic": spec["diagnostic"],
        "dino_block": spec["dino_block"],
        "feat_level": spec["feat_level"],
        "dift_ensemble": spec["ensemble_size"],
        "scene": scene,
        "solver_mode": "calibrated",
    }
    row = ch5.row_from_metrics(spec["config_key"], scene, "calibrated")
    return {**base, **row}


def write_outputs() -> None:
    ch5.ensure_dirs()
    csv_path = ch5.CSV_DIR / "chapter5_projection_old_settings_ablation.csv"
    fields = [
        "config_key",
        "checkpoint_path",
        "diagnostic",
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
    rows = [row_for(spec, scene) for spec in DIAG_SPECS for scene in ch5.SCENES]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result_payload = {
        "generated": ch5.now(),
        "diagnostic_only": True,
        "baseline_final_protocol": "ch5_eval_proj_wide_h1024_d256_sp_mnn_mp2048",
        "old_result_reference": "projection_wide_sp_mnn_mp2048",
        "rows": rows,
        "averages": {},
    }
    for spec in DIAG_SPECS:
        values = []
        for scene in ch5.SCENES:
            metrics = ch5.target_metrics(spec["config_key"], scene, "calibrated")
            if metrics and metrics.get("mAA@10") != "":
                values.append(float(metrics["mAA@10"]))
        result_payload["averages"][spec["config_key"]] = (sum(values) / len(values)) if values else None

    json_path = ch5.REPORT_DIR / "chapter5_projection_old_settings_ablation_result.json"
    json_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")

    md_path = ch5.ROOT / "output_v2" / "reports" / "chapter5_projection_old_settings_ablation.md"
    lines = [
        "# Chapter 5 Projection Old-Settings Diagnostic",
        "",
        f"Generated: `{result_payload['generated']}`",
        "",
        "Diagnostic only; not part of the main Chapter 5 selected-protocol tables.",
        "",
        "| Config | Avg calibrated mAA@10 | DINO block | DIFT ensemble | Purpose |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for spec in DIAG_SPECS:
        avg = result_payload["averages"][spec["config_key"]]
        avg_text = "" if avg is None else f"{avg:.6f}"
        lines.append(
            f"| `{spec['config_key']}` | {avg_text} | {spec['dino_block']} | {spec['ensemble_size']} | {spec['diagnostic']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Visible GPU id to use.")
    parser.add_argument("--scene", choices=ch5.SCENES, action="append", help="Optional scene subset; repeatable.")
    parser.add_argument("--finalize-only", action="store_true", help="Only rewrite diagnostic CSV/JSON/Markdown.")
    args = parser.parse_args()

    scenes = args.scene or ch5.SCENES
    if not args.finalize_only:
        for spec in DIAG_SPECS:
            for scene in scenes:
                run_diag_projection(spec, scene, args.gpu)
    write_outputs()


if __name__ == "__main__":
    main()
