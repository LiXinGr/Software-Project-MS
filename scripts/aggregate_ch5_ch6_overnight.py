#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]
MODES = ["calibrated", "shared_focal", "varying_focal"]
RESULTS_ROOT = ROOT / "output_v2" / "results_v2"
BENCH_ROOT = ROOT / "output_v2" / "benchmarks_v2"
MATCHES_ROOT = ROOT / "output_v2" / "matches_v2"
TIMING_ROOT = ROOT / "output_v2" / "timing"
CSV_ROOT = ROOT / "output_v2" / "csv"
REPORT_ROOT = ROOT / "output_v2" / "reports"
LOG_ROOT = ROOT / "output_v2" / "logs" / "ch5_ch6_overnight"
STATUS_DIR = LOG_ROOT / "status"
FAILURES_TSV = LOG_ROOT / "failures.tsv"

NEW_TRAIN_KEY = "ch5_b16_train_proj_temp005_h1024_d256"
NEW_EVAL_KEY = "ch5_b16_eval_proj_temp005_h1024_d256_sp_mnn_mp2048"
NEW_PROJ_CKPT = ROOT / "experiments" / NEW_TRAIN_KEY / "best.pt"

TARGET_SOLVERS = {
    "calibrated": "3p_ours_shift_scale+12",
    "shared_focal": "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}


@dataclass(frozen=True)
class Method:
    method: str
    config_key: str
    matcher_checkpoint: str
    projection_checkpoint: str
    descriptor_setup: str
    final_descriptor_match: str
    notes: str
    modes: tuple[str, ...] = tuple(MODES)


MAIN_METHODS = [
    Method(
        "Projection head + MNN",
        NEW_EVAL_KEY,
        "MNN",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "DINOv3 block16 feat_level=-8 + DIFT t0 up2 ens2 + h1024 tau005 projection",
        "yes",
        "Descriptor-only baseline from Part A.",
    ),
    Method(
        "Zero-shot LightGlue",
        "ch6_zeroshot_lg_proj_h1024_temp005_ft010_sp_mnn_mp2048",
        "pretrained SuperPoint+LightGlue",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "final h1024 tau005 projected descriptor",
        "yes",
        "No matcher fine-tuning.",
    ),
    Method(
        "Warm-start LightGlue fine-tuning",
        "ch6_warmstart_lg_full_v1_proj_h1024_temp005_ft010_sp_mnn_mp2048",
        "external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "final h1024 tau005 projected descriptor",
        "descriptor eval yes; matcher trained on older descriptor",
        "Historical matcher checkpoint trained with older projection, evaluated here with final descriptor if compatible.",
    ),
    Method(
        "From-scratch LightGlue",
        "ch6_scratch_stage2_lg_v1_proj_h1024_temp005_ft010_sp_mnn_mp2048",
        "external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "final h1024 tau005 projected descriptor",
        "descriptor eval yes; matcher trained on older descriptor",
        "Stage1 homography then Stage2 60-scene MegaDepth checkpoint.",
    ),
    Method(
        "Expanded MegaDepth LightGlue",
        "ch6_expanded151_lg_proj_h1024_temp005_ft010_sp_mnn_mp2048",
        "external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "final h1024 tau005 projected descriptor",
        "descriptor eval yes; matcher trained on older descriptor",
        "Expected strongest learned-matcher checkpoint, but matcher training used older descriptor.",
    ),
    Method(
        "Joint descriptor-matcher optimization",
        "ch6_joint_unfrozen_proj60_proj_h1024_temp005_ft010_sp_mnn_mp2048",
        "external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar",
        f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "final h1024 tau005 projected descriptor if compatible",
        "diagnostic",
        "May be incompatible or may expect its native projection head.",
    ),
    Method(
        "SuperPoint+LightGlue",
        "superpoint_lg_mp2048",
        "pretrained SuperPoint+LightGlue",
        "n/a",
        "SuperPoint descriptors",
        "fixed baseline",
        "Existing final output_v2 baseline; not regenerated.",
    ),
    Method(
        "RoMa",
        "roma_outdoor_mp2048",
        "RoMa outdoor",
        "n/a",
        "RoMa dense matcher",
        "fixed baseline",
        "Existing final output_v2 baseline; not regenerated.",
    ),
    Method(
        "RoMaV2",
        "romav2_precise_mp2048",
        "RoMaV2 precise",
        "n/a",
        "RoMaV2 dense matcher",
        "fixed baseline",
        "Existing final output_v2 baseline; not regenerated.",
    ),
]


COMPARE_CH5_KEYS = [
    NEW_EVAL_KEY,
    "ch5_b16_eval_proj_temp005_h512_d256_sp_mnn_mp2048",
    "ch5_b16_eval_proj_wide_h1024_d256_sp_mnn_mp2048",
    "ch5_b16_eval_proj_temp003_h512_d256_sp_mnn_mp2048",
    "ch5_b16_eval_proj_temp007_h512_d256_sp_mnn_mp2048",
    "ch5_b16_eval_proj_deep_h512x512_d256_sp_mnn_mp2048",
    "ch5_b16_eval_proj_wide_h1024_d128_sp_mnn_mp2048",
    "ch5_b16_eval_proj_wide_h1024_d512_sp_mnn_mp2048",
]

CH5_LABELS = {
    NEW_EVAL_KEY: "new h1024 tau005",
    "ch5_b16_eval_proj_temp005_h512_d256_sp_mnn_mp2048": "h512 tau005",
    "ch5_b16_eval_proj_wide_h1024_d256_sp_mnn_mp2048": "h1024 tau007",
    "ch5_b16_eval_proj_temp003_h512_d256_sp_mnn_mp2048": "h512 tau003",
    "ch5_b16_eval_proj_temp007_h512_d256_sp_mnn_mp2048": "h512 tau007",
    "ch5_b16_eval_proj_deep_h512x512_d256_sp_mnn_mp2048": "deep h512x512 tau007",
    "ch5_b16_eval_proj_wide_h1024_d128_sp_mnn_mp2048": "h1024 d128 tau007",
    "ch5_b16_eval_proj_wide_h1024_d512_sp_mnn_mp2048": "h1024 d512 tau007",
}


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None, ndigits: int = 6) -> str:
    return "" if value is None else f"{value:.{ndigits}f}"


def first_summary(config_key: str, scene: str, mode: str) -> Path | None:
    pattern = RESULTS_ROOT / config_key / scene / f"{mode}-{config_key}_{scene}*_summary.json"
    matches = sorted(glob.glob(str(pattern)))
    return Path(matches[-1]) if matches else None


def pick_experiment(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    target = TARGET_SOLVERS.get(mode)
    experiments = summary.get("experiments", [])
    for item in experiments:
        if item.get("solver") == target:
            return item
    if not experiments:
        return {}
    return max(experiments, key=lambda x: float(x.get("mAA@10", float("-inf"))))


def avg_matches(config_key: str, scene: str) -> float | None:
    timing_path = TIMING_ROOT / f"{config_key}_{scene}_timing.json"
    if timing_path.exists():
        try:
            payload = read_json(timing_path)
            vals = [float(x["num_matches"]) for x in payload.get("pair_timings", []) if "num_matches" in x]
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
    bench = BENCH_ROOT / f"{config_key}_{scene}.h5"
    if bench.exists():
        try:
            import h5py  # type: ignore

            counts = []
            with h5py.File(bench, "r") as handle:
                for key in handle.keys():
                    obj = handle[key]
                    if hasattr(obj, "shape") and len(obj.shape) >= 1 and key.startswith("corr_"):
                        counts.append(int(obj.shape[0]))
            if counts:
                return sum(counts) / len(counts)
        except Exception:
            pass
    match_dir = MATCHES_ROOT / config_key / scene
    if match_dir.exists():
        try:
            import numpy as np  # type: ignore

            counts = []
            for path in match_dir.glob("*.npz"):
                with np.load(path) as data:
                    if "mkpts0" in data:
                        counts.append(int(data["mkpts0"].shape[0]))
            if counts:
                return sum(counts) / len(counts)
        except Exception:
            pass
    return None


def metric_row(method: str, config_key: str, scene: str, mode: str) -> dict[str, Any]:
    summary_path = first_summary(config_key, scene, mode)
    matches = avg_matches(config_key, scene)
    if not summary_path:
        return {
            "method": method,
            "config_key": config_key,
            "scene": scene,
            "solver_mode": mode,
            "mAA@10": "",
            "inlier_ratio": "",
            "avg_matches": fmt(matches),
            "median_pose_error": "",
            "summary_json_path": "",
        }
    summary = read_json(summary_path)
    exp = pick_experiment(summary, mode)
    return {
        "method": method,
        "config_key": config_key,
        "scene": scene,
        "solver_mode": mode,
        "mAA@10": exp.get("mAA@10", ""),
        "inlier_ratio": exp.get("mean_inlier_ratio", ""),
        "avg_matches": fmt(matches),
        "median_pose_error": exp.get("median_pose_error", ""),
        "summary_json_path": rel(summary_path),
    }


def rows_for_key(method: str, config_key: str, modes: tuple[str, ...] = tuple(MODES)) -> list[dict[str, Any]]:
    return [metric_row(method, config_key, scene, mode) for scene in SCENES for mode in modes]


def avg_for_key(config_key: str, mode: str, field: str = "mAA@10") -> float | None:
    vals = []
    for scene in SCENES:
        row = metric_row("", config_key, scene, mode)
        value = row.get(field)
        if value == "" or value is None:
            return None
        vals.append(float(value))
    return mean(vals)


def avg_matches_for_key(config_key: str) -> float | None:
    return mean([avg_matches(config_key, scene) for scene in SCENES])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def chapter5_rows() -> list[dict[str, Any]]:
    rows = []
    for key in COMPARE_CH5_KEYS:
        for scene in SCENES:
            for mode in MODES:
                row = metric_row(CH5_LABELS.get(key, key), key, scene, mode)
                row.update(
                    {
                        "architecture": architecture_for_key(key),
                        "projection_checkpoint": projection_checkpoint_for_key(key),
                        "selected_candidate": "yes" if key == NEW_EVAL_KEY else "no",
                    }
                )
                rows.append(row)
        rows.append(
            {
                "method": CH5_LABELS.get(key, key),
                "config_key": key,
                "scene": "AVERAGE",
                "solver_mode": "calibrated",
                "mAA@10": fmt(avg_for_key(key, "calibrated")),
                "inlier_ratio": fmt(avg_for_key(key, "calibrated", "inlier_ratio")),
                "avg_matches": fmt(avg_matches_for_key(key)),
                "median_pose_error": "",
                "summary_json_path": "",
                "architecture": architecture_for_key(key),
                "projection_checkpoint": projection_checkpoint_for_key(key),
                "selected_candidate": "yes" if key == NEW_EVAL_KEY else "no",
            }
        )
    return rows


def architecture_for_key(key: str) -> str:
    if key == NEW_EVAL_KEY:
        return "1664 -> 1024 -> 256, tau=0.05"
    if "temp005_h512" in key:
        return "1664 -> 512 -> 256, tau=0.05"
    if "wide_h1024_d256" in key:
        return "1664 -> 1024 -> 256, tau=0.07"
    if "temp003" in key:
        return "1664 -> 512 -> 256, tau=0.03"
    if "temp007" in key:
        return "1664 -> 512 -> 256, tau=0.07"
    if "deep" in key:
        return "1664 -> 512 -> 512 -> 256, tau=0.07"
    if "d128" in key:
        return "1664 -> 1024 -> 128, tau=0.07"
    if "d512" in key:
        return "1664 -> 1024 -> 512, tau=0.07"
    return ""


def projection_checkpoint_for_key(key: str) -> str:
    mapping = {
        NEW_EVAL_KEY: f"experiments/{NEW_TRAIN_KEY}/best.pt",
        "ch5_b16_eval_proj_temp005_h512_d256_sp_mnn_mp2048": "experiments/phase2_projection_temp005/best.pt",
        "ch5_b16_eval_proj_wide_h1024_d256_sp_mnn_mp2048": "experiments/phase2_projection_wide/best.pt",
        "ch5_b16_eval_proj_temp003_h512_d256_sp_mnn_mp2048": "experiments/phase2_projection_temp003/best.pt",
        "ch5_b16_eval_proj_temp007_h512_d256_sp_mnn_mp2048": "experiments/phase2_projection_v1/best.pt",
        "ch5_b16_eval_proj_deep_h512x512_d256_sp_mnn_mp2048": "experiments/phase2_projection_deep/best.pt",
        "ch5_b16_eval_proj_wide_h1024_d128_sp_mnn_mp2048": "output_v2/checkpoints/chapter5_supervised/ch5_train_proj_wide_t007_h1024_d128/best.pt",
        "ch5_b16_eval_proj_wide_h1024_d512_sp_mnn_mp2048": "output_v2/checkpoints/chapter5_supervised/ch5_train_proj_wide_t007_h1024_d512/best.pt",
    }
    return mapping.get(key, "")


def main_summary_rows() -> list[dict[str, Any]]:
    rows = []
    for method in MAIN_METHODS:
        rows.append(
            {
                "method": method.method,
                "config_key": method.config_key,
                "calibrated_avg": fmt(avg_for_key(method.config_key, "calibrated")),
                "shared_focal_avg": fmt(avg_for_key(method.config_key, "shared_focal")),
                "varying_focal_avg": fmt(avg_for_key(method.config_key, "varying_focal")),
                "calibrated_inliers_avg": fmt(avg_for_key(method.config_key, "calibrated", "inlier_ratio")),
                "avg_matches": fmt(avg_matches_for_key(method.config_key)),
                "scenes": "; ".join(SCENES),
                "descriptor_setup": method.descriptor_setup,
                "matcher_checkpoint": method.matcher_checkpoint,
                "projection_checkpoint": method.projection_checkpoint,
                "final_descriptor_match": method.final_descriptor_match,
                "notes": method.notes,
            }
        )
    return rows


def per_scene_rows() -> list[dict[str, Any]]:
    rows = []
    for method in MAIN_METHODS:
        rows.extend(rows_for_key(method.method, method.config_key, method.modes))
    return rows


def threshold_keys(family: str) -> list[tuple[str, str]]:
    if family == "zeroshot":
        return [
            ("0.00", "ch6_zeroshot_lg_proj_h1024_temp005_ft000_sp_mnn_mp2048"),
            ("0.02", "ch6_zeroshot_lg_proj_h1024_temp005_ft002_sp_mnn_mp2048"),
            ("0.05", "ch6_zeroshot_lg_proj_h1024_temp005_ft005_sp_mnn_mp2048"),
            ("0.10", "ch6_zeroshot_lg_proj_h1024_temp005_ft010_sp_mnn_mp2048"),
            ("0.15", "ch6_zeroshot_lg_proj_h1024_temp005_ft015_sp_mnn_mp2048"),
            ("0.20", "ch6_zeroshot_lg_proj_h1024_temp005_ft020_sp_mnn_mp2048"),
        ]
    return [
        ("0.05", "ch6_expanded151_lg_proj_h1024_temp005_ft005_sp_mnn_mp2048"),
        ("0.10", "ch6_expanded151_lg_proj_h1024_temp005_ft010_sp_mnn_mp2048"),
        ("0.15", "ch6_expanded151_lg_proj_h1024_temp005_ft015_sp_mnn_mp2048"),
        ("0.20", "ch6_expanded151_lg_proj_h1024_temp005_ft020_sp_mnn_mp2048"),
    ]


def threshold_rows(family: str) -> list[dict[str, Any]]:
    rows = []
    for threshold, key in threshold_keys(family):
        vals = []
        inliers = []
        matches = []
        for scene in SCENES:
            row = metric_row(family, key, scene, "calibrated")
            row["threshold"] = threshold
            rows.append(row)
            if row["mAA@10"] != "":
                vals.append(float(row["mAA@10"]))
            if row["inlier_ratio"] != "":
                inliers.append(float(row["inlier_ratio"]))
            if row["avg_matches"] != "":
                matches.append(float(row["avg_matches"]))
        rows.append(
            {
                "method": family,
                "config_key": key,
                "threshold": threshold,
                "scene": "AVERAGE",
                "solver_mode": "calibrated",
                "mAA@10": fmt(mean(vals)),
                "inlier_ratio": fmt(mean(inliers)),
                "avg_matches": fmt(mean(matches)),
                "median_pose_error": "",
                "summary_json_path": "",
            }
        )
    return rows


def best_threshold_key(family: str) -> str:
    best_key = ""
    best_score = float("-inf")
    best_threshold = ""
    for threshold, key in threshold_keys(family):
        avg = avg_for_key(key, "calibrated")
        if avg is None:
            return ""
        if avg > best_score:
            best_score = avg
            best_key = key
            best_threshold = threshold
    if best_threshold == "0.10":
        return ""
    return best_key


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.exists():
        return []
    with FAILURES_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def diagnostics_rows() -> list[dict[str, Any]]:
    rows = []
    diagnostic_configs = [
        ("Zero-shot noadapt", "ch6_zeroshot_lg_proj_h1024_temp005_noadapt_ft010_sp_mnn_mp2048", "disabled adaptivity diagnostic"),
        ("Joint descriptor-matcher optimization", "ch6_joint_unfrozen_proj60_proj_h1024_temp005_ft010_sp_mnn_mp2048", "joint/native projection diagnostic"),
    ]
    for label, key, note in diagnostic_configs:
        for scene in SCENES:
            row = metric_row(label, key, scene, "calibrated")
            row.update({"status": "evaluated" if row["summary_json_path"] else "missing_or_failed", "notes": note})
            rows.append(row)
    for failure in read_failures():
        rows.append(
            {
                "method": failure.get("stage", ""),
                "config_key": failure.get("config_key", ""),
                "scene": failure.get("scene", ""),
                "solver_mode": failure.get("solver_mode", ""),
                "mAA@10": "",
                "inlier_ratio": "",
                "avg_matches": "",
                "median_pose_error": "",
                "summary_json_path": "",
                "status": failure.get("status", ""),
                "notes": f"{failure.get('reason', '')}; log={failure.get('log_path', '')}; {failure.get('notes', '')}",
            }
        )
    for method in MAIN_METHODS:
        if "older descriptor" in method.notes or method.final_descriptor_match not in {"yes", "fixed baseline"}:
            rows.append(
                {
                    "method": method.method,
                    "config_key": method.config_key,
                    "scene": "all",
                    "solver_mode": "all",
                    "mAA@10": "",
                    "inlier_ratio": "",
                    "avg_matches": "",
                    "median_pose_error": "",
                    "summary_json_path": "",
                    "status": "warning",
                    "notes": method.notes,
                }
            )
    return rows


def status_time(name: str) -> int | None:
    path = STATUS_DIR / name
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def elapsed_minutes(start: int | None, end: int | None) -> str:
    if start is None or end is None or end < start:
        return ""
    return f"{(end - start) / 60.0:.1f}"


def all_summary_paths_for_prefixes(prefixes: list[str]) -> list[str]:
    paths = []
    for prefix in prefixes:
        for path in sorted(RESULTS_ROOT.glob(f"{prefix}/*/*summary.json")):
            paths.append(rel(path))
    return paths


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return out


def write_ch5_report(ch5_rows_data: list[dict[str, Any]]) -> None:
    new_cal = avg_for_key(NEW_EVAL_KEY, "calibrated")
    h512 = avg_for_key("ch5_b16_eval_proj_temp005_h512_d256_sp_mnn_mp2048", "calibrated")
    h1024_t007 = avg_for_key("ch5_b16_eval_proj_wide_h1024_d256_sp_mnn_mp2048", "calibrated")
    selected = "pending"
    if new_cal is not None:
        old_vals = [v for v in [h512, h1024_t007] if v is not None]
        selected = "yes" if old_vals and new_cal >= max(old_vals) else "no"
    train_start = status_time("train.started")
    train_done = status_time("train.done")
    eval_done = max([status_time(f"eval_shard_{i}.done") or 0 for i in range(16)] + [0]) or None
    lines = [
        "# Chapter 5 Projection Final Wide Temp005 Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Training Setup",
        "",
        f"- Experiment: `{NEW_TRAIN_KEY}`.",
        "- Architecture: `1664 -> 1024 -> 256`.",
        "- Temperature: `tau=0.05`.",
        "- Training scenes: `0080`, `0042`, `0380`, `0000`, `0366`, `0001`, `0005`, `0237`, `0011`, `0148`.",
        "- DINOv3/DIFT are frozen; only the projection head is trained.",
        f"- Checkpoint: `{rel(NEW_PROJ_CKPT)}`.",
        f"- Training time minutes: `{elapsed_minutes(train_start, train_done)}`.",
        f"- Evaluation time minutes from train completion to last shard: `{elapsed_minutes(train_done, eval_done)}`.",
        "",
        "## Evaluation Setup",
        "",
        "- SuperPoint keypoints, cosine MNN, 2048 correspondence cap.",
        "- Raw-image final protocol, Sampson threshold 2.0 px, reprojection threshold 16.0 px.",
        "- Solver modes: calibrated, shared_focal, varying_focal.",
        "",
        "## Per-Scene Calibrated Results",
        "",
    ]
    scene_rows = []
    for scene in SCENES:
        row = metric_row("new", NEW_EVAL_KEY, scene, "calibrated")
        scene_rows.append([scene, row["mAA@10"], row["inlier_ratio"], row["avg_matches"], row["summary_json_path"]])
    lines.extend(markdown_table(["scene", "mAA@10", "inlier ratio", "avg matches", "summary JSON"], scene_rows))
    lines += [
        "",
        "## All-Mode Averages",
        "",
    ]
    lines.extend(
        markdown_table(
            ["mode", "average mAA@10"],
            [[mode, fmt(avg_for_key(NEW_EVAL_KEY, mode))] for mode in MODES],
        )
    )
    lines += [
        "",
        "## Comparisons",
        "",
        f"- New h1024 tau005 calibrated avg: `{fmt(new_cal)}`.",
        f"- Existing h512 tau005 calibrated avg: `{fmt(h512)}`.",
        f"- Existing h1024 tau007 calibrated avg: `{fmt(h1024_t007)}`.",
        f"- Selected as final Chapter 5 projection head: `{selected}`.",
        "",
        "See `output_v2/csv/chapter5_projection_final_wide_temp005.csv` for full rows.",
    ]
    (REPORT_ROOT / "chapter5_projection_final_wide_temp005_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_ch6_method(summary_rows: list[dict[str, Any]]) -> str:
    candidates = []
    for row in summary_rows:
        if row["config_key"].startswith("ch6_") or row["config_key"] == NEW_EVAL_KEY:
            try:
                score = float(row["calibrated_avg"])
            except Exception:
                continue
            if row.get("final_descriptor_match", "").startswith("yes") or "descriptor eval yes" in row.get("final_descriptor_match", ""):
                candidates.append((score, row["method"], row["config_key"]))
    if not candidates:
        return "pending"
    score, method, key = max(candidates)
    return f"{method} (`{key}`), calibrated avg {score:.6f}"


def write_ch6_report(summary_rows: list[dict[str, Any]], per_scene: list[dict[str, Any]], zero_rows: list[dict[str, Any]], expanded_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> None:
    launch_start = status_time("launch.started")
    final_done = status_time("finalize.done") or int(time.time())
    gpu_csv = (STATUS_DIR / "gpus.csv").read_text(encoding="utf-8").strip() if (STATUS_DIR / "gpus.csv").exists() else ""
    lines = [
        "# Chapter 6 Learned Matcher Evaluation Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Executive Summary",
        "",
        f"- Selected Chapter 6 result: {selected_ch6_method(summary_rows)}.",
        f"- Projection checkpoint: `{rel(NEW_PROJ_CKPT)}`.",
        "- Historical learned matcher checkpoints are evaluated with the final descriptor when technically compatible; their training descriptor mismatch is recorded.",
        "",
        "## Exact Protocol",
        "",
        "- DINOv3 ViT-L/16 block 16, `feat_level=-8`.",
        "- DIFT Stable Diffusion v1.5, `t=0`, `up_ft_index=2`, ensemble 2.",
        "- Independent branch L2 normalization, equal fusion weights, h1024 tau005 projection, L2-normalized 256D output.",
        "- SuperPoint keypoints, 2048 correspondence cap, raw image coordinates.",
        "- RePoseD threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, 25 LO iterations.",
        "",
        "## Runtime",
        "",
        f"- GPU allocation: `{gpu_csv}`.",
        "- Estimate: 8-12 hours with all free GPUs; 18-30 hours on one GPU.",
        f"- Actual elapsed minutes so far: `{elapsed_minutes(launch_start, final_done)}`.",
        "",
        "## Main Table",
        "",
    ]
    lines.extend(
        markdown_table(
            ["method", "config", "calibrated", "shared", "varying", "inliers", "matches", "notes"],
            [
                [
                    r["method"],
                    f"`{r['config_key']}`",
                    r["calibrated_avg"],
                    r["shared_focal_avg"],
                    r["varying_focal_avg"],
                    r["calibrated_inliers_avg"],
                    r["avg_matches"],
                    r["notes"],
                ]
                for r in summary_rows
            ],
        )
    )
    lines += ["", "## Per-Scene Calibrated Table", ""]
    cal_rows = [r for r in per_scene if r["solver_mode"] == "calibrated"]
    lines.extend(markdown_table(["method", "scene", "mAA@10", "inliers", "matches", "summary"], [[r["method"], r["scene"], r["mAA@10"], r["inlier_ratio"], r["avg_matches"], r["summary_json_path"]] for r in cal_rows]))
    lines += ["", "## Zero-Shot Threshold Sweep", ""]
    lines.extend(markdown_table(["threshold", "scene", "mAA@10", "summary"], [[r.get("threshold", ""), r["scene"], r["mAA@10"], r["summary_json_path"]] for r in zero_rows]))
    lines += ["", "## Expanded Threshold Sweep", ""]
    lines.extend(markdown_table(["threshold", "scene", "mAA@10", "summary"], [[r.get("threshold", ""), r["scene"], r["mAA@10"], r["summary_json_path"]] for r in expanded_rows]))
    lines += ["", "## Diagnostics", ""]
    lines.extend(markdown_table(["method", "config", "scene", "status", "notes"], [[r.get("method", ""), r.get("config_key", ""), r.get("scene", ""), r.get("status", ""), r.get("notes", "")] for r in diag_rows]))
    paths = all_summary_paths_for_prefixes(["ch6_*", NEW_EVAL_KEY])
    missing = [r for r in per_scene if not r["summary_json_path"] and (r["config_key"].startswith("ch6_") or r["config_key"] == NEW_EVAL_KEY)]
    lines += [
        "",
        "## Warnings",
        "",
        "- Warm-start, scratch, expanded, and joint LightGlue checkpoints were trained historically with an older projection setup.",
        "- The reported Chapter 6 final-descriptor rows use the new h1024 tau005 projection checkpoint at evaluation time when compatible.",
        "",
        "## Missing Or Failed Evaluations",
        "",
    ]
    if missing:
        lines.extend(markdown_table(["config", "scene", "mode"], [[r["config_key"], r["scene"], r["solver_mode"]] for r in missing]))
    else:
        lines.append("No missing main-method rows detected.")
    lines += ["", "## Output JSON Summary Paths", ""]
    lines.extend([f"- `{p}`" for p in paths])
    (REPORT_ROOT / "chapter6_learned_matcher_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(summary_rows: list[dict[str, Any]]) -> None:
    ch5_selected = "pending"
    new_avg = avg_for_key(NEW_EVAL_KEY, "calibrated")
    if new_avg is not None:
        old_best = max([v for k in COMPARE_CH5_KEYS if k != NEW_EVAL_KEY for v in [avg_for_key(k, "calibrated")] if v is not None], default=None)
        ch5_selected = "yes" if old_best is not None and new_avg >= old_best else "no"
    failures = read_failures()
    lines = [
        "# Chapter 5/6 Overnight Final Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## What Was Run",
        "",
        f"- Trained projection head `{NEW_TRAIN_KEY}` if not already complete.",
        f"- Evaluated `{NEW_EVAL_KEY}` with MNN.",
        "- Evaluated Chapter 6 zero-shot, threshold sweeps, noadapt diagnostic, warm-start, scratch, expanded 151-scene, and joint checkpoints where compatible.",
        "- Collected fixed SuperPoint+LightGlue, RoMa, and RoMaV2 baselines from existing output_v2 summaries.",
        "",
        "## Success / Failure",
        "",
        f"- New checkpoint exists: `{'yes' if NEW_PROJ_CKPT.exists() else 'no'}`.",
        f"- Failure records: `{len(failures)}`.",
        "",
        "## Chapter 5 Selection",
        "",
        f"- New h1024 tau005 calibrated average: `{fmt(new_avg)}`.",
        f"- Selected as Chapter 5 projection head: `{ch5_selected}`.",
        "",
        "## Chapter 6 Selection",
        "",
        f"- Selected matcher: {selected_ch6_method(summary_rows)}.",
        "",
        "## Still Needs Work",
        "",
    ]
    if failures:
        lines.append("- Inspect `output_v2/csv/chapter6_diagnostics.csv` and worker logs for failures or incompatibilities.")
    else:
        lines.append("- No failure rows recorded by the overnight launcher.")
    (REPORT_ROOT / "ch5_ch6_overnight_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure() -> None:
    projection_avg = avg_for_key(NEW_EVAL_KEY, "calibrated")
    if projection_avg is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        labels = ["Training-free fusion", "Projection head", "Best LoRA variant", "SuperPoint+LightGlue", "RoMa", "RoMaV2"]
        values = [76.2, projection_avg, 79.2, 84.0, 87.5, 86.3]
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
        bars = ax.bar(labels, values, color="#3b82f6")
        ax.set_ylabel("Calibrated mAA@10")
        ax.set_ylim(0, max(values) + 8)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out_png = ROOT / "Pictures" / "fig_ch5_supervised_adaptation_comparison.png"
        out_pdf = ROOT / "Pictures" / "fig_ch5_supervised_adaptation_comparison.pdf"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_pdf)
        plt.close(fig)
    except Exception as exc:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        (LOG_ROOT / "figure_error.txt").write_text(str(exc) + "\n", encoding="utf-8")


def write_outputs(final: bool = False) -> None:
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    ch5 = chapter5_rows()
    summary = main_summary_rows()
    per_scene = per_scene_rows()
    zero = threshold_rows("zeroshot")
    expanded = threshold_rows("expanded")
    diagnostics = diagnostics_rows()

    write_csv(
        CSV_ROOT / "chapter5_projection_final_wide_temp005.csv",
        ch5,
        ["method", "config_key", "architecture", "projection_checkpoint", "selected_candidate", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"],
    )
    write_csv(
        CSV_ROOT / "chapter6_learned_matcher_summary.csv",
        summary,
        ["method", "config_key", "calibrated_avg", "shared_focal_avg", "varying_focal_avg", "calibrated_inliers_avg", "avg_matches", "scenes", "descriptor_setup", "matcher_checkpoint", "projection_checkpoint", "final_descriptor_match", "notes"],
    )
    write_csv(
        CSV_ROOT / "chapter6_learned_matcher_per_scene.csv",
        per_scene,
        ["method", "config_key", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"],
    )
    write_csv(
        CSV_ROOT / "chapter6_zeroshot_threshold_sweep.csv",
        zero,
        ["method", "config_key", "threshold", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"],
    )
    write_csv(
        CSV_ROOT / "chapter6_expanded_threshold_sweep.csv",
        expanded,
        ["method", "config_key", "threshold", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"],
    )
    write_csv(
        CSV_ROOT / "chapter6_diagnostics.csv",
        diagnostics,
        ["method", "config_key", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path", "status", "notes"],
    )

    write_ch5_report(ch5)
    write_ch6_report(summary, per_scene, zero, expanded, diagnostics)
    write_final_report(summary)
    write_figure()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Chapter 5/6 overnight outputs.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--best-threshold-family", choices=["zeroshot", "expanded"], default=None)
    args = parser.parse_args()

    if args.best_threshold_family:
        print(best_threshold_key(args.best_threshold_family))
        return
    if args.write or args.final:
        write_outputs(final=args.final)


if __name__ == "__main__":
    main()
