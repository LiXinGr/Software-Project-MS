#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
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
LOG_ROOT = ROOT / "output_v2" / "logs" / "ch6_oldproj_eval"
FAILURES_TSV = LOG_ROOT / "failures.tsv"

OLD_PROJ_CKPT = ROOT / "experiments" / "phase2_projection_wide" / "best.pt"
OLD_PROJ_CONFIG = ROOT / "experiments" / "phase2_projection_wide" / "config.json"
MANIFEST_PATH = REPORT_ROOT / "chapter6_oldproj_eval_manifest.json"
ENSEMBLE_SIZE_LABEL = os.environ.get("ENSEMBLE_SIZE", "8")
SKIP_MNN_BASELINE = os.environ.get("SKIP_MNN_BASELINE", "0") == "1"

TARGET_SOLVERS = {
    "calibrated": "3p_ours_shift_scale+12",
    "shared_focal": "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}

MNN_KEY = "ch6_input_oldproj_wide_t007_mnn_sp_mnn_mp2048"
NOADAPT_KEY = "ch6_oldproj_zeroshot_lg_noadapt_ft010_sp_mnn_mp2048"
JOINT_KEY = "ch6_oldproj_joint_unfrozen_proj60_ft010_sp_mnn_mp2048"

ZERO_KEYS = [
    ("0.00", "ch6_oldproj_zeroshot_lg_ft000_sp_mnn_mp2048"),
    ("0.02", "ch6_oldproj_zeroshot_lg_ft002_sp_mnn_mp2048"),
    ("0.05", "ch6_oldproj_zeroshot_lg_ft005_sp_mnn_mp2048"),
    ("0.10", "ch6_oldproj_zeroshot_lg_ft010_sp_mnn_mp2048"),
    ("0.15", "ch6_oldproj_zeroshot_lg_ft015_sp_mnn_mp2048"),
    ("0.20", "ch6_oldproj_zeroshot_lg_ft020_sp_mnn_mp2048"),
]
WARM_KEYS = [
    ("0.10", "ch6_oldproj_warmstart_lg_full_v1_ft010_sp_mnn_mp2048"),
    ("0.15", "ch6_oldproj_warmstart_lg_full_v1_ft015_sp_mnn_mp2048"),
]
SCRATCH_KEYS = [
    ("0.10", "ch6_oldproj_scratch_stage2_lg_v1_ft010_sp_mnn_mp2048"),
    ("0.15", "ch6_oldproj_scratch_stage2_lg_v1_ft015_sp_mnn_mp2048"),
]
EXPANDED_KEYS = [
    ("0.05", "ch6_oldproj_expanded151_lg_ft005_sp_mnn_mp2048"),
    ("0.10", "ch6_oldproj_expanded151_lg_ft010_sp_mnn_mp2048"),
    ("0.15", "ch6_oldproj_expanded151_lg_ft015_sp_mnn_mp2048"),
    ("0.20", "ch6_oldproj_expanded151_lg_ft020_sp_mnn_mp2048"),
]

FIXED_BASELINES = [
    ("SuperPoint+LightGlue", "superpoint_lg_mp2048", "pretrained SuperPoint+LightGlue", "SuperPoint descriptors"),
    ("RoMa", "roma_outdoor_mp2048", "RoMa outdoor", "RoMa dense matcher"),
    ("RoMaV2", "romav2_precise_mp2048", "RoMaV2 precise", "RoMaV2 dense matcher"),
]

MATCHER_CKPTS = {
    "warmstart": "external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar",
    "scratch": "external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar",
    "expanded151": "external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar",
    "joint": "external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar",
}

TRAINING_CONFIGS = [
    ROOT / "external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/config.yaml",
    ROOT / "external/glue-factory/outputs/training/stage2_dinov3_lg_v1/config.yaml",
    ROOT / "external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/config.yaml",
    ROOT / "external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/config.yaml",
]


@dataclass(frozen=True)
class MainMethod:
    method: str
    config_key: str
    matcher_checkpoint: str
    descriptor_setup: str
    projection_checkpoint: str
    final_descriptor_match: str
    notes: str


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


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.{ndigits}f}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


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


def best_key(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    best_threshold = candidates[0][0]
    best_config = candidates[0][1]
    best_score = float("-inf")
    for threshold, key in candidates:
        score = avg_for_key(key, "calibrated")
        if score is not None and score > best_score:
            best_threshold = threshold
            best_config = key
            best_score = score
    return best_threshold, best_config


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.exists():
        return []
    with FAILURES_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def projection_config() -> dict[str, Any]:
    if not OLD_PROJ_CONFIG.exists():
        return {}
    try:
        return read_json(OLD_PROJ_CONFIG)
    except Exception:
        return {}


def training_config_lines() -> list[dict[str, str]]:
    rows = []
    for path in TRAINING_CONFIGS:
        item = {
            "path": rel(path),
            "projection_checkpoint": "",
            "dift_ensemble_size": "",
            "feat_level": "",
            "dift_t": "",
            "dift_up_ft_index": "",
        }
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                for key in ["projection_checkpoint", "dift_ensemble_size", "feat_level", "dift_t", "dift_up_ft_index"]:
                    if stripped.startswith(f"{key}:"):
                        item[key] = stripped.split(":", 1)[1].strip()
        rows.append(item)
    return rows


def main_methods() -> list[MainMethod]:
    zero_th, zero_key = best_key(ZERO_KEYS)
    warm_th, warm_key = best_key(WARM_KEYS)
    scratch_th, scratch_key = best_key(SCRATCH_KEYS)
    expanded_th, expanded_key = best_key(EXPANDED_KEYS)
    methods: list[MainMethod] = []
    if not SKIP_MNN_BASELINE:
        methods.append(
            MainMethod(
                "Projection head + MNN",
                MNN_KEY,
                "MNN",
                f"DINOv3 block16 feat_level=-8 + DIFT t0 up2 ens{ENSEMBLE_SIZE_LABEL} + old h1024 tau007 projection",
                rel(OLD_PROJ_CKPT),
                "old projection/matcher-compatible descriptor",
                "Descriptor-only baseline using the projection checkpoint referenced by learned-matcher training configs.",
            )
        )
    methods.extend([
        MainMethod(
            f"Zero-shot LightGlue best ft={zero_th}",
            zero_key,
            "pretrained SuperPoint+LightGlue",
            "old h1024 tau007 projected descriptor",
            rel(OLD_PROJ_CKPT),
            "old projection descriptor",
            f"Best zero-shot threshold among evaluated sweep: {zero_th}.",
        ),
        MainMethod(
            f"Warm-start LightGlue fine-tuning best ft={warm_th}",
            warm_key,
            MATCHER_CKPTS["warmstart"],
            "old h1024 tau007 projected descriptor",
            rel(OLD_PROJ_CKPT),
            "matcher/projection consistent; eval ensemble follows launch setting",
            f"Best warm-start threshold among evaluated {', '.join(t for t, _ in WARM_KEYS)}: {warm_th}.",
        ),
        MainMethod(
            f"From-scratch LightGlue best ft={scratch_th}",
            scratch_key,
            MATCHER_CKPTS["scratch"],
            "old h1024 tau007 projected descriptor",
            rel(OLD_PROJ_CKPT),
            "matcher/projection consistent; eval ensemble follows launch setting",
            f"Best scratch threshold among evaluated {', '.join(t for t, _ in SCRATCH_KEYS)}: {scratch_th}.",
        ),
        MainMethod(
            f"Expanded 151-scene LightGlue best ft={expanded_th}",
            expanded_key,
            MATCHER_CKPTS["expanded151"],
            "old h1024 tau007 projected descriptor",
            rel(OLD_PROJ_CKPT),
            "matcher/projection consistent; eval ensemble follows launch setting",
            f"Best expanded threshold among evaluated sweep: {expanded_th}.",
        ),
        MainMethod(
            "Joint descriptor-matcher optimization diagnostic",
            JOINT_KEY,
            MATCHER_CKPTS["joint"],
            "old projection descriptor or native joint expectation",
            rel(OLD_PROJ_CKPT),
            "diagnostic",
            "Use only if it produces non-empty stable matches; otherwise reject as incompatible.",
        ),
    ])
    for label, key, ckpt, desc in FIXED_BASELINES:
        methods.append(
            MainMethod(
                label,
                key,
                ckpt,
                desc,
                "n/a",
                "fixed baseline",
                "Existing final output_v2 baseline; not regenerated.",
            )
        )
    return methods


def main_summary_rows() -> list[dict[str, Any]]:
    rows = []
    for method in main_methods():
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
    for method in main_methods():
        for scene in SCENES:
            for mode in MODES:
                rows.append(metric_row(method.method, method.config_key, scene, mode))
    return rows


def threshold_rows(candidates: list[tuple[str, str]], family: str) -> list[dict[str, Any]]:
    rows = []
    for threshold, key in candidates:
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


def diagnostics_rows() -> list[dict[str, Any]]:
    rows = []
    for label, key, note in [
        ("Zero-shot noadapt", NOADAPT_KEY, "disabled adaptivity diagnostic"),
        ("Joint descriptor-matcher optimization", JOINT_KEY, "joint/native projection diagnostic"),
    ]:
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

    for item in training_config_lines():
        rows.append(
            {
                "method": "training-config provenance",
                "config_key": "",
                "scene": "all",
                "solver_mode": "all",
                "mAA@10": "",
                "inlier_ratio": "",
                "avg_matches": "",
                "median_pose_error": "",
                "summary_json_path": "",
                "status": "warning" if item.get("dift_ensemble_size") != ENSEMBLE_SIZE_LABEL else "ok",
                "notes": (
                    f"{item['path']}: projection_checkpoint={item['projection_checkpoint']}; "
                    f"dift_ensemble_size={item['dift_ensemble_size']}; feat_level={item['feat_level']}; "
                    f"dift_t={item['dift_t']}; dift_up_ft_index={item['dift_up_ft_index']}"
                ),
            }
        )
    return rows


def failures_csv_rows() -> list[dict[str, Any]]:
    return read_failures()


def all_config_keys() -> list[str]:
    keys = [NOADAPT_KEY, JOINT_KEY]
    if not SKIP_MNN_BASELINE:
        keys.append(MNN_KEY)
    keys.extend(key for _, key in ZERO_KEYS)
    keys.extend(key for _, key in WARM_KEYS)
    keys.extend(key for _, key in SCRATCH_KEYS)
    keys.extend(key for _, key in EXPANDED_KEYS)
    return sorted(set(keys))


def all_summary_paths() -> list[str]:
    paths = []
    for key in all_config_keys() + [key for _, key, _, _ in FIXED_BASELINES]:
        for scene in SCENES:
            for mode in MODES:
                p = first_summary(key, scene, mode)
                if p:
                    paths.append(rel(p))
    return sorted(set(paths))


def best_learned_method() -> dict[str, Any] | None:
    candidates = []
    for row in main_summary_rows():
        if row["method"].startswith(("Zero-shot", "Warm-start", "From-scratch", "Expanded")):
            try:
                score = float(row["calibrated_avg"])
            except Exception:
                continue
            candidates.append((score, row))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def per_scene_calibrated_table(methods: list[MainMethod]) -> str:
    rows = []
    for method in methods:
        vals = [avg_for_scene(method.config_key, scene, "calibrated") for scene in SCENES]
        rows.append(
            [
                method.method,
                fmt(vals[0]),
                fmt(vals[1]),
                fmt(vals[2]),
                fmt(mean(vals)),
            ]
        )
    return markdown_table(["method", "sacre_coeur", "reichstag", "st_peters_square", "average"], rows)


def avg_for_scene(config_key: str, scene: str, mode: str) -> float | None:
    row = metric_row("", config_key, scene, mode)
    return None if row["mAA@10"] == "" else float(row["mAA@10"])


def threshold_table(candidates: list[tuple[str, str]]) -> str:
    rows = []
    for threshold, key in candidates:
        vals = [avg_for_scene(key, scene, "calibrated") for scene in SCENES]
        rows.append([threshold, fmt(vals[0]), fmt(vals[1]), fmt(vals[2]), fmt(mean(vals))])
    return markdown_table(["threshold", "sacre_coeur", "reichstag", "st_peters_square", "average"], rows)


def write_report(command: str, gpus: str, ensemble_size: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report = REPORT_ROOT / "chapter6_oldproj_learned_matcher_eval_report.md"
    proj = projection_config()
    methods = main_methods()
    main_rows = main_summary_rows()
    best = best_learned_method()
    sp_avg = avg_for_key("superpoint_lg_mp2048", "calibrated")
    roma_avg = avg_for_key("roma_outdoor_mp2048", "calibrated")
    romav2_avg = avg_for_key("romav2_precise_mp2048", "calibrated")
    mnn_avg = None if SKIP_MNN_BASELINE else avg_for_key(MNN_KEY, "calibrated")

    lines = [
        "# Chapter 6 Old-Projection Learned Matcher Evaluation Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Executive Summary",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Best learned matcher: {best['method']} (`{best['config_key']}`), calibrated avg {best['calibrated_avg']}.",
                f"- Shared focal avg for best: {best['shared_focal_avg']}; varying focal avg: {best['varying_focal_avg']}.",
                "- Projection+MNN old-projection baseline omitted by this launch." if SKIP_MNN_BASELINE else f"- Projection+MNN old-projection baseline calibrated avg: {fmt(mnn_avg)}.",
                f"- Fixed baselines calibrated avg: SuperPoint+LightGlue {fmt(sp_avg)}, RoMa {fmt(roma_avg)}, RoMaV2 {fmt(romav2_avg)}.",
            ]
        )
    else:
        lines.append("- No complete learned-matcher result is available yet.")
    lines.extend(
        [
            "",
            "## Exact Protocol",
            "",
            "- Evaluation only; no training and no checkpoint modification.",
            "- Raw-image coordinates, final raw depth/cache protocol, SuperPoint keypoints, 2048 correspondence cap.",
            "- DINOv3 ViT-L/16 block 16, `feat_level=-8`.",
            f"- DIFT Stable Diffusion v1.5, `t=0`, `up_ft_index=2`, ensemble `{ensemble_size}`.",
            "- Independent branch L2 normalization, equal-weight fusion, projected L2-normalized 256D descriptor.",
            "- RePoseD threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, 25 LO iterations.",
            "",
            "## Projection/Matcher Consistency",
            "",
            f"- Projection checkpoint used: `{rel(OLD_PROJ_CKPT)}`.",
            f"- Projection architecture: `1664 -> {proj.get('hidden_dims', ['?'])[0] if proj.get('hidden_dims') else proj.get('hidden_dim', '?')} -> {proj.get('output_dim', '?')}`.",
            f"- Projection temperature: `{proj.get('temperature', 'unknown')}`.",
            "- Learned matcher training configs inspected:",
        ]
    )
    for item in training_config_lines():
        lines.append(
            f"  - `{item['path']}`: projection `{item['projection_checkpoint']}`, "
            f"DIFT ensemble `{item['dift_ensemble_size']}`, feat_level `{item['feat_level']}`."
        )
    lines.extend(
        [
            "",
            "Caveat: the learned-matcher training configs record `dift_ensemble_size: 1`, while the historical Chapter 7 evaluation script used `ensemble_size=8`. This run uses the launch ensemble shown above so the report can test the historical old-projection evaluation distribution under the final raw-image protocol.",
            "",
            "## Runtime",
            "",
            f"- Command: `{command}`",
            f"- GPU allocation: `{gpus}`",
            f"- Git commit: `{git_commit()}`",
            "",
            "## Main Table",
            "",
        ]
    )
    lines.append(
        markdown_table(
            ["method", "config", "calibrated", "shared", "varying", "inliers", "matches", "notes"],
            [
                [
                    row["method"],
                    f"`{row['config_key']}`",
                    row["calibrated_avg"],
                    row["shared_focal_avg"],
                    row["varying_focal_avg"],
                    row["calibrated_inliers_avg"],
                    row["avg_matches"],
                    row["notes"],
                ]
                for row in main_rows
            ],
        )
    )
    lines.extend(["", "## Per-Scene Calibrated Table", "", per_scene_calibrated_table(methods), ""])
    lines.extend(["## Zero-Shot Threshold Sweep", "", threshold_table(ZERO_KEYS), ""])
    lines.extend(["## Expanded 151-Scene Threshold Sweep", "", threshold_table(EXPANDED_KEYS), ""])
    lines.extend(["## Diagnostics", ""])
    diag = diagnostics_rows()
    lines.append(
        markdown_table(
            ["method", "config", "scene", "status", "notes"],
            [[r.get("method", ""), f"`{r.get('config_key', '')}`", r.get("scene", ""), r.get("status", ""), r.get("notes", "")] for r in diag],
        )
    )
    lines.extend(["", "## Final Recommendation", ""])
    if best:
        replace = ""
        try:
            replace = "yes" if float(best["calibrated_avg"]) >= 81.790571 else "no"
        except Exception:
            replace = "undetermined"
        lines.extend(
            [
                f"- Use `{best['config_key']}` as the old-projection-consistent Chapter 6 learned-matcher candidate.",
                f"- This run should replace the previous h1024-temp005 learned-matcher evaluation only if the thesis chooses matcher/projection consistency over the newer Chapter 5 projection checkpoint. Current automatic replace flag versus previous warm-start 81.790571: `{replace}`.",
            ]
        )
    else:
        lines.append("- No final recommendation yet because the main learned-matcher rows are incomplete.")
    lines.extend(["", "## Missing Or Failed Evaluations", ""])
    failures = read_failures()
    if failures:
        lines.append(markdown_table(["stage", "config", "scene", "mode", "status", "reason"], [[f.get("stage", ""), f.get("config_key", ""), f.get("scene", ""), f.get("solver_mode", ""), f.get("status", ""), f.get("reason", "")] for f in failures]))
    else:
        lines.append("No failures recorded.")
    lines.extend(["", "## Output JSON Summary Paths", ""])
    paths = all_summary_paths()
    lines.extend([f"- `{p}`" for p in paths] if paths else ["No summary paths found yet."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(command: str, gpus: str, ensemble_size: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "command": command,
        "gpu_ids": [x for x in gpus.split(",") if x],
        "config_keys": all_config_keys(),
        "projection_checkpoint": rel(OLD_PROJ_CKPT),
        "projection_config": rel(OLD_PROJ_CONFIG),
        "learned_matcher_checkpoints": MATCHER_CKPTS,
        "dinov3": {"model": "ViT-L/16", "block": 16, "feat_level": -8},
        "dift": {"model": "Stable Diffusion v1.5", "t": 0, "up_ft_index": 2, "ensemble_size": int(ensemble_size)},
        "skip_mnn_baseline": SKIP_MNN_BASELINE,
        "prioritize_expanded151": os.environ.get("PRIORITIZE_EXPANDED151", "0") == "1",
        "fusion": {"branch_l2": True, "alpha_dift": 0.5, "alpha_dinov3": 0.5, "projected_dim": 256, "projected_l2": True},
        "scenes": SCENES,
        "solver_modes": MODES,
        "output_paths": {
            "main_csv": "output_v2/csv/chapter6_oldproj_learned_matcher_main.csv",
            "per_scene_csv": "output_v2/csv/chapter6_oldproj_learned_matcher_per_scene.csv",
            "zeroshot_sweep_csv": "output_v2/csv/chapter6_oldproj_zeroshot_threshold_sweep.csv",
            "expanded_sweep_csv": "output_v2/csv/chapter6_oldproj_expanded_threshold_sweep.csv",
            "diagnostics_csv": "output_v2/csv/chapter6_oldproj_diagnostics.csv",
            "failures_csv": "output_v2/csv/chapter6_oldproj_failures.csv",
            "report": "output_v2/reports/chapter6_oldproj_learned_matcher_eval_report.md",
            "summary_json_paths": all_summary_paths(),
        },
        "training_config_provenance": training_config_lines(),
        "failures": read_failures(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_outputs(command: str, gpus: str, ensemble_size: str) -> None:
    main_fields = [
        "method",
        "config_key",
        "calibrated_avg",
        "shared_focal_avg",
        "varying_focal_avg",
        "calibrated_inliers_avg",
        "avg_matches",
        "scenes",
        "descriptor_setup",
        "matcher_checkpoint",
        "projection_checkpoint",
        "final_descriptor_match",
        "notes",
    ]
    per_scene_fields = ["method", "config_key", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"]
    sweep_fields = ["method", "config_key", "threshold", "scene", "solver_mode", "mAA@10", "inlier_ratio", "avg_matches", "median_pose_error", "summary_json_path"]
    diag_fields = per_scene_fields + ["status", "notes"]
    failure_fields = ["time", "stage", "config_key", "scene", "solver_mode", "status", "reason", "log_path", "notes"]

    write_csv(CSV_ROOT / "chapter6_oldproj_learned_matcher_main.csv", main_summary_rows(), main_fields)
    write_csv(CSV_ROOT / "chapter6_oldproj_learned_matcher_per_scene.csv", per_scene_rows(), per_scene_fields)
    write_csv(CSV_ROOT / "chapter6_oldproj_zeroshot_threshold_sweep.csv", threshold_rows(ZERO_KEYS, "zeroshot"), sweep_fields)
    write_csv(CSV_ROOT / "chapter6_oldproj_expanded_threshold_sweep.csv", threshold_rows(EXPANDED_KEYS, "expanded151"), sweep_fields)
    write_csv(CSV_ROOT / "chapter6_oldproj_diagnostics.csv", diagnostics_rows(), diag_fields)
    write_csv(CSV_ROOT / "chapter6_oldproj_failures.csv", failures_csv_rows(), failure_fields)
    write_report(command, gpus, ensemble_size)
    write_manifest(command, gpus, ensemble_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--command", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--ensemble-size", default="8")
    args = parser.parse_args()

    if args.manifest_only:
        write_manifest(args.command, args.gpus, args.ensemble_size)
        return
    if args.write:
        write_outputs(args.command, args.gpus, args.ensemble_size)
        return
    print(json.dumps(main_summary_rows(), indent=2))


if __name__ == "__main__":
    main()
