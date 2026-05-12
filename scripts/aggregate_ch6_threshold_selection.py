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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]
THRESHOLDS = ["0.00", "0.02", "0.05", "0.10", "0.15", "0.20"]
MODES = ["calibrated", "shared_focal", "varying_focal"]

RESULTS_ROOT = ROOT / "output_v2" / "results_v2"
BENCH_ROOT = ROOT / "output_v2" / "benchmarks_v2"
MATCHES_ROOT = ROOT / "output_v2" / "matches_v2"
TIMING_ROOT = ROOT / "output_v2" / "timing"
CSV_ROOT = ROOT / "output_v2" / "csv"
REPORT_ROOT = ROOT / "output_v2" / "reports"
LOG_ROOT = ROOT / "output_v2" / "logs" / "ch6_threshold_sweeps"
STATUS_DIR = LOG_ROOT / "status"
TASKS_TSV = LOG_ROOT / "tasks.tsv"
FAILURES_TSV = LOG_ROOT / "failures.tsv"

TARGET_SOLVER = "3p_ours_shift_scale+12"


FAMILIES: dict[str, dict[str, str]] = {
    "Zero-shot LightGlue": {
        "prefix": "ch6_oldproj_zeroshot_lg",
        "checkpoint": "pretrained SuperPoint+LightGlue",
        "category": "zero-shot",
    },
    "Warm-start LightGlue": {
        "prefix": "ch6_oldproj_warmstart_lg_full_v1",
        "checkpoint": "external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar",
        "category": "warm-start",
    },
    "From-scratch LightGlue": {
        "prefix": "ch6_oldproj_scratch_stage2_lg_v1",
        "checkpoint": "external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar",
        "category": "from-scratch",
    },
    "Expanded 151-scene LightGlue": {
        "prefix": "ch6_oldproj_expanded151_lg",
        "checkpoint": "external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar",
        "category": "expanded MegaDepth",
    },
    "Joint optimization diagnostic": {
        "prefix": "ch6_oldproj_joint_unfrozen_proj60",
        "checkpoint": "external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar",
        "category": "joint optimization",
    },
}


def suffix(threshold: str) -> str:
    return {
        "0.00": "ft000",
        "0.02": "ft002",
        "0.05": "ft005",
        "0.10": "ft010",
        "0.15": "ft015",
        "0.20": "ft020",
    }[threshold]


def config_key(family: str, threshold: str) -> str:
    return f"{FAMILIES[family]['prefix']}_{suffix(threshold)}_sp_mnn_mp2048"


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.{ndigits}f}"


def mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def first_summary(key: str, scene: str, mode: str) -> Path | None:
    pattern = RESULTS_ROOT / key / scene / f"{mode}-{key}_{scene}*_summary.json"
    matches = sorted(glob.glob(str(pattern)))
    return Path(matches[-1]) if matches else None


def pick_experiment(summary: dict[str, Any]) -> dict[str, Any]:
    experiments = summary.get("experiments", [])
    for item in experiments:
        if item.get("solver") == TARGET_SOLVER:
            return item
    if not experiments:
        return {}
    return max(experiments, key=lambda x: float(x.get("mAA@10", float("-inf"))))


def avg_matches(key: str, scene: str) -> float | None:
    timing_path = TIMING_ROOT / f"{key}_{scene}_timing.json"
    if timing_path.exists():
        try:
            payload = read_json(timing_path)
            vals = [float(x["num_matches"]) for x in payload.get("pair_timings", []) if "num_matches" in x]
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
    bench = BENCH_ROOT / f"{key}_{scene}.h5"
    if bench.exists():
        try:
            import h5py  # type: ignore

            counts = []
            with h5py.File(bench, "r") as handle:
                for name in handle.keys():
                    obj = handle[name]
                    if hasattr(obj, "shape") and len(obj.shape) >= 1 and name.startswith("corr_"):
                        counts.append(int(obj.shape[0]))
            if counts:
                return sum(counts) / len(counts)
        except Exception:
            pass
    match_dir = MATCHES_ROOT / key / scene
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


def metric(key: str, scene: str, mode: str = "calibrated") -> dict[str, Any]:
    path = first_summary(key, scene, mode)
    matches = avg_matches(key, scene)
    if not path:
        return {
            "mAA@10": None,
            "inlier_ratio": None,
            "avg_matches": matches,
            "summary_json_path": "",
        }
    summary = read_json(path)
    exp = pick_experiment(summary)
    return {
        "mAA@10": exp.get("mAA@10"),
        "inlier_ratio": exp.get("mean_inlier_ratio"),
        "avg_matches": matches,
        "summary_json_path": rel(path),
    }


def complete_for_mode(key: str, mode: str) -> tuple[bool, list[str]]:
    missing = [scene for scene in SCENES if first_summary(key, scene, mode) is None]
    return not missing, missing


def threshold_complete(family: str, threshold: str) -> bool:
    key = config_key(family, threshold)
    return all(first_summary(key, scene, "calibrated") is not None for scene in SCENES)


def threshold_average(family: str, threshold: str) -> float | None:
    key = config_key(family, threshold)
    vals: list[float | None] = []
    for scene in SCENES:
        val = metric(key, scene, "calibrated")["mAA@10"]
        vals.append(None if val is None or val == "" else float(val))
    if any(v is None for v in vals):
        return None
    return mean(vals)


def planned_new_thresholds() -> dict[str, set[str]]:
    planned: dict[str, set[str]] = {family: set() for family in FAMILIES}
    if not TASKS_TSV.exists():
        return planned
    with TASKS_TSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            family = row.get("family", "")
            threshold = row.get("threshold", "")
            if family in planned and threshold:
                planned[family].add(threshold)
    return planned


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.exists():
        return []
    with FAILURES_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def status_time(name: str) -> int | None:
    path = STATUS_DIR / name
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def runtime_text() -> str:
    start = status_time("supervisor.started")
    done = status_time("supervisor.done")
    failed = status_time("supervisor.failed")
    end = done or failed
    if not start:
        return "not launched"
    if not end:
        elapsed = int(time.time()) - start
        return f"running; elapsed so far {elapsed / 3600:.2f} hours"
    elapsed = end - start
    status = "completed" if done else "failed"
    return f"{status}; actual elapsed {elapsed / 3600:.2f} hours ({elapsed} seconds)"


def sweep_rows() -> list[dict[str, Any]]:
    rows = []
    for family, meta in FAMILIES.items():
        for threshold in THRESHOLDS:
            key = config_key(family, threshold)
            scene_values = {}
            scene_inliers = {}
            scene_matches = {}
            summary_paths = {}
            for scene in SCENES:
                item = metric(key, scene, "calibrated")
                scene_values[scene] = item["mAA@10"]
                scene_inliers[scene] = item["inlier_ratio"]
                scene_matches[scene] = item["avg_matches"]
                summary_paths[scene] = item["summary_json_path"]
            avg = mean([None if scene_values[s] in ("", None) else float(scene_values[s]) for s in SCENES])
            rows.append(
                {
                    "family": family,
                    "category": meta["category"],
                    "threshold": threshold,
                    "config_key": key,
                    "sacre_coeur": fmt(scene_values["sacre_coeur"]),
                    "reichstag": fmt(scene_values["reichstag"]),
                    "st_peters_square": fmt(scene_values["st_peters_square"]),
                    "average": fmt(avg),
                    "sacre_coeur_inliers": fmt(scene_inliers["sacre_coeur"]),
                    "reichstag_inliers": fmt(scene_inliers["reichstag"]),
                    "st_peters_square_inliers": fmt(scene_inliers["st_peters_square"]),
                    "avg_matches": fmt(mean([scene_matches[s] for s in SCENES])),
                    "complete_calibrated": "yes" if threshold_complete(family, threshold) else "no",
                    "summary_json_paths": "; ".join(p for p in summary_paths.values() if p),
                    "matcher_checkpoint": meta["checkpoint"],
                }
            )
    return rows


def selected_rows() -> list[dict[str, Any]]:
    rows = []
    for family, meta in FAMILIES.items():
        candidates = []
        for threshold in THRESHOLDS:
            avg = threshold_average(family, threshold)
            if avg is not None:
                candidates.append((avg, threshold, config_key(family, threshold)))
        if candidates:
            best_avg, threshold, key = max(candidates, key=lambda item: item[0])
        else:
            best_avg, threshold, key = (None, "", "")
        shared_ok, shared_missing = complete_for_mode(key, "shared_focal") if key else (False, SCENES[:])
        varying_ok, varying_missing = complete_for_mode(key, "varying_focal") if key else (False, SCENES[:])
        needed = []
        if key:
            if not shared_ok:
                needed.append(f"{key}:shared_focal:{','.join(shared_missing)}")
            if not varying_ok:
                needed.append(f"{key}:varying_focal:{','.join(varying_missing)}")
        rows.append(
            {
                "family": family,
                "category": meta["category"],
                "selected_threshold": threshold,
                "selected_config_key": key,
                "calibrated_average": fmt(best_avg),
                "shared_focal_complete": "yes" if shared_ok else "no",
                "varying_focal_complete": "yes" if varying_ok else "no",
                "missing_all_mode_evaluations": "; ".join(needed),
                "matcher_checkpoint": meta["checkpoint"],
            }
        )
    return rows


def warnings_rows() -> list[str]:
    warnings: list[str] = []
    for row in sweep_rows():
        if row["complete_calibrated"] != "yes":
            missing = []
            key = row["config_key"]
            for scene in SCENES:
                if first_summary(key, scene, "calibrated") is None:
                    missing.append(scene)
            warnings.append(f"missing calibrated JSONs for {key}: {', '.join(missing)}")
        try:
            avg_m = float(row["avg_matches"]) if row["avg_matches"] else None
        except Exception:
            avg_m = None
        if avg_m is not None and avg_m <= 0:
            warnings.append(f"empty-match output for {row['config_key']}: average matches {row['avg_matches']}")
        elif avg_m is not None and avg_m < 20:
            warnings.append(f"very-low-match warning for {row['config_key']}: average matches {row['avg_matches']}")
    for failure in read_failures():
        warnings.append(
            "failure recorded: "
            f"{failure.get('stage', '')} {failure.get('config_key', '')} "
            f"{failure.get('scene', '')} {failure.get('solver_mode', '')}: "
            f"{failure.get('reason', '')}"
        )
    return warnings


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(args: argparse.Namespace) -> None:
    planned = planned_new_thresholds()
    selected = selected_rows()
    sweep = sweep_rows()
    warnings = warnings_rows()

    lines = [
        "# Chapter 6 LightGlue Threshold Selection Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Protocol",
        "",
        "- Evaluation only; no training or checkpoint modification.",
        "- Selection solver mode: `calibrated`.",
        "- Scenes: `sacre_coeur`, `reichstag`, `st_peters_square`.",
        "- Raw-image final protocol with RePoseD threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, and 25 LO iterations.",
        "- Correspondence cap: 2048.",
        "- Descriptor: SuperPoint keypoints, DINOv3 ViT-L/16 block 16 `feat_level=-8`, DIFT SD v1.5 `t=0`, `up_ft_index=2`, ensemble 2.",
        "- Fusion/projection: independent branch L2 normalization, equal DIFT/DINOv3 weights, old projection checkpoint `experiments/phase2_projection_wide/best.pt`, architecture `1664 -> 1024 -> 256`, L2-normalized 256D output.",
        "",
        "## Runtime",
        "",
        f"- Runtime estimate before launch: `{args.runtime_estimate}`.",
        f"- Actual runtime: `{runtime_text()}`.",
        f"- Launch command: `{args.command}`.",
        f"- GPU allocation: `{args.gpus}`.",
        f"- Git commit: `{git_commit()}`.",
        "",
        "## Reused And Newly Evaluated Thresholds",
        "",
    ]
    reuse_rows = []
    for family in FAMILIES:
        new_thresholds = sorted(planned.get(family, set()), key=THRESHOLDS.index)
        complete_existing = [
            threshold
            for threshold in THRESHOLDS
            if threshold_complete(family, threshold) and threshold not in planned.get(family, set())
        ]
        reuse_rows.append(
            [
                family,
                ", ".join(complete_existing) if complete_existing else "",
                ", ".join(new_thresholds) if new_thresholds else "",
            ]
        )
    lines.append(markdown_table(["family", "already present / reused", "newly evaluated by this launch"], reuse_rows))

    lines.extend(["", "## Selected Thresholds", ""])
    lines.append(
        markdown_table(
            [
                "family",
                "selected threshold",
                "config",
                "calibrated avg",
                "shared_focal complete",
                "varying_focal complete",
                "still needed if selected",
            ],
            [
                [
                    row["family"],
                    row["selected_threshold"],
                    f"`{row['selected_config_key']}`",
                    row["calibrated_average"],
                    row["shared_focal_complete"],
                    row["varying_focal_complete"],
                    row["missing_all_mode_evaluations"],
                ]
                for row in selected
            ],
        )
    )

    lines.extend(["", "## Per-Family Calibrated Sweeps", ""])
    for family in FAMILIES:
        lines.extend([f"### {family}", ""])
        family_rows = [row for row in sweep if row["family"] == family]
        lines.append(
            markdown_table(
                ["threshold", "sacre_coeur", "reichstag", "st_peters_square", "average"],
                [
                    [
                        row["threshold"],
                        row["sacre_coeur"],
                        row["reichstag"],
                        row["st_peters_square"],
                        row["average"],
                    ]
                    for row in family_rows
                ],
            )
        )
        lines.append("")

    lines.extend(["## Missing All-Mode Evaluations For Selected Thresholds", ""])
    needed_rows = []
    for row in selected:
        if row["missing_all_mode_evaluations"]:
            needed_rows.append([row["family"], row["selected_config_key"], row["missing_all_mode_evaluations"]])
    if needed_rows:
        lines.append(markdown_table(["family", "selected config", "exact configs/modes/scenes needed"], needed_rows))
    else:
        lines.append("All selected thresholds already have `shared_focal` and `varying_focal` summaries.")

    lines.extend(["", "## Failures And Warnings", ""])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("No failures, missing calibrated JSONs, or empty-match outputs were detected.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `output_v2/csv/chapter6_all_threshold_sweeps.csv`",
            "- `output_v2/csv/chapter6_selected_thresholds.csv`",
            "- `output_v2/reports/chapter6_threshold_selection_report.md`",
        ]
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "chapter6_threshold_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(args: argparse.Namespace) -> None:
    sweep_fields = [
        "family",
        "category",
        "threshold",
        "config_key",
        "sacre_coeur",
        "reichstag",
        "st_peters_square",
        "average",
        "sacre_coeur_inliers",
        "reichstag_inliers",
        "st_peters_square_inliers",
        "avg_matches",
        "complete_calibrated",
        "summary_json_paths",
        "matcher_checkpoint",
    ]
    selected_fields = [
        "family",
        "category",
        "selected_threshold",
        "selected_config_key",
        "calibrated_average",
        "shared_focal_complete",
        "varying_focal_complete",
        "missing_all_mode_evaluations",
        "matcher_checkpoint",
    ]
    write_csv(CSV_ROOT / "chapter6_all_threshold_sweeps.csv", sweep_rows(), sweep_fields)
    write_csv(CSV_ROOT / "chapter6_selected_thresholds.csv", selected_rows(), selected_fields)
    write_report(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Chapter 6 LightGlue threshold sweeps.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--command", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--runtime-estimate", default="3-6 hours on GPUs 0,1,3; 9-18 hours on one GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        write_outputs(args)
    else:
        for row in selected_rows():
            print(row)


if __name__ == "__main__":
    main()
