#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import subprocess
import time
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
LOG_ROOT = ROOT / "output_v2" / "logs" / "ch6_selected_all_modes"
STATUS_DIR = LOG_ROOT / "status"
FAILURES_TSV = LOG_ROOT / "failures.tsv"

TARGET_SOLVERS = {
    "calibrated": "3p_ours_shift_scale+12",
    "shared_focal": "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}

METHODS = [
    {
        "family": "Descriptor-only control",
        "method": "Projection head + MNN",
        "config": "ch5_b16_eval_proj_temp005_h512_d256_sp_mnn_mp2048",
        "threshold": "",
        "notes": "Descriptor-only MNN control from the corrected Chapter 5 selected projection-head baseline.",
    },
    {
        "family": "Zero-shot LightGlue",
        "method": "Zero-shot LightGlue",
        "config": "ch6_oldproj_zeroshot_lg_ft015_sp_mnn_mp2048",
        "threshold": "0.15",
        "notes": "Selected by calibrated threshold sweep; no matcher training on projected descriptors.",
    },
    {
        "family": "Warm-start LightGlue",
        "method": "Warm-start LightGlue fine-tuning",
        "config": "ch6_oldproj_warmstart_lg_full_v1_ft002_sp_mnn_mp2048",
        "threshold": "0.02",
        "notes": "Selected by calibrated threshold sweep; 60-scene MegaDepth warm-start model.",
    },
    {
        "family": "From-scratch LightGlue",
        "method": "From-scratch LightGlue",
        "config": "ch6_oldproj_scratch_stage2_lg_v1_ft002_sp_mnn_mp2048",
        "threshold": "0.02",
        "notes": "Selected by calibrated threshold sweep; Stage 1 homography then 60-scene MegaDepth.",
    },
    {
        "family": "Expanded 151-scene LightGlue",
        "method": "Expanded 151-scene LightGlue",
        "config": "ch6_oldproj_expanded151_lg_ft002_sp_mnn_mp2048",
        "threshold": "0.02",
        "notes": "Selected by calibrated threshold sweep; expanded MegaDepth data scaling.",
    },
    {
        "family": "Joint optimization diagnostic",
        "method": "Joint optimization diagnostic",
        "config": "ch6_oldproj_joint_unfrozen_proj60_ft000_sp_mnn_mp2048",
        "threshold": "0.00",
        "notes": "Selected by calibrated threshold sweep; diagnostic with trainable projection during training.",
    },
    {
        "family": "Fixed baseline",
        "method": "SuperPoint+LightGlue",
        "config": "superpoint_lg_mp2048",
        "threshold": "",
        "notes": "Existing fixed baseline; not regenerated.",
    },
    {
        "family": "Fixed baseline",
        "method": "RoMa",
        "config": "roma_outdoor_mp2048",
        "threshold": "",
        "notes": "Existing fixed baseline; not regenerated.",
    },
    {
        "family": "Fixed baseline",
        "method": "RoMaV2",
        "config": "romav2_precise_mp2048",
        "threshold": "",
        "notes": "Existing fixed baseline; not regenerated.",
    },
]


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


def first_summary(config: str, scene: str, mode: str) -> Path | None:
    pattern = RESULTS_ROOT / config / scene / f"{mode}-{config}_{scene}*_summary.json"
    matches = sorted(glob.glob(str(pattern)))
    return Path(matches[-1]) if matches else None


def pick_experiment(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    target = TARGET_SOLVERS[mode]
    experiments = summary.get("experiments", [])
    for item in experiments:
        if item.get("solver") == target:
            return item
    if not experiments:
        return {}
    return max(experiments, key=lambda item: float(item.get("mAA@10", float("-inf"))))


def avg_matches(config: str, scene: str) -> float | None:
    timing_path = TIMING_ROOT / f"{config}_{scene}_timing.json"
    if timing_path.exists():
        try:
            payload = read_json(timing_path)
            vals = [float(x["num_matches"]) for x in payload.get("pair_timings", []) if "num_matches" in x]
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
    bench = BENCH_ROOT / f"{config}_{scene}.h5"
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
    match_dir = MATCHES_ROOT / config / scene
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


def metric(config: str, scene: str, mode: str) -> dict[str, Any]:
    summary_path = first_summary(config, scene, mode)
    matches = avg_matches(config, scene)
    if not summary_path:
        return {
            "mAA@10": None,
            "inlier_ratio": None,
            "avg_matches": matches,
            "summary_json_path": "",
        }
    summary = read_json(summary_path)
    exp = pick_experiment(summary, mode)
    return {
        "mAA@10": exp.get("mAA@10"),
        "inlier_ratio": exp.get("mean_inlier_ratio"),
        "avg_matches": matches,
        "summary_json_path": rel(summary_path),
    }


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.exists():
        return []
    with FAILURES_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_notes() -> list[str]:
    notes: list[str] = []
    archived_failures = sorted(LOG_ROOT.glob("archive_failed_*"))
    if archived_failures:
        notes.append(
            "An initial launch attempt failed before accepted outputs were produced because RePoseD could not locate "
            "`libmkl_intel_lp64.so.2`; the launcher was updated to prepend MKL/OpenMP library directories to "
            "`LD_LIBRARY_PATH`, then the run was relaunched successfully."
        )
    return notes


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
        return f"running; elapsed so far {(int(time.time()) - start) / 3600:.2f} hours"
    elapsed = end - start
    status = "completed" if done else "failed"
    return f"{status}; actual elapsed {elapsed / 3600:.2f} hours ({elapsed} seconds)"


def all_modes_rows() -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        config = method["config"]
        for mode in MODES:
            scene_scores = []
            scene_inliers = []
            scene_matches = []
            for scene in SCENES:
                item = metric(config, scene, mode)
                scene_scores.append(None if item["mAA@10"] in ("", None) else float(item["mAA@10"]))
                scene_inliers.append(None if item["inlier_ratio"] in ("", None) else float(item["inlier_ratio"]))
                scene_matches.append(item["avg_matches"])
            rows.append(
                {
                    "family": method["family"],
                    "method": method["method"],
                    "config": config,
                    "threshold": method["threshold"],
                    "solver_mode": mode,
                    "sacre_coeur": fmt(scene_scores[0]),
                    "reichstag": fmt(scene_scores[1]),
                    "st_peters_square": fmt(scene_scores[2]),
                    "average": fmt(mean(scene_scores)),
                    "inlier_ratio": fmt(mean(scene_inliers)),
                    "avg_matches": fmt(mean(scene_matches)),
                    "notes": method["notes"],
                }
            )
    return rows


def per_scene_rows() -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        config = method["config"]
        for scene in SCENES:
            calibrated = metric(config, scene, "calibrated")
            shared = metric(config, scene, "shared_focal")
            varying = metric(config, scene, "varying_focal")
            rows.append(
                {
                    "family": method["family"],
                    "method": method["method"],
                    "config": config,
                    "threshold": method["threshold"],
                    "scene": scene,
                    "calibrated": fmt(calibrated["mAA@10"]),
                    "shared_focal": fmt(shared["mAA@10"]),
                    "varying_focal": fmt(varying["mAA@10"]),
                    "inlier_ratio": fmt(calibrated["inlier_ratio"]),
                    "avg_matches": fmt(calibrated["avg_matches"]),
                    "notes": method["notes"],
                }
            )
    return rows


def missing_outputs() -> list[str]:
    missing = []
    for method in METHODS:
        config = method["config"]
        for scene in SCENES:
            for mode in MODES:
                if first_summary(config, scene, mode) is None:
                    missing.append(f"{config}:{mode}:{scene}")
    return missing


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


def threshold_table() -> str:
    rows = []
    for method in METHODS:
        if method["threshold"]:
            rows.append([method["method"], method["config"], method["threshold"]])
    return markdown_table(["method", "selected config", "threshold"], rows)


def write_report(args: argparse.Namespace) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report = REPORT_ROOT / "chapter6_selected_all_modes_report.md"
    all_rows = all_modes_rows()
    per_scene = per_scene_rows()
    missing = missing_outputs()
    failures = read_failures()
    notes = run_notes()

    lines = [
        "# Chapter 6 Selected LightGlue All-Modes Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Protocol",
        "",
        "- Evaluation only; no training was run.",
        "- No checkpoints were modified.",
        "- Calibrated results were reused; this pass only completed missing `shared_focal` and `varying_focal` summaries for selected learned-matcher configs.",
        "- Scenes: `sacre_coeur`, `reichstag`, `st_peters_square`.",
        "- Raw-image coordinate protocol, SuperPoint keypoints, correspondence cap 2048.",
        "- DINOv3 ViT-L/16 block 16 with `feat_level=-8`; DIFT SD v1.5 with `t=0`, `up_ft_index=2`, ensemble 2.",
        "- Independent branch L2 normalization, equal-weight fusion, Chapter 5 selected projection checkpoint `experiments/phase2_projection_temp005/best.pt`, architecture `1664 -> 512 -> 256`, temperature `tau=0.05`, L2-normalized 256D projected descriptors.",
        "- RePoseD threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, 25 LO iterations.",
        "",
        "## Runtime",
        "",
        f"- Runtime estimate before launch: `{args.runtime_estimate}`.",
        f"- Actual runtime: `{runtime_text()}`.",
        f"- Launch command: `{args.command}`.",
        f"- GPU allocation requested: `{args.gpus}`.",
        f"- Git commit: `{git_commit()}`.",
        "",
        "## Configs Evaluated In This Pass",
        "",
        markdown_table(
            ["method", "config", "threshold", "new modes"],
            [
                [m["method"], f"`{m['config']}`", m["threshold"], "`shared_focal`, `varying_focal`"]
                for m in METHODS
                if m["method"] in {
                    "Warm-start LightGlue fine-tuning",
                    "From-scratch LightGlue",
                    "Expanded 151-scene LightGlue",
                    "Joint optimization diagnostic",
                }
            ],
        ),
        "",
        "## Missing Or Failed Evaluations",
        "",
    ]
    if missing:
        lines.extend([f"- Missing summary: `{item}`" for item in missing])
    elif failures:
        lines.extend([f"- Failure: `{f.get('config_key', '')}:{f.get('solver_mode', '')}:{f.get('scene', '')}` {f.get('reason', '')}" for f in failures])
    else:
        lines.append("No missing or failed selected all-mode evaluations were detected.")

    if notes:
        lines.extend(["", "## Run Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    lines.extend(["", "## Main All-Mode Table", ""])
    lines.append(
        markdown_table(
            ["method", "config", "threshold", "mode", "calibrated/shared/varying avg", "inliers", "matches"],
            [
                [
                    row["method"],
                    f"`{row['config']}`",
                    row["threshold"],
                    row["solver_mode"],
                    row["average"],
                    row["inlier_ratio"],
                    row["avg_matches"],
                ]
                for row in all_rows
            ],
        )
    )

    lines.extend(["", "## Per-Scene Calibrated Table", ""])
    calibrated_rows = [row for row in all_rows if row["solver_mode"] == "calibrated"]
    lines.append(
        markdown_table(
            ["method", "config", "threshold", "sacre_coeur", "reichstag", "st_peters_square", "average"],
            [
                [
                    row["method"],
                    f"`{row['config']}`",
                    row["threshold"],
                    row["sacre_coeur"],
                    row["reichstag"],
                    row["st_peters_square"],
                    row["average"],
                ]
                for row in calibrated_rows
            ],
        )
    )

    lines.extend(["", "## Threshold Selection Table", "", threshold_table(), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The projection-head MNN row is the corrected descriptor-only control from the selected Chapter 5 projection-head baseline.",
            "- Zero-shot LightGlue tests compatibility between the projected descriptor and an unfine-tuned LightGlue matcher.",
            "- Warm-start and from-scratch LightGlue test matcher adaptation on the 60-scene MegaDepth setup.",
            "- Expanded 151-scene LightGlue tests whether scaling the MegaDepth training data improves the selected learned matcher.",
            "- The joint optimization diagnostic tests whether making the projection trainable during matcher training helps under this evaluation protocol.",
            "- The numeric conclusions should be read directly from the all-mode and per-scene tables above.",
            "",
            "## Output Files",
            "",
            "- `output_v2/csv/chapter6_selected_all_modes.csv`",
            "- `output_v2/csv/chapter6_selected_per_scene.csv`",
            "- `output_v2/reports/chapter6_selected_all_modes_report.md`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(args: argparse.Namespace) -> None:
    write_csv(
        CSV_ROOT / "chapter6_selected_all_modes.csv",
        all_modes_rows(),
        ["family", "method", "config", "threshold", "solver_mode", "sacre_coeur", "reichstag", "st_peters_square", "average", "inlier_ratio", "avg_matches", "notes"],
    )
    write_csv(
        CSV_ROOT / "chapter6_selected_per_scene.csv",
        per_scene_rows(),
        ["family", "method", "config", "threshold", "scene", "calibrated", "shared_focal", "varying_focal", "inlier_ratio", "avg_matches", "notes"],
    )
    write_report(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Chapter 6 selected all-mode evaluations.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--command", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--runtime-estimate", default="20-60 minutes; RePoseD-only pass reusing existing benchmarks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        write_outputs(args)
    else:
        for row in all_modes_rows():
            print(row)


if __name__ == "__main__":
    main()
