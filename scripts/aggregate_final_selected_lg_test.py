#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

CONFIG = "final_selected_expanded151_lg_proj_dinov3_dift_ft002_mp2048"
RESULTS_ROOT = ROOT / "output_v2" / "results_v2" / CONFIG
BENCH_ROOT = ROOT / "output_v2" / "benchmarks_v2"
CSV_ROOT = ROOT / "output_v2" / "csv"
REPORT_ROOT = ROOT / "output_v2" / "reports"
LOG_ROOT = ROOT / "output_v2" / "logs" / "final_selected_expanded151_lg_test"
STATUS_DIR = LOG_ROOT / "status"
FAILURES_TSV = LOG_ROOT / "failures.tsv"

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

MODES = ["calibrated", "shared_focal", "varying_focal"]
TARGET_SOLVERS = {
    "calibrated": "3p_ours_shift_scale+12",
    "shared_focal": "4p_ours_scale_shift+12",
    "varying_focal": "4p_ours_scale_shift+12",
}

CHECKPOINT = "external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
PROJECTION = "experiments/phase2_projection_wide/best.pt"


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


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


def pairs_file(scene: str) -> Path:
    return ROOT / "output" / f"pairs_{scene}.txt"


def pair_count(scene: str) -> int:
    path = pairs_file(scene)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def pair_limit(scene: str) -> int:
    count = pair_count(scene)
    return min(count, 15000) if count else 0


def benchmark_path(scene: str) -> Path:
    return BENCH_ROOT / f"{CONFIG}_{scene}.h5"


def summary_path(scene: str, mode: str) -> Path:
    return RESULTS_ROOT / scene / f"{mode}-{CONFIG}_{scene}-2.0t_summary.json"


def raw_json_path(scene: str, mode: str) -> Path:
    return RESULTS_ROOT / scene / f"{mode}-{CONFIG}_{scene}-2.0t.json"


@lru_cache(maxsize=None)
def avg_matches(scene: str) -> float | None:
    path = benchmark_path(scene)
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as handle:
            corr_keys = [key for key in handle.keys() if key.startswith("corr_")]
            if not corr_keys:
                return None
            return sum(float(handle[key].shape[0]) for key in corr_keys) / len(corr_keys)
    except Exception:
        return None


def read_metric(scene: str, mode: str) -> dict[str, Any]:
    path = summary_path(scene, mode)
    if not path.exists():
        return {
            "scene": scene,
            "solver_mode": mode,
            "solver": TARGET_SOLVERS[mode],
            "mAA@10": None,
            "inlier_ratio": None,
            "solver_runtime_ms_per_pair": None,
            "num_evaluated_pairs": None,
            "avg_matches": avg_matches(scene),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = TARGET_SOLVERS[mode]
    experiments = payload.get("experiments", [])
    selected = None
    for item in experiments:
        if item.get("solver") == target:
            selected = item
            break
    if selected is None and experiments:
        selected = max(experiments, key=lambda item: float(item.get("mAA@10") or -1.0))
    selected = selected or {}
    return {
        "scene": scene,
        "solver_mode": mode,
        "solver": selected.get("solver", target),
        "mAA@10": selected.get("mAA@10"),
        "inlier_ratio": selected.get("mean_inlier_ratio"),
        "solver_runtime_ms_per_pair": selected.get("solver_runtime_ms_per_pair"),
        "num_evaluated_pairs": selected.get("num_evaluated_pairs"),
        "avg_matches": avg_matches(scene),
    }


def mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value not in (None, "")]
    if not nums:
        return None
    return sum(nums) / len(nums)


def all_mode_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        metrics = [read_metric(scene, mode) for scene in SCENES]
        row = {
            "config": CONFIG,
            "solver_mode": mode,
            "solver": TARGET_SOLVERS[mode],
            "average": fmt(mean([item["mAA@10"] for item in metrics])),
            "inlier_ratio": fmt(mean([item["inlier_ratio"] for item in metrics])),
            "avg_matches": fmt(mean([item["avg_matches"] for item in metrics])),
            "num_evaluated_pairs": fmt(mean([item["num_evaluated_pairs"] for item in metrics])),
            "notes": "Selected final sparse generic-feature pipeline from Chapter 6.",
        }
        for scene, metric in zip(SCENES, metrics):
            row[scene] = fmt(metric["mAA@10"])
        rows.append(row)
    return rows


def per_scene_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        for mode in MODES:
            metric = read_metric(scene, mode)
            rows.append(
                {
                    "config": CONFIG,
                    "scene": scene,
                    "solver_mode": mode,
                    "solver": metric["solver"],
                    "mAA@10": fmt(metric["mAA@10"]),
                    "inlier_ratio": fmt(metric["inlier_ratio"]),
                    "avg_matches": fmt(metric["avg_matches"]),
                    "num_evaluated_pairs": fmt(metric["num_evaluated_pairs"]),
                    "pair_file": str(pairs_file(scene).relative_to(ROOT)),
                    "pair_count": pair_count(scene),
                    "pair_limit": pair_limit(scene),
                    "notes": "",
                }
            )
    return rows


def missing_outputs() -> list[str]:
    missing = []
    for scene in SCENES:
        if not benchmark_path(scene).exists():
            missing.append(f"{CONFIG}:benchmark:{scene}")
        for mode in MODES:
            if not summary_path(scene, mode).exists():
                missing.append(f"{CONFIG}:{mode}:{scene}")
    return missing


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.exists():
        return []
    with FAILURES_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = all_mode_rows()
    scene_rows = per_scene_rows()
    missing = missing_outputs()
    failures = read_failures()

    lines = [
        "# Final Selected Expanded-151 LightGlue Test Report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Protocol",
        "",
        "- Evaluation only; no training was run.",
        "- No checkpoints were modified.",
        "- This is the selected final sparse generic-feature pipeline from Chapter 6.",
        f"- Config: `{CONFIG}`.",
        f"- Matcher checkpoint: `{CHECKPOINT}`.",
        f"- Projection checkpoint: `{PROJECTION}`.",
        "- Matcher: LightGlue, 9 layers, 4 heads, filter threshold 0.02.",
        "- Keypoints: SuperPoint detections; correspondence cap 2048.",
        "- Descriptor: projected DINOv3 + DIFT descriptor; projection architecture `1664 -> 1024 -> 256`; L2-normalized 256D output.",
        "- DINOv3: ViT-L/16, block 16, `feat_level=-8`.",
        "- DIFT: Stable Diffusion v1.5, `t=0`, `up_ft_index=2`, `ensemble_size=2`.",
        "- Fusion: independent branch L2 normalization and equal DINOv3/DIFT weights.",
        "- Pose protocol: raw-image coordinate final protocol with RePoseD/Sampson threshold 2.0 px, reprojection threshold 16.0 px, 1000 RANSAC iterations, and 25 LO iterations.",
        "- Solver modes: `calibrated`, `shared_focal`, `varying_focal`.",
        "",
        "## Runtime",
        "",
        f"- Runtime estimate before launch: `{args.runtime_estimate}`.",
        f"- Actual runtime: `{runtime_text()}`.",
        f"- Launch command: `{args.command}`.",
        f"- GPU allocation requested: `{args.gpus}`.",
        f"- Git commit: `{git_commit()}`.",
        "",
        "## Test Scenes And Pairs",
        "",
        markdown_table(
            ["scene", "pair file", "total pairs", "evaluated pair limit"],
            [[scene, f"`{pairs_file(scene).relative_to(ROOT)}`", pair_count(scene), pair_limit(scene)] for scene in SCENES],
        ),
        "",
        "## Missing Or Failed Evaluations",
        "",
    ]
    if missing:
        lines.extend([f"- Missing: `{item}`" for item in missing])
    elif failures:
        lines.extend([f"- Failure: `{item.get('scene', '')}:{item.get('solver_mode', '')}` {item.get('reason', '')}" for item in failures])
    else:
        lines.append("No missing or failed final selected test evaluations were detected.")

    lines.extend(["", "## Main All-Mode Table", ""])
    lines.append(
        markdown_table(
            ["mode", "solver", "average mAA@10", "inlier ratio", "avg matches", "avg evaluated pairs"],
            [
                [
                    row["solver_mode"],
                    row["solver"],
                    row["average"],
                    row["inlier_ratio"],
                    row["avg_matches"],
                    row["num_evaluated_pairs"],
                ]
                for row in rows
            ],
        )
    )

    lines.extend(["", "## Per-Scene Tables", ""])
    for mode in MODES:
        mode_rows = [row for row in scene_rows if row["solver_mode"] == mode]
        lines.extend([f"### {mode}", ""])
        lines.append(
            markdown_table(
                ["scene", "mAA@10", "inlier ratio", "avg matches", "evaluated pairs"],
                [
                    [row["scene"], row["mAA@10"], row["inlier_ratio"], row["avg_matches"], row["num_evaluated_pairs"]]
                    for row in mode_rows
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is the selected final sparse generic-feature pipeline from Chapter 6.",
            "- It should be compared against the existing final-test baselines: SuperPoint+LightGlue, RoMa, RoMaV2, and descriptor-only MNN if available.",
            "- The numeric conclusions should be read from the all-mode and per-scene tables above.",
            "",
            "## Output Files",
            "",
            "- `output_v2/csv/final_selected_expanded151_lg_test_summary.csv`",
            "- `output_v2/csv/final_selected_expanded151_lg_test_per_scene.csv`",
            "- `output_v2/reports/final_selected_expanded151_lg_test_report.md`",
        ]
    )

    (REPORT_ROOT / "final_selected_expanded151_lg_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(args: argparse.Namespace) -> None:
    summary_fields = [
        "config",
        "solver_mode",
        "solver",
        *SCENES,
        "average",
        "inlier_ratio",
        "avg_matches",
        "num_evaluated_pairs",
        "notes",
    ]
    per_scene_fields = [
        "config",
        "scene",
        "solver_mode",
        "solver",
        "mAA@10",
        "inlier_ratio",
        "avg_matches",
        "num_evaluated_pairs",
        "pair_file",
        "pair_count",
        "pair_limit",
        "notes",
    ]
    write_csv(CSV_ROOT / "final_selected_expanded151_lg_test_summary.csv", all_mode_rows(), summary_fields)
    write_csv(CSV_ROOT / "final_selected_expanded151_lg_test_per_scene.csv", per_scene_rows(), per_scene_fields)
    write_report(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate final selected expanded-151 LightGlue test run.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--command", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--runtime-estimate", default="overnight run; depends on online descriptor extraction and 10 held-out scenes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        write_outputs(args)
    else:
        for row in all_mode_rows():
            print(row)


if __name__ == "__main__":
    main()
