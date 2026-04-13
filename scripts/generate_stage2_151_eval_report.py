#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "external" / "glue-factory" / "outputs" / "training" / "stage2_dinov3_lg_151scenes_v1"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "stage2_151scenes_training_report.md"

SCENES = ("sacre_coeur", "reichstag", "st_peters_square")
CONFIGS = (
    ("stage2_151scenes_lg_ft010", 0.10),
    ("stage2_151scenes_lg_ft015", 0.15),
)
PRIMARY_SOLVER = "3p_ours_shift_scale+12"

REFERENCE_ROWS = [
    ("Phase 4b warm-start 60sc (ft=0.10)", {"sacre_coeur": 85.8, "reichstag": 84.1, "st_peters_square": 73.8}),
    ("Phase 4c from-scratch 60sc (ft=0.10)", {"sacre_coeur": 85.3, "reichstag": 82.9, "st_peters_square": 74.9}),
    ("SP+LG (target)", {"sacre_coeur": 86.2, "reichstag": 84.5, "st_peters_square": 75.3}),
]


@dataclass
class SceneSummary:
    maa10: float
    num_pairs: int
    inliers: float
    avg_matches: float
    avg_conf: float
    zero_match_pairs: int


def read_solver_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Solver") == PRIMARY_SOLVER:
                return row
    raise RuntimeError(f"Missing solver row {PRIMARY_SOLVER} in {csv_path}")


def compute_match_stats(matches_dir: Path) -> tuple[float, float, int]:
    counts: list[int] = []
    confs: list[float] = []
    zero_match_pairs = 0

    for match_path in sorted(matches_dir.glob("*.npz")):
        with np.load(match_path) as data:
            mkpts0 = data["mkpts0"]
            scores = data["scores"] if "scores" in data.files else np.zeros((len(mkpts0),), dtype=np.float32)
        counts.append(int(len(mkpts0)))
        pair_conf = float(scores.mean()) if scores.size else 0.0
        confs.append(pair_conf)
        if len(mkpts0) == 0:
            zero_match_pairs += 1

    if not counts:
        raise RuntimeError(f"No match files found in {matches_dir}")

    return float(np.mean(counts)), float(np.mean(confs)), zero_match_pairs


def load_config_summary(config_key: str) -> dict[str, SceneSummary]:
    summaries: dict[str, SceneSummary] = {}
    for scene in SCENES:
        csv_path = PROJECT_ROOT / "output" / "results" / config_key / scene / f"results_{config_key}_{scene}.csv"
        row = read_solver_row(csv_path)
        matches_dir = PROJECT_ROOT / "output" / "matches" / config_key / scene
        avg_matches, avg_conf, zero_match_pairs = compute_match_stats(matches_dir)
        summaries[scene] = SceneSummary(
            maa10=float(row["mAA@10"]),
            num_pairs=int(row["Num_Pairs"]),
            inliers=float(row["Inliers"]),
            avg_matches=avg_matches,
            avg_conf=avg_conf,
            zero_match_pairs=zero_match_pairs,
        )
    return summaries


def average_maa(scene_data: dict[str, SceneSummary]) -> float:
    return float(np.mean([scene_data[scene].maa10 for scene in SCENES]))


def training_window() -> tuple[str, str, str]:
    log_path = TRAIN_DIR / "log.txt"
    lines = log_path.read_text(encoding="utf-8").splitlines()

    start_line = next(line for line in lines if "Will fine-tune from weights" in line)
    end_line = next(line for line in reversed(lines) if "Finished training on process 0." in line)

    ts_pattern = r"\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
    start_ts = datetime.strptime(re.search(ts_pattern, start_line).group(1), "%m/%d/%Y %H:%M:%S")
    end_ts = datetime.strptime(re.search(ts_pattern, end_line).group(1), "%m/%d/%Y %H:%M:%S")

    duration = end_ts - start_ts
    hours, rem = divmod(int(duration.total_seconds()), 3600)
    minutes, _ = divmod(rem, 60)
    return (
        start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        f"{hours}h {minutes:02d}m",
    )


def load_best_checkpoint_metrics() -> tuple[int, dict[str, float]]:
    ckpt = torch.load(TRAIN_DIR / "checkpoint_best.tar", map_location="cpu", weights_only=False)
    return int(ckpt["epoch"]), {k: float(v) for k, v in ckpt["eval"].items()}


def format_scene_row(label: str, values: dict[str, float]) -> str:
    avg = np.mean([values[scene] for scene in SCENES])
    return (
        f"| {label} | {values['sacre_coeur']:.1f} | {values['reichstag']:.1f} | "
        f"{values['st_peters_square']:.1f} | {avg:.1f} |"
    )


def make_comparison_table(config_summaries: dict[str, dict[str, SceneSummary]]) -> str:
    rows = [
        "| Configuration | sacre | reichstag | st_peters | avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in REFERENCE_ROWS[:2]:
        rows.append(format_scene_row(label, values))
    for config_key, threshold in CONFIGS:
        row_label = f"**Stage 2 151sc (ft={threshold:.2f})**"
        values = {scene: config_summaries[config_key][scene].maa10 for scene in SCENES}
        rows.append(format_scene_row(row_label, values))
    rows.append(format_scene_row(*REFERENCE_ROWS[2]))
    return "\n".join(rows)


def make_match_stats_table(config_summaries: dict[str, dict[str, SceneSummary]]) -> str:
    rows = [
        "| Configuration | Scene | pairs | avg_matches | avg_conf | zero_match_pairs | inlier_ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for config_key, threshold in CONFIGS:
        label = f"ft={threshold:.2f}"
        for scene in SCENES:
            summary = config_summaries[config_key][scene]
            rows.append(
                f"| {label} | `{scene}` | {summary.num_pairs} | {summary.avg_matches:.1f} | "
                f"{summary.avg_conf:.3f} | {summary.zero_match_pairs} | {summary.inliers:.1f} |"
            )
    return "\n".join(rows)


def choose_best_config(config_summaries: dict[str, dict[str, SceneSummary]]) -> tuple[str, float]:
    best_key = max((key for key, _ in CONFIGS), key=lambda key: average_maa(config_summaries[key]))
    return best_key, average_maa(config_summaries[best_key])


def assessment_text(best_avg: float) -> tuple[str, str]:
    phase4b = np.mean([85.8, 84.1, 73.8])
    delta = best_avg - phase4b
    if delta > 0.3:
        assessment = f"Yes. The 151-scene Stage 2 setup improved over the 60-scene warm-start by `{delta:+.1f}` mAA points on average."
    elif delta < -0.3:
        assessment = f"No. The 151-scene Stage 2 setup is `{delta:+.1f}` mAA points below the 60-scene warm-start baseline on average."
    else:
        assessment = f"Not meaningfully. The 151-scene Stage 2 setup is essentially tied with the 60-scene warm-start baseline (`{delta:+.1f}` mAA)."

    recommendation = (
        "More epochs alone are unlikely to help much: the training checkpoint selected by validation loss was at epoch `45`, "
        "and the final epochs remained in the same narrow validation-loss band. If the new evaluation does not clearly beat the 60-scene run, "
        "the next gains are more likely to come from validation/threshold calibration, scene selection, or matcher/extractor recipe changes than from simply extending training."
    )
    return assessment, recommendation


def main() -> None:
    config_summaries = {config_key: load_config_summary(config_key) for config_key, _ in CONFIGS}
    start_time, end_time, duration = training_window()
    best_epoch, best_metrics = load_best_checkpoint_metrics()
    best_key, best_avg = choose_best_config(config_summaries)
    assessment, recommendation = assessment_text(best_avg)

    best_threshold = next(threshold for key, threshold in CONFIGS if key == best_key)

    report = f"""# Stage 2 151-Scene Training Report

Date: 2026-04-13

Status:
- 151-scene MegaDepth Stage 2 training completed successfully
- PhotoTourism evaluation completed for `ft=0.10` and `ft=0.15`
- Headline metric below is calibrated `mAA@10` for solver `{PRIMARY_SOLVER}`

## Training Summary

- Hardware: `4 x NVIDIA RTX A5000`
- Training output: [stage2_dinov3_lg_151scenes_v1]({TRAIN_DIR})
- Start: `{start_time} CEST`
- End: `{end_time} CEST`
- Duration: `{duration}`
- Best checkpoint: [checkpoint_best.tar]({TRAIN_DIR / 'checkpoint_best.tar'})
- Best checkpoint selected by validation loss at epoch: `{best_epoch}`
- Best training validation metrics:
  - `loss/total = {best_metrics['loss/total']:.4f}`
  - `accuracy = {best_metrics['accuracy']:.4f}`
  - `average_precision = {best_metrics['average_precision']:.4f}`
  - `match_precision = {best_metrics['match_precision']:.4f}`
  - `match_recall = {best_metrics['match_recall']:.4f}`

## Config Summary

- Optimizer: `Adam`, `weight_decay=0.0`
- Learning rate: `1e-4`
- LR schedule: constant for first `9` epochs, then exponential decay `gamma=0.95`
- Epochs: `50`
- Batch size: `32` global over `4` GPUs
- Keypoints per image: `2048`
- Matcher: `LightGlue`, `9` layers, `4` heads, `input_dim=256`
- Loss: `NLL`, `gamma=1.0`, `nll_balancing=0.5`
- GT thresholds: positive `3px`, negative `5px`
- Descriptor pipeline:
  - SuperPoint keypoints
  - DINOv3 block `16`
  - DIFT `t=0`, `up_ft_index=2`, `ensemble_size=8` at evaluation time
  - projection head [best.pt]({PROJECT_ROOT / 'experiments' / 'phase2_projection_wide' / 'best.pt'})
- Fine-tuning warm-start: [stage1_dinov3_lg_v8/checkpoint_best.tar]({PROJECT_ROOT / 'external' / 'glue-factory' / 'outputs' / 'training' / 'stage1_dinov3_lg_v8' / 'checkpoint_best.tar'})
- Training cache: [gf_cache_stage2_151]({PROJECT_ROOT / 'data' / 'gf_cache_stage2_151'})

## Downstream Results

{make_comparison_table(config_summaries)}

Best new configuration by average `mAA@10`:
- `{best_key}` with `ft={best_threshold:.2f}`
- average `mAA@10 = {best_avg:.1f}`

## Match Statistics

{make_match_stats_table(config_summaries)}

## Assessment

- {assessment}
- Best new run compared to Phase 4c from-scratch 60-scene baseline (`81.0` avg): `{best_avg - 81.0:+.1f}` points.
- Best new run compared to SP+LG target (`82.0` avg): `{best_avg - 82.0:+.1f}` points.

## Recommendation

- {recommendation}

## Raw Artifacts

- Training log: [log.txt]({TRAIN_DIR / 'log.txt'})
- Training TensorBoard: [events.out.tfevents.1775913536.marr.4044889.0]({TRAIN_DIR / 'events.out.tfevents.1775913536.marr.4044889.0'})
- `ft=0.10` results root: [{PROJECT_ROOT / 'output' / 'results' / 'stage2_151scenes_lg_ft010'}]({PROJECT_ROOT / 'output' / 'results' / 'stage2_151scenes_lg_ft010'})
- `ft=0.15` results root: [{PROJECT_ROOT / 'output' / 'results' / 'stage2_151scenes_lg_ft015'}]({PROJECT_ROOT / 'output' / 'results' / 'stage2_151scenes_lg_ft015'})
- `ft=0.10` matches root: [{PROJECT_ROOT / 'output' / 'matches' / 'stage2_151scenes_lg_ft010'}]({PROJECT_ROOT / 'output' / 'matches' / 'stage2_151scenes_lg_ft010'})
- `ft=0.15` matches root: [{PROJECT_ROOT / 'output' / 'matches' / 'stage2_151scenes_lg_ft015'}]({PROJECT_ROOT / 'output' / 'matches' / 'stage2_151scenes_lg_ft015'})
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")
    print(f"Best config: {best_key} (avg mAA@10={best_avg:.1f})")


if __name__ == "__main__":
    main()
