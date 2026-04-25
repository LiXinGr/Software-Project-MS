#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "output" / "results"
SUMMARY_PATH = RESULTS_ROOT / "phototourism_test_summary.md"
SUMMARY_CSV_PATH = RESULTS_ROOT / "phototourism_test_summary.csv"
SUMMARY_JSON_PATH = RESULTS_ROOT / "phototourism_test_summary.json"

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

METHODS = [
    ("test_dinov3_mnn", "DINOv3 MNN", 63.6),
    ("test_dift_mnn", "DIFT MNN", 66.5),
    ("test_ours_151sc_ft010", "Ours (frozen 151sc)", 81.4),
    ("test_splg", "SP+LG", 81.7),
    ("test_roma", "RoMa", 87.1),
    ("test_romav2", "RoMaV2", 86.4),
]

TARGET_SOLVER = "3p_ours_shift_scale+12"
TARGET_EXP_TYPE = "calibrated"


def read_scene_score(method_key: str, scene: str) -> float | None:
    csv_path = RESULTS_ROOT / method_key / scene / f"results_{method_key}_{scene}.csv"
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Solver") == TARGET_SOLVER and row.get("Exp.Type") == TARGET_EXP_TYPE:
                try:
                    return float(row["mAA@10"])
                except (KeyError, ValueError):
                    return None
    return None


def read_runtime(method_key: str, scene: str) -> float | None:
    runtime_path = RESULTS_ROOT / method_key / scene / "runtime_summary.json"
    if not runtime_path.exists():
        return None
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        return float(payload["wall_seconds"])
    except Exception:
        return None


def read_runtime_payload(method_key: str, scene: str) -> dict | None:
    runtime_path = RESULTS_ROOT / method_key / scene / "runtime_summary.json"
    if not runtime_path.exists():
        return None
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def fmt_score(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"


def fmt_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes = seconds / 60.0
    if minutes >= 120.0:
        return f"{minutes / 60.0:.2f} h"
    return f"{minutes:.1f} min"


def collect_summary_rows() -> list[dict]:
    rows: list[dict] = []
    for method_key, label, val_avg in METHODS:
        scores = [read_scene_score(method_key, scene) for scene in SCENES]
        runtime_payloads = [read_runtime_payload(method_key, scene) for scene in SCENES]
        present_scores = [score for score in scores if score is not None]
        stage_names = ["wall_seconds", "match_seconds", "pack_seconds", "eval_seconds", "csv_seconds"]
        stage_totals: dict[str, float | None] = {}
        stage_avgs: dict[str, float | None] = {}
        for stage_name in stage_names:
            values = [
                float(payload[stage_name])
                for payload in runtime_payloads
                if payload is not None and stage_name in payload
            ]
            stage_totals[stage_name] = sum(values) if values else None
            stage_avgs[stage_name] = (sum(values) / len(values)) if values else None
        test_avg = sum(present_scores) / len(present_scores) if present_scores else None
        delta = None if test_avg is None else test_avg - val_avg
        failed_scenes = [scene for scene, score in zip(SCENES, scores) if score is None]
        row = {
            "method_key": method_key,
            "method_label": label,
            "validation_avg_3scenes": val_avg,
            "test_avg_10scenes": test_avg,
            "delta_vs_validation": delta,
            "completed_scenes": sum(score is not None for score in scores),
            "total_scenes": len(SCENES),
            "total_runtime_seconds": stage_totals["wall_seconds"],
            "avg_runtime_seconds": stage_avgs["wall_seconds"],
            "total_match_seconds": stage_totals["match_seconds"],
            "avg_match_seconds": stage_avgs["match_seconds"],
            "total_pack_seconds": stage_totals["pack_seconds"],
            "avg_pack_seconds": stage_avgs["pack_seconds"],
            "total_eval_seconds": stage_totals["eval_seconds"],
            "avg_eval_seconds": stage_avgs["eval_seconds"],
            "total_csv_seconds": stage_totals["csv_seconds"],
            "avg_csv_seconds": stage_avgs["csv_seconds"],
            "failed_scenes": failed_scenes,
            "per_scene_mAA10": {scene: score for scene, score in zip(SCENES, scores)},
            "per_scene_runtime": {
                scene: runtime_payloads[idx]
                for idx, scene in enumerate(SCENES)
                if runtime_payloads[idx] is not None
            },
        }
        rows.append(row)
    return rows


def build_tables(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# PhotoTourism Test Summary")
    lines.append("")
    lines.append("## Table 1: Full per-scene results")
    lines.append("")
    lines.append("| Method | british | florence | lincoln | milan | rushmore | piazza | sagrada | st_pauls | taj | temple | AVG |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        label = row["method_label"]
        scores = [row["per_scene_mAA10"][scene] for scene in SCENES]
        avg = row["test_avg_10scenes"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                *(fmt_score(score) for score in scores),
                fmt_score(avg),
            )
        )

    lines.append("")
    lines.append("## Table 2: Test vs Validation comparison")
    lines.append("")
    lines.append("| Method | Val Avg (3 scenes) | Test Avg (10 scenes) | Delta |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        label = row["method_label"]
        val_avg = row["validation_avg_3scenes"]
        test_avg = row["test_avg_10scenes"]
        delta = row["delta_vs_validation"]
        delta_str = "?" if delta is None else f"{delta:+.1f}"
        lines.append(f"| {label} | {val_avg:.1f} | {fmt_score(test_avg)} | {delta_str} |")

    lines.append("")
    lines.append("## Runtime summary")
    lines.append("")
    lines.append("| Method | Completed scenes | Match total | Pack total | Eval total | CSV total | Total wall time | Avg / scene | Failed scenes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        label = row["method_label"]
        completed = row["completed_scenes"]
        total_runtime = row["total_runtime_seconds"]
        avg_runtime = row["avg_runtime_seconds"]
        total_match = row["total_match_seconds"]
        total_pack = row["total_pack_seconds"]
        total_eval = row["total_eval_seconds"]
        total_csv = row["total_csv_seconds"]
        failed_scenes = row["failed_scenes"]
        failed_str = ", ".join(failed_scenes) if failed_scenes else "-"
        lines.append(
            f"| {label} | {completed}/{len(SCENES)} | {fmt_runtime(total_match)} | {fmt_runtime(total_pack)} | {fmt_runtime(total_eval)} | {fmt_runtime(total_csv)} | {fmt_runtime(total_runtime)} | {fmt_runtime(avg_runtime)} | {failed_str} |"
        )

    return "\n".join(lines) + "\n"


def write_summary_csv(rows: list[dict]) -> None:
    fieldnames = [
        "method_key",
        "method_label",
        "validation_avg_3scenes",
        "test_avg_10scenes",
        "delta_vs_validation",
        "completed_scenes",
        "total_scenes",
        "total_runtime_seconds",
        "avg_runtime_seconds",
        "total_match_seconds",
        "avg_match_seconds",
        "total_pack_seconds",
        "avg_pack_seconds",
        "total_eval_seconds",
        "avg_eval_seconds",
        "total_csv_seconds",
        "avg_csv_seconds",
        "failed_scenes",
        *SCENES,
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "method_key": row["method_key"],
                "method_label": row["method_label"],
                "validation_avg_3scenes": row["validation_avg_3scenes"],
                "test_avg_10scenes": row["test_avg_10scenes"],
                "delta_vs_validation": row["delta_vs_validation"],
                "completed_scenes": row["completed_scenes"],
                "total_scenes": row["total_scenes"],
                "total_runtime_seconds": row["total_runtime_seconds"],
                "avg_runtime_seconds": row["avg_runtime_seconds"],
                "total_match_seconds": row["total_match_seconds"],
                "avg_match_seconds": row["avg_match_seconds"],
                "total_pack_seconds": row["total_pack_seconds"],
                "avg_pack_seconds": row["avg_pack_seconds"],
                "total_eval_seconds": row["total_eval_seconds"],
                "avg_eval_seconds": row["avg_eval_seconds"],
                "total_csv_seconds": row["total_csv_seconds"],
                "avg_csv_seconds": row["avg_csv_seconds"],
                "failed_scenes": ",".join(row["failed_scenes"]),
            }
            flat.update(row["per_scene_mAA10"])
            writer.writerow(flat)


def main() -> None:
    rows = collect_summary_rows()
    summary = build_tables(rows)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    write_summary_csv(rows)
    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(summary, end="")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved summary CSV to {SUMMARY_CSV_PATH}")
    print(f"Saved summary JSON to {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
