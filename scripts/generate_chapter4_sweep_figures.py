#!/usr/bin/env python3
"""Generate Chapter 4 sweep diagrams from stored experiment CSVs."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "output" / "results"
FIGURES_ROOT = PROJECT_ROOT / "figures"
TARGET_SOLVER = "3p_ours_shift_scale+12"


@dataclass(frozen=True)
class SweepPoint:
    x: int
    maa: float
    inliers: float
    csv_path: Path


def read_target_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Solver") == TARGET_SOLVER and row.get("Exp.Type") == "calibrated":
                return row
    raise ValueError(f"Missing calibrated {TARGET_SOLVER} row in {csv_path}")


def latest_csv(directory: Path, prefix: str) -> Path:
    matches = sorted(directory.glob(prefix))
    if not matches:
        raise FileNotFoundError(f"No CSV matching {prefix} in {directory}")
    return matches[-1]


def load_dinov3_layer_sweep() -> list[SweepPoint]:
    points = []
    for path in sorted(RESULTS_ROOT.glob("phase1_layer_study_b*/sacre_coeur")):
        match = re.search(r"phase1_layer_study_b(\d+)", str(path))
        if not match:
            continue
        block = int(match.group(1))
        csv_path = latest_csv(path, "results_dinov3_sacre_coeur_*.csv")
        row = read_target_row(csv_path)
        points.append(
            SweepPoint(
                x=block,
                maa=float(row["mAA@10"]),
                inliers=float(row["Inliers"]),
                csv_path=csv_path,
            )
        )
    return sorted(points, key=lambda point: point.x)


def load_dift_timestep_sweep() -> list[SweepPoint]:
    points = []
    for path in sorted(RESULTS_ROOT.glob("phase1_dift_temporal_t*_up2_s768/sacre_coeur")):
        match = re.search(r"phase1_dift_temporal_t(\d+)_", str(path))
        if not match:
            continue
        timestep = int(match.group(1))
        csv_path = latest_csv(path, "results_dift_sacre_coeur_*.csv")
        row = read_target_row(csv_path)
        points.append(
            SweepPoint(
                x=timestep,
                maa=float(row["mAA@10"]),
                inliers=float(row["Inliers"]),
                csv_path=csv_path,
            )
        )
    return sorted(points, key=lambda point: point.x)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, figure_stem: Path) -> None:
    svg_path = figure_stem.with_suffix(".svg")
    png_path = figure_stem.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {svg_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {png_path.relative_to(PROJECT_ROOT)}")


def plot_dinov3_layer_sweep(points: list[SweepPoint]) -> Path:
    blocks = [point.x for point in points]
    maa = [point.maa for point in points]
    inliers = [point.inliers for point in points]
    point_by_block = {point.x: point for point in points}
    baseline = point_by_block[23].maa

    fig, ax = plt.subplots(figsize=(5.5, 3.45))
    blue = "#2563eb"
    red = "#dc2626"

    line_maa, = ax.plot(
        blocks,
        maa,
        color=blue,
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        label="mAA@10",
    )
    ax.set_xlabel("DINOv3 block number")
    ax.set_ylabel("mAA@10", color=blue)
    ax.tick_params(axis="y", labelcolor=blue)
    ax.set_xticks(blocks)
    ax.set_xlim(3, 24)
    ax.set_ylim(15, 80)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    depth_labels = ["17%", "33%", "50%", "67%", "83%", "100%"]
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(blocks)
    secax.set_xticklabels(depth_labels)
    secax.set_xlabel("Network depth")
    secax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    line_inliers, = ax2.plot(
        blocks,
        inliers,
        color=red,
        marker="^",
        linewidth=1.8,
        markersize=5.0,
        linestyle="--",
        label="Inlier ratio",
    )
    ax2.set_ylabel("Inlier ratio (%)", color=red)
    ax2.tick_params(axis="y", labelcolor=red)
    ax2.set_ylim(0, 75)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#cbd5e1")

    best_block = 16
    ax.axvline(best_block, color="#6b7280", linestyle="--", linewidth=1.2, zorder=0)
    ax.annotate(
        "Best (block 16)",
        xy=(best_block, point_by_block[best_block].maa),
        xytext=(15.3, 78),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 0.9},
        color="#111827",
    )

    ax.annotate(
        "High inliers,\nlow accuracy",
        xy=(12, point_by_block[12].maa),
        xytext=(8.3, 30),
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": "#cbd5e1", "lw": 0.8},
        color="#111827",
    )

    baseline_line = ax.axhline(
        baseline,
        color="#64748b",
        linestyle="--",
        linewidth=1.1,
        label="Baseline (last block)",
    )
    ax.text(3.25, baseline + 1.0, f"Baseline (last block) {baseline:.1f}", color="#475569", va="bottom")

    ax.set_title("DINOv3 Layer Sweep on sacre_coeur", pad=11)
    ax.legend([line_maa, line_inliers, baseline_line], ["mAA@10", "Inlier ratio", "Baseline (last block)"], loc="upper right", frameon=True)
    fig.tight_layout()

    path = FIGURES_ROOT / "dinov3_layer_sweep"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".svg")


def plot_ch4_layer_sweep(points: list[SweepPoint]) -> Path:
    blocks = [point.x for point in points]
    maa = [point.maa for point in points]
    inliers = [point.inliers for point in points]
    point_by_block = {point.x: point for point in points}

    fig, ax = plt.subplots(figsize=(5.65, 3.35))
    blue = "#2563eb"
    orange = "#f97316"
    gray = "#6b7280"

    for block, label in [(12, "Inlier peak"), (16, "Pose peak")]:
        ax.axvline(block, color=gray, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.72, zorder=0)
        ax.text(block, 78.0, label, ha="center", va="top", color="#374151", fontsize=7.8)

    line_maa, = ax.plot(
        blocks,
        maa,
        color=blue,
        marker="o",
        linewidth=2.2,
        markersize=4.8,
        label="mAA@10",
        zorder=3,
    )
    ax.set_xlabel("DINOv3 block number")
    ax.set_ylabel("mAA@10", color=blue)
    ax.tick_params(axis="y", labelcolor=blue)
    ax.set_xticks(blocks)
    ax.set_xlim(3, 24)
    ax.set_ylim(20, 80)
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80])
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    ax2 = ax.twinx()
    line_inliers, = ax2.plot(
        blocks,
        inliers,
        color=orange,
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        linestyle="--",
        label="MNN inlier ratio",
        zorder=3,
    )
    ax2.set_ylabel("MNN inlier ratio (%)", color=orange)
    ax2.tick_params(axis="y", labelcolor=orange)
    ax2.set_ylim(10, 70)
    ax2.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#cbd5e1")

    ax2.annotate(
        f"{point_by_block[12].inliers:.1f}% inliers",
        xy=(12, point_by_block[12].inliers),
        xytext=(9.25, 63.0),
        textcoords="data",
        color="#9a3412",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#9a3412", "lw": 0.9},
    )
    ax.annotate(
        f"{point_by_block[12].maa:.1f} mAA",
        xy=(12, point_by_block[12].maa),
        xytext=(10.0, 33.0),
        textcoords="data",
        color="#1d4ed8",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 0.9},
    )
    ax.annotate(
        f"{point_by_block[16].maa:.1f} mAA",
        xy=(16, point_by_block[16].maa),
        xytext=(18.0, 75.0),
        textcoords="data",
        color="#1d4ed8",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 0.9},
    )
    ax2.annotate(
        f"{point_by_block[16].inliers:.1f}% inliers",
        xy=(16, point_by_block[16].inliers),
        xytext=(18.8, 46.5),
        textcoords="data",
        color="#9a3412",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#9a3412", "lw": 0.9},
    )

    ax.set_title("DINOv3 block sweep: inliers and pose accuracy diverge", pad=8)
    ax.legend([line_maa, line_inliers], ["mAA@10", "MNN inlier ratio"], loc="lower right", frameon=True)
    fig.tight_layout()

    path = FIGURES_ROOT / "ch4_layer_sweep"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".png")


def plot_dift_timestep_sweep(points: list[SweepPoint], dino_block16_maa: float) -> Path:
    timesteps = [point.x for point in points]
    maa = [point.maa for point in points]
    point_by_t = {point.x: point for point in points}

    fig, ax = plt.subplots(figsize=(5.5, 3.35))
    blue = "#2563eb"

    ax.axvspan(0, 30, color="#bfdbfe", alpha=0.36, zorder=0)
    ax.text(
        15,
        max(maa) + 0.72,
        "Geometric regime\n(plateau)",
        ha="center",
        va="top",
        fontsize=8,
        color="#1e40af",
    )

    ax.plot(timesteps, maa, color=blue, marker="o", linewidth=2.0, markersize=4.0, label="DIFT mAA@10")
    ax.set_xlabel("DIFT timestep t")
    ax.set_ylabel("mAA@10")
    ax.set_title("DIFT Timestep Sweep on sacre_coeur", pad=8)
    ax.set_xlim(min(timesteps) - 4, max(timesteps) + 9)
    ax.set_ylim(min(min(maa), dino_block16_maa) - 1.2, max(maa) + 1.0)
    major_timesteps = [0, 30, 70, 125, 175, 225, 261]
    ax.set_xticks(major_timesteps)
    ax.set_xticks(timesteps, minor=True)
    ax.tick_params(axis="x", which="major", rotation=0)
    ax.tick_params(axis="x", which="minor", length=2.5, color="#64748b")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    ref_line = ax.axhline(
        dino_block16_maa,
        color="#64748b",
        linestyle="--",
        linewidth=1.1,
        label="DINOv3 block 16",
    )
    ax.text(timesteps[-1] + 4, dino_block16_maa + 0.08, "DINOv3 block 16", color="#475569", va="bottom", ha="right")

    if 261 in point_by_t:
        semantic = point_by_t[261]
        ax.scatter([semantic.x], [semantic.maa], color="#dc2626", marker="*", s=95, zorder=5, label="Original config")
        ax.annotate(
            "Original config\n(semantic)",
            xy=(semantic.x, semantic.maa),
            xytext=(197, semantic.maa + 2.25),
            arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 0.9},
            color="#991b1b",
            ha="left",
        )

    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    path = FIGURES_ROOT / "dift_timestep_sweep"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".svg")


def main() -> None:
    configure_style()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    dinov3_points = load_dinov3_layer_sweep()
    dift_points = load_dift_timestep_sweep()

    expected_blocks = [4, 8, 12, 16, 20, 23]
    found_blocks = [point.x for point in dinov3_points]
    if found_blocks != expected_blocks:
        raise RuntimeError(f"Expected DINOv3 blocks {expected_blocks}, found {found_blocks}")

    dino_block16 = next(point.maa for point in dinov3_points if point.x == 16)

    print("DINOv3 layer sweep:")
    for point in dinov3_points:
        print(f"  block={point.x:2d} mAA@10={point.maa:5.2f} inliers={point.inliers:5.2f} source={point.csv_path.relative_to(PROJECT_ROOT)}")
    print("DIFT timestep sweep:")
    for point in dift_points:
        print(f"  t={point.x:3d} mAA@10={point.maa:5.2f} source={point.csv_path.relative_to(PROJECT_ROOT)}")

    plot_dinov3_layer_sweep(dinov3_points)
    plot_ch4_layer_sweep(dinov3_points)
    plot_dift_timestep_sweep(dift_points, dino_block16)


if __name__ == "__main__":
    main()
