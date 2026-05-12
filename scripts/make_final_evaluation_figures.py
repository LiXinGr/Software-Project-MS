#!/usr/bin/env python3
"""Generate final-evaluation thesis figures from existing CSV artifacts only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "output_v2" / "csv"
FIG_DIR = ROOT / "output_v2" / "figures"

ALL_MODES_CSV = CSV_DIR / "final_figure_all_modes_accuracy.csv"
PER_SCENE_CSV = CSV_DIR / "final_test_per_scene_calibrated.csv"
RUNTIME_BREAKDOWN_CSV = CSV_DIR / "final_selected_runtime_breakdown_table.csv"
ALL_MODE_FALLBACK_CSV = CSV_DIR / "chapter9_test_eval.csv"

METHODS = ["Final selected", "SP+LG", "RoMa", "RoMaV2"]
MODES = ["calibrated", "shared_focal", "varying_focal"]
MODE_LABELS = {
    "calibrated": "calibrated",
    "shared_focal": "shared focal",
    "varying_focal": "varying focal",
}
MODE_COLORS = {
    "calibrated": "#1f77b4",
    "shared_focal": "#ff7f0e",
    "varying_focal": "#2ca02c",
}
MODE_HATCHES = {
    "calibrated": "",
    "shared_focal": "//",
    "varying_focal": "xx",
}
METHOD_COLORS = {
    "Final selected": "#1f77b4",
    "SP+LG": "#7f7f7f",
    "RoMa": "#ff7f0e",
    "RoMaV2": "#2ca02c",
}

FINAL_SCENES = [
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
SCENE_LABELS = {
    "british_museum": "British Museum",
    "florence_cathedral_side": "Florence",
    "lincoln_memorial_statue": "Lincoln",
    "milan_cathedral": "Milan",
    "mount_rushmore": "Rushmore",
    "piazza_san_marco": "San Marco",
    "sagrada_familia": "Sagrada",
    "st_pauls_cathedral": "St Pauls",
    "taj_mahal": "Taj Mahal",
    "temple_nara_japan": "Nara",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    return float(value.strip())


def normalize_method(method: str, config_name: str = "") -> str | None:
    if method.startswith("Final selected"):
        return "Final selected"
    if method == "SP+LG" or config_name == "test_splg":
        return "SP+LG"
    if method == "RoMa" or config_name == "test_roma":
        return "RoMa"
    if method == "RoMaV2" or config_name == "test_romav2":
        return "RoMaV2"
    return None


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", dy: float = 1.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def load_all_mode_values() -> tuple[dict[tuple[str, str], float], bool]:
    values: dict[tuple[str, str], float] = {}
    for row in read_rows(ALL_MODES_CSV):
        method = normalize_method(row.get("method", ""), row.get("config_name", ""))
        mode = row.get("solver_mode", "")
        if method in METHODS and mode in MODES and row.get("mAA10", "").strip():
            values[(method, mode)] = to_float(row["mAA10"])

    used_fallback = False
    missing = [
        (method, mode)
        for method in METHODS
        for mode in MODES
        if (method, mode) not in values
    ]
    if missing and ALL_MODE_FALLBACK_CSV.exists():
        config_to_method = {
            "test_splg": "SP+LG",
            "test_roma": "RoMa",
            "test_romav2": "RoMaV2",
        }
        buckets: dict[tuple[str, str], list[float]] = {}
        for row in read_rows(ALL_MODE_FALLBACK_CSV):
            method = config_to_method.get(row.get("config_key", ""))
            mode = row.get("solver_mode", "")
            scene = row.get("scene", "")
            if method not in METHODS or mode not in MODES or scene not in FINAL_SCENES:
                continue
            buckets.setdefault((method, mode), []).append(to_float(row["mAA@10"]))
        for key, vals in buckets.items():
            if key not in values and len(vals) == len(FINAL_SCENES):
                values[key] = sum(vals) / len(vals)
                used_fallback = True

    still_missing = [
        (method, mode)
        for method in METHODS
        for mode in MODES
        if (method, mode) not in values
    ]
    if still_missing:
        raise RuntimeError(f"Missing all-mode accuracy values: {still_missing}")
    return values, used_fallback


def plot_all_modes() -> tuple[list[Path], dict[tuple[str, str], float], bool]:
    values, used_fallback = load_all_mode_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    x = list(range(len(METHODS)))
    width = 0.23
    offsets = {
        "calibrated": -width,
        "shared_focal": 0.0,
        "varying_focal": width,
    }

    for mode in MODES:
        heights = [values[(method, mode)] for method in METHODS]
        bars = ax.bar(
            [pos + offsets[mode] for pos in x],
            heights,
            width=width,
            color=MODE_COLORS[mode],
            edgecolor="#303030",
            linewidth=0.8,
            hatch=MODE_HATCHES[mode],
            label=MODE_LABELS[mode],
        )
        label_bars(ax, bars, dy=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "Final selected":
            tick.set_fontweight("bold")
    ax.set_ylim(0, 92)
    ax.set_ylabel("mAA@10")
    ax.set_title("Final-test accuracy across solver modes", pad=16)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.tight_layout()
    return save_figure(fig, "fig_final_test_all_modes_accuracy"), values, used_fallback


def load_per_scene_values() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for row in read_rows(PER_SCENE_CSV):
        method = normalize_method(row.get("method_name", ""), row.get("config_name", ""))
        scene = row.get("scene", "")
        val = row.get("calibrated_mAA10", "").strip()
        if method in METHODS and scene in FINAL_SCENES and val:
            values[(method, scene)] = to_float(val)

    missing = [
        (method, scene)
        for method in METHODS
        for scene in FINAL_SCENES
        if (method, scene) not in values
    ]
    if missing:
        raise RuntimeError(f"Missing per-scene calibrated values: {missing}")
    return values


def plot_per_scene() -> tuple[list[Path], dict[tuple[str, str], float]]:
    values = load_per_scene_values()
    fig, ax = plt.subplots(figsize=(13.8, 6.6))

    x = list(range(len(FINAL_SCENES)))
    width = 0.19
    offsets = {
        "Final selected": -1.5 * width,
        "SP+LG": -0.5 * width,
        "RoMa": 0.5 * width,
        "RoMaV2": 1.5 * width,
    }

    for method in METHODS:
        heights = [values[(method, scene)] for scene in FINAL_SCENES]
        bars = ax.bar(
            [pos + offsets[method] for pos in x],
            heights,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#303030",
            linewidth=0.6,
            label=method,
        )
        if method in {"Final selected", "SP+LG"}:
            label_bars(ax, bars, dy=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABELS[scene] for scene in FINAL_SCENES], rotation=42, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Calibrated mAA@10")
    ax.set_title("Per-scene calibrated final-test accuracy", pad=16)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout()
    return save_figure(fig, "fig_final_test_per_scene_calibrated"), values


def plot_gain_vs_splg(per_scene_values: dict[tuple[str, str], float]) -> tuple[list[Path], float, list[str]]:
    diffs = [
        per_scene_values[("Final selected", scene)] - per_scene_values[("SP+LG", scene)]
        for scene in FINAL_SCENES
    ]
    mean_diff = sum(diffs) / len(diffs)
    negative_scenes = [scene for scene, diff in zip(FINAL_SCENES, diffs) if diff < 0]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = list(range(len(FINAL_SCENES)))
    colors = ["#1f77b4" if diff >= 0 else "#9a9a9a" for diff in diffs]
    bars = ax.bar(x, diffs, width=0.68, color=colors, edgecolor="#303030", linewidth=0.7)

    ax.axhline(0, color="#202020", linewidth=1.0)
    ax.axhline(mean_diff, color="#d62728", linestyle="--", linewidth=1.8)
    ax.text(
        len(FINAL_SCENES) - 0.45,
        mean_diff + 0.08,
        f"mean {mean_diff:+.2f}",
        color="#d62728",
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

    for bar, diff in zip(bars, diffs):
        y = diff + 0.12 if diff >= 0 else diff - 0.18
        va = "bottom" if diff >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{diff:+.1f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABELS[scene] for scene in FINAL_SCENES], rotation=42, ha="right")
    ax.set_ylabel("mAA@10 difference vs SP+LG")
    ax.set_title("Per-scene calibrated gain over SP+LG", pad=14)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    y_min = min(-1.5, min(diffs) - 0.8)
    y_max = max(3.5, max(diffs) + 0.9)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    return save_figure(fig, "fig_final_test_gain_vs_splg_per_scene"), mean_diff, negative_scenes


def plot_runtime_breakdown() -> tuple[list[Path], list[tuple[str, float]], float, float]:
    rows = read_rows(RUNTIME_BREAKDOWN_CSV)
    component_labels = {
        "DIFT feature extraction": "DIFT feature extraction",
        "DINOv3 forward pass": "DINOv3 forward pass",
        "SuperPoint keypoint extraction": "SuperPoint keypoint extraction",
        "benchmark packing / depth sampling": "benchmark packing / depth sampling",
        "RePoseD calibrated solver": "RePoseD calibrated solver",
        "image loading + preprocessing": "image loading + preprocessing",
        "LightGlue matching": "LightGlue matching",
        "fusion + projection head": "fusion + projection head",
        "DINOv3/DIFT descriptor sampling": "Descriptor sampling",
    }

    components: list[tuple[str, float]] = []
    total_online = 1680.314480
    total_cached = 66.664759
    for row in rows:
        label = row.get("component_label", "")
        value = row.get("pair_equivalent_mean_ms", "").strip()
        if not value:
            continue
        if label == "total online":
            total_online = to_float(value)
        elif label == "total cached descriptors":
            total_cached = to_float(value)
        elif label in component_labels:
            components.append((component_labels[label], to_float(value)))

    wanted = {
        "DIFT feature extraction",
        "DINOv3 forward pass",
        "SuperPoint keypoint extraction",
        "benchmark packing / depth sampling",
        "RePoseD calibrated solver",
        "image loading + preprocessing",
        "LightGlue matching",
        "fusion + projection head",
        "Descriptor sampling",
    }
    missing = sorted(wanted - {label for label, _ in components})
    if missing:
        raise RuntimeError(f"Missing runtime components: {missing}")

    components.sort(key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in components]
    values = [value for _, value in components]
    colors = [
        "#ff7f0e" if label == "DIFT feature extraction" else
        "#1f77b4" if label == "DINOv3 forward pass" else
        "#9a9a9a"
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=colors, edgecolor="#303030", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Milliseconds per pair")
    ax.set_title(
        "Runtime breakdown of the final selected method\n"
        "Feature extraction dominates online runtime.",
        pad=14,
    )
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=11,
        )

    legend_handles = [
        Patch(facecolor="#ff7f0e", edgecolor="#303030", label="DIFT feature extraction"),
        Patch(facecolor="#1f77b4", edgecolor="#303030", label="DINOv3 forward pass"),
        Patch(facecolor="#9a9a9a", edgecolor="#303030", label="other components"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.text(
        0.5,
        0.02,
        f"Total online: {total_online:.1f} ms/pair    Cached descriptors: {total_cached:.1f} ms/pair",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return save_figure(fig, "fig_final_selected_runtime_breakdown"), components, total_online, total_cached


def main() -> None:
    configure_style()

    saved_paths: list[Path] = []
    all_mode_paths, all_mode_values, used_fallback = plot_all_modes()
    saved_paths.extend(all_mode_paths)

    per_scene_paths, per_scene_values = plot_per_scene()
    saved_paths.extend(per_scene_paths)

    gain_paths, mean_gain, negative_scenes = plot_gain_vs_splg(per_scene_values)
    saved_paths.extend(gain_paths)

    runtime_paths, runtime_components, total_online, total_cached = plot_runtime_breakdown()
    saved_paths.extend(runtime_paths)

    expected_all_modes = {
        ("Final selected", "calibrated"): 74.3,
        ("Final selected", "shared_focal"): 62.1,
        ("Final selected", "varying_focal"): 37.7,
        ("SP+LG", "calibrated"): 72.8,
        ("SP+LG", "shared_focal"): 64.6,
        ("SP+LG", "varying_focal"): 41.7,
        ("RoMa", "calibrated"): 81.6,
        ("RoMa", "shared_focal"): 74.0,
        ("RoMa", "varying_focal"): 53.1,
        ("RoMaV2", "calibrated"): 81.5,
        ("RoMaV2", "shared_focal"): 73.0,
        ("RoMaV2", "varying_focal"): 51.3,
    }
    for key, expected in expected_all_modes.items():
        actual = round(all_mode_values[key], 1)
        if actual != expected:
            raise RuntimeError(f"All-mode validation failed for {key}: {actual} != {expected}")

    if abs(mean_gain - 1.507511) > 1e-5:
        raise RuntimeError(f"Mean gain validation failed: {mean_gain:.6f}")
    if negative_scenes != ["piazza_san_marco"]:
        raise RuntimeError(f"Negative-scene validation failed: {negative_scenes}")
    if runtime_components[0][0] != "DIFT feature extraction" or runtime_components[1][0] != "DINOv3 forward pass":
        raise RuntimeError("Runtime breakdown validation failed: DIFT/DINOv3 are not the top two components")

    print("Saved figures:")
    for path in saved_paths:
        print(path.relative_to(ROOT))
    if used_fallback:
        print(
            "Note: final_figure_all_modes_accuracy.csv lacks baseline shared/varying rows; "
            "missing all-mode baseline values were filled from output_v2/csv/chapter9_test_eval.csv."
        )
    print("Validation:")
    print("  all-mode values match requested one-decimal values")
    print(f"  mean gain over SP+LG: {mean_gain:+.6f}; negative scene(s): {', '.join(negative_scenes)}")
    print(
        "  runtime top components: "
        f"{runtime_components[0][0]} ({runtime_components[0][1]:.1f} ms), "
        f"{runtime_components[1][0]} ({runtime_components[1][1]:.1f} ms)"
    )
    print(f"  totals: online {total_online:.1f} ms/pair, cached descriptors {total_cached:.1f} ms/pair")


if __name__ == "__main__":
    main()
