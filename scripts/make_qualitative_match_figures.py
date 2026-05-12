#!/usr/bin/env python3
"""Create qualitative final-test match visualizations from saved artifacts only.

This script does not run feature extraction, matching, model inference, or
evaluation. It reads existing image files and saved .npz correspondence files.
Images are resized only for visualization; saved match coordinates are scaled
to the visualization canvas after loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "phototourism"
FIG_DIR = ROOT / "output_v2" / "figures"
REPORT_PATH = ROOT / "output_v2" / "reports" / "qualitative_match_visualization_report.md"

FINAL_CONFIG = "final_selected_expanded151_lg_proj_dinov3_dift_ft002_mp2048"
MATCH_ROOTS = {
    "final_selected": ROOT / "output_v2" / "matches_v2" / FINAL_CONFIG,
    "splg": ROOT / "output_v2" / "matches_v2" / "test_splg",
    "roma": ROOT / "output_v2" / "matches_v2" / "test_roma",
    "romav2": ROOT / "output_v2" / "matches_v2" / "test_romav2",
}
METHOD_TITLES = {
    "final_selected": "Final selected",
    "splg": "SP+LG",
    "roma": "RoMa",
    "romav2": "RoMaV2",
}
PANEL_ORDER = ["final_selected", "splg", "roma", "romav2"]

GREEN = "#00a651"
VIS_SIZE = (720, 480)  # width, height for each image inside a panel
MAX_DISPLAY_MATCHES = 120


@dataclass(frozen=True)
class PairSpec:
    scene: str
    pair_id: str
    slug: str
    reason: str


SELECTED_PAIRS = [
    PairSpec(
        scene="temple_nara_japan",
        pair_id="59390642_9181109188__71571234_236438589",
        slug="temple_nara",
        reason=(
            "Primary thesis qualitative pair: visually clear temple facade, "
            "strong overlap, and saved matches are available for all four methods."
        ),
    ),
    PairSpec(
        scene="piazza_san_marco",
        pair_id="00509209_2257958656__00911019_30410665",
        slug="san_marco",
        reason=(
            "Optional contrast pair from the weaker San Marco scene, where the "
            "final selected method is not better than SP+LG in the per-scene table."
        ),
    ),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


def pil_resample_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def image_path(scene: str, image_id: str) -> Path:
    path = DATA_ROOT / scene / "images_preprocessed" / f"{image_id}.jpg"
    if not path.exists():
        path = DATA_ROOT / scene / "dense" / "images" / f"{image_id}.jpg"
    return path


def load_visual_image_and_scale(path: Path) -> tuple[np.ndarray, float, float]:
    image = Image.open(path).convert("RGB")
    original_w, original_h = image.size
    target_w, target_h = VIS_SIZE
    image = image.resize((target_w, target_h), pil_resample_filter())
    sx = target_w / float(original_w)
    sy = target_h / float(original_h)
    return np.asarray(image), sx, sy


def load_matches(scene: str, pair_id: str, method_key: str) -> tuple[Path, np.ndarray, np.ndarray]:
    match_path = MATCH_ROOTS[method_key] / scene / f"{pair_id}.npz"
    if not match_path.exists():
        raise FileNotFoundError(match_path)
    with np.load(match_path, allow_pickle=False) as data:
        mkpts0 = np.asarray(data["mkpts0"], dtype=np.float32)
        mkpts1 = np.asarray(data["mkpts1"], dtype=np.float32)
    return match_path, mkpts0, mkpts1


def deterministic_match_indices(num_matches: int, max_display: int = MAX_DISPLAY_MATCHES) -> np.ndarray:
    shown = min(max_display, num_matches)
    if shown <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.linspace(0, num_matches - 1, shown).round().astype(np.int64)


def make_canvas(
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    sx0: float,
    sy0: float,
    sx1: float,
    sy1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_w, target_h = VIS_SIZE
    canvas = np.full((target_h, target_w * 2, 3), 255, dtype=np.uint8)
    canvas[:, :target_w] = img0
    canvas[:, target_w:] = img1

    pts0 = mkpts0.copy()
    pts1 = mkpts1.copy()
    pts0[:, 0] *= sx0
    pts0[:, 1] *= sy0
    pts1[:, 0] = pts1[:, 0] * sx1 + target_w
    pts1[:, 1] *= sy1
    return canvas, pts0, pts1


def draw_matches_panel(
    ax: plt.Axes,
    canvas: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    title: str,
    total_matches: int,
    shown_indices: np.ndarray,
) -> None:
    ax.imshow(canvas)
    ax.set_axis_off()
    for idx in shown_indices:
        x0, y0 = pts0[idx]
        x1, y1 = pts1[idx]
        ax.plot([x0, x1], [y0, y1], color=GREEN, alpha=0.52, linewidth=0.7)
    if len(shown_indices):
        ax.scatter(pts0[shown_indices, 0], pts0[shown_indices, 1], s=5, c=GREEN, alpha=0.78, linewidths=0)
        ax.scatter(pts1[shown_indices, 0], pts1[shown_indices, 1], s=5, c=GREEN, alpha=0.78, linewidths=0)
    ax.set_title(f"{title}: {len(shown_indices)}/{total_matches} matches", pad=5)


def build_pair_visuals(pair: PairSpec):
    img0_id, img1_id = pair.pair_id.split("__")
    img0_path = image_path(pair.scene, img0_id)
    img1_path = image_path(pair.scene, img1_id)
    if not img0_path.exists() or not img1_path.exists():
        raise FileNotFoundError(f"Missing source image(s): {img0_path}, {img1_path}")

    img0, sx0, sy0 = load_visual_image_and_scale(img0_path)
    img1, sx1, sy1 = load_visual_image_and_scale(img1_path)

    method_data = {}
    for method_key in PANEL_ORDER:
        match_path, mkpts0, mkpts1 = load_matches(pair.scene, pair.pair_id, method_key)
        if len(mkpts0) != len(mkpts1):
            raise ValueError(f"Mismatched point arrays in {match_path}")
        canvas, pts0, pts1 = make_canvas(img0, img1, mkpts0, mkpts1, sx0, sy0, sx1, sy1)
        shown_indices = deterministic_match_indices(len(mkpts0))
        method_data[method_key] = {
            "match_path": match_path,
            "canvas": canvas,
            "pts0": pts0,
            "pts1": pts1,
            "total_matches": len(mkpts0),
            "shown_indices": shown_indices,
        }

    return img0_path, img1_path, method_data


def save_combined_figure(pair: PairSpec, method_data) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.0, 6.25),
        gridspec_kw={"wspace": 0.035, "hspace": 0.34},
    )
    for ax, method_key in zip(axes.ravel(), PANEL_ORDER):
        data = method_data[method_key]
        draw_matches_panel(
            ax,
            data["canvas"],
            data["pts0"],
            data["pts1"],
            METHOD_TITLES[method_key],
            data["total_matches"],
            data["shown_indices"],
        )
    fig.suptitle(f"{pair.scene}: {pair.pair_id}", y=0.985, fontsize=14)

    stem = f"fig_final_test_qualitative_{pair.slug}_comparison"
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def save_method_figures(pair: PairSpec, method_data) -> list[Path]:
    outputs: list[Path] = []
    for method_key in PANEL_ORDER:
        data = method_data[method_key]
        fig, ax = plt.subplots(figsize=(12.5, 4.7))
        draw_matches_panel(
            ax,
            data["canvas"],
            data["pts0"],
            data["pts1"],
            METHOD_TITLES[method_key],
            data["total_matches"],
            data["shown_indices"],
        )
        fig.tight_layout(pad=0.3)
        stem = f"fig_final_test_qualitative_{pair.slug}_{method_key}"
        output = FIG_DIR / f"{stem}.png"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_report(records: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qualitative Match Visualization Report",
        "",
        "Generated from existing saved image and match artifacts only. No feature extraction, matching, model inference, evaluation, or runtime benchmarking was run.",
        "",
        "Images are resized to a common visualization size of `720 x 480` pixels per side. The saved raw match coordinates are scaled only for drawing; no resized image is used to compute matches.",
        "",
        "All displayed correspondences are raw saved matches drawn in green. Inlier/outlier coloring is not used because the available `.npz` match artifacts contain matched keypoints but no trustworthy per-match inlier mask, and the evaluation JSON files store aggregate inlier counts with empty per-match inlier lists.",
        "",
    ]

    for record in records:
        pair = record["pair"]
        lines.extend(
            [
                f"## {pair.scene}: `{pair.pair_id}`",
                "",
                f"- Selection note: {pair.reason}",
                f"- Left image: `{rel(record['img0_path'])}`",
                f"- Right image: `{rel(record['img1_path'])}`",
                f"- Inlier/outlier coloring available: no; raw matches only.",
                "",
                "| method | match file | raw matches | displayed matches |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for method_key in PANEL_ORDER:
            data = record["method_data"][method_key]
            lines.append(
                f"| {METHOD_TITLES[method_key]} | `{rel(data['match_path'])}` | "
                f"{data['total_matches']} | {len(data['shown_indices'])} |"
            )
        lines.extend(["", "Generated figures:"])
        for output in record["outputs"]:
            lines.append(f"- `{rel(output)}`")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    all_outputs: list[Path] = []
    for pair in SELECTED_PAIRS:
        img0_path, img1_path, method_data = build_pair_visuals(pair)
        outputs = []
        outputs.extend(save_combined_figure(pair, method_data))
        outputs.extend(save_method_figures(pair, method_data))
        all_outputs.extend(outputs)
        records.append(
            {
                "pair": pair,
                "img0_path": img0_path,
                "img1_path": img1_path,
                "method_data": method_data,
                "outputs": outputs,
            }
        )

    write_report(records)

    print("Generated qualitative match figures:")
    for output in all_outputs:
        print(rel(output))
    print(rel(REPORT_PATH))


if __name__ == "__main__":
    main()
