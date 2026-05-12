#!/usr/bin/env python3
"""Generate Figure 4.1: grid alignment illustration."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).absolute().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures"


COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "grid": "#cbd5e1",
    "grid_soft": "#e2e8f0",
    "blue": "#2563eb",
    "blue_soft": "#eff6ff",
    "teal": "#0f766e",
    "violet": "#7c3aed",
    "amber": "#d97706",
    "orange": "#ea580c",
    "red": "#dc2626",
    "rose": "#e11d48",
}


def add_label(ax, x, y, text, size=14, weight="regular", color=None, ha="left", va="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        family="DejaVu Sans",
    )


def add_box(ax, x, y, w, h, face, edge, lw=1.4, zorder=1):
    box = Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def add_token(ax, x, y, radius=0.12, face="#e2e8f0", edge="#94a3b8", lw=1.1, zorder=4):
    token = Circle((x, y), radius=radius, facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(token)
    return token


def add_arrow(ax, start, end, color, lw=2.0, alpha=1.0, style="-|>", zorder=6, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        alpha=alpha,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=7,
        shrinkB=7,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_grid(ax, x0, y0, snapped=False):
    stride = 1.0
    centers_x = [x0 + 0.5 + i * stride for i in range(4)]
    centers_y = [y0 + 0.5 + i * stride for i in range(4)]
    boundaries_x = [x0 + i * stride for i in range(5)]
    boundaries_y = [y0 + i * stride for i in range(5)]

    focus_face = "#eff6ff" if not snapped else "#fff7ed"
    focus_edge = COLORS["blue"] if not snapped else COLORS["orange"]
    add_box(ax, centers_x[1], centers_y[1], stride, stride, focus_face, focus_edge, lw=1.8, zorder=0)

    for x in boundaries_x:
        ax.plot([x, x], [y0, y0 + 4 * stride], color=COLORS["grid"], lw=1.0, zorder=1)
    for y in boundaries_y:
        ax.plot([x0, x0 + 4 * stride], [y, y], color=COLORS["grid"], lw=1.0, zorder=1)
    for x in centers_x:
        ax.plot([x, x], [y0, y0 + 4 * stride], color=COLORS["grid_soft"], lw=0.8, ls=(0, (2, 5)), zorder=1)
    for y in centers_y:
        ax.plot([x0, x0 + 4 * stride], [y, y], color=COLORS["grid_soft"], lw=0.8, ls=(0, (2, 5)), zorder=1)

    highlighted = {
        (1, 1): ("#dbeafe", COLORS["blue"]),
        (2, 1): ("#ccfbf1", COLORS["teal"]),
        (1, 2): ("#ede9fe", COLORS["violet"]),
        (2, 2): ("#fef3c7", COLORS["amber"]),
    }
    for row, cy in enumerate(centers_y):
        for col, cx in enumerate(centers_x):
            if snapped:
                if (col, row) == (2, 2):
                    add_token(ax, cx, cy, radius=0.17, face="#ffedd5", edge=COLORS["orange"], lw=2.0)
                else:
                    add_token(ax, cx, cy, radius=0.11, face="#f1f5f9", edge="#cbd5e1", lw=0.9, zorder=3)
            elif (col, row) in highlighted:
                face, edge = highlighted[(col, row)]
                add_token(ax, cx, cy, radius=0.16, face=face, edge=edge, lw=1.8)
            else:
                add_token(ax, cx, cy, radius=0.11)

    add_label(ax, x0, y0 + 4.18, "patch-token grid", size=9, color="#64748b")
    return centers_x, centers_y, stride


def draw_keypoint(ax, x, y, ghost=False):
    if ghost:
        circle = Circle(
            (x, y),
            radius=0.13,
            facecolor="#fda4af",
            edgecolor="#be123c",
            linewidth=1.6,
            linestyle=(0, (3, 2)),
            zorder=8,
        )
    else:
        circle = Circle((x, y), radius=0.14, facecolor=COLORS["rose"], edgecolor="white", linewidth=2.0, zorder=8)
    ax.add_patch(circle)
    cross_color = "#be123c" if ghost else COLORS["rose"]
    ax.plot([x - 0.28, x + 0.28], [y, y], color=cross_color, lw=1.2, zorder=7, linestyle=(0, (3, 2)) if ghost else "solid")
    ax.plot([x, x], [y - 0.28, y + 0.28], color=cross_color, lw=1.2, zorder=7, linestyle=(0, (3, 2)) if ghost else "solid")


def draw_descriptor_icon(ax, x, y, color, title):
    add_box(ax, x, y, 1.45, 0.95, "#ffffff", "#cbd5e1", lw=1.0, zorder=3)
    add_label(ax, x + 0.72, y + 0.73, title, size=8.5, weight="bold", color="#64748b", ha="center")
    heights = [0.24, 0.38, 0.20, 0.48]
    for idx, height in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (x + 0.33 + idx * 0.22, y + 0.16),
                0.11,
                height,
                facecolor=color,
                edgecolor="none",
                alpha=[0.45, 0.78, 0.55, 0.9][idx],
                zorder=4,
            )
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.0, 8.0), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    add_label(
        ax,
        7.5,
        7.65,
        "Grid Alignment for Patch-Token Descriptors",
        size=20,
        weight="bold",
        ha="center",
    )
    add_label(
        ax,
        7.5,
        7.32,
        "The descriptor grid is coarse; a SuperPoint keypoint usually lies between token centers.",
        size=10.5,
        color="#64748b",
        ha="center",
    )

    add_box(ax, 0.45, 0.45, 6.65, 6.55, "#fbfcfe", "#d8dee8", lw=1.2, zorder=-2)
    add_box(ax, 7.9, 0.45, 6.65, 6.55, "#fbfcfe", "#d8dee8", lw=1.2, zorder=-2)

    add_label(ax, 0.78, 6.62, "Bilinear sampling", size=18, weight="bold")
    add_label(ax, 0.78, 6.28, "Keep the SuperPoint location p = (u, v)", size=10.8, color=COLORS["muted"])
    add_label(ax, 8.23, 6.62, "Nearest-center lookup", size=18, weight="bold")
    add_label(ax, 8.23, 6.28, "Read one token directly, but move the keypoint", size=10.8, color=COLORS["muted"])

    left_x, left_y = 1.25, 1.78
    lcx, lcy, stride = draw_grid(ax, left_x, left_y, snapped=False)
    p_left = (lcx[1] + 0.58, lcy[1] + 0.55)
    surrounding = [
        ((lcx[1], lcy[1]), COLORS["blue"], 1.8, "w00", (-0.42, -0.23)),
        ((lcx[2], lcy[1]), COLORS["teal"], 2.4, "w10", (0.2, -0.23)),
        ((lcx[1], lcy[2]), COLORS["violet"], 1.9, "w01", (-0.42, 0.25)),
        ((lcx[2], lcy[2]), COLORS["amber"], 3.0, "w11", (0.2, 0.25)),
    ]
    for center, color, lw, weight, offset in surrounding:
        add_arrow(ax, center, p_left, color=color, lw=lw, alpha=0.78)
        add_label(ax, center[0] + offset[0], center[1] + offset[1], weight, size=8.5, weight="bold", color="#64748b")
    draw_keypoint(ax, *p_left)
    add_label(ax, p_left[0] + 0.35, p_left[1] + 0.08, "p = (u, v)", size=10.5, weight="bold")
    add_label(ax, p_left[0] + 0.35, p_left[1] - 0.22, "not snapped", size=9.2, color=COLORS["muted"])
    add_label(ax, left_x + 4.28, left_y + 2.85, "4 surrounding", size=9.5, color=COLORS["muted"])
    add_label(ax, left_x + 4.28, left_y + 2.6, "patch tokens", size=9.5, color=COLORS["muted"])
    draw_descriptor_icon(ax, 5.55, 3.0, COLORS["blue"], "blended")

    add_box(ax, 1.05, 0.72, 5.5, 0.82, COLORS["blue_soft"], "#bfdbfe", lw=1.0, zorder=1)
    add_label(ax, 1.3, 1.25, "d(p) = w00 d00 + w10 d10 + w01 d01 + w11 d11", size=10.4, weight="bold")
    add_label(ax, 1.3, 0.95, "Descriptor is sampled at the original, sub-patch keypoint location.", size=8.7, color=COLORS["muted"])

    right_x, right_y = 8.65, 1.78
    rcx, rcy, _ = draw_grid(ax, right_x, right_y, snapped=True)
    p_right = (rcx[1] + 0.58, rcy[1] + 0.55)
    snapped_center = (rcx[2], rcy[2])
    draw_keypoint(ax, *p_right, ghost=True)
    add_label(ax, p_right[0] - 1.55, p_right[1] + 0.05, "original p", size=9.5, color=COLORS["muted"])
    add_arrow(ax, (p_right[0] - 0.75, p_right[1] + 0.03), p_right, color="#64748b", lw=1.1, style="-|>")
    add_arrow(ax, p_right, snapped_center, color=COLORS["red"], lw=2.7)
    add_token(ax, *snapped_center, radius=0.055, face="white", edge="white", lw=0.6, zorder=9)
    add_label(ax, snapped_center[0] + 0.34, snapped_center[1] + 0.28, "snapped center", size=10.5, weight="bold")
    add_label(ax, p_right[0] - 0.05, p_right[1] - 0.5, "localization error", size=10.5, weight="bold", color=COLORS["red"])
    add_label(ax, p_right[0] + 0.52, p_right[1] + 0.26, "delta p", size=9.5, color=COLORS["red"])
    draw_descriptor_icon(ax, 12.95, 2.65, COLORS["orange"], "single token")

    add_box(ax, 8.5, 0.72, 5.5, 0.82, "#fff1f2", "#fecdd3", lw=1.0, zorder=1)
    add_label(ax, 8.75, 1.25, "d(p) ~= d_nearest", size=10.4, weight="bold")
    add_label(ax, 8.75, 0.95, "Descriptor lookup is simple, but the geometric point is displaced.", size=8.7, color=COLORS["muted"])

    add_arrow(ax, (6.85, 1.95), (8.2, 1.95), color="#94a3b8", lw=1.4, style="-|>")
    add_label(
        ax,
        7.5,
        1.62,
        "Bilinear interpolation keeps descriptor sampling aligned with the detected keypoint.",
        size=9.8,
        color=COLORS["muted"],
        ha="center",
    )

    png_path = OUTPUT_DIR / "grid_alignment_illustration.png"
    pdf_path = OUTPUT_DIR / "grid_alignment_illustration.pdf"
    svg_path = OUTPUT_DIR / "grid_alignment_illustration.svg"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    print("Saved {}".format(png_path.relative_to(PROJECT_ROOT)))
    print("Saved {}".format(pdf_path.relative_to(PROJECT_ROOT)))
    print("Saved {}".format(svg_path.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
