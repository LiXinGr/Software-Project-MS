import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import sys  # 🔹 add this
import numpy as np
import torch
from PIL import Image

# 🔹 Add these lines BEFORE importing from external.DIFT
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIFT_ROOT = PROJECT_ROOT / "external" / "DIFT"

# Put both the project root and DIFT root on sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DIFT_ROOT))

from external.DIFT.extract_dift import main as dift_extract_main


def run_dift_extractor(img_path, ft_path, shared_args):
    """
    Call the original extract_dift.py main(args) for one image
    using exactly the same parameters style as the CLI.
    """
    dift_args = argparse.Namespace(
        img_size=shared_args.img_size,          # e.g. [768, 768]
        model_id=shared_args.model_id,          # e.g. "stable-diffusion-v1-5/stable-diffusion-v1-5"
        t=shared_args.t,                        # e.g. 261
        up_ft_index=shared_args.up_ft_index,    # e.g. 1
        prompt=shared_args.prompt,              # e.g. "a photo of a cat"
        ensemble_size=shared_args.ensemble_size,# e.g. 8
        input_path=str(img_path),
        output_path=str(ft_path),
    )
    dift_extract_main(dift_args)               # runs your original script
    ft = torch.load(ft_path)                   # [C, H, W]
    return ft


# ---- helpers reused from before (same as previous message) ----

def compute_matches(
    ft1,
    ft2,
    max_points=2000,
    use_mutual=False,
    ratio_thresh=None,
):
    """
    Match DIFT feature maps using cosine similarity.

    Args:
        ft1, ft2: feature maps [C, H, W]
        max_points: max number of points sampled from image 1
        use_mutual: if True, apply mutual nearest neighbor filtering
        ratio_thresh: if not None, apply a Lowe-style ratio test on similarities.
                      (e.g. 1.05 or 1.1 for sim-based ratio)

    Returns:
        x1, y1, x2, y2: numpy arrays of matched coordinates in feature space
        (H, W): spatial size of feature maps
    """
    assert ft1.shape == ft2.shape, "Feature maps must have the same shape"
    C, H, W = ft1.shape
    N = H * W

    # Flatten to [N, C]
    ft1_flat = ft1.view(C, -1).t()   # [N, C]
    ft2_flat = ft2.view(C, -1).t()   # [N, C]

    # L2-normalize (cosine similarity)
    ft1_flat = ft1_flat / (ft1_flat.norm(dim=1, keepdim=True) + 1e-8)
    ft2_flat = ft2_flat / (ft2_flat.norm(dim=1, keepdim=True) + 1e-8)

    # Sample points in image 1
    if N > max_points:
        idx1_full = torch.randperm(N)[:max_points]  # indices in ft1_flat
    else:
        idx1_full = torch.arange(N)

    desc1 = ft1_flat[idx1_full]      # [M, C], M <= max_points
    desc2 = ft2_flat                 # [N, C]

    # Similarity matrix: [M, N]
    sim = desc1 @ desc2.t()

    # Base: best match in image 2 for each sampled point in image 1
    best_sim, best_j = sim.max(dim=1)   # [M], indices in desc2 (0..N-1)

    # Track row indices of desc1 (0..M-1)
    rows = torch.arange(desc1.shape[0])

    # ---------------- Ratio test (optional) ----------------
    if ratio_thresh is not None:
        # top-2 similarities for each row
        top2_sim, top2_idx = sim.topk(2, dim=1)   # [M, 2]
        best = top2_sim[:, 0]
        second = top2_sim[:, 1]

        # similarity-based ratio: we want best significantly larger than second
        ratio = best / (second + 1e-8)
        good = ratio >= ratio_thresh

        rows = rows[good]
        best_j = best_j[good]

        # If everything got filtered, bail out early
        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W)

    # ---------------- Mutual NN (optional) ----------------
    if use_mutual:
        # Best match in image 1 for each point in image 2
        # (full reverse mapping over ALL desc1 rows)
        rev_best_i = sim.argmax(dim=0)       # [N], indices in 0..M-1

        # Enforce: for match (i -> j) we require rev_best_i[j] == i
        mutual_mask = rev_best_i[best_j] == rows
        rows = rows[mutual_mask]
        best_j = best_j[mutual_mask]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W)

    # ---------------- Convert to (x, y) coords ----------------
    # rows: indices in desc1 (0..M-1)
    # idx1_full: corresponding flat indices in ft1_flat (0..N-1)
    flat1 = idx1_full[rows]          # [K], K = #matches
    flat2 = best_j                   # [K]

    y1 = (flat1 // W).cpu().numpy()
    x1 = (flat1 % W).cpu().numpy()
    y2 = (flat2 // W).cpu().numpy()
    x2 = (flat2 % W).cpu().numpy()

    return x1, y1, x2, y2, (H, W)



def load_and_resize(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    if img_size[0] > 0:
        img = img.resize(tuple(img_size))
    return np.array(img)


def visualize_matches(img1_np, img2_np, x1, y1, x2, y2,
                      feat_hw, out_path=None, max_lines=200):
    H_feat, W_feat = feat_hw
    h, w, _ = img1_np.shape
    assert img2_np.shape[0] == h and img2_np.shape[1] == w

    canvas = np.concatenate([img1_np, img2_np], axis=1)

    scale_x = w / W_feat
    scale_y = h / H_feat

    x1_img = (x1 + 0.5) * scale_x
    y1_img = (y1 + 0.5) * scale_y
    x2_img = (x2 + 0.5) * scale_x + w
    y2_img = (y2 + 0.5) * scale_y

    num_matches = len(x1)
    if num_matches > max_lines:
        idx = np.random.choice(num_matches, size=max_lines, replace=False)
        x1_img = x1_img[idx]
        y1_img = y1_img[idx]
        x2_img = x2_img[idx]
        y2_img = y2_img[idx]

    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    plt.axis("off")
    for xa, ya, xb, yb in zip(x1_img, y1_img, x2_img, y2_img):
        plt.plot([xa, xb], [ya, yb], linewidth=0.5)

    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ---------------- main WITH ALL PARAMETERS ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Match two images using DIFT features (calls extract_dift.py twice)."
    )

    # images (now used exactly as provided)
    parser.add_argument("--img1", required=True, type=str,
                        help="Path to first image")
    parser.add_argument("--img2", required=True, type=str,
                        help="Path to second image")

    # matching / viz options
    parser.add_argument("--max_points", type=int, default=2000,
                        help="Max number of feature points to match")
    parser.add_argument("--max_lines", type=int, default=200,
                        help="Max number of lines to draw in visualization")

    # ---- DIFT PARAMETERS: same as original script ----
    parser.add_argument(
        "--model_id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="model_id of the diffusion model (same as extract_dift.py)",
    )
    parser.add_argument(
        "--input_dummy",  # not used
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output_dummy",  # not used
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=[768, 768],
        help="Resize input images to [w h] before DIFT (same as extract_dift.py)",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=261,
        help="Diffusion time step (same as extract_dift.py)",
    )
    parser.add_argument(
        "--up_ft_index",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Which upsampling block of UNet to extract features from",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Stable Diffusion prompt (shared for both images)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=8,
        help="Number of repeated images in each batch (same as extract_dift.py)",
    )

    args = parser.parse_args()

    # base dirs (still use datasets/ for outputs if you want)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # 🔹 use image paths EXACTLY as provided
    img1_path = Path(args.img1)   # no extra logic
    img2_path = Path(args.img2)

    # outputs go to datasets/ (you can change this if you prefer)
    ft1_path = datasets_dir / f"{img1_path.stem}_dift.pt"
    ft2_path = datasets_dir / f"{img2_path.stem}_dift.pt"
    vis_path = datasets_dir / f"dift_matches_{img1_path.stem}_{img2_path.stem}_improved_NN.png"

    # 1) extract features with original script
    ft1 = run_dift_extractor(img1_path, ft1_path, args)
    ft2 = run_dift_extractor(img2_path, ft2_path, args)

    # 2) compute NN matches
    x1, y1, x2, y2, feat_hw = compute_matches(ft1, ft2, max_points=args.max_points, use_mutual=True, ratio_thresh=1.1)

    # 3) load images for viz (using the exact paths you gave)
    img1_np = load_and_resize(img1_path, args.img_size)
    img2_np = load_and_resize(img2_path, args.img_size)

    # 4) visualize & save
    visualize_matches(
        img1_np, img2_np, x1, y1, x2, y2,
        feat_hw, out_path=vis_path, max_lines=args.max_lines
    )

    print(f"[DIFT] Saved feature maps to:\n  {ft1_path}\n  {ft2_path}")
    print(f"[DIFT] Saved match visualization to:\n  {vis_path}")


if __name__ == "__main__":
    main()

# python scripts/extract_dift.py \
#   --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
#   --input_path ./datasets/cat.png \
#   --output_path ./datasets/dift_cat.pt \
#   --img_size 768 768 \
#   --t 261 \
#   --up_ft_index 1 \
#   --prompt "a photo of a cat" \
#   --ensemble_size 8