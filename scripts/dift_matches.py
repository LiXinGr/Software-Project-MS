"""
Script accepts all the flags required for the DIFT model. It also
doesn't requires the paths for the 2 images on which it will perform
matching. In case they were processed before and it already has it
features, it just reuses it.

For now, resizing is made only inside the DIFT itself. For convinience
it depict both images with the same width (the biggest between two) but keeping the aspect ratio.
You can also manage the NN process by setting ration_tresh value
(default: None) and use_mutal (default: False) for mutual NN. You can
use them separatly of together.

We may need to write an own code for resizing dut to the fact that DIFT resizing doesn't preserve aspect ratio and not use padding.
It just stretches/squeashed the image to exactly [width, height]
"""


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
    # Check if features already exist
    if Path(ft_path).exists():
        print(f"[DIFT] Found existing features at {ft_path}, skipping extraction.")
        return torch.load(ft_path)

    dift_args = argparse.Namespace(
        img_size=shared_args.img_size,          # e.g. [768, 768]
        model_id=shared_args.model_id,          # e.g. "stable-diffusion-v1-5/stable-diffusion-v1-5"
        t=shared_args.t,                        # e.g. 261
        up_ft_index=shared_args.up_ft_index,    # e.g. 1
        prompt=shared_args.prompt,              # e.g. "a photo of a cat"
        ensemble_size=shared_args.ensemble_size,# e.g. 8
        input_path=str(img_path),
        output_path=str(ft_path),
        device=shared_args.device,              # e.g. "cuda:3"
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
    Match feature maps using cosine similarity (dot product after L2 norm).

    Args:
        ft1, ft2: feature maps [C, H, W]
        max_points: max number of points sampled from image 1
        use_mutual: if True, apply mutual nearest neighbor filtering
        ratio_thresh: if not None, apply a Lowe-style ratio test on similarities
                     (e.g. 1.05 or 1.1 for sim-based ratio)

    Returns:
        x1, y1, x2, y2: numpy arrays of matched coordinates in feature space
        (H, W): spatial size of feature maps
    """
    device = ft1.device

    C, H1, W1 = ft1.shape
    assert C == ft2.shape[0], "Feature maps must have the same channel dimension"
    
    C2, H2, W2 = ft2.shape
    N1 = H1 * W1
    N2 = H2 * W2

    # Flatten to [N, C]
    ft1_flat = ft1.view(C, -1).t()   # [N1, C]
    ft2_flat = ft2.view(C, -1).t()   # [N2, C]

    # L2-normalize (so dot product == cosine similarity)
    ft1_flat = ft1_flat / (ft1_flat.norm(dim=1, keepdim=True) + 1e-8)
    ft2_flat = ft2_flat / (ft2_flat.norm(dim=1, keepdim=True) + 1e-8)

    # Sample points in image 1
    # Sample points in image 1
    if N1 > max_points:
        idx1_full = torch.randperm(N1, device=device)[:max_points]
    else:
        idx1_full = torch.arange(N1, device=device)

    desc1 = ft1_flat[idx1_full]      # [M, C]
    desc2 = ft2_flat                 # [N, C]
    M = desc1.shape[0]

    # Similarity matrix: [M, N]
    sim = desc1 @ desc2.t()

    # 1->2 best matches
    best_sim, best_j = sim.max(dim=1)          # [M]
    rows_all = torch.arange(M, device=device)  # [M]

    # ---------------- Mutual NN (optional) ----------------
    # IMPORTANT: do this BEFORE ratio test, so reverse NN is computed on the same candidate set.
    rows = rows_all
    if use_mutual:
        rev_best_i = sim.argmax(dim=0)  # [N], for each j in image2: best i in image1 (0..M-1)
        mutual_mask = rev_best_i[best_j] == rows_all

        rows = rows_all[mutual_mask]
        best_j = best_j[mutual_mask]
        best_sim = best_sim[mutual_mask]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W)

    # ---------------- Ratio test (optional) ----------------
    # Apply ratio only to rows that survived mutual (if mutual is enabled).
    if ratio_thresh is not None:
        if N2 < 2:
            # Not enough candidates for a top-2 ratio test; keep what we have.
            pass
        else:
            top2_sim, _ = sim.topk(2, dim=1)  # [M, 2]
            best = top2_sim[:, 0]
            second = top2_sim[:, 1]
            ratio = best / (second + 1e-8)

            good = ratio[rows] >= ratio_thresh
            rows = rows[good]
            best_j = best_j[good]
            best_sim = best_sim[good]

            if rows.numel() == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]), (H1, W1), (H2, W2)

    # ---------------- Convert to (x, y) coords ----------------
    flat1 = idx1_full[rows]   # [K] indices into ft1_flat (0..N-1)
    flat2 = best_j            # [K] indices into ft2_flat (0..N-1)

    y1 = (flat1 // W1).cpu().numpy()
    x1 = (flat1 % W1).cpu().numpy()
    y2 = (flat2 // W2).cpu().numpy()
    x2 = (flat2 % W2).cpu().numpy()

    return x1, y1, x2, y2, (H1, W1), (H2, W2)



def load_and_resize(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    if img_size[0] > 0:
        img = img.resize(tuple(img_size))
    return np.array(img)


def visualize_matches(img1_np, img2_np, x1, y1, x2, y2,
                      feat_hw1, feat_hw2, out_path=None, max_lines=200):
    H_feat1, W_feat1 = feat_hw1
    H_feat2, W_feat2 = feat_hw2
    
    h1, w1, _ = img1_np.shape
    h2, w2, _ = img2_np.shape
    
    # Resize so axis 0 (height) is the same for the biggest image (max height)
    # This implies stacking horizontally (concatenating on axis 1)
    target_h = max(h1, h2)
    
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    # Resize images using PIL, maintaining aspect ratio
    i1 = Image.fromarray(img1_np).resize((int(w1 * scale1), int(h1 * scale1)))
    i2 = Image.fromarray(img2_np).resize((int(w2 * scale2), int(h2 * scale2)))
    
    img1_viz = np.array(i1)
    img2_viz = np.array(i2)
    
    # Ensure exact height match for concatenation due to potential rounding
    if img1_viz.shape[0] != target_h:
        img1_viz = np.array(i1.resize((img1_viz.shape[1], target_h)))
    if img2_viz.shape[0] != target_h:
        img2_viz = np.array(i2.resize((img2_viz.shape[1], target_h)))

    canvas = np.concatenate([img1_viz, img2_viz], axis=1)

    # Transform coordinates
    # 1. Feature Map -> Original Image
    scale_x1_orig = w1 / W_feat1
    scale_y1_orig = h1 / H_feat1
    x1_orig = (x1 + 0.5) * scale_x1_orig
    y1_orig = (y1 + 0.5) * scale_y1_orig
    
    scale_x2_orig = w2 / W_feat2
    scale_y2_orig = h2 / H_feat2
    x2_orig = (x2 + 0.5) * scale_x2_orig
    y2_orig = (y2 + 0.5) * scale_y2_orig
    
    # 2. Original Image -> Visualization Image
    x1_img = x1_orig * scale1
    y1_img = y1_orig * scale1
    
    # Image 2 is stacked to the right of Image 1
    x_offset = img1_viz.shape[1]
    x2_img = x2_orig * scale2 + x_offset
    y2_img = y2_orig * scale2

    num_matches = len(x1)
    # TODO: do i really need to limit lines
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g. cuda, cuda:3, cuda:4)",
    )
    parser.add_argument(
        "--use_mutual",
        action="store_true",
        help="Enable mutual nearest neighbor filtering",
    )
    parser.add_argument(
        "--ratio_thresh",
        type=float,
        default=None,
        help="Lowe-style ratio test threshold (e.g. 1.05 or 1.1). None disables the test.",
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
    x1, y1, x2, y2, feat_hw1, feat_hw2 = compute_matches(
        ft1, ft2,
        max_points=args.max_points,
        use_mutual=args.use_mutual,
        ratio_thresh=args.ratio_thresh
    )

    # 3) load images for viz (using the exact paths you gave)
    img1_np = load_and_resize(img1_path, args.img_size)
    img2_np = load_and_resize(img2_path, args.img_size)

    # 4) visualize & save
    visualize_matches(
        img1_np, img2_np, x1, y1, x2, y2,
        feat_hw1, feat_hw2, out_path=vis_path, max_lines=args.max_lines
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


# python scripts/DIFT_two_image_correspondence.py \
#   --img1 datasets/bran1.jpg \
#   --img2 datasets/bran2.jpg

# python scripts/DIFT_two_image_correspondence.py \
#   --img1 datasets/bran1.jpg \
#   --img2 datasets/bran2.jpg \
#   --img_size 768 768 \
#   --max_points 2000 \
#   --max_lines 200 \
#   --prompt "a photo of a cat" \
#   --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#   --t 261 \
#   --up_ft_index 1 \
#   --ensemble_size 8

