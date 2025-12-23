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
from util import compute_matches, visualize_matches

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
def load_and_resize(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    if img_size[0] > 0:
        img = img.resize(tuple(img_size))
    return np.array(img)
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

