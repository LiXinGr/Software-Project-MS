"""
DIFT Feature Matching Script

Extracts features using DIFT (Diffusion Features) and matches them using MNN + ratio test.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import argparse
from pathlib import Path
import sys
import warnings

# Suppress diffusers safety checker warnings
warnings.filterwarnings("ignore", message=".*safety checker.*")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIFT_ROOT = PROJECT_ROOT / "external" / "DIFT"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DIFT_ROOT))

from util import compute_matches, visualize_matches, match_dense_features, save_matches
from external.DIFT.extract_dift import main as dift_extract_main


def run_dift_extractor(img_path, feature_cache_dir, shared_args):
    """
    Extract DIFT features for one image.
    Returns feature map [C, H, W] and original image size (H, W).
    """
    # Get original image size
    img_orig = Image.open(img_path).convert("RGB")
    orig_size = (img_orig.height, img_orig.width)
    
    # Determine cache path
    ft_path = Path(feature_cache_dir) / f"{Path(img_path).stem}_dift.pt"
    
    if ft_path.exists():
        return torch.load(ft_path), orig_size

    dift_args = argparse.Namespace(
        img_size=shared_args.img_size,
        model_id=shared_args.model_id,
        t=shared_args.t,
        up_ft_index=shared_args.up_ft_index,
        prompt=shared_args.prompt,
        ensemble_size=shared_args.ensemble_size,
        input_path=str(img_path),
        output_path=str(ft_path),
        device=shared_args.device,
    )
    
    ft_path.parent.mkdir(parents=True, exist_ok=True)
    dift_extract_main(dift_args)
    ft = torch.load(ft_path)
    return ft, orig_size


def process_pair(img1_path, img2_path, feature_cache_dir, args):
    """Process a single pair and return mkpts0, mkpts1."""
    ft1, orig_size1 = run_dift_extractor(img1_path, feature_cache_dir, args)
    ft2, orig_size2 = run_dift_extractor(img2_path, feature_cache_dir, args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ft1 = ft1.to(device)
    ft2 = ft2.to(device)
    
    mkpts0, mkpts1 = match_dense_features(
        ft1, ft2,
        img_size1=orig_size1,
        img_size2=orig_size2,
        max_points=args.max_points,
        use_mutual=args.use_mutual,
        ratio_thresh=args.ratio_thresh,
    )
    
    return mkpts0, mkpts1, orig_size1, orig_size2


def main():
    parser = argparse.ArgumentParser(
        description="Match images using DIFT features"
    )

    # Single-pair mode
    parser.add_argument("--img1", type=str, help="Path to first image (single-pair mode)")
    parser.add_argument("--img2", type=str, help="Path to second image (single-pair mode)")
    
    # Batch mode
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt (batch mode)")
    parser.add_argument("--images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches (batch mode)")
    
    # Matching options
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=0.8)
    
    # DIFT parameters
    parser.add_argument("--model_id", type=str, 
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--img_size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--t", type=int, default=261)
    parser.add_argument("--up_ft_index", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--ensemble_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None,
                        help="Directory to cache extracted features")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    # Set default feature cache
    if args.feature_cache is None:
        args.feature_cache = PROJECT_ROOT / "datasets" / "dift_features"
    feature_cache_dir = Path(args.feature_cache)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_file and args.output_dir:
        # Batch mode
        print(f"[DIFT] Batch mode: processing pairs from {args.pairs_file}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        
        with open(args.pairs_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        # Apply limit if specified
        if args.limit:
            pairs = pairs[:args.limit]
        
        print(f"[DIFT] Processing {len(pairs)} pairs...")
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            # Check if output already exists
            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            output_path = output_dir / f"{pair_name}.npz"
            
            if output_path.exists():
                continue
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[DIFT] Skipping pair: {img1_name}, {img2_name}")
                continue
            
            mkpts0, mkpts1, _, _ = process_pair(
                img1_path, img2_path, feature_cache_dir, args
            )
            
            save_matches(output_path, mkpts0, mkpts1)
        
        print(f"[DIFT] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        mkpts0, mkpts1, orig_size1, orig_size2 = process_pair(
            img1_path, img2_path, feature_cache_dir, args
        )
        
        print(f"[DIFT] Found {len(mkpts0)} matches")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        
        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB").resize(tuple(args.img_size)))
            img2_np = np.array(Image.open(img2_path).convert("RGB").resize(tuple(args.img_size)))
            
            vis_path = f"datasets/dift_matches_{img1_path.stem}_{img2_path.stem}.png"
            
            visualize_matches(
                img1_np, img2_np,
                mkpts0[:, 0], mkpts0[:, 1],
                mkpts1[:, 0], mkpts1[:, 1],
                orig_size1, orig_size2,
                out_path=vis_path,
                max_lines=args.max_lines,
            )
            print(f"[DIFT] Saved visualization to {vis_path}")
    else:
        parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()

# Single-pair example:
# python3 scripts/dift_matches.py --img1 datasets/bran1.jpg --img2 datasets/bran2.jpg

# Batch mode example:
# python3 scripts/dift_matches.py \
#   --pairs_file datasets/phototourism/sacre_coeur/pairs.txt \
#   --images_dir datasets/phototourism/sacre_coeur/dense/images \
#   --output_dir output/matches/dift
