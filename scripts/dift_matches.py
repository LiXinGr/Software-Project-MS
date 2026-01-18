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

from util import compute_matches, visualize_matches, match_dense_features, save_matches, match_at_keypoints, get_superpoint_keypoints
from torchvision.transforms import PILToTensor

# Import DIFT's SDFeaturizer for efficient single-load feature extraction
from src.models.dift_sd import SDFeaturizer


def run_dift_extractor(img_path, dift_model, feature_cache_dir, shared_args):
    """
    Extract DIFT features for one image using pre-loaded model.
    Returns feature map [C, H, W] and original image size (H, W).
    
    Args:
        img_path: Path to the image
        dift_model: Pre-loaded SDFeaturizer instance (loaded once, reused for all images)
        feature_cache_dir: Directory to cache extracted features
        shared_args: Arguments containing t, up_ft_index, ensemble_size, etc.
    """
    # Get original image size
    img_orig = Image.open(img_path).convert("RGB")
    orig_size = (img_orig.height, img_orig.width)
    
    # Determine cache path
    # Include parameters in cache key to avoid stale data
    if isinstance(shared_args.img_size, list):
        sz_str = f"{shared_args.img_size[0]}x{shared_args.img_size[1]}" if len(shared_args.img_size) > 1 else str(shared_args.img_size[0])
    else:
        sz_str = str(shared_args.img_size)
    
    cache_key = f"{Path(img_path).stem}_dift_sz{sz_str}_t{shared_args.t}_up{shared_args.up_ft_index}_ens{shared_args.ensemble_size}.pt"
    ft_path = Path(feature_cache_dir) / cache_key
    
    if ft_path.exists():
        return torch.load(ft_path), orig_size

    # Resize image to target size
    img_size = shared_args.img_size
    if isinstance(img_size, list):
        img_size = tuple(img_size)
    img_resized = img_orig.resize(img_size)
    
    # Convert to tensor: [1, C, H, W], range [-1, 1]
    img_tensor = (PILToTensor()(img_resized) / 255.0 - 0.5) * 2
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    # Extract features using pre-loaded model
    ft = dift_model.forward(
        img_tensor,
        prompt=shared_args.prompt,
        t=shared_args.t,
        up_ft_index=shared_args.up_ft_index,
        ensemble_size=shared_args.ensemble_size
    )
    
    # Squeeze batch dimension: [1, C, H, W] -> [C, H, W]
    ft = ft.squeeze(0)
    
    # Save to cache
    ft_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ft.cpu(), ft_path)
    
    return ft, orig_size


def process_pair(img1_path, img2_path, dift_model, feature_cache_dir, args):
    """Process a single pair and return mkpts0, mkpts1."""
    ft1, orig_size1 = run_dift_extractor(img1_path, dift_model, feature_cache_dir, args)
    ft2, orig_size2 = run_dift_extractor(img2_path, dift_model, feature_cache_dir, args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ft1 = ft1.to(device)
    ft2 = ft2.to(device)
    
    if args.use_sp_keypoints:
        # Use SuperPoint-detected keypoints for fair comparison
        sp_cache = Path(feature_cache_dir).parent / 'superpoint_kpts' if feature_cache_dir else None
        kpts1 = get_superpoint_keypoints(img1_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts2 = get_superpoint_keypoints(img2_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        
        # DIFT extracts features at args.img_size (e.g., 512x512), which may differ from
        # the preprocessed image size. We need to map SuperPoint keypoints from preprocessed
        # image coords to DIFT's input coords.
        img_size = args.img_size
        if isinstance(img_size, list):
            img_size = tuple(img_size)
        dift_h, dift_w = img_size if len(img_size) == 2 else (img_size[0], img_size[0])
        
        # Scale keypoints from preprocessed image coords to DIFT input coords
        # orig_size is (H, W) format
        preproc_h1, preproc_w1 = orig_size1  # (H, W)
        preproc_h2, preproc_w2 = orig_size2
        
        kpts1_scaled = kpts1.clone().float()
        kpts1_scaled[:, 0] *= dift_w / preproc_w1  # x: scale by width ratio
        kpts1_scaled[:, 1] *= dift_h / preproc_h1  # y: scale by height ratio
        
        kpts2_scaled = kpts2.clone().float()
        kpts2_scaled[:, 0] *= dift_w / preproc_w2  # x
        kpts2_scaled[:, 1] *= dift_h / preproc_h2  # y
        
        # Note: Ratio test is disabled for DIFT keypoint matching because the
        # similarity-based ratio test doesn't work well with dense diffusion features.
        # Mutual matching alone provides sufficient filtering.
        mkpts0_scaled, mkpts1_scaled = match_at_keypoints(
            ft1, ft2,
            kpts1_scaled, kpts2_scaled,
            img_size1=(dift_h, dift_w),  # DIFT's input size
            img_size2=(dift_h, dift_w),
            use_mutual=args.use_mutual,
            ratio_thresh=None,  # Disabled for DIFT - ratio test incompatible with dense features
        )
        
        # Scale matched keypoints back to preprocessed image coords
        if len(mkpts0_scaled) > 0:
            mkpts0 = mkpts0_scaled.copy()
            mkpts0[:, 0] *= preproc_w1 / dift_w
            mkpts0[:, 1] *= preproc_h1 / dift_h
            
            mkpts1 = mkpts1_scaled.copy()
            mkpts1[:, 0] *= preproc_w2 / dift_w
            mkpts1[:, 1] *= preproc_h2 / dift_h
        else:
            mkpts0, mkpts1 = mkpts0_scaled, mkpts1_scaled
    else:
        # Use dense matching with random sampling (original behavior)
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
    parser.add_argument("--img_size", nargs="+", type=int, default=[1120, 1120],
                        help="Image size for DIFT extraction (default: 1120x1120 to match preprocessing)")
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
    parser.add_argument("--use_sp_keypoints", action="store_true",
                        help="Use SuperPoint-detected keypoints for fair comparison")

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
        
        # Initialize DIFT model ONCE (Stable Diffusion)
        print(f"[DIFT] Loading Stable Diffusion model: {args.model_id}")
        dift_model = SDFeaturizer(sd_id=args.model_id, device=args.device)
        print(f"[DIFT] Model loaded successfully")
        
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
                img1_path, img2_path, dift_model, feature_cache_dir, args
            )
            
            save_matches(output_path, mkpts0, mkpts1)
        
        print(f"[DIFT] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        # Single-pair mode - need to load model
        print(f"[DIFT] Loading Stable Diffusion model: {args.model_id}")
        dift_model = SDFeaturizer(sd_id=args.model_id, device=args.device)
        
        mkpts0, mkpts1, orig_size1, orig_size2 = process_pair(
            img1_path, img2_path, dift_model, feature_cache_dir, args
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
