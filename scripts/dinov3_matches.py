"""
DINOv3 Feature Matching Script

Extracts features using DINOv3 ViT-L/16 and matches them using MNN + ratio test.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import torch
from PIL import Image
import torchvision.transforms as T
import timm
from util import compute_matches, visualize_matches, match_dense_features, save_matches, match_at_keypoints, get_superpoint_keypoints
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def run_dinov3_extractor(
    img_path,
    model,
    transform,
    img_size,
    feat_level,
    cache_dir=None,
):
    """
    Extract a single DINOv3 feature map [C, H, W] for one image.
    Returns the feature map and original image size (H, W).
    """
    # Load original image to get its size
    img_orig = Image.open(img_path).convert("RGB")
    orig_size = (img_orig.height, img_orig.width)  # (H, W)
    
    # Check cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{Path(img_path).stem}.dinov3.pt"
        if cache_path.exists():
            return torch.load(cache_path), orig_size
    
    # Resize and extract features
    img = img_orig.resize((img_size, img_size), Image.BILINEAR)
    x = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        feats = model(x)

    ft = feats[feat_level].squeeze(0).cpu()  # [C, H, W]

    # Save to cache if specified
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{Path(img_path).stem}.dinov3.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ft, cache_path)

    return ft, orig_size


def process_pair(img1_path, img2_path, model, transform, args, feature_cache=None):
    """Process a single pair and return mkpts0, mkpts1."""
    ft1, orig_size1 = run_dinov3_extractor(
        img1_path, model, transform,
        img_size=args.img_size,
        feat_level=args.feat_level,
        cache_dir=feature_cache,
    )
    
    ft2, orig_size2 = run_dinov3_extractor(
        img2_path, model, transform,
        img_size=args.img_size,
        feat_level=args.feat_level,
        cache_dir=feature_cache,
    )
    
    # Move to device for matching
    device = next(model.parameters()).device
    ft1 = ft1.to(device)
    ft2 = ft2.to(device)
    
    if args.use_sp_keypoints:
        # Use SuperPoint-detected keypoints for fair comparison
        sp_cache = Path(feature_cache).parent / 'superpoint_kpts' if feature_cache else None
        kpts1 = get_superpoint_keypoints(img1_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts2 = get_superpoint_keypoints(img2_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        
        mkpts0, mkpts1 = match_at_keypoints(
            ft1, ft2,
            kpts1, kpts2,
            img_size1=orig_size1,
            img_size2=orig_size2,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh,
        )
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
        description="Two-image correspondence using DINOv3 features"
    )

    # Single-pair mode
    parser.add_argument("--img1", type=str, help="Path to first image (single-pair mode)")
    parser.add_argument("--img2", type=str, help="Path to second image (single-pair mode)")
    
    # Batch mode
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt (batch mode)")
    parser.add_argument("--images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches (batch mode)")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=518, 
                        help="Image size for feature extraction (default: 518 for ViT patch=14)")
    parser.add_argument("--feat_level", type=int, default=-1)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--use_mutual", action="store_true", default=True,
                        help="Use mutual nearest neighbor (default: True)")
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=None,
                        help="Lowe's ratio test threshold (default: None = disabled, use 0.8 for stricter filtering)")
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None,
                        help="Directory to cache extracted features")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization images")
    parser.add_argument("--use_sp_keypoints", action="store_true",
                        help="Use SuperPoint-detected keypoints for fair comparison")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model
    transform = T.Compose([T.ToTensor()])
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        features_only=True,
    )
    model.to(device)
    model.eval()

    # Determine mode
    if args.pairs_file and args.output_dir:
        # Batch mode
        print(f"[DINOv3] Batch mode: processing pairs from {args.pairs_file}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        
        # Read pairs file
        with open(args.pairs_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        # Apply limit if specified
        if args.limit:
            pairs = pairs[:args.limit]
        
        print(f"[DINOv3] Processing {len(pairs)} pairs...")
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            # Check if output already exists
            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            output_path = output_dir / f"{pair_name}.npz"
            
            if output_path.exists():
                continue
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[DINOv3] Skipping pair: {img1_name}, {img2_name} (file not found)")
                continue
            
            mkpts0, mkpts1, _, _ = process_pair(
                img1_path, img2_path, model, transform, args,
                feature_cache=args.feature_cache
            )
            
            # Save matches
            save_matches(output_path, mkpts0, mkpts1)
        
        print(f"[DINOv3] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode (backward compatible)
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        mkpts0, mkpts1, orig_size1, orig_size2 = process_pair(
            img1_path, img2_path, model, transform, args,
            feature_cache=args.feature_cache
        )
        
        print(f"[DINOv3] Found {len(mkpts0)} matches")
        
        # Save matches
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        
        # Visualize if requested or in single-pair mode by default
        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            
            vis_path = f"datasets/{img1_path.stem}_{img2_path.stem}_dinov3_matches.png"
            
            # Convert mkpts back to feature coords for visualization
            # (This maintains compatibility with visualize_matches)
            x1 = mkpts0[:, 0]
            y1 = mkpts0[:, 1]
            x2 = mkpts1[:, 0]
            y2 = mkpts1[:, 1]
            
            # For visualization, use pixel coords directly (1:1 mapping)
            visualize_matches(
                img1_np, img2_np,
                x1, y1, x2, y2,
                orig_size1, orig_size2,  # Treat as if feature map = image size
                out_path=vis_path,
                max_lines=args.max_lines,
            )
            print(f"[DINOv3] Saved visualization to {vis_path}")
    else:
        parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()


# Single-pair example:
# python3 scripts/dinov3_matches.py \
#   --img1 datasets/bran1.jpg \
#   --img2 datasets/bran2.jpg \
#   --img_size 518 \
#   --use_mutual \
#   --ratio_thresh 0.8

# Batch mode example:
# python3 scripts/dinov3_matches.py \
#   --pairs_file datasets/phototourism/sacre_coeur/pairs.txt \
#   --images_dir datasets/phototourism/sacre_coeur/dense/images \
#   --output_dir output/matches/dinov3 \
#   --use_mutual \
#   --ratio_thresh 0.8