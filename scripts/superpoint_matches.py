"""
SuperPoint + LightGlue Feature Matching Script

Extracts features using SuperPoint and matches using LightGlue or NN.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from util import compute_matches, visualize_matches, save_matches


def extract_features(img_path, extractor, device, cache_dir=None):
    """Extract SuperPoint features, with optional caching."""
    if cache_dir:
        cache_path = Path(cache_dir) / f"{Path(img_path).stem}_superpoint.pt"
        if cache_path.exists():
            return torch.load(cache_path)
    
    image_tensor = load_image(img_path).to(device)
    
    with torch.no_grad():
        feats = extractor.extract(image_tensor.unsqueeze(0))
    
    if cache_dir:
        cache_path = Path(cache_dir) / f"{Path(img_path).stem}_superpoint.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
    
    return feats


def match_lightglue(feats0, feats1, matcher):
    """Match using LightGlue."""
    with torch.no_grad():
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        matches = matches01['matches'][0]  # [M, 2]
        
        kpts0 = feats0['keypoints'][0]
        kpts1 = feats1['keypoints'][0]
        
        mkpts0 = kpts0[matches[:, 0]].cpu().numpy()
        mkpts1 = kpts1[matches[:, 1]].cpu().numpy()
    
    return mkpts0, mkpts1


def match_nn(feats0, feats1, max_points, use_mutual, ratio_thresh):
    """Match using nearest neighbor with optional MNN and ratio test."""
    kpts0 = feats0['keypoints'][0]
    kpts1 = feats1['keypoints'][0]
    desc0 = feats0['descriptors'][0]
    desc1 = feats1['descriptors'][0]
    
    # Transpose descriptors if needed [N, D] -> [D, N]
    if desc0.shape[-1] == 256:
        desc0 = desc0.t()
        desc1 = desc1.t()
    
    # Create pseudo feature maps [D, 1, N]
    ft1 = desc0.unsqueeze(1)
    ft2 = desc1.unsqueeze(1)
    
    # Match
    x1_idx, _, x2_idx, _, _, _ = compute_matches(
        ft1, ft2,
        max_points=max_points,
        use_mutual=use_mutual,
        ratio_thresh=ratio_thresh
    )
    
    if len(x1_idx) > 0:
        mkpts0 = kpts0[x1_idx].cpu().numpy()
        mkpts1 = kpts1[x2_idx].cpu().numpy()
    else:
        mkpts0 = np.zeros((0, 2))
        mkpts1 = np.zeros((0, 2))
    
    return mkpts0, mkpts1


def process_pair(img1_path, img2_path, extractor, matcher, args, device, cache_dir=None):
    """Process a single pair."""
    feats0 = extract_features(img1_path, extractor, device, cache_dir)
    feats1 = extract_features(img2_path, extractor, device, cache_dir)
    
    if args.matcher == 'lightglue':
        mkpts0, mkpts1 = match_lightglue(feats0, feats1, matcher)
    else:
        mkpts0, mkpts1 = match_nn(
            feats0, feats1, 
            args.max_points,
            args.use_mutual,
            args.ratio_thresh
        )
    
    # Get image sizes
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    size1 = (img1.height, img1.width)
    size2 = (img2.height, img2.width)
    
    return mkpts0, mkpts1, size1, size2


def main():
    parser = argparse.ArgumentParser(description="Image matching with SuperPoint + LightGlue/NN")
    
    # Single-pair mode
    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    
    # Batch mode
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt")
    parser.add_argument("--images_dir", type=str, help="Base directory for images")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    
    # Matching options
    parser.add_argument("--matcher", type=str, choices=["nn", "lightglue"], default="lightglue")
    parser.add_argument("--max_points", type=int, default=2048)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=0.8)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[SuperPoint] Using device: {device}")

    # Initialize models
    extractor = SuperPoint(max_num_keypoints=args.max_points).eval().to(device)
    matcher = None
    if args.matcher == 'lightglue':
        matcher = LightGlue(features='superpoint').eval().to(device)
    
    if args.pairs_file and args.output_dir:
        # Batch mode
        print(f"[SuperPoint] Batch mode: processing pairs from {args.pairs_file}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        
        with open(args.pairs_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        print(f"[SuperPoint] Processing {len(pairs)} pairs with {args.matcher}...")
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[SuperPoint] Skipping: {img1_name}, {img2_name}")
                continue
            
            try:
                mkpts0, mkpts1, _, _ = process_pair(
                    img1_path, img2_path, 
                    extractor, matcher, args, device,
                    cache_dir=args.feature_cache
                )
                
                pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
                save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
            except Exception as e:
                print(f"[SuperPoint] Error: {e}")
        
        print(f"[SuperPoint] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        mkpts0, mkpts1, size1, size2 = process_pair(
            img1_path, img2_path,
            extractor, matcher, args, device,
            cache_dir=args.feature_cache
        )
        
        print(f"[SuperPoint] Found {len(mkpts0)} matches")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        
        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            
            vis_path = f"datasets/{img1_path.stem}_{img2_path.stem}_superpoint_{args.matcher}_matches.png"
            
            if len(mkpts0) > 0:
                visualize_matches(
                    img1_np, img2_np,
                    mkpts0[:, 0], mkpts0[:, 1],
                    mkpts1[:, 0], mkpts1[:, 1],
                    size1, size2,
                    out_path=vis_path,
                    max_lines=args.max_lines,
                )
                print(f"[SuperPoint] Saved visualization to {vis_path}")
    else:
        parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()