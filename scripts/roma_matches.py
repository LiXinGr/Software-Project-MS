"""
RoMa Feature Matching Script

Uses RoMa (Robust Dense Feature Matching) for correspondence finding.
RoMa has its own matching head - no MNN needed.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add external/RoMa to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROMA_ROOT = PROJECT_ROOT / "external" / "RoMa"
sys.path.insert(0, str(ROMA_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from util import visualize_matches, save_matches

try:
    from romatch import roma_outdoor
except ImportError as e:
    print(f"Error importing RoMa: {e}")
    print(f"Make sure you are running with the correct environment.")
    sys.exit(1)


def process_pair(img1_path, img2_path, roma_model, args, device):
    """Process a single pair and return mkpts0, mkpts1."""
    # RoMa match
    warp, certainty = roma_model.match(str(img1_path), str(img2_path), device=device)
    
    # Sample matches
    matches, match_certainty = roma_model.sample(warp, certainty, num=args.max_points)
    
    # Get image dimensions
    im1 = Image.open(img1_path).convert("RGB")
    im2 = Image.open(img2_path).convert("RGB")
    W1, H1 = im1.size
    W2, H2 = im2.size
    
    # Convert to pixel coordinates
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H1, W1, H2, W2)
    
    # Convert to numpy arrays [N, 2]
    mkpts0 = kpts1.cpu().numpy()  # [N, 2] (x, y)
    mkpts1 = kpts2.cpu().numpy()  # [N, 2] (x, y)
    
    return mkpts0, mkpts1, (H1, W1), (H2, W2)


def main():
    parser = argparse.ArgumentParser(description="Match images using RoMa.")
    
    # Single-pair mode
    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    
    # Batch mode
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt")
    parser.add_argument("--images_dir", type=str, help="Base directory for images")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    
    # RoMa options
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[RoMa] Using device: {device}")

    # Initialize RoMa model
    roma_model = roma_outdoor(device=device)

    if args.pairs_file and args.output_dir:
        # Batch mode
        print(f"[RoMa] Batch mode: processing pairs from {args.pairs_file}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        
        with open(args.pairs_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        print(f"[RoMa] Processing {len(pairs)} pairs...")
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[RoMa] Skipping pair: {img1_name}, {img2_name}")
                continue
            
            try:
                mkpts0, mkpts1, _, _ = process_pair(
                    img1_path, img2_path, roma_model, args, device
                )
                
                pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
                save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
            except Exception as e:
                print(f"[RoMa] Error processing {img1_name}, {img2_name}: {e}")
        
        print(f"[RoMa] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        if not img1_path.exists() or not img2_path.exists():
            print(f"Error: Image not found")
            sys.exit(1)

        mkpts0, mkpts1, size1, size2 = process_pair(
            img1_path, img2_path, roma_model, args, device
        )
        
        print(f"[RoMa] Found {len(mkpts0)} matches")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        
        if args.visualize or not args.output_dir:
            datasets_dir = PROJECT_ROOT / "datasets"
            vis_path = datasets_dir / f"{img1_path.stem}_{img2_path.stem}_roma_matches.png"
            
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            
            visualize_matches(
                img1_np, img2_np,
                mkpts0[:, 0], mkpts0[:, 1],
                mkpts1[:, 0], mkpts1[:, 1],
                size1, size2,
                out_path=str(vis_path),
                max_lines=args.max_lines
            )
            print(f"[RoMa] Saved visualization to {vis_path}")
    else:
        parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()
