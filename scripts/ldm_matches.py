"""
LDM (Latent Diffusion Model) Feature Matching Script

Extracts features using LDM and matches them using MNN + ratio test.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LDM_ROOT = PROJECT_ROOT / "external" / "LDM_correspondences"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LDM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from util import match_dense_features, save_matches, visualize_matches

try:
    from utils.optimize_token import load_ldm
except ImportError as e:
    print(f"Error importing LDM: {e}")
    print("Make sure you are running with the LDM environment.")
    sys.exit(1)


def extract_ldm_features(img_path, ldm_model, args, device):
    """Extract LDM features for a single image."""
    from PIL import Image
    import torchvision.transforms as T
    
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    orig_size = (img.height, img.width)
    
    # Resize to model input size
    img_resized = img.resize((args.img_size, args.img_size), Image.BILINEAR)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # Extract features using LDM
    with torch.no_grad():
        # Get latent features from the model
        # Note: This may need adjustment based on the actual LDM API
        latent = ldm_model.encode(img_tensor)
        if hasattr(latent, 'sample'):
            features = latent.sample()
        else:
            features = latent
    
    # Features shape: [1, C, H, W] -> [C, H, W]
    features = features.squeeze(0)
    
    return features, orig_size


def process_pair(img1_path, img2_path, ldm_model, args, device):
    """Process a single pair and return mkpts0, mkpts1."""
    ft1, orig_size1 = extract_ldm_features(img1_path, ldm_model, args, device)
    ft2, orig_size2 = extract_ldm_features(img2_path, ldm_model, args, device)
    
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
        description="Match images using LDM features"
    )

    # Single-pair mode
    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    
    # Batch mode
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt")
    parser.add_argument("--images_dir", type=str, help="Base directory for images")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    
    # Matching options
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=0.8)
    
    # LDM parameters
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[LDM] Using device: {device}")

    # Load LDM model
    print("[LDM] Loading model...")
    ldm_model = load_ldm(device=device)
    ldm_model.eval()

    if args.pairs_file and args.output_dir:
        # Batch mode
        print(f"[LDM] Batch mode: processing pairs from {args.pairs_file}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        
        with open(args.pairs_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        print(f"[LDM] Processing {len(pairs)} pairs...")
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[LDM] Skipping: {img1_name}, {img2_name}")
                continue
            
            try:
                mkpts0, mkpts1, _, _ = process_pair(
                    img1_path, img2_path, ldm_model, args, device
                )
                
                pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
                save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
            except Exception as e:
                print(f"[LDM] Error: {e}")
        
        print(f"[LDM] Saved matches to {output_dir}")
        
    elif args.img1 and args.img2:
        # Single-pair mode
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        mkpts0, mkpts1, orig_size1, orig_size2 = process_pair(
            img1_path, img2_path, ldm_model, args, device
        )
        
        print(f"[LDM] Found {len(mkpts0)} matches")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        
        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            
            vis_path = f"datasets/{img1_path.stem}_{img2_path.stem}_ldm_matches.png"
            
            if len(mkpts0) > 0:
                visualize_matches(
                    img1_np, img2_np,
                    mkpts0[:, 0], mkpts0[:, 1],
                    mkpts1[:, 0], mkpts1[:, 1],
                    orig_size1, orig_size2,
                    out_path=vis_path,
                    max_lines=args.max_lines,
                )
                print(f"[LDM] Saved visualization to {vis_path}")
    else:
        parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()

# Single-pair example:
# python3 scripts/ldm_matches.py --img1 datasets/bran1.jpg --img2 datasets/bran2.jpg

# Batch mode example:
# python3 scripts/ldm_matches.py \
#   --pairs_file datasets/phototourism/sacre_coeur/pairs.txt \
#   --images_dir datasets/phototourism/sacre_coeur/dense/images \
#   --output_dir output/matches/ldm
