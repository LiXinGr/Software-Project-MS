import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import warnings

# Add external/RoMa to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROMA_ROOT = PROJECT_ROOT / "external" / "RoMa"
sys.path.insert(0, str(ROMA_ROOT))

# Import util from current script directory
sys.path.insert(0, str(SCRIPT_DIR))
from util import visualize_matches

try:
    from romatch import roma_outdoor
except ImportError as e:
    print(f"Error importing RoMa: {e}")
    print(f"Make sure you are running this script with the correct environment and that '{ROMA_ROOT}' contains the RoMa code.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Match two images using RoMa.")
    parser.add_argument("--img1", required=True, type=str, help="Path to first image")
    parser.add_argument("--img2", required=True, type=str, help="Path to second image")
    parser.add_argument("--max_points", type=int, default=2000, help="Max number of keypoints to sample/visualize")
    parser.add_argument("--max_lines", type=int, default=200, help="Max number of lines to draw in visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g. cuda, cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    
    if not img1_path.exists():
        print(f"Error: Image 1 not found at {img1_path}")
        sys.exit(1)
    if not img2_path.exists():
        print(f"Error: Image 2 not found at {img2_path}")
        sys.exit(1)

    # Initialize RoMa model
    # Note: RoMa defaults: coarse_res=560, upsample_res=(864, 1152)
    # We can stick to defaults or allow configuration. For now, defaults are fine.
    roma_model = roma_outdoor(device=device)

    print(f"Matching {img1_path} and {img2_path}...")
    
    # RoMa match
    # match() takes paths or PIL images
    warp, certainty = roma_model.match(str(img1_path), str(img2_path), device=device)

    # Sample matches
    # sample() returns matches in [-1, 1] coordinate space
    # The 'num' argument controls how many matches to sample based on certainty
    matches, match_certainty = roma_model.sample(warp, certainty, num=args.max_points)
    
    # Convert to pixel coordinates
    # We need image dimensions for this
    im1 = Image.open(img1_path).convert("RGB")
    im2 = Image.open(img2_path).convert("RGB")
    W1, H1 = im1.size
    W2, H2 = im2.size
    
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H1, W1, H2, W2)
    
    # Convert to numpy for visualization
    x1 = kpts1[:, 0].cpu().numpy()
    y1 = kpts1[:, 1].cpu().numpy()
    x2 = kpts2[:, 0].cpu().numpy()
    y2 = kpts2[:, 1].cpu().numpy()
    
    # Prepare pseudo "feature map sizes" for util.visualize_matches
    # Since we have exact pixel coordinates, we can tell visualize_matches that 
    # the feature map size IS the image size. This essentially disables the scaling 
    # logic in visualize_matches (scale = 1).
    feat_hw1 = (H1, W1)
    feat_hw2 = (H2, W2)
    
    # Prepare output path
    # Using the same naming convention as other scripts
    # assuming they save to datasets/ or current dir relative to script
    datasets_dir = PROJECT_ROOT / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    vis_path = datasets_dir / f"{img1_path.stem}_{img2_path.stem}_roma_matches.png"
    
    print(f"Found {len(x1)} matches. Visualizing...")
    
    visualize_matches(
        np.array(im1), 
        np.array(im2), 
        x1, y1, x2, y2,
        feat_hw1, feat_hw2, 
        out_path=str(vis_path), 
        max_lines=args.max_lines
    )
    
    print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    main()
