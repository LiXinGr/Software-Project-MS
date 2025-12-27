import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from util import compute_matches, visualize_matches

def main():
    parser = argparse.ArgumentParser(description="Image matching with SuperPoint and LightGlue/NN")
    
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")
    parser.add_argument("--matcher", type=str, choices=["nn", "lightglue"], default="lightglue",
                        help="Choose 'nn' for simple nearest neighbor matching or 'lightglue' for the transformer-based matcher")
    parser.add_argument("--max_points", type=int, default=2048, help="Maximum number of keypoints to extract")
    parser.add_argument("--use_mutual", action="store_true", help="Use mutual nearest neighbors (only for NN matcher)")
    parser.add_argument("--ratio_thresh", type=float, default=None, help="Lowe's ratio test threshold (only for NN matcher)")
    parser.add_argument("--max_lines", type=int, default=200, help="Maximum number of matches to visualize")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load images as tensors
    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    try:
        # load_image returns [3, H, W]
        image0_tensor = load_image(img1_path).to(device)
        image1_tensor = load_image(img2_path).to(device)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # SuperPoint Extractor
    extractor = SuperPoint(max_num_keypoints=args.max_points).eval().to(device)

    print("Extracting features with SuperPoint...")
    with torch.no_grad():
        feats0 = extractor.extract(image0_tensor.unsqueeze(0)) # [1, 3, H, W]
        feats1 = extractor.extract(image1_tensor.unsqueeze(0))

    kpts0 = feats0['keypoints'][0] # [N, 2]
    desc0 = feats0['descriptors'][0] # Check shape below
    kpts1 = feats1['keypoints'][0]
    desc1 = feats1['descriptors'][0]
    
    # Ensure descriptors are [D, N] for util compatibility if valid
    # LightGlue features usually [B, N, D] or [B, D, N]. 
    # If [N, D], we need transpose for NN (to match expected "channels" dim).
    # If [D, N], it is fine.
    # Note: we will check shape dynamically.
    if desc0.shape[-1] == 256: # If [N, 256]
        desc0_t = desc0.t() # [256, N]
        desc1_t = desc1.t()
    else:
        desc0_t = desc0
        desc1_t = desc1

    x1_out, y1_out, x2_out, y2_out = [], [], [], []
    H1, W1 = image0_tensor.shape[1], image0_tensor.shape[2]
    H2, W2 = image1_tensor.shape[1], image1_tensor.shape[2]

    if args.matcher == 'nn':
        print("Matching with Nearest Neighbors...")
        # Prepare for compute_matches: expects [C, H, W]
        # We can fake it as [C, 1, N]
        ft1 = desc0_t.unsqueeze(1) # [256, 1, N]
        ft2 = desc1_t.unsqueeze(1) # [256, 1, N]

        # match
        x1_idx, _, x2_idx, _, _, _ = compute_matches(
            ft1, ft2,
            max_points=args.max_points,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh
        )
        
        # x1_idx corresponds to flat index, which is index in N
        if len(x1_idx) > 0:
            kpts0_matched = kpts0[x1_idx].cpu().numpy() # [K, 2]
            kpts1_matched = kpts1[x2_idx].cpu().numpy()
            
            x1_out = kpts0_matched[:, 0]
            y1_out = kpts0_matched[:, 1]
            x2_out = kpts1_matched[:, 0]
            y2_out = kpts1_matched[:, 1]
        
    else:
        print("Matching with LightGlue...")
        """
        n_layers: Number of stacked self+cross attention layers. Reduce this value for faster inference at the cost of accuracy (continuous red line in the plot above). Default: 9 (all layers).
        flash: Enable FlashAttention. Significantly increases the speed and reduces the memory consumption without any impact on accuracy. Default: True (LightGlue automatically detects if FlashAttention is available).
        mp: Enable mixed precision inference. Default: False (off)
        depth_confidence: Controls the early stopping. A lower values stops more often at earlier layers. Default: 0.95, disable with -1.
        width_confidence: Controls the iterative point pruning. A lower value prunes more points earlier. Default: 0.99, disable with -1.
        filter_threshold: Match confidence. Increase this value to obtain less, but stronger matches. Default: 0.1
        """
        
        matcher = LightGlue(features='superpoint').eval().to(device)
        
        with torch.no_grad():
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            matches = matches01['matches'][0] # [M, 2] indices
            scores = matches01['scores'][0]
            
            m_kpts0 = kpts0[matches[:, 0]].cpu().numpy()
            m_kpts1 = kpts1[matches[:, 1]].cpu().numpy()
            
            if len(m_kpts0) > 0:
                x1_out = m_kpts0[:, 0]
                y1_out = m_kpts0[:, 1]
                x2_out = m_kpts1[:, 0]
                y2_out = m_kpts1[:, 1]

    print(f"Found {len(x1_out)} matches.")

    # Visualization
    print("Visualizing...")
    img1_np = np.array(Image.open(img1_path).convert("RGB"))
    img2_np = np.array(Image.open(img2_path).convert("RGB"))

    vis_path = f"datasets/{img1_path.stem}_{img2_path.stem}_superpoint_{args.matcher}_matches.png"
    
    # Adjust coordinates for visualize_matches:
    # It assumes (x + 0.5) * scale = original_coord
    # We want original_coord. So we pass (original_coord) - 0.5 if scale=1.
    
    if len(x1_out) > 0:
        visualize_matches(
            img1_np,
            img2_np,
            x1_out - 0.5, y1_out - 0.5,
            x2_out - 0.5, y2_out - 0.5,
            (H1, W1), (H2, W2),
            out_path=vis_path,
            max_lines=args.max_lines,
        )
        print(f"Saved visualization to {vis_path}")
    else:
        print("No matches found to visualize.")

if __name__ == "__main__":
    main()


# # Test NN matcher
# python3 scripts/superpoint_matches.py --img1 datasets/bran1.jpg --img2 datasets/bran2.jpg --matcher nn --max_points 100 --device cuda
# # Test LightGlue matcher (using CPU for generic compatibility if GPU not available, or cuda)
# python3 scripts/superpoint_matches.py --img1 datasets/bran1.jpg --img2 datasets/bran2.jpg --matcher lightglue --max_points 100 --device cpu