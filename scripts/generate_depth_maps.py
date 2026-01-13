"""
UniDepth Depth Map Generation Script

Generates metric depth maps using UniDepth.
Can process images from a pairs file (only needed images) or all images in a directory.

Run in the UniDepth conda environment.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys

# Try importing UniDepth V2 first (doesn't need xformers), then V1
try:
    from unidepth.models import UniDepthV2
    UNIDEPTH_VERSION = 2
except ImportError:
    try:
        from unidepth.models import UniDepthV1
        UNIDEPTH_VERSION = 1
    except ImportError:
        print("Error: UniDepth not found. Make sure you're in the unidepth conda environment.")
        sys.exit(1)


def load_unidepth_model(backbone="vitl14", device="cuda"):
    """Load UniDepth model (V2 preferred, V1 fallback)."""
    print(f"[UniDepth] Loading UniDepthV{UNIDEPTH_VERSION} with backbone: {backbone}")
    
    if UNIDEPTH_VERSION == 2:
        model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{backbone}")
    else:
        model = UniDepthV1.from_pretrained(f"lpiccinelli/unidepth-v1-{backbone}")
    
    model = model.to(device)
    model.eval()
    return model


def process_image(img_path, model, device, max_size=None):
    """Process a single image and return depth map and intrinsics."""
    img = Image.open(img_path).convert("RGB")
    orig_size = (img.height, img.width)
    
    # Resize if image is too large (to save GPU memory)
    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.BILINEAR)
    
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        predictions = model.infer(img_tensor)
    
    depth = predictions["depth"].squeeze().cpu().numpy()
    K = predictions["intrinsics"].squeeze().cpu().numpy()
    
    # If resized, scale depth back to original resolution
    if max_size and max(orig_size) > max_size:
        from scipy.ndimage import zoom
        scale_h = orig_size[0] / depth.shape[0]
        scale_w = orig_size[1] / depth.shape[1]
        depth = zoom(depth, (scale_h, scale_w), order=1)
        # Adjust intrinsics for original size
        K[0, 0] *= scale_w  # fx
        K[1, 1] *= scale_h  # fy
        K[0, 2] *= scale_w  # cx
        K[1, 2] *= scale_h  # cy
    
    return depth, K, orig_size


def get_images_from_pairs(pairs_file, images_dir):
    """Extract unique image names from pairs file."""
    images = set()
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.add(parts[0])
                images.add(parts[1])
    
    # Convert to full paths
    image_paths = []
    for img_name in images:
        img_path = Path(images_dir) / img_name
        if img_path.exists():
            image_paths.append(img_path)
    
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps using UniDepth")
    
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save depth maps")
    parser.add_argument("--pairs_file", type=str, default=None,
                        help="Only process images in this pairs file (optional)")
    parser.add_argument("--backbone", type=str, default="vitl14",
                        choices=["vitl14", "vits14", "cnvnxtl"],
                        help="UniDepth backbone (default: vitl14)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_size", type=int, default=None,
                        help="Max image dimension to reduce GPU memory (e.g., 1024)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have depth maps")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[UniDepth] Using device: {device}")
    if args.max_size:
        print(f"[UniDepth] Max image size: {args.max_size}px")
    
    # Load model
    model = load_unidepth_model(backbone=args.backbone, device=device)
    
    # Setup directories
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image list
    if args.pairs_file:
        image_files = get_images_from_pairs(args.pairs_file, images_dir)
        print(f"[UniDepth] Processing {len(image_files)} images from pairs file")
    else:
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        print(f"[UniDepth] Processing all {len(image_files)} images in directory")
    
    # Process images
    processed = 0
    skipped = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Depth maps", ncols=80):
        stem = img_path.stem
        depth_path = output_dir / f"{stem}_depth.npy"
        K_path = output_dir / f"{stem}_K.npy"
        
        if args.skip_existing and depth_path.exists() and K_path.exists():
            skipped += 1
            continue
        
        try:
            depth, K, size = process_image(img_path, model, device, max_size=args.max_size)
            np.save(depth_path, depth)
            np.save(K_path, K)
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only show first 3 errors
                print(f"\n[UniDepth] Error: {img_path.name}: {e}")
            elif errors == 4:
                print("\n[UniDepth] Suppressing further error messages...")
    
    print(f"\n[UniDepth] Done. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
