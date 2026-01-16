#!/usr/bin/env python3
"""
Preprocess images for fair matcher comparison.

This script preprocesses all images using the resize + letterbox strategy:
1. Resize long edge to 1120px (divisible by 14 and 16)
2. Scale short edge proportionally (maintains aspect ratio)
3. Pad with black to make both dimensions divisible by 16
4. Save preprocessed images and transformation info

Usage:
    python3 scripts/preprocess_images.py \
        --images_dir datasets/phototourism/sacre_coeur/images \
        --output_dir datasets/phototourism/sacre_coeur/images_preprocessed \
        --target_size 1120
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add scripts to path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from util import preprocess_image, PreprocessInfo


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for fair matcher comparison")
    parser.add_argument("--images_dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed images")
    parser.add_argument("--target_size", type=int, default=1120, help="Target long edge size (default: 1120)")
    parser.add_argument("--divisibility", type=int, default=16, help="Pad dimensions to be divisible by (default: 16)")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"], help="Output format")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in images_dir.iterdir() if f.suffix in image_extensions]
    
    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Target long edge: {args.target_size}px")
    print(f"Divisibility: {args.divisibility}")
    print(f"Output directory: {output_dir}")
    
    # Store preprocessing info for all images
    preprocess_info = {}
    
    for img_path in tqdm(image_files, desc="Preprocessing"):
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess
        img_preprocessed, info = preprocess_image(
            img,
            target_long_edge=args.target_size,
            divisibility=args.divisibility,
            return_info=True,
        )
        
        # Save preprocessed image
        output_path = output_dir / f"{img_path.stem}.{args.format}"
        if args.format == "jpg":
            img_preprocessed.save(output_path, "JPEG", quality=args.quality)
        else:
            img_preprocessed.save(output_path, "PNG")
        
        # Store info
        preprocess_info[img_path.name] = {
            "scale": info.scale,
            "pad_left": info.pad_left,
            "pad_top": info.pad_top,
            "orig_size": list(info.orig_size),
            "resized_size": list(info.resized_size),
            "final_size": list(info.final_size),
        }
    
    # Save preprocessing info
    info_path = output_dir / "preprocess_info.json"
    with open(info_path, 'w') as f:
        json.dump(preprocess_info, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"  Preprocessed images: {output_dir}")
    print(f"  Preprocessing info: {info_path}")
    
    # Print sample info
    if preprocess_info:
        sample_name = next(iter(preprocess_info))
        sample = preprocess_info[sample_name]
        print(f"\nSample preprocessing ({sample_name}):")
        print(f"  Original size: {sample['orig_size']}")
        print(f"  Scale factor: {sample['scale']:.4f}")
        print(f"  Resized size: {sample['resized_size']}")
        print(f"  Final size (after padding): {sample['final_size']}")
        print(f"  Padding: left={sample['pad_left']}, top={sample['pad_top']}")


if __name__ == "__main__":
    main()
