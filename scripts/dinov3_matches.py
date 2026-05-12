"""
DINOv3 Feature Matching Script

Extracts features using DINOv3 ViT-Large/patch16 (vit_large_patch16_dinov3.lvd1689m)
and matches them using MNN + ratio test.

Supports two modes:
1. Single-pair mode: --img1 and --img2 arguments
2. Batch mode: --pairs_file and --output_dir arguments for benchmarking
"""

import sys
import torch
from PIL import Image
import torchvision.transforms as T
import timm
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from util import (
    compute_matches,
    visualize_matches,
    match_dense_features,
    save_matches,
    match_at_keypoints,
    get_superpoint_keypoints,
    preprocess_image,
    map_points_to_original,
    reset_peak_memory,
    current_peak_memory_mb,
    timed_feature_load,
    save_timing_json,
)


PATCH_SIZE = 16  # DINOv3 ViT-Large uses patch size 16


def _map_keypoints_to_preprocessed(kpts, info, device):
    if kpts.numel() == 0:
        return kpts.to(device)
    out = kpts.clone().float().to(device)
    out[:, 0] = out[:, 0] * info.scale + info.pad_left
    out[:, 1] = out[:, 1] * info.scale + info.pad_top
    return out


def _map_points_to_original_np(points, info):
    if len(points) == 0:
        return points
    mapped = map_points_to_original(points, info).astype(np.float32, copy=False)
    h, w = info.orig_size
    mapped[:, 0] = np.clip(mapped[:, 0], 0, w - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, h - 1)
    return mapped


def get_config_key(args):
    """Return a canonical string identifying this configuration."""
    parts = ["dinov3", f"l{args.feat_level}"]
    if args.snap_to_grid:
        parts.append("gridalign")
        parts.append(f"out{args.snap_output_coords}")
    elif args.dense_grid:
        parts.append("dense16")
    elif args.use_sp_keypoints:
        parts.append("sp")
    else:
        parts.append("dense")
    parts.append("mnn" if args.use_mutual else "nn")
    if args.ratio_thresh:
        parts.append(f"rt{args.ratio_thresh}")
    parts.append(f"mp{args.max_points}")
    return "_".join(parts)


def _snap_to_grid_indices(kpts, feat_shape, device):
    """
    Snap pixel keypoints to nearest DINOv3 patch center grid indices.

    Args:
        kpts: [N, 2] tensor of (x, y) pixel coords
        feat_shape: (H_feat, W_feat)
        device: torch device

    Returns:
        hi: [M] long tensor of feature row indices (deduplicated)
        wi: [M] long tensor of feature col indices (deduplicated)
        px_snap: [M] float tensor of snapped pixel x coords
        py_snap: [M] float tensor of snapped pixel y coords
        px_orig: [M] float tensor of original keypoint x coords (representative per snapped cell)
        py_orig: [M] float tensor of original keypoint y coords (representative per snapped cell)
    """
    H_feat, W_feat = feat_shape
    x = kpts[:, 0].float()
    y = kpts[:, 1].float()

    wi = ((x - 8) / PATCH_SIZE).round().long().clamp(0, W_feat - 1)
    hi = ((y - 8) / PATCH_SIZE).round().long().clamp(0, H_feat - 1)

    flat_idx = hi * W_feat + wi
    # Keep one representative original keypoint per snapped grid cell.
    # np.unique returns sorted unique values and the first index where each occurs.
    flat_idx_np = flat_idx.cpu().numpy()
    unique_flat_np, first_idx_np = np.unique(flat_idx_np, return_index=True)
    unique_flat = torch.from_numpy(unique_flat_np).to(device)
    first_idx = torch.from_numpy(first_idx_np).long()

    hi_out = (unique_flat // W_feat).to(device)
    wi_out = (unique_flat % W_feat).to(device)
    px_snap = (8 + PATCH_SIZE * wi_out).float()
    py_snap = (8 + PATCH_SIZE * hi_out).float()
    px_orig = kpts[first_idx, 0].float().to(device)
    py_orig = kpts[first_idx, 1].float().to(device)

    return hi_out, wi_out, px_snap, py_snap, px_orig, py_orig


def _get_dense_grid_indices(feat_shape, device):
    """
    Generate all DINOv3 patch center grid indices for an image.

    Args:
        feat_shape: (H_feat, W_feat)
        device: torch device

    Returns:
        hi: [N] long tensor of feature row indices
        wi: [N] long tensor of feature col indices
        px: [N] float tensor of pixel x coords (patch centers)
        py: [N] float tensor of pixel y coords (patch centers)
    """
    H_feat, W_feat = feat_shape
    hi = torch.arange(H_feat, device=device).repeat_interleave(W_feat)
    wi = torch.arange(W_feat, device=device).repeat(H_feat)
    px = (8 + PATCH_SIZE * wi).float()
    py = (8 + PATCH_SIZE * hi).float()
    return hi, wi, px, py


def _match_at_grid_indices(
    ft1, ft2,
    hi1, wi1, px1, py1,
    hi2, wi2, px2, py2,
    use_mutual=True,
    ratio_thresh=None,
    out_px1=None,
    out_py1=None,
    out_px2=None,
    out_py2=None,
    max_matches=None,
):
    """
    Match DINOv3 features at exact grid indices (no bilinear interpolation).

    Args:
        ft1, ft2: [C, H_feat, W_feat] feature maps on device
        hi1, wi1: [N1] row/col indices into ft1
        px1, py1: [N1] pixel x/y coords for ft1 descriptor lookup points
        hi2, wi2: [N2] row/col indices into ft2
        px2, py2: [N2] pixel x/y coords for ft2 descriptor lookup points
        use_mutual: apply mutual nearest neighbor
        ratio_thresh: Lowe's ratio test threshold (None = disabled)
        out_px1, out_py1: optional [N1] output coords for image 1 (defaults to px1/py1)
        out_px2, out_py2: optional [N2] output coords for image 2 (defaults to px2/py2)

    Returns:
        mkpts0: [M, 2] numpy array (x, y) in image 1 pixel space
        mkpts1: [M, 2] numpy array (x, y) in image 2 pixel space
    """
    device = ft1.device
    if out_px1 is None:
        out_px1, out_py1 = px1, py1
    if out_px2 is None:
        out_px2, out_py2 = px2, py2

    desc1 = ft1[:, hi1, wi1].t()  # [N1, C]
    desc2 = ft2[:, hi2, wi2].t()  # [N2, C]

    desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + 1e-8)
    desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + 1e-8)

    N1, N2 = desc1.shape[0], desc2.shape[0]
    if N1 == 0 or N2 == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    sim = desc1 @ desc2.t()  # [N1, N2]

    best_sim, best_j = sim.max(dim=1)
    rows_all = torch.arange(N1, device=device)
    rows = rows_all

    if use_mutual:
        rev_best_i = sim.argmax(dim=0)
        mutual_mask = rev_best_i[best_j] == rows_all
        rows = rows_all[mutual_mask]
        best_j = best_j[mutual_mask]
        best_sim = best_sim[mutual_mask]
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

    if ratio_thresh is not None and N2 >= 2:
        sim_ratio_thresh = ratio_thresh if ratio_thresh > 1.0 else 1.0 / ratio_thresh
        top2_sim, _ = sim.topk(2, dim=1)
        ratio = top2_sim[:, 0] / (top2_sim[:, 1] + 1e-8)
        good = ratio[rows] >= sim_ratio_thresh
        rows = rows[good]
        best_j = best_j[good]
        best_sim = best_sim[good]
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

    if max_matches is not None and rows.numel() > max_matches:
        order = torch.argsort(best_sim, descending=True, stable=True)[:max_matches]
        rows = rows[order]
        best_j = best_j[order]

    mkpts0 = torch.stack([out_px1[rows], out_py1[rows]], dim=1).cpu().numpy()
    mkpts1 = torch.stack([out_px2[best_j], out_py2[best_j]], dim=1).cpu().numpy()
    return mkpts0, mkpts1


def run_dinov3_extractor(
    img_path,
    model,
    transform,
    feat_level,
    target_long_edge=1120,
    cache_dir=None,
):
    """
    Extract a single DINOv3 feature map [C, H_feat, W_feat] for one image.
    Loads the raw image, resizes its long edge to target_long_edge, then pads to
    a PATCH_SIZE multiple for DINOv3. Returns the feature map, raw image size,
    and preprocessing info for coordinate mapping.
    """
    img = Image.open(img_path).convert("RGB")
    raw_W, raw_H = img.size  # PIL gives (W, H)
    raw_size = (raw_H, raw_W)
    img_proc, prep_info = preprocess_image(
        img,
        target_long_edge=target_long_edge,
        divisibility=PATCH_SIZE,
        return_info=True,
    )
    W, H = img_proc.size

    # Check cache — key encodes actual image dimensions
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / (
            f"{Path(img_path).stem}_dinov3_raw{raw_W}x{raw_H}_prep{W}x{H}_l{feat_level}.pt"
        )
        if cache_path.exists():
            return (torch.load(cache_path), raw_size, prep_info), True

    x = transform(img_proc).unsqueeze(0).to(next(model.parameters()).device)  # [1, 3, H, W]

    # Pad to nearest multiple of patch size (preprocessed images are already div-by-16,
    # but guard against edge cases)
    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

    model.eval()
    with torch.no_grad():
        feats = model(x)

    ft = feats[0].squeeze(0).cpu()  # [C, H_feat, W_feat]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ft, cache_path)

    return (ft, raw_size, prep_info), False


def process_pair(
    img1_path,
    img2_path,
    model,
    transform,
    args,
    feature_cache=None,
    feature_timings=None,
    seen_images=None,
):
    """Process a single pair and return mkpts0, mkpts1."""
    ft1, orig_size1, prep_info1 = timed_feature_load(
        str(img1_path),
        feature_timings,
        seen_images,
        lambda: run_dinov3_extractor(
            img1_path, model, transform,
            args.feat_level,
            target_long_edge=args.img_size,
            cache_dir=feature_cache,
        ),
    )

    ft2, orig_size2, prep_info2 = timed_feature_load(
        str(img2_path),
        feature_timings,
        seen_images,
        lambda: run_dinov3_extractor(
            img2_path, model, transform,
            args.feat_level,
            target_long_edge=args.img_size,
            cache_dir=feature_cache,
        ),
    )
    

    
    # Move to device for matching
    device = next(model.parameters()).device
    ft1 = ft1.to(device)
    ft2 = ft2.to(device)
    
    if args.snap_to_grid:
        # Mode A: Snap SuperPoint keypoints to nearest DINOv3 patch center
        sp_cache = Path(args.sp_cache_dir) if args.sp_cache_dir else (Path(feature_cache).parent / 'superpoint_kpts' if feature_cache else None)
        kpts1_original = get_superpoint_keypoints(img1_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts2_original = get_superpoint_keypoints(img2_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts1_prep = _map_keypoints_to_preprocessed(kpts1_original, prep_info1, device)
        kpts2_prep = _map_keypoints_to_preprocessed(kpts2_original, prep_info2, device)

        hi1, wi1, px1_snap, py1_snap, px1_orig, py1_orig = _snap_to_grid_indices(
            kpts1_prep.cpu(), ft1.shape[1:], device
        )
        hi2, wi2, px2_snap, py2_snap, px2_orig, py2_orig = _snap_to_grid_indices(
            kpts2_prep.cpu(), ft2.shape[1:], device
        )

        if args.snap_output_coords == "snapped":
            out_px1, out_py1 = px1_snap, py1_snap
            out_px2, out_py2 = px2_snap, py2_snap
        else:
            out_px1, out_py1 = px1_orig, py1_orig
            out_px2, out_py2 = px2_orig, py2_orig

        mkpts0, mkpts1 = _match_at_grid_indices(
            ft1, ft2,
            hi1, wi1, px1_snap, py1_snap,
            hi2, wi2, px2_snap, py2_snap,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh,
            out_px1=out_px1,
            out_py1=out_py1,
            out_px2=out_px2,
            out_py2=out_py2,
            max_matches=args.max_points,
        )
        mkpts0 = _map_points_to_original_np(mkpts0, prep_info1)
        mkpts1 = _map_points_to_original_np(mkpts1, prep_info2)
    elif args.dense_grid:
        # Mode B: Use all DINOv3 patch centers as keypoints
        hi1, wi1, px1, py1 = _get_dense_grid_indices(ft1.shape[1:], device)
        hi2, wi2, px2, py2 = _get_dense_grid_indices(ft2.shape[1:], device)

        mkpts0, mkpts1 = _match_at_grid_indices(
            ft1, ft2,
            hi1, wi1, px1, py1,
            hi2, wi2, px2, py2,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh,
            max_matches=args.max_points,
        )
        mkpts0 = _map_points_to_original_np(mkpts0, prep_info1)
        mkpts1 = _map_points_to_original_np(mkpts1, prep_info2)
    elif args.use_sp_keypoints:
        # Original SP mode: bilinear interpolation at SuperPoint keypoints
        sp_cache = Path(args.sp_cache_dir) if args.sp_cache_dir else (Path(feature_cache).parent / 'superpoint_kpts' if feature_cache else None)
        kpts1_raw = get_superpoint_keypoints(img1_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts2_raw = get_superpoint_keypoints(img2_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache)
        kpts1 = _map_keypoints_to_preprocessed(kpts1_raw, prep_info1, device)
        kpts2 = _map_keypoints_to_preprocessed(kpts2_raw, prep_info2, device)

        mkpts0_prep, mkpts1_prep = match_at_keypoints(
            ft1, ft2,
            kpts1, kpts2,
            img_size1=prep_info1.final_size,
            img_size2=prep_info2.final_size,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh,
        )
        mkpts0 = _map_points_to_original_np(mkpts0_prep, prep_info1)
        mkpts1 = _map_points_to_original_np(mkpts1_prep, prep_info2)
    else:
        # Original dense mode: random sampling with bilinear interpolation
        mkpts0_prep, mkpts1_prep = match_dense_features(
            ft1, ft2,
            img_size1=prep_info1.final_size,
            img_size2=prep_info2.final_size,
            max_points=args.max_points,
            use_mutual=args.use_mutual,
            ratio_thresh=args.ratio_thresh,
        )
        mkpts0 = _map_points_to_original_np(mkpts0_prep, prep_info1)
        mkpts1 = _map_points_to_original_np(mkpts1_prep, prep_info2)
    
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
    parser.add_argument("--images_dir", "--image_dir", dest="images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches (batch mode)")
    parser.add_argument("--scene", type=str, default=None, help="Scene name for timing metadata")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=1120, 
                        help="Image size for feature extraction (default: 1120 to match preprocessing)")
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
    parser.add_argument("--sp_cache_dir", type=str, default=None,
                        help="Directory containing/raw-writing SuperPoint keypoint cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization images")
    parser.add_argument("--use_sp_keypoints", action="store_true",
                        help="Use SuperPoint-detected keypoints with bilinear interpolation (original SP mode)")
    parser.add_argument("--snap_to_grid", action="store_true",
                        help="Mode A: snap SuperPoint keypoints to nearest DINOv3 patch center (no interpolation)")
    parser.add_argument("--snap_output_coords", choices=["original", "snapped"], default="original",
                        help="For --snap_to_grid, report either original SuperPoint coords or snapped patch-center coords")
    parser.add_argument("--dense_grid", action="store_true",
                        help="Mode B: use all DINOv3 patch centers as keypoints (no SuperPoint, no interpolation)")
    parser.add_argument("--print_config_key", action="store_true",
                        help="Print the config key for this configuration and exit")
    parser.add_argument("--raw_images", action="store_true",
                        help="Compatibility flag: input images are raw and internally resized for DINOv3")
    parser.add_argument("--timing_output", type=str, default=None,
                        help="Path to write per-pair and per-image timing JSON")

    args = parser.parse_args()

    if args.print_config_key:
        print(get_config_key(args))
        sys.exit(0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model with specific layer extraction
    # DINOv3 ViT-L has 24 transformer blocks (indices 0-23)
    # Negative indices work: -1 = last (23), -12 = block 12

    # Convert negative index to positive for timm
    num_blocks = 24  # ViT-L has 24 blocks
    if args.feat_level < 0:
        out_idx = num_blocks + args.feat_level
    else:
        out_idx = args.feat_level

    print(f"[DINOv3] Extracting features from block {out_idx} (feat_level={args.feat_level})")

    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True,
        features_only=True,
        out_indices=[out_idx],  # Extract only this specific layer
        dynamic_img_size=True,  # Support non-square images via RoPE
    )
    model.to(device)
    model.eval()
    reset_peak_memory(device)

    # Use model-specific normalization (ImageNet stats for DINOv3)
    data_config = timm.data.resolve_model_data_config(model)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

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
        pair_timings = []
        feature_timings = []
        seen_images = set()
        skipped_existing = 0
        skipped_missing = 0
        
        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            
            # Check if output already exists
            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            output_path = output_dir / f"{pair_name}.npz"
            
            if output_path.exists():
                skipped_existing += 1
                continue
            
            if not img1_path.exists() or not img2_path.exists():
                print(f"[DINOv3] Skipping pair: {img1_name}, {img2_name} (file not found)")
                skipped_missing += 1
                continue
            
            t_start = time.time()
            mkpts0, mkpts1, _, _ = process_pair(
                img1_path, img2_path, model, transform, args,
                feature_cache=args.feature_cache,
                feature_timings=feature_timings,
                seen_images=seen_images,
            )
            
            # Save matches
            save_matches(output_path, mkpts0, mkpts1)
            pair_timings.append(
                {
                    "img0": img1_name,
                    "img1": img2_name,
                    "time_ms": (time.time() - t_start) * 1000.0,
                    "num_matches": int(len(mkpts0)),
                }
            )
        
        print(f"[DINOv3] Saved matches to {output_dir}")
        save_timing_json(
            args.timing_output,
            config_key=get_config_key(args),
            scene=args.scene or output_dir.name,
            pair_timings=pair_timings,
            feature_timings=feature_timings,
            peak_mem_mb=current_peak_memory_mb(device),
            extra={
                "skipped_existing": skipped_existing,
                "skipped_missing": skipped_missing,
                "target_long_edge": args.img_size,
                "coordinate_frame": "raw",
            },
        )
        
    elif args.img1 and args.img2:
        # Single-pair mode (backward compatible)
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        
        feature_timings = []
        seen_images = set()
        t_start = time.time()
        mkpts0, mkpts1, orig_size1, orig_size2 = process_pair(
            img1_path, img2_path, model, transform, args,
            feature_cache=args.feature_cache,
            feature_timings=feature_timings,
            seen_images=seen_images,
        )
        pair_timings = [{
            "img0": img1_path.name,
            "img1": img2_path.name,
            "time_ms": (time.time() - t_start) * 1000.0,
            "num_matches": int(len(mkpts0)),
        }]
        
        print(f"[DINOv3] Found {len(mkpts0)} matches")
        
        # Save matches
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{img1_path.stem}__{img2_path.stem}"
            save_matches(output_dir / f"{pair_name}.npz", mkpts0, mkpts1)
        save_timing_json(
            args.timing_output,
            config_key=get_config_key(args),
            scene=args.scene or "single_pair",
            pair_timings=pair_timings,
            feature_timings=feature_timings,
            peak_mem_mb=current_peak_memory_mb(device),
            extra={"target_long_edge": args.img_size, "coordinate_frame": "raw"},
        )
        
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
