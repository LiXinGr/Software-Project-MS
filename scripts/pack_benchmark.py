"""
Benchmark Packing Script

Aggregates matches, depth, and ground truth into a single HDF5 file
compatible with RePoseD evaluation.

Reads:
- Matches from .npz files (output of matcher scripts)
- Depth maps from .npy files (output of UniDepth)
- Ground truth poses from COLMAP sparse reconstruction

Outputs:
- Single .h5 file with all pairs packed for RePoseD

Run after generating matches and depth maps.
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import sys
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Add path to RePoseD utils
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPOSED_ROOT = PROJECT_ROOT / "external" / "RePoseD"
sys.path.insert(0, str(REPOSED_ROOT))

from utils.read_write_colmap import read_cameras_binary, read_images_binary


def load_colmap_model(sparse_dir):
    """Load COLMAP reconstruction data."""
    cameras_bin = Path(sparse_dir) / "cameras.bin"
    images_bin = Path(sparse_dir) / "images.bin"
    
    cameras = read_cameras_binary(str(cameras_bin))
    images = read_images_binary(str(images_bin))
    
    return cameras, images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    q = np.array(qvec)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def get_camera_intrinsics(camera):
    """Extract intrinsics matrix from COLMAP camera."""
    params = camera.params
    
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        # Default to first params as focal, assume centered
        fx = fy = params[0]
        cx, cy = params[1] if len(params) > 1 else 0, params[2] if len(params) > 2 else 0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def get_relative_pose(img1_data, img2_data):
    """Compute relative pose from image 1 to image 2."""
    # Get world-to-camera transforms
    R1 = qvec2rotmat(img1_data.qvec)
    t1 = np.array(img1_data.tvec)
    R2 = qvec2rotmat(img2_data.qvec)
    t2 = np.array(img2_data.tvec)
    
    # Relative pose: from camera 1 to camera 2
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    return R_rel, t_rel


def sample_depth_at_keypoints(depth_map, keypoints):
    """
    Sample depth values at keypoint locations.
    
    Args:
        depth_map: [H, W] depth array
        keypoints: [N, 2] array of (x, y) pixel coordinates
    
    Returns:
        depths: [N] array of depth values
    """
    H, W = depth_map.shape
    depths = np.zeros(len(keypoints))
    
    for i, (x, y) in enumerate(keypoints):
        # Round to nearest pixel
        px = int(round(x))
        py = int(round(y))
        
        # Clamp to valid range
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        
        depths[i] = depth_map[py, px]
    
    return depths


def generate_pairs_from_colmap(images, min_common_points=100):
    """
    Generate image pairs based on COLMAP covisibility.
    
    Returns list of (img_name1, img_name2) tuples.
    """
    # Build point visibility map
    point_to_images = {}
    for img_id, img in images.items():
        for pt_id in img.point3D_ids:
            if pt_id != -1:
                if pt_id not in point_to_images:
                    point_to_images[pt_id] = set()
                point_to_images[pt_id].add(img.name)
    
    # Count common points between image pairs
    pair_counts = {}
    for pt_id, img_names in point_to_images.items():
        img_list = list(img_names)
        for i in range(len(img_list)):
            for j in range(i + 1, len(img_list)):
                pair = tuple(sorted([img_list[i], img_list[j]]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Filter pairs with enough common points
    pairs = [pair for pair, count in pair_counts.items() if count >= min_common_points]
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Pack benchmark data into HDF5")
    
    parser.add_argument("--matches_dir", type=str, required=True,
                        help="Directory containing match .npz files")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Directory containing depth .npy files")
    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="Directory containing COLMAP sparse reconstruction")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .h5 file path")
    parser.add_argument("--pairs_file", type=str, default=None,
                        help="Optional pairs.txt file (if not provided, generates from COLMAP)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N pairs (for dry-run)")
    parser.add_argument("--min_common_points", type=int, default=100,
                        help="Minimum common points for pair generation")
    
    args = parser.parse_args()
    
    # Load COLMAP model
    print("[Pack] Loading COLMAP model...")
    cameras, images = load_colmap_model(args.sparse_dir)
    
    # Build name-to-image mapping
    name_to_img = {img.name: img for img in images.values()}
    name_to_camera = {img.name: cameras[img.camera_id] for img in images.values()}
    
    # Get pairs
    if args.pairs_file:
        print(f"[Pack] Loading pairs from {args.pairs_file}")
        with open(args.pairs_file, 'r') as f:
            pairs = [tuple(line.strip().split()) for line in f if line.strip()]
    else:
        print("[Pack] Generating pairs from COLMAP covisibility...")
        pairs = generate_pairs_from_colmap(images, args.min_common_points)
    
    print(f"[Pack] Total pairs: {len(pairs)}")
    
    # Apply limit
    if args.limit:
        pairs = pairs[:args.limit]
        print(f"[Pack] Limited to {len(pairs)} pairs")
    
    # Create output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    matches_dir = Path(args.matches_dir)
    depth_dir = Path(args.depth_dir)
    
    with h5py.File(output_path, 'w') as h5f:
        packed = 0
        skipped = 0
        
        for img1_name, img2_name in tqdm(pairs, desc="Packing pairs"):
            # Check if images exist in COLMAP
            if img1_name not in name_to_img or img2_name not in name_to_img:
                skipped += 1
                continue
            
            img1_stem = Path(img1_name).stem
            img2_stem = Path(img2_name).stem
            
            # Load matches
            match_file = matches_dir / f"{img1_stem}__{img2_stem}.npz"
            if not match_file.exists():
                # Try reversed order
                match_file = matches_dir / f"{img2_stem}__{img1_stem}.npz"
                if not match_file.exists():
                    skipped += 1
                    continue
            
            match_data = np.load(match_file)
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            
            if len(mkpts0) < 5:
                skipped += 1
                continue
            
            # Load depth maps
            depth0_file = depth_dir / f"{img1_stem}_depth.npy"
            depth1_file = depth_dir / f"{img2_stem}_depth.npy"
            
            if not depth0_file.exists() or not depth1_file.exists():
                skipped += 1
                continue
            
            depth0_map = np.load(depth0_file)
            depth1_map = np.load(depth1_file)
            
            # Sample depth at keypoints
            depths0 = sample_depth_at_keypoints(depth0_map, mkpts0)
            depths1 = sample_depth_at_keypoints(depth1_map, mkpts1)
            
            # Get intrinsics and relative pose
            K0 = get_camera_intrinsics(name_to_camera[img1_name])
            K1 = get_camera_intrinsics(name_to_camera[img2_name])
            R_rel, t_rel = get_relative_pose(name_to_img[img1_name], name_to_img[img2_name])
            
            # Create pair name for RePoseD format
            # Format: img0_o_img1 (using '_o_' as separator)
            pair_key = f"{img1_stem}_o_{img2_stem}"
            
            # Pack correspondence data
            # Format: [kp0_x, kp0_y, kp1_x, kp1_y, score, ..., depth0, depth1, ...]
            # For now we use a simpler format matching RePoseD
            corr_data = np.column_stack([
                mkpts0[:, 0], mkpts0[:, 1],  # kp0_x, kp0_y
                mkpts1[:, 0], mkpts1[:, 1],  # kp1_x, kp1_y
                np.ones(len(mkpts0)),        # placeholder score
                np.zeros(len(mkpts0)),       # placeholder
                np.zeros(len(mkpts0)),       # placeholder
                np.zeros(len(mkpts0)),       # placeholder (total 8 cols before depths)
                depths0, depths1,            # cols 8, 9 (real depth indices 1)
                depths0, depths1,            # cols 10, 11 (midas placeholder - we use same)
                depths0, depths1,            # cols 12-13
                depths0, depths1,            # cols 14-15
                depths0, depths1,            # cols 16-17
                depths0, depths1,            # cols 18-19
                depths0, depths1,            # cols 20-21
                depths0, depths1,            # cols 22-23
                depths0, depths1,            # cols 24-25
                depths0, depths1,            # cols 26-27
                depths0, depths1,            # cols 28-29
                depths0, depths1,            # cols 30-31 (unidepth - index 12)
            ])
            
            # Create pose matrix [3, 4] - RePoseD expects [R|t] not 4x4
            pose = np.zeros((3, 4))
            pose[:3, :3] = R_rel
            pose[:3, 3] = t_rel
            
            # Save to HDF5
            # Note: RePoseD parses pairs as (stem1_o, stem2) so K keys must match
            h5f.create_dataset(f"corr_{pair_key}", data=corr_data)
            h5f.create_dataset(f"pose_{pair_key}", data=pose)
            
            # K keys: first image gets _o suffix to match RePoseD parsing
            K1_key = f"K_{img1_stem}_o"
            K2_key = f"K_{img2_stem}"
            if K1_key not in h5f:
                h5f.create_dataset(K1_key, data=K0)
            if K2_key not in h5f:
                h5f.create_dataset(K2_key, data=K1)
            
            packed += 1
        
        print(f"[Pack] Done. Packed: {packed}, Skipped: {skipped}")
        print(f"[Pack] Output saved to: {output_path}")


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/pack_benchmark.py \
#   --matches_dir output/matches/dinov3 \
#   --depth_dir datasets/phototourism/sacre_coeur/depth_unidepth \
#   --sparse_dir datasets/phototourism/sacre_coeur/dense/sparse \
#   --output output/benchmark_dinov3_sacre_coeur.h5 \
#   --limit 10
