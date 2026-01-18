import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PreprocessInfo:
    """Information about image preprocessing for coordinate transformation."""
    scale: float           # Resize scale factor
    pad_left: int          # Padding added to left
    pad_top: int           # Padding added to top
    orig_size: tuple       # Original (H, W)
    resized_size: tuple    # Size after resize, before padding (H, W)
    final_size: tuple      # Final size after padding (H, W)


def preprocess_image(
    img,
    target_long_edge=1120,
    divisibility=16,
    return_info=True,
):
    """
    Preprocess image for fair comparison across matchers.
    
    Steps:
    1. Resize so long edge = target_long_edge (1120 = LCM of 14 and 16)
    2. Scale short edge proportionally (maintains aspect ratio)
    3. Pad with zeros to make both dimensions divisible by divisibility
    
    Args:
        img: PIL Image or numpy array [H, W, 3]
        target_long_edge: Target size for long edge (default 1120, divisible by 14 & 16)
        divisibility: Pad so dimensions are divisible by this (default 16)
        return_info: If True, return PreprocessInfo for coordinate mapping
    
    Returns:
        img_out: Preprocessed PIL Image
        info: PreprocessInfo (if return_info=True)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    orig_w, orig_h = img.size
    orig_size = (orig_h, orig_w)
    
    # Step 1: Resize so long edge = target_long_edge
    if orig_w >= orig_h:
        # Width is long edge
        new_w = target_long_edge
        scale = target_long_edge / orig_w
        new_h = int(round(orig_h * scale))
    else:
        # Height is long edge
        new_h = target_long_edge
        scale = target_long_edge / orig_h
        new_w = int(round(orig_w * scale))
    
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    resized_size = (new_h, new_w)
    
    # Step 2: Pad to divisibility
    pad_h = (divisibility - new_h % divisibility) % divisibility
    pad_w = (divisibility - new_w % divisibility) % divisibility
    
    # Pad evenly (or slightly more on bottom/right)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    final_h = new_h + pad_h
    final_w = new_w + pad_w
    final_size = (final_h, final_w)
    
    # Create padded image (black padding)
    img_padded = Image.new('RGB', (final_w, final_h), (0, 0, 0))
    img_padded.paste(img_resized, (pad_left, pad_top))
    
    if return_info:
        info = PreprocessInfo(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_size=orig_size,
            resized_size=resized_size,
            final_size=final_size,
        )
        return img_padded, info
    
    return img_padded


def map_points_to_original(points, info):
    """
    Map keypoint coordinates from preprocessed image back to original image.
    
    Args:
        points: [N, 2] array of (x, y) coordinates in preprocessed image
        info: PreprocessInfo from preprocess_image
    
    Returns:
        points_orig: [N, 2] array of (x, y) coordinates in original image
    """
    if len(points) == 0:
        return points
    
    points = np.asarray(points)
    
    # Undo padding
    x = points[:, 0] - info.pad_left
    y = points[:, 1] - info.pad_top
    
    # Undo resize
    x = x / info.scale
    y = y / info.scale
    
    return np.stack([x, y], axis=1)


def update_intrinsics(K, info):
    """
    Update camera intrinsics matrix for preprocessed image.
    
    Args:
        K: [3, 3] camera intrinsics matrix
        info: PreprocessInfo from preprocess_image
    
    Returns:
        K_new: [3, 3] updated intrinsics for preprocessed image
    """
    K_new = K.copy()
    
    # Scale focal length
    K_new[0, 0] *= info.scale  # fx
    K_new[1, 1] *= info.scale  # fy
    
    # Scale and shift principal point
    K_new[0, 2] = K_new[0, 2] * info.scale + info.pad_left  # cx
    K_new[1, 2] = K_new[1, 2] * info.scale + info.pad_top   # cy
    
    return K_new


def get_superpoint_keypoints(img_path, device='cuda', max_keypoints=2048, cache_dir=None):
    """
    Extract keypoints using SuperPoint detector.
    
    This function provides keypoint locations that can be used with
    other feature extractors (DINOv3, DIFT) for fair comparison.
    
    Args:
        img_path: Path to image
        device: torch device
        max_keypoints: Maximum number of keypoints to detect
        cache_dir: Optional directory to cache keypoints
    
    Returns:
        kpts: [N, 2] tensor of keypoint coordinates (x, y) in pixel space
    """
    try:
        from lightglue import SuperPoint
    except ImportError:
        raise ImportError("LightGlue not installed. Run: pip install git+https://github.com/cvg/LightGlue.git")
    
    img_path = Path(img_path)
    
    # Check cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{img_path.stem}_sp_kpts_k{max_keypoints}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location=device)
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Extract keypoints
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    with torch.no_grad():
        feats = extractor.extract(img_tensor)
        kpts = feats['keypoints'][0]  # [N, 2] in (x, y) format
    
    # Cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{img_path.stem}_sp_kpts.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(kpts.cpu(), cache_path)
    
    return kpts


def compute_matches(
    ft1,
    ft2,
    max_points=2000,
    use_mutual=False,
    ratio_thresh=None,
):
    """
    Match feature maps using cosine similarity (dot product after L2 norm).

    Args:
        ft1, ft2: feature maps [C, H, W]
        max_points: max number of points sampled from image 1
        use_mutual: if True, apply mutual nearest neighbor filtering
        ratio_thresh: if not None, apply a Lowe-style ratio test on similarities
                     (e.g. 1.05 or 1.1 for sim-based ratio)

    Returns:
        x1, y1, x2, y2: numpy arrays of matched coordinates in feature space
        (H, W): spatial size of feature maps
    """
    device = ft1.device

    C, H1, W1 = ft1.shape
    assert C == ft2.shape[0], "Feature maps must have the same channel dimension"
    
    C2, H2, W2 = ft2.shape
    N1 = H1 * W1
    N2 = H2 * W2

    # Flatten to [N, C]
    ft1_flat = ft1.view(C, -1).t()   # [N1, C]
    ft2_flat = ft2.view(C, -1).t()   # [N2, C]

    # L2-normalize (so dot product == cosine similarity)
    ft1_flat = ft1_flat / (ft1_flat.norm(dim=1, keepdim=True) + 1e-8)
    ft2_flat = ft2_flat / (ft2_flat.norm(dim=1, keepdim=True) + 1e-8)

    # Sample points in image 1
    # Sample points in image 1
    if N1 > max_points:
        idx1_full = torch.randperm(N1, device=device)[:max_points]
    else:
        idx1_full = torch.arange(N1, device=device)

    desc1 = ft1_flat[idx1_full]      # [M, C]
    desc2 = ft2_flat                 # [N, C]
    M = desc1.shape[0]

    # Similarity matrix: [M, N]
    sim = desc1 @ desc2.t()

    # 1->2 best matches
    best_sim, best_j = sim.max(dim=1)          # [M]
    rows_all = torch.arange(M, device=device)  # [M]

    # ---------------- Mutual NN (optional) ----------------
    # IMPORTANT: do this BEFORE ratio test, so reverse NN is computed on the same candidate set.
    rows = rows_all
    if use_mutual:
        rev_best_i = sim.argmax(dim=0)  # [N], for each j in image2: best i in image1 (0..M-1)
        mutual_mask = rev_best_i[best_j] == rows_all

        rows = rows_all[mutual_mask]
        best_j = best_j[mutual_mask]
        best_sim = best_sim[mutual_mask]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W)

    # ---------------- Ratio test (optional) ----------------
    # Apply ratio only to rows that survived mutual (if mutual is enabled).
    if ratio_thresh is not None:
        if N2 < 2:
            # Not enough candidates for a top-2 ratio test; keep what we have.
            pass
        else:
            top2_sim, _ = sim.topk(2, dim=1)  # [M, 2]
            best = top2_sim[:, 0]
            second = top2_sim[:, 1]
            ratio = best / (second + 1e-8)

            good = ratio[rows] >= ratio_thresh
            rows = rows[good]
            best_j = best_j[good]
            best_sim = best_sim[good]

            if rows.numel() == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]), (H1, W1), (H2, W2)

    # ---------------- Convert to (x, y) coords ----------------
    flat1 = idx1_full[rows]   # [K] indices into ft1_flat (0..N-1)
    flat2 = best_j            # [K] indices into ft2_flat (0..N-1)

    y1 = (flat1 // W1).cpu().numpy()
    x1 = (flat1 % W1).cpu().numpy()
    y2 = (flat2 // W2).cpu().numpy()
    x2 = (flat2 % W2).cpu().numpy()

    return x1, y1, x2, y2, (H1, W1), (H2, W2)


def match_dense_features(
    ft1,
    ft2,
    img_size1,
    img_size2,
    max_points=2000,
    use_mutual=True,
    ratio_thresh=0.8,
):
    """
    Match dense feature maps and return pixel-space keypoint matches.
    
    This is the standard interface for benchmarking - returns matches in 
    original image pixel coordinates.
    
    Args:
        ft1, ft2: feature maps [C, H, W] (torch tensors)
        img_size1: (height, width) of original image 1
        img_size2: (height, width) of original image 2
        max_points: max number of points to sample from image 1
        use_mutual: if True, apply mutual nearest neighbor filtering (recommended)
        ratio_thresh: Lowe's ratio test threshold (lower = stricter, None = disabled)
                     For similarity-based ratio, use values > 1.0 (e.g. 1.1)
                     For distance-based (inverted), use values < 1.0 (e.g. 0.8)
    
    Returns:
        mkpts0: [N, 2] numpy array of (x, y) pixel coordinates in image 1
        mkpts1: [N, 2] numpy array of (x, y) pixel coordinates in image 2
    """
    # Convert ratio threshold: our compute_matches uses similarity (higher = better)
    # so ratio_thresh should be > 1.0 for sim-based. If user passes < 1, invert it.
    sim_ratio_thresh = ratio_thresh
    if ratio_thresh is not None and ratio_thresh < 1.0:
        # User passed distance-based ratio (0.8 means best/second < 0.8)
        # Convert to similarity-based: best/second > 1/0.8 = 1.25
        sim_ratio_thresh = 1.0 / ratio_thresh
    
    x1, y1, x2, y2, feat_hw1, feat_hw2 = compute_matches(
        ft1, ft2,
        max_points=max_points,
        use_mutual=use_mutual,
        ratio_thresh=sim_ratio_thresh,
    )
    
    if len(x1) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    
    H_feat1, W_feat1 = feat_hw1
    H_feat2, W_feat2 = feat_hw2
    h1, w1 = img_size1
    h2, w2 = img_size2
    
    # Convert feature-space coords to pixel coords
    # Feature cell center to pixel: (feat_coord + 0.5) * scale
    scale_x1 = w1 / W_feat1
    scale_y1 = h1 / H_feat1
    scale_x2 = w2 / W_feat2
    scale_y2 = h2 / H_feat2
    
    px1 = (x1 + 0.5) * scale_x1
    py1 = (y1 + 0.5) * scale_y1
    px2 = (x2 + 0.5) * scale_x2
    py2 = (y2 + 0.5) * scale_y2
    
    mkpts0 = np.stack([px1, py1], axis=1)  # [N, 2]
    mkpts1 = np.stack([px2, py2], axis=1)  # [N, 2]
    
    return mkpts0, mkpts1


def match_at_keypoints(
    ft1,
    ft2,
    kpts1,
    kpts2,
    img_size1,
    img_size2,
    use_mutual=True,
    ratio_thresh=None,
):
    """
    Match dense feature maps using precomputed keypoint locations.
    
    This allows using an external detector (e.g., SuperPoint) to select
    keypoint locations, then extract dense features at those locations
    for matching. This ensures fair comparison across different feature
    extractors.
    
    Args:
        ft1, ft2: feature maps [C, H, W] (torch tensors)
        kpts1: [N1, 2] keypoint coordinates (x, y) in pixel space for image 1
        kpts2: [N2, 2] keypoint coordinates (x, y) in pixel space for image 2
        img_size1: (height, width) of original image 1
        img_size2: (height, width) of original image 2
        use_mutual: if True, apply mutual nearest neighbor filtering
        ratio_thresh: Lowe's ratio test threshold (None = disabled)
    
    Returns:
        mkpts0: [M, 2] numpy array of matched keypoints in image 1
        mkpts1: [M, 2] numpy array of matched keypoints in image 2
    """
    device = ft1.device
    C, H1, W1 = ft1.shape
    C2, H2, W2 = ft2.shape
    h1, w1 = img_size1
    h2, w2 = img_size2
    
    # Convert keypoints to feature map coordinates
    scale_x1 = W1 / w1
    scale_y1 = H1 / h1
    scale_x2 = W2 / w2
    scale_y2 = H2 / h2
    
    # Scale keypoints to feature map space
    kpts1_feat = kpts1.clone().float()
    kpts1_feat[:, 0] *= scale_x1
    kpts1_feat[:, 1] *= scale_y1
    
    kpts2_feat = kpts2.clone().float()
    kpts2_feat[:, 0] *= scale_x2
    kpts2_feat[:, 1] *= scale_y2
    
    # Clamp to valid range
    kpts1_feat[:, 0] = kpts1_feat[:, 0].clamp(0, W1 - 1)
    kpts1_feat[:, 1] = kpts1_feat[:, 1].clamp(0, H1 - 1)
    kpts2_feat[:, 0] = kpts2_feat[:, 0].clamp(0, W2 - 1)
    kpts2_feat[:, 1] = kpts2_feat[:, 1].clamp(0, H2 - 1)
    
    # Sample features at keypoint locations using bilinear interpolation
    # grid_sample expects grid in [-1, 1] range
    grid1 = kpts1_feat.clone()
    grid1[:, 0] = (grid1[:, 0] / (W1 - 1)) * 2 - 1  # x
    grid1[:, 1] = (grid1[:, 1] / (H1 - 1)) * 2 - 1  # y
    grid1 = grid1.view(1, -1, 1, 2).to(device)  # [1, N, 1, 2]
    
    grid2 = kpts2_feat.clone()
    grid2[:, 0] = (grid2[:, 0] / (W2 - 1)) * 2 - 1
    grid2[:, 1] = (grid2[:, 1] / (H2 - 1)) * 2 - 1
    grid2 = grid2.view(1, -1, 1, 2).to(device)
    
    import torch.nn.functional as F
    
    # Extract features: [1, C, N, 1] -> [N, C]
    desc1 = F.grid_sample(ft1.unsqueeze(0), grid1, mode='bilinear', align_corners=True)
    desc1 = desc1.squeeze(0).squeeze(-1).t()  # [N1, C]
    
    desc2 = F.grid_sample(ft2.unsqueeze(0), grid2, mode='bilinear', align_corners=True)
    desc2 = desc2.squeeze(0).squeeze(-1).t()  # [N2, C]
    
    # L2-normalize
    desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + 1e-8)
    desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + 1e-8)
    
    N1 = desc1.shape[0]
    N2 = desc2.shape[0]
    
    if N1 == 0 or N2 == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    
    # Compute similarity matrix
    sim = desc1 @ desc2.t()  # [N1, N2]
    
    # Find best matches: 1 -> 2
    best_sim, best_j = sim.max(dim=1)  # [N1]
    rows_all = torch.arange(N1, device=device)
    
    rows = rows_all
    
    # Mutual nearest neighbor
    if use_mutual:
        rev_best_i = sim.argmax(dim=0)  # [N2]
        mutual_mask = rev_best_i[best_j] == rows_all
        rows = rows_all[mutual_mask]
        best_j = best_j[mutual_mask]
        best_sim = best_sim[mutual_mask]
        
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
    
    # Ratio test
    if ratio_thresh is not None and N2 >= 2:
        # Convert ratio_thresh for similarity-based matching
        sim_ratio_thresh = ratio_thresh if ratio_thresh > 1.0 else 1.0 / ratio_thresh
        
        top2_sim, _ = sim.topk(2, dim=1)
        best = top2_sim[:, 0]
        second = top2_sim[:, 1]
        ratio = best / (second + 1e-8)
        
        good = ratio[rows] >= sim_ratio_thresh
        rows = rows[good]
        best_j = best_j[good]
        
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
    
    # Get matched keypoints in original pixel space
    mkpts0 = kpts1[rows].cpu().numpy()
    mkpts1 = kpts2[best_j].cpu().numpy()
    
    return mkpts0, mkpts1


def save_matches(output_path, mkpts0, mkpts1):
    """
    Save matches in standardized .npz format.
    
    Args:
        output_path: path to save .npz file
        mkpts0: [N, 2] numpy array of keypoints in image 1
        mkpts1: [N, 2] numpy array of keypoints in image 2
    """
    np.savez(output_path, mkpts0=mkpts0, mkpts1=mkpts1)


def load_matches(input_path):
    """
    Load matches from standardized .npz format.
    
    Returns:
        mkpts0: [N, 2] numpy array of keypoints in image 1
        mkpts1: [N, 2] numpy array of keypoints in image 2
    """
    data = np.load(input_path)
    return data['mkpts0'], data['mkpts1']


def visualize_matches(img1_np, img2_np, x1, y1, x2, y2,
                      feat_hw1, feat_hw2, out_path=None, max_lines=200):
    H_feat1, W_feat1 = feat_hw1
    H_feat2, W_feat2 = feat_hw2
    
    h1, w1, _ = img1_np.shape
    h2, w2, _ = img2_np.shape
    
    # Resize so axis 0 (height) is the same for the biggest image (max height)
    # This implies stacking horizontally (concatenating on axis 1)
    target_h = max(h1, h2)
    
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    # Resize images using PIL, maintaining aspect ratio
    i1 = Image.fromarray(img1_np).resize((int(w1 * scale1), int(h1 * scale1)))
    i2 = Image.fromarray(img2_np).resize((int(w2 * scale2), int(h2 * scale2)))
    
    img1_viz = np.array(i1)
    img2_viz = np.array(i2)
    
    # Ensure exact height match for concatenation due to potential rounding
    if img1_viz.shape[0] != target_h:
        img1_viz = np.array(i1.resize((img1_viz.shape[1], target_h)))
    if img2_viz.shape[0] != target_h:
        img2_viz = np.array(i2.resize((img2_viz.shape[1], target_h)))

    canvas = np.concatenate([img1_viz, img2_viz], axis=1)

    # Transform coordinates
    # 1. Feature Map -> Original Image
    scale_x1_orig = w1 / W_feat1
    scale_y1_orig = h1 / H_feat1
    x1_orig = (x1 + 0.5) * scale_x1_orig
    y1_orig = (y1 + 0.5) * scale_y1_orig
    
    scale_x2_orig = w2 / W_feat2
    scale_y2_orig = h2 / H_feat2
    x2_orig = (x2 + 0.5) * scale_x2_orig
    y2_orig = (y2 + 0.5) * scale_y2_orig
    
    # 2. Original Image -> Visualization Image
    x1_img = x1_orig * scale1
    y1_img = y1_orig * scale1
    
    # Image 2 is stacked to the right of Image 1
    x_offset = img1_viz.shape[1]
    x2_img = x2_orig * scale2 + x_offset
    y2_img = y2_orig * scale2

    num_matches = len(x1)
    # TODO: do i really need to limit lines
    if num_matches > max_lines:
        idx = np.random.choice(num_matches, size=max_lines, replace=False)
        x1_img = x1_img[idx]
        y1_img = y1_img[idx]
        x2_img = x2_img[idx]
        y2_img = y2_img[idx]

    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    plt.axis("off")
    for xa, ya, xb, yb in zip(x1_img, y1_img, x2_img, y2_img):
        plt.plot([xa, xb], [ya, yb], linewidth=0.5)

    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()