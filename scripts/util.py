import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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