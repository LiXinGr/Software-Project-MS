import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DINOv3] Using device: {device}")

# ---------- paths (same pattern as your DIFT script) ----------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


# ---------- DINOv3 feature extractor ----------

def make_transform():
    """ImageNet-style normalization, no resize here."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def create_dinov3_model(model_name: str):
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
    )
    model.eval()
    model.to(device)       # ✅ move to GPU if available
    return model


def run_dinov3_extractor(
    img_path,
    ft_path,
    model,
    transform,
    img_size,
    feat_level: int = -1,
):
    """
    Load image, resize for the model, compute a feature map from the requested
    feature level, and save it. Returns [C, Hf, Wf] feature map and the resized
    image as a numpy array for visualization.
    """
    img = Image.open(img_path).convert("RGB")

    # Resize image to (img_size, img_size) ONCE
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)

    # This is what we will show in the visualization
    img_np = np.array(img_resized)

    x = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        feats_list = model(x)
    ft = feats_list[feat_level][0].detach().cpu()

    torch.save(ft, ft_path)
    return ft, img_np


# ---------- matching ----------

def compute_matches(
    ft1: torch.Tensor,
    ft2: torch.Tensor,
    max_points: int = 2000,
    use_mutual: bool = False,
    ratio_thresh: float | None = None,
    sim_thresh: float | None = None,
    topk: int | None = None,
):
    """
    Match dense feature maps using cosine similarity.

    Args:
        ft1, ft2: feature maps [C, H, W]
        max_points: max number of points sampled from image 1
        use_mutual: if True, apply mutual nearest neighbor filtering
        ratio_thresh: if not None, apply a Lowe-style ratio test on similarities.
        sim_thresh: if not None, drop matches with cosine similarity < sim_thresh
        topk: if not None, keep only top-k matches by similarity after filtering

    Returns:
        x1, y1, x2, y2: numpy arrays of matched coordinates in feature space
        (H, W): spatial size of feature maps
        scores: numpy array of similarity scores for each match
    """
    assert ft1.shape == ft2.shape, "Feature maps must have the same shape"
    C, H, W = ft1.shape
    N = H * W

    # Flatten features to [N, C]
    ft1_flat = ft1.view(C, -1).t()  # [N, C]
    ft2_flat = ft2.view(C, -1).t()  # [N, C]

    # L2 normalize for cosine similarity
    ft1_flat = ft1_flat / (ft1_flat.norm(dim=1, keepdim=True) + 1e-8)
    ft2_flat = ft2_flat / (ft2_flat.norm(dim=1, keepdim=True) + 1e-8)

    # Randomly sample points in image 1 to limit compute
    if N > max_points:
        idx1_full = torch.randperm(N)[:max_points]
    else:
        idx1_full = torch.arange(N)

    desc1 = ft1_flat[idx1_full]  # [M, C]
    desc2 = ft2_flat             # [N, C]
    M = desc1.shape[0]

    # Similarity matrix: [M, N]
    sim = desc1 @ desc2.t()

    # Best match in image 2 for each sampled point in image 1
    best_sim, best_j = sim.max(dim=1)  # [M]
    rows = torch.arange(M)

    # --- ratio test (optional) ---
    if ratio_thresh is not None:
        # top-2 similarities for each row
        top2_sim, _ = sim.topk(2, dim=1)  # [M, 2]
        best = top2_sim[:, 0]
        second = top2_sim[:, 1]
        ratio = best / (second + 1e-8)
        good = ratio >= ratio_thresh

        rows = rows[good]
        best_j = best_j[good]
        best_sim = best_sim[good]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W), np.array([])

    # --- similarity threshold (optional) ---
    if sim_thresh is not None:
        good = best_sim >= sim_thresh
        rows = rows[good]
        best_j = best_j[good]
        best_sim = best_sim[good]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W), np.array([])

    # --- mutual NN (optional) ---
    if use_mutual:
        # for each point in image 2, best match index in desc1
        rev_best_i = sim.argmax(dim=0)  # [N]

        mutual_mask = rev_best_i[best_j] == rows
        rows = rows[mutual_mask]
        best_j = best_j[mutual_mask]
        best_sim = best_sim[mutual_mask]

        if rows.numel() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), (H, W), np.array([])

    # --- keep only top-k matches by similarity (optional) ---
    if topk is not None and rows.numel() > topk:
        topk_sim, idx_topk = torch.topk(best_sim, topk)
        rows = rows[idx_topk]
        best_j = best_j[idx_topk]
        best_sim = topk_sim

    # Convert to (x, y) coords in feature grid
    flat1 = idx1_full[rows]  # [K]
    flat2 = best_j           # [K]

    y1 = (flat1 // W).cpu().numpy()
    x1 = (flat1 % W).cpu().numpy()
    y2 = (flat2 // W).cpu().numpy()
    x2 = (flat2 % W).cpu().numpy()
    scores = best_sim.cpu().numpy()

    return x1, y1, x2, y2, (H, W), scores


def visualize_matches(
    img1_np,
    img2_np,
    x1,
    y1,
    x2,
    y2,
    feat_hw,
    scores=None,
    out_path=None,
    max_lines=200,
):
    """
    Visualize matches on a side-by-side canvas.

    If scores are provided, we draw the top `max_lines` highest-scoring matches.
    Otherwise, we randomly subsample.
    """
    H_feat, W_feat = feat_hw
    h, w, _ = img1_np.shape
    assert img2_np.shape[0] == h and img2_np.shape[1] == w

    canvas = np.concatenate([img1_np, img2_np], axis=1)

    scale_x = w / W_feat
    scale_y = h / H_feat

    x1_img = (x1 + 0.5) * scale_x
    y1_img = (y1 + 0.5) * scale_y
    x2_img = (x2 + 0.5) * scale_x + w
    y2_img = (y2 + 0.5) * scale_y

    num_matches = len(x1)
    if num_matches == 0:
        print("[DINOv3] No matches to visualize.")
    elif num_matches > max_lines:
        if scores is not None and len(scores) == num_matches:
            order = np.argsort(-scores)[:max_lines]  # top by score
        else:
            order = np.random.choice(num_matches, size=max_lines, replace=False)
        x1_img = x1_img[order]
        y1_img = y1_img[order]
        x2_img = x2_img[order]
        y2_img = y2_img[order]

    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    plt.axis("off")

    # thicker lines + visible endpoints
    for xa, ya, xb, yb in zip(x1_img, y1_img, x2_img, y2_img):
        plt.plot([xa, xb], [ya, yb], linewidth=1.5)
        plt.scatter([xa, xb], [ya, yb], s=8)

    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ---------- main CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Match two images using DINOv3 ViT-L/16 dense features."
    )

    parser.add_argument("--img1", required=True, type=str,
                        help="Path to first image")
    parser.add_argument("--img2", required=True, type=str,
                        help="Path to second image")

    parser.add_argument("--max_points", type=int, default=2000,
                        help="Max number of feature points to sample from image 1")
    parser.add_argument("--max_lines", type=int, default=200,
                        help="Max number of lines to draw in visualization")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Resize images to this size before DINOv3")

    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_7b_patch16_dinov3.lvd1689m",   # <- 7B teacher
        help="timm DINOv3 backbone (e.g. vit_7b_patch16_dinov3.lvd1689m).",
    )

    parser.add_argument(
        "--feat_level",
        type=int,
        default=-1,
        help="Index in timm features list (-1: deepest, -2: slightly shallower).",
    )

    parser.add_argument(
        "--no_mutual",
        action="store_true",
        help="Disable mutual nearest-neighbor filtering.",
    )
    parser.add_argument(
        "--ratio_thresh",
        type=float,
        default=1.2,
        help="Similarity ratio threshold (>=) for ratio test; set <=1.0 to disable.",
    )
    parser.add_argument(
        "--sim_thresh",
        type=float,
        default=0.7,
        help="Cosine similarity threshold; set <=0 to disable.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=500,
        help="Keep at most this many best matches after filtering; set <=0 to disable.",
    )

    args = parser.parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    ft1_path = DATASETS_DIR / f"{img1_path.stem}_dinov3_vitl16.pt"
    ft2_path = DATASETS_DIR / f"{img2_path.stem}_dinov3_vitl16.pt"
    vis_path = DATASETS_DIR / f"dinov3_matches_{img1_path.stem}_{img2_path.stem}.png"

    # 1) model + transform
    transform = make_transform()
    model = create_dinov3_model(args.model_name)

    # 2) extract features
    ft1, img1_np = run_dinov3_extractor(
        img1_path, ft1_path, model, transform, args.img_size, feat_level=args.feat_level
    )
    ft2, img2_np = run_dinov3_extractor(
        img2_path, ft2_path, model, transform, args.img_size, feat_level=args.feat_level
    )

    # 3) matching
    ratio = args.ratio_thresh if args.ratio_thresh > 1.0 else None
    sim_thresh = args.sim_thresh if args.sim_thresh > 0 else None
    topk = args.topk if args.topk > 0 else None

    x1, y1, x2, y2, feat_hw, scores = compute_matches(
        ft1,
        ft2,
        max_points=args.max_points,
        use_mutual=not args.no_mutual,
        ratio_thresh=ratio,
        sim_thresh=sim_thresh,
        topk=topk,
    )

    print(f"[DINOv3] matches after filters: {len(x1)}")

    # Optional: fallback if everything got filtered out
    if len(x1) == 0:
        print("[DINOv3] No matches after filters, recomputing without mutual/ratio/sim...")
        x1, y1, x2, y2, feat_hw, scores = compute_matches(
            ft1,
            ft2,
            max_points=args.max_points,
            use_mutual=False,
            ratio_thresh=None,
            sim_thresh=None,
            topk=topk,
        )
        print(f"[DINOv3] fallback matches: {len(x1)}")

    # 4) visualization
    visualize_matches(
        img1_np, img2_np, x1, y1, x2, y2,
        feat_hw, scores=scores, out_path=vis_path, max_lines=args.max_lines
    )

    print(f"[DINOv3] Saved feature maps to:\n  {ft1_path}\n  {ft2_path}")
    print(f"[DINOv3] Saved match visualization to:\n  {vis_path}")


if __name__ == "__main__":
    main()
