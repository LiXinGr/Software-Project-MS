"""
Fusion matcher for cached DINOv3 + DIFT features.

This matcher reuses existing per-image feature caches:
- DINOv3 features sampled at SuperPoint keypoints on the preprocessed image grid
- DIFT features sampled at the same SuperPoint keypoints after scaling them into
  the square DIFT input grid

For each keypoint, it:
1. L2-normalizes DINOv3 and DIFT descriptors independently
2. Applies optional weighting
3. Concatenates them into a fused descriptor
4. Optionally fits/applies a scene-level PCA projection
5. L2-normalizes again before cosine-similarity MNN matching

The output format matches the existing pipeline: one .npz per image pair
containing mkpts0 and mkpts1 in preprocessed-image pixel coordinates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from util import get_superpoint_keypoints, save_matches, visualize_matches


DINOV3_NUM_BLOCKS = 24
DEFAULT_ALPHA = 0.5


def feat_level_to_block(feat_level):
    if feat_level >= 0:
        return feat_level
    return DINOV3_NUM_BLOCKS + feat_level


def canonical_img_size(img_size):
    if isinstance(img_size, int):
        return [img_size, img_size]
    if len(img_size) == 1:
        return [img_size[0], img_size[0]]
    return [img_size[0], img_size[1]]


def alpha_tag(alpha):
    alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    if "." not in alpha_str:
        alpha_str = f"{alpha_str}.0"
    return f"a{alpha_str.replace('.', '')}"


def get_config_key(args):
    block = feat_level_to_block(args.feat_level)
    parts = ["fusion", f"dinov3b{block}", f"dift_t{args.t}up{args.up_ft_index}"]
    if args.ensemble_size != 8:
        parts.append(f"ens{args.ensemble_size}")
    if canonical_img_size(args.img_size) != [768, 768]:
        h, w = canonical_img_size(args.img_size)
        parts.append(f"sz{h}x{w}")
    if abs(args.alpha - DEFAULT_ALPHA) > 1e-9:
        parts.append(alpha_tag(args.alpha))
    if args.pca_dim > 0:
        parts.append(f"pca{args.pca_dim}")
    parts.append("sp")
    parts.append("mnn" if args.use_mutual else "nn")
    if args.ratio_thresh is not None:
        parts.append(f"rt{args.ratio_thresh}")
    parts.append(f"mp{args.max_points}")
    return "_".join(parts)


def infer_scene_name(args):
    if getattr(args, "scene", None):
        return args.scene
    if getattr(args, "images_dir", None):
        return Path(args.images_dir).resolve().parent.name
    if getattr(args, "img1", None):
        p = Path(args.img1).resolve()
        for parent in p.parents:
            if parent.name in {"sacre_coeur", "reichstag", "st_peters_square"}:
                return parent.name
    return "single_pair"


def l2_normalize(desc):
    return F.normalize(desc, p=2, dim=1, eps=1e-8)


def sample_feature_map(ft, kpts, img_size, device):
    c, h_feat, w_feat = ft.shape
    h_img, w_img = img_size
    if kpts.numel() == 0:
        return torch.empty((0, c), dtype=ft.dtype, device=device)

    kpts_feat = kpts.clone().float()
    kpts_feat[:, 0] *= w_feat / w_img
    kpts_feat[:, 1] *= h_feat / h_img
    kpts_feat[:, 0] = kpts_feat[:, 0].clamp(0, w_feat - 1)
    kpts_feat[:, 1] = kpts_feat[:, 1].clamp(0, h_feat - 1)

    grid = kpts_feat.clone()
    grid[:, 0] = (grid[:, 0] / (w_feat - 1)) * 2 - 1
    grid[:, 1] = (grid[:, 1] / (h_feat - 1)) * 2 - 1
    grid = grid.view(1, -1, 1, 2).to(device)

    desc = F.grid_sample(ft.unsqueeze(0), grid, mode="bilinear", align_corners=True)
    return desc.squeeze(0).squeeze(-1).t()


def match_descriptor_sets(desc1, desc2, kpts1, kpts2, use_mutual=True, ratio_thresh=None):
    if desc1.numel() == 0 or desc2.numel() == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    desc1 = l2_normalize(desc1)
    desc2 = l2_normalize(desc2)
    sim = desc1 @ desc2.t()

    n1 = desc1.shape[0]
    rows_all = torch.arange(n1, device=desc1.device)
    _, best_j = sim.max(dim=1)
    rows = rows_all

    if use_mutual:
        rev_best_i = sim.argmax(dim=0)
        mutual_mask = rev_best_i[best_j] == rows_all
        rows = rows_all[mutual_mask]
        best_j = best_j[mutual_mask]
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

    if ratio_thresh is not None and desc2.shape[0] >= 2:
        sim_ratio_thresh = ratio_thresh if ratio_thresh > 1.0 else 1.0 / ratio_thresh
        top2_sim, _ = sim.topk(2, dim=1)
        ratios = top2_sim[:, 0] / (top2_sim[:, 1] + 1e-8)
        keep = ratios[rows] >= sim_ratio_thresh
        rows = rows[keep]
        best_j = best_j[keep]
        if rows.numel() == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

    return kpts1[rows].cpu().numpy(), kpts2[best_j].cpu().numpy()


def load_dinov3_feature(img_path, args, source_cache_dir):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    cache_path = source_cache_dir / f"{img_path.stem}_dinov3_{w}x{h}_l{args.feat_level}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing DINOv3 cache for {img_path.name}: {cache_path}")
    return torch.load(cache_path, map_location="cpu"), (h, w)


def load_dift_feature(img_path, args, source_cache_dir):
    img = Image.open(img_path).convert("RGB")
    orig_size = (img.height, img.width)
    dift_h, dift_w = canonical_img_size(args.img_size)
    cache_path = source_cache_dir / (
        f"{img_path.stem}_dift_sz{dift_h}x{dift_w}_t{args.t}_up{args.up_ft_index}_ens{args.ensemble_size}.pt"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing DIFT cache for {img_path.name}: {cache_path}")
    return torch.load(cache_path, map_location="cpu"), orig_size


def raw_cache_path(img_path, feature_cache_dir):
    return feature_cache_dir / f"{img_path.stem}_fusion_raw.pt"


def final_cache_path(img_path, feature_cache_dir):
    return feature_cache_dir / f"{img_path.stem}_fusion_desc.pt"


def pca_cache_path(feature_cache_dir):
    return feature_cache_dir / "pca_model.pt"


def source_cache_dirs(args, scene_name):
    dino_key = f"dinov3_l{args.feat_level}_sp_mnn_mp{args.max_points}"
    dift_key = f"dift_t{args.t}_up{args.up_ft_index}_ens{args.ensemble_size}_sp_mnn_mp{args.max_points}"
    cache_root = Path(args.cache_root)
    return cache_root / dino_key / scene_name, cache_root / dift_key / scene_name


def get_or_build_raw_bundle(img_path, args, device, feature_cache_dir, dino_cache_dir, dift_cache_dir, sp_cache_dir):
    cache_path = raw_cache_path(img_path, feature_cache_dir)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    kpts = get_superpoint_keypoints(
        img_path,
        device=device,
        max_keypoints=args.max_points,
        cache_dir=sp_cache_dir,
    ).cpu().float()

    dino_ft, orig_size = load_dinov3_feature(img_path, args, dino_cache_dir)
    dift_ft, _ = load_dift_feature(img_path, args, dift_cache_dir)

    dino_ft = dino_ft.to(device)
    dift_ft = dift_ft.to(device)

    dino_desc = sample_feature_map(dino_ft, kpts, orig_size, device)

    dift_h, dift_w = canonical_img_size(args.img_size)
    kpts_dift = kpts.clone()
    kpts_dift[:, 0] *= dift_w / orig_size[1]
    kpts_dift[:, 1] *= dift_h / orig_size[0]
    dift_desc = sample_feature_map(dift_ft, kpts_dift, (dift_h, dift_w), device)

    dino_desc = l2_normalize(dino_desc)
    dift_desc = l2_normalize(dift_desc)
    fused = torch.cat(
        [args.alpha * dift_desc, (1.0 - args.alpha) * dino_desc],
        dim=1,
    ).cpu()

    bundle = {"kpts": kpts.cpu(), "desc": fused}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)

    del dino_ft, dift_ft, dino_desc, dift_desc, fused
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return bundle


def fit_pca_model(image_paths, args, device, feature_cache_dir, dino_cache_dir, dift_cache_dir, sp_cache_dir):
    cache_path = pca_cache_path(feature_cache_dir)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    rng = np.random.default_rng(args.pca_seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)
    samples = []
    total = 0

    iterator = tqdm(shuffled, desc="PCA fit", leave=False)
    for img_path in iterator:
        bundle = get_or_build_raw_bundle(
            img_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
        )
        desc = bundle["desc"].float()
        if desc.numel() == 0:
            continue

        remaining = args.max_pca_samples - total
        if remaining <= 0:
            break

        if desc.shape[0] > remaining:
            take_idx = rng.choice(desc.shape[0], size=remaining, replace=False)
            desc = desc[torch.from_numpy(take_idx).long()]

        samples.append(desc)
        total += desc.shape[0]
        iterator.set_postfix(samples=total)

        if total >= args.max_pca_samples:
            break

    if not samples:
        raise RuntimeError("Could not collect descriptors for PCA fitting.")

    x = torch.cat(samples, dim=0)
    q = min(args.pca_dim, x.shape[0], x.shape[1])
    if q < args.pca_dim:
        raise RuntimeError(
            f"PCA dim {args.pca_dim} is too large for collected samples {tuple(x.shape)}."
        )

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    torch.manual_seed(args.pca_seed)
    _, _, v = torch.pca_lowrank(x_centered, q=q, center=False)

    model = {
        "mean": mean.squeeze(0).contiguous(),
        "components": v[:, :args.pca_dim].contiguous(),
        "n_samples": int(x.shape[0]),
        "dim": int(args.pca_dim),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, cache_path)
    return model


def project_desc(desc, pca_model):
    mean = pca_model["mean"].float()
    components = pca_model["components"].float()
    projected = (desc.float() - mean) @ components
    return l2_normalize(projected)


def get_or_build_final_bundle(
    img_path,
    args,
    device,
    feature_cache_dir,
    dino_cache_dir,
    dift_cache_dir,
    sp_cache_dir,
    pca_model=None,
):
    cache_path = final_cache_path(img_path, feature_cache_dir)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    raw_bundle = get_or_build_raw_bundle(
        img_path,
        args,
        device,
        feature_cache_dir,
        dino_cache_dir,
        dift_cache_dir,
        sp_cache_dir,
    )

    desc = raw_bundle["desc"].float()
    if pca_model is not None:
        desc = project_desc(desc, pca_model)
    else:
        desc = l2_normalize(desc)

    bundle = {"kpts": raw_bundle["kpts"].float(), "desc": desc.cpu()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    return bundle


def prepare_descriptor_cache(image_paths, args, device, feature_cache_dir, dino_cache_dir, dift_cache_dir, sp_cache_dir):
    pca_model = None
    if args.pca_dim > 0:
        pca_model = fit_pca_model(
            image_paths,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
        )

    for img_path in tqdm(image_paths, desc="Descriptor cache", leave=False):
        get_or_build_final_bundle(
            img_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            pca_model=pca_model,
        )

    return pca_model


def process_pair(img1_path, img2_path, args, device, feature_cache_dir, dino_cache_dir, dift_cache_dir, sp_cache_dir, pca_model):
    bundle1 = get_or_build_final_bundle(
        img1_path,
        args,
        device,
        feature_cache_dir,
        dino_cache_dir,
        dift_cache_dir,
        sp_cache_dir,
        pca_model=pca_model,
    )
    bundle2 = get_or_build_final_bundle(
        img2_path,
        args,
        device,
        feature_cache_dir,
        dino_cache_dir,
        dift_cache_dir,
        sp_cache_dir,
        pca_model=pca_model,
    )

    desc1 = bundle1["desc"].to(device)
    desc2 = bundle2["desc"].to(device)
    kpts1 = bundle1["kpts"].to(device)
    kpts2 = bundle2["kpts"].to(device)

    return match_descriptor_sets(
        desc1,
        desc2,
        kpts1,
        kpts2,
        use_mutual=args.use_mutual,
        ratio_thresh=args.ratio_thresh,
    )


def parse_pairs_file(pairs_file):
    with open(pairs_file, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Match images using fused DINOv3 + DIFT descriptors")
    parser.add_argument("--img1", type=str, help="Path to first image (single-pair mode)")
    parser.add_argument("--img2", type=str, help="Path to second image (single-pair mode)")
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt (batch mode)")
    parser.add_argument("--images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (optional, auto-detected if omitted)")
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=None)
    parser.add_argument("--feat_level", type=int, default=-8, help="DINOv3 feature level to fuse")
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768], help="DIFT input size used by the source cache")
    parser.add_argument("--t", type=int, default=0, help="DIFT timestep used by the source cache")
    parser.add_argument("--up_ft_index", type=int, choices=[0, 1, 2, 3], default=2, help="DIFT upsampling block used by the source cache")
    parser.add_argument("--ensemble_size", type=int, default=8, help="DIFT ensemble size used by the source cache")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weight applied to DIFT before concatenation")
    parser.add_argument("--pca_dim", type=int, default=0, help="Optional PCA output dimension (0 disables PCA)")
    parser.add_argument("--max_pca_samples", type=int, default=50000, help="Maximum descriptors used to fit PCA")
    parser.add_argument("--pca_seed", type=int, default=0, help="Random seed for PCA sampling")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None, help="Directory for fused descriptor cache")
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "cache" / "features"), help="Root directory containing cached matcher features")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to process")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--print_config_key", action="store_true", help="Print the config key for this configuration and exit")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0, 1].")
    if args.pca_dim < 0:
        parser.error("--pca_dim must be >= 0.")

    args.img_size = canonical_img_size(args.img_size)

    if args.print_config_key:
        print(get_config_key(args))
        return

    scene_name = infer_scene_name(args)
    if args.feature_cache is None:
        args.feature_cache = PROJECT_ROOT / "cache" / "features" / get_config_key(args) / scene_name
    feature_cache_dir = Path(args.feature_cache)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    dino_cache_dir, dift_cache_dir = source_cache_dirs(args, scene_name)
    if not dino_cache_dir.exists():
        raise FileNotFoundError(f"Missing DINOv3 source cache directory: {dino_cache_dir}")
    if not dift_cache_dir.exists():
        raise FileNotFoundError(f"Missing DIFT source cache directory: {dift_cache_dir}")

    sp_cache_dir = Path(args.cache_root) / "superpoint_kpts" / scene_name
    sp_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.pairs_file and args.output_dir:
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = parse_pairs_file(args.pairs_file)
        if args.limit:
            pairs = pairs[: args.limit]

        image_paths = sorted({images_dir / p[0] for p in pairs} | {images_dir / p[1] for p in pairs})
        pca_model = prepare_descriptor_cache(
            image_paths,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
        )

        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            if not img1_path.exists() or not img2_path.exists():
                print(f"[FUSION] Skipping missing pair: {img1_name}, {img2_name}")
                continue

            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            output_path = output_dir / f"{pair_name}.npz"
            if output_path.exists():
                continue

            mkpts0, mkpts1 = process_pair(
                img1_path,
                img2_path,
                args,
                device,
                feature_cache_dir,
                dino_cache_dir,
                dift_cache_dir,
                sp_cache_dir,
                pca_model,
            )
            save_matches(output_path, mkpts0, mkpts1)

        print(f"[FUSION] Saved matches to {output_dir}")
        return

    if args.img1 and args.img2:
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        pca_model = None
        if args.pca_dim > 0:
            pca_model = prepare_descriptor_cache(
                [img1_path, img2_path],
                args,
                device,
                feature_cache_dir,
                dino_cache_dir,
                dift_cache_dir,
                sp_cache_dir,
            )

        mkpts0, mkpts1 = process_pair(
            img1_path,
            img2_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            pca_model,
        )
        print(f"[FUSION] Found {len(mkpts0)} matches")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_matches(output_dir / f"{img1_path.stem}__{img2_path.stem}.npz", mkpts0, mkpts1)

        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            vis_path = PROJECT_ROOT / f"datasets/fusion_matches_{img1_path.stem}_{img2_path.stem}.png"
            visualize_matches(
                img1_np,
                img2_np,
                mkpts0[:, 0],
                mkpts0[:, 1],
                mkpts1[:, 0],
                mkpts1[:, 1],
                (img1_np.shape[0], img1_np.shape[1]),
                (img2_np.shape[0], img2_np.shape[1]),
                out_path=str(vis_path),
                max_lines=args.max_lines,
            )
            print(f"[FUSION] Saved visualization to {vis_path}")
        return

    parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()
