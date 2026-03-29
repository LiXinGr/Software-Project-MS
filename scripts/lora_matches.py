"""LoRA-DINOv3 + DIFT projection matcher for eval pipeline.

Loads DINOv3 with LoRA adapters from a train_lora.py checkpoint, merges the
adapters into the base weights (zero inference overhead), and runs the same
fused-feature + projection-head matching pipeline as projection_matches.py.

DIFT features are loaded from the precomputed cache (unchanged from Phase 2a).
DINOv3 features are computed online with the merged LoRA model.

Config key format: lora_r{rank}_proj_wide_sp_mnn_mp{max_points}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import timm.data
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_matches import (
    DEFAULT_ALPHA,
    canonical_img_size,
    infer_scene_name,
    l2_normalize,
    match_descriptor_sets,
    parse_pairs_file,
    sample_feature_map as fusion_sample,
)
from train_lora import (
    DINOV3_FEAT_LEVEL,
    DINOV3_MODEL_NAME,
    DINOV3_NUM_BLOCKS,
    DIFT_DIM,
    FUSED_DIM,
    PATCH_SIZE,
    LoRALinear,
    LoRALinearSplitQKV,
    inject_lora,
    merge_lora_into_model,
)
from train_projection_head import ProjectionHead
from util import get_superpoint_keypoints, save_matches, visualize_matches


# ---------------------------------------------------------------------------
# LoRA model loading
# ---------------------------------------------------------------------------

def load_lora_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load DINOv3 with LoRA merged into base weights + projection head.

    Returns (dinov3_model, transform, proj_head, config).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    rank = int(cfg.get("lora_rank", 4))
    lora_alpha = float(cfg.get("lora_alpha", 8.0))
    lora_dropout = float(cfg.get("lora_dropout", 0.0))
    feat_level = int(cfg.get("feat_level", DINOV3_FEAT_LEVEL))

    out_idx = DINOV3_NUM_BLOCKS + feat_level if feat_level < 0 else feat_level

    # Build model and inject LoRA structure (needed to load state dict)
    model = timm.create_model(
        DINOV3_MODEL_NAME,
        pretrained=True,
        features_only=True,
        out_indices=[out_idx],
        dynamic_img_size=True,
    )
    data_config = timm.data.resolve_model_data_config(model)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    for param in model.parameters():
        param.requires_grad_(False)

    inject_lora(model, target_suffixes=["attn.qkv"], rank=rank, alpha=lora_alpha, dropout=lora_dropout,
                qkv_suffixes=["attn.qkv"])

    # Load LoRA weights
    lora_state = ckpt["lora_state_dict"]
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    unexpected_non_lora = [k for k in unexpected if "lora_" not in k]
    if unexpected_non_lora:
        print(f"[LORA] Warning: unexpected non-LoRA keys: {unexpected_non_lora[:5]}")

    # Merge LoRA adapters into base weights (zero overhead at inference)
    n_merged = _merge_lora(model)

    model.eval().to(device)
    print(f"[EVAL] Using LoRA checkpoint: {checkpoint_path}")
    print(f"[EVAL] LoRA merged: {n_merged} layers modified (rank={rank})")

    # Projection head — dimensions read from checkpoint config so all modes work:
    #   Stage 1 (joint):   input_dim=1664 (FUSED), hidden=[1024], output=256
    #   Stage 4 (dinov3_only, frozen): input_dim=1024, hidden=[512], output=256
    #   Stage 5 (fusion, frozen):      input_dim=1664, hidden=[1024], output=256
    hidden_dims = cfg.get("hidden_dims") or [int(cfg.get("hidden_dim", 1024))]
    proj_head = ProjectionHead(
        input_dim=int(cfg.get("input_dim", FUSED_DIM)),
        hidden_dims=hidden_dims,
        output_dim=int(cfg.get("output_dim", 256)),
    )
    proj_head.load_state_dict(ckpt["proj_state_dict"])
    proj_head.eval().to(device)

    return model, transform, proj_head, cfg


def _merge_lora(model: torch.nn.Module) -> int:
    """Delegate to the canonical merge_lora_into_model from train_lora. Returns merged count."""
    return merge_lora_into_model(model)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_lora_dinov3(
    img_path: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    feat_level: int,
    cache_dir: Optional[Path],
) -> tuple:
    """Extract [C, H/16, W/16] feature map and original (H, W)."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    orig_size = (H, W)

    if cache_dir is not None:
        cache_path = cache_dir / f"{img_path.stem}_lora_dinov3_{W}x{H}_l{feat_level}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu"), orig_size

    device = next(model.parameters()).device
    x = transform(img).unsqueeze(0).to(device)
    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    ft = model(x)[0].squeeze(0).cpu()  # (C, H_feat, W_feat)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ft, cache_path)

    return ft, orig_size


def load_dift_feature(img_path: Path, args: argparse.Namespace, dift_cache_dir: Path):
    """Load precomputed DIFT feature map from cache."""
    img = Image.open(img_path).convert("RGB")
    orig_size = (img.height, img.width)
    dift_h, dift_w = canonical_img_size(args.img_size)
    cache_path = dift_cache_dir / (
        f"{img_path.stem}_dift_sz{dift_h}x{dift_w}"
        f"_t{args.t}_up{args.up_ft_index}_ens{args.ensemble_size}.pt"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing DIFT cache: {cache_path}")
    return torch.load(cache_path, map_location="cpu"), orig_size


# ---------------------------------------------------------------------------
# Bundle building (keypoints -> fused + projected desc)
# ---------------------------------------------------------------------------

_bundle_cache: dict = {}
_MAX_BUNDLE_CACHE = 500


def _get_bundle(
    img_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    lora_model: torch.nn.Module,
    transform: T.Compose,
    proj_model: ProjectionHead,
    dino_cache_dir: Optional[Path],
    dift_cache_dir: Optional[Path],   # None when dinov3_only
    sp_cache_dir: Path,
    feat_level: int,
    dinov3_only: bool = False,
):
    key = str(img_path)
    if key in _bundle_cache:
        return _bundle_cache[key]

    kpts = get_superpoint_keypoints(
        img_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache_dir
    ).cpu().float()

    # LoRA DINOv3 features
    dino_ft, orig_size = run_lora_dinov3(img_path, lora_model, transform, feat_level, dino_cache_dir)
    dino_ft = dino_ft.to(device)
    dino_desc = l2_normalize(fusion_sample(dino_ft, kpts, orig_size, device))  # (N, 1024)

    if dinov3_only:
        # Stage 4: projection head operates on 1024-dim DINOv3 features only
        fused = dino_desc
    else:
        # Stage 5 / original: fuse with precomputed DIFT features
        dift_ft, _ = load_dift_feature(img_path, args, dift_cache_dir)
        dift_ft = dift_ft.to(device)
        dift_h, dift_w = canonical_img_size(args.img_size)
        kpts_dift = kpts.clone()
        kpts_dift[:, 0] *= dift_w / orig_size[1]
        kpts_dift[:, 1] *= dift_h / orig_size[0]
        dift_desc = l2_normalize(fusion_sample(dift_ft, kpts_dift, (dift_h, dift_w), device))
        fused = torch.cat([args.alpha * dift_desc, (1.0 - args.alpha) * dino_desc], dim=1)

    # Project through MLP
    with torch.no_grad():
        chunks = []
        for s in range(0, fused.shape[0], 4096):
            chunks.append(proj_model(fused[s : s + 4096].float()))
        projected = torch.cat(chunks, dim=0).cpu()

    bundle = {"kpts": kpts.cpu(), "desc": projected}

    if len(_bundle_cache) >= _MAX_BUNDLE_CACHE:
        _bundle_cache.pop(next(iter(_bundle_cache)))
    _bundle_cache[key] = bundle

    del dino_ft, dino_desc, fused
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return bundle


# ---------------------------------------------------------------------------
# Config key
# ---------------------------------------------------------------------------

def get_config_key(args: argparse.Namespace) -> str:
    rank = getattr(args, "lora_rank_from_ckpt", args.lora_rank_hint)
    dinov3_only = getattr(args, "dinov3_only", False)
    proj_tag = "proj_dinov3only" if dinov3_only else "proj_fusion"
    parts = [f"phase2_lora_r{rank}_{proj_tag}", "sp", "mnn" if args.use_mutual else "nn"]
    if args.ratio_thresh is not None:
        parts.append(f"rt{args.ratio_thresh}")
    parts.append(f"mp{args.max_points}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA-DINOv3 + projection head matcher")
    parser.add_argument("--lora_checkpoint", required=True, help="Path to train_lora.py checkpoint")
    parser.add_argument("--img1", type=str)
    parser.add_argument("--img2", type=str)
    parser.add_argument("--pairs_file", type=str)
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=None)
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768])
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--up_ft_index", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "cache" / "features"))
    parser.add_argument("--lora_cache", type=str, default=None, help="Cache dir for LoRA DINOv3 features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora_rank_hint", type=int, default=4, help="Used in config key if rank not in ckpt")
    parser.add_argument("--print_config_key", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max_lines", type=int, default=200)
    args = parser.parse_args()

    args.img_size = canonical_img_size(args.img_size)

    if args.print_config_key:
        print(get_config_key(args))
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lora_ckpt_path = Path(args.lora_checkpoint)

    lora_model, transform, proj_head, ckpt_cfg = load_lora_checkpoint(lora_ckpt_path, device)
    args.lora_rank_from_ckpt = int(ckpt_cfg.get("lora_rank", args.lora_rank_hint))
    args.dinov3_only = ckpt_cfg.get("dinov3_only", False)

    feat_level = int(ckpt_cfg.get("feat_level", DINOV3_FEAT_LEVEL))
    scene_name = infer_scene_name(args)

    # Cache dirs
    cache_root = Path(args.cache_root)
    sp_cache_dir = cache_root / "superpoint_kpts" / scene_name
    sp_cache_dir.mkdir(parents=True, exist_ok=True)

    # DIFT cache is only needed when not dinov3_only
    if args.dinov3_only:
        dift_cache_dir = None
    else:
        dift_key = f"dift_t{args.t}_up{args.up_ft_index}_ens{args.ensemble_size}_sp_mnn_mp{args.max_points}"
        dift_cache_dir = cache_root / dift_key / scene_name
        if not dift_cache_dir.exists():
            raise FileNotFoundError(f"Missing DIFT source cache: {dift_cache_dir}")

    dino_cache_dir = (
        Path(args.lora_cache) / scene_name if args.lora_cache
        else cache_root / get_config_key(args) / scene_name
    )

    def process_pair(img1_path: Path, img2_path: Path):
        b1 = _get_bundle(img1_path, args, device, lora_model, transform, proj_head,
                         dino_cache_dir, dift_cache_dir, sp_cache_dir, feat_level,
                         dinov3_only=args.dinov3_only)
        b2 = _get_bundle(img2_path, args, device, lora_model, transform, proj_head,
                         dino_cache_dir, dift_cache_dir, sp_cache_dir, feat_level,
                         dinov3_only=args.dinov3_only)
        return match_descriptor_sets(
            b1["desc"].to(device), b2["desc"].to(device),
            b1["kpts"].to(device), b2["kpts"].to(device),
            use_mutual=args.use_mutual, ratio_thresh=args.ratio_thresh,
        )

    if args.pairs_file and args.output_dir:
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = parse_pairs_file(args.pairs_file)
        if args.limit:
            pairs = pairs[: args.limit]

        for img1_name, img2_name in tqdm(pairs, desc="Matching"):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            if not img1_path.exists() or not img2_path.exists():
                continue
            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            out_path = output_dir / f"{pair_name}.npz"
            if out_path.exists():
                continue
            mkpts0, mkpts1 = process_pair(img1_path, img2_path)
            save_matches(out_path, mkpts0, mkpts1)

        print(f"[LORA] Saved matches to {output_dir}")

    elif args.img1 and args.img2:
        mkpts0, mkpts1 = process_pair(Path(args.img1), Path(args.img2))
        print(f"[LORA] Found {len(mkpts0)} matches")
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_matches(
                out_dir / f"{Path(args.img1).stem}__{Path(args.img2).stem}.npz",
                mkpts0, mkpts1,
            )
        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(args.img1).convert("RGB"))
            img2_np = np.array(Image.open(args.img2).convert("RGB"))
            vis_path = PROJECT_ROOT / f"datasets/lora_{Path(args.img1).stem}_{Path(args.img2).stem}.png"
            visualize_matches(
                img1_np, img2_np,
                mkpts0[:, 0], mkpts0[:, 1], mkpts1[:, 0], mkpts1[:, 1],
                img1_np.shape[:2], img2_np.shape[:2],
                out_path=str(vis_path), max_lines=args.max_lines,
            )
    else:
        parser.error("Provide --img1/--img2 or --pairs_file/--output_dir")


if __name__ == "__main__":
    main()
