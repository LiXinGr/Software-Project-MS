"""Eval B: LoRA-DINOv3 + DIFT fusion (1664-dim), no projection head.

Loads a Stage-1 LoRA checkpoint, merges into DINOv3. For each image:
  1. Extract 1024-dim LoRA-DINOv3 features at SuperPoint keypoints
  2. Load precomputed DIFT 640-dim features from cache
  3. L2-normalise each, concatenate → 1664-dim
  4. L2-normalise fused vector, MNN match

Config key: lora_r{rank}_fusion_noproj_sp_mnn[_rt{thresh}]_mp{max_points}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
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
from lora_matches import load_lora_checkpoint, load_dift_feature, run_lora_dinov3
from train_lora import DINOV3_FEAT_LEVEL
from util import get_superpoint_keypoints, save_matches

_bundle_cache: dict = {}
_MAX_BUNDLE_CACHE = 500


def get_config_key(args: argparse.Namespace) -> str:
    rank = getattr(args, "lora_rank_from_ckpt", args.lora_rank_hint)
    parts = [f"phase2_lora_r{rank}_fusion_noproj", "sp", "mnn" if args.use_mutual else "nn"]
    if args.ratio_thresh is not None:
        parts.append(f"rt{args.ratio_thresh}")
    parts.append(f"mp{args.max_points}")
    return "_".join(parts)


def _get_bundle(
    img_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    lora_model: torch.nn.Module,
    transform: T.Compose,
    feat_level: int,
    dino_cache_dir: Optional[Path],
    dift_cache_dir: Path,
    sp_cache_dir: Path,
):
    key = str(img_path)
    if key in _bundle_cache:
        return _bundle_cache[key]

    kpts = get_superpoint_keypoints(
        img_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache_dir
    ).cpu().float()

    # LoRA-DINOv3 features
    run_lora_dinov3.target_long_edge = int(getattr(args, "dino_img_size", 1120))
    dino_ft, orig_size, prep_info = run_lora_dinov3(
        img_path, lora_model, transform, feat_level, dino_cache_dir
    )
    dino_ft = dino_ft.to(device)
    kpts_dino = kpts.clone()
    kpts_dino[:, 0] = kpts_dino[:, 0] * prep_info.scale + prep_info.pad_left
    kpts_dino[:, 1] = kpts_dino[:, 1] * prep_info.scale + prep_info.pad_top
    dino_desc = l2_normalize(fusion_sample(dino_ft, kpts_dino, prep_info.final_size, device))  # (N, 1024)

    # Precomputed DIFT features
    dift_ft, _ = load_dift_feature(img_path, args, dift_cache_dir)
    dift_ft = dift_ft.to(device)
    dift_h, dift_w = canonical_img_size(args.img_size)
    kpts_dift = kpts.clone()
    kpts_dift[:, 0] *= dift_w / orig_size[1]
    kpts_dift[:, 1] *= dift_h / orig_size[0]
    dift_desc = l2_normalize(fusion_sample(dift_ft, kpts_dift, (dift_h, dift_w), device))  # (N, 640)

    # Fuse: [alpha*DIFT, (1-alpha)*DINOv3] → 1664-dim, then L2-norm
    fused = torch.cat([args.alpha * dift_desc, (1.0 - args.alpha) * dino_desc], dim=1)
    desc = l2_normalize(fused).cpu()

    bundle = {"kpts": kpts, "desc": desc}
    if len(_bundle_cache) >= _MAX_BUNDLE_CACHE:
        _bundle_cache.pop(next(iter(_bundle_cache)))
    _bundle_cache[key] = bundle

    del dino_ft, dift_ft, dino_desc, dift_desc, fused
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval B: LoRA-DINOv3 + DIFT fusion, no projection head"
    )
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--pairs_file", type=str)
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--dino_img_size", type=int, default=1120)
    parser.add_argument("--feat_level", type=int, default=None)
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
    parser.add_argument("--lora_cache", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora_rank_hint", type=int, default=4)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--print_config_key", action="store_true")
    args = parser.parse_args()

    args.img_size = canonical_img_size(args.img_size)

    if args.print_config_key:
        print(get_config_key(args))
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lora_ckpt_path = Path(args.lora_checkpoint)

    lora_model, transform, _, ckpt_cfg = load_lora_checkpoint(
        lora_ckpt_path, device, feat_level_override=args.feat_level
    )
    args.lora_rank_from_ckpt = int(ckpt_cfg.get("lora_rank", args.lora_rank_hint))
    feat_level = int(args.feat_level if args.feat_level is not None else ckpt_cfg.get("feat_level", DINOV3_FEAT_LEVEL))
    scene_name = infer_scene_name(args)

    cache_root = Path(args.cache_root)
    config_key = get_config_key(args)
    sp_cache_dir = cache_root / "superpoint_kpts" / scene_name
    sp_cache_dir.mkdir(parents=True, exist_ok=True)

    dift_key = f"dift_t{args.t}_up{args.up_ft_index}_ens{args.ensemble_size}_sp_mnn_mp{args.max_points}"
    dift_cache_dir = cache_root / dift_key / scene_name
    if not dift_cache_dir.exists():
        raise FileNotFoundError(f"Missing DIFT source cache: {dift_cache_dir}")

    dino_cache_dir = (
        Path(args.lora_cache) / scene_name if args.lora_cache
        else cache_root / config_key / scene_name
    )

    def process_pair(img1_path: Path, img2_path: Path):
        b1 = _get_bundle(img1_path, args, device, lora_model, transform, feat_level,
                         dino_cache_dir, dift_cache_dir, sp_cache_dir)
        b2 = _get_bundle(img2_path, args, device, lora_model, transform, feat_level,
                         dino_cache_dir, dift_cache_dir, sp_cache_dir)
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

        for img1_name, img2_name in tqdm(pairs, desc="Matching (LoRA fusion no-proj)"):
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

        print(f"[Eval B] Saved matches to {output_dir}")
    else:
        parser.error("Provide --pairs_file and --output_dir")


if __name__ == "__main__":
    main()
