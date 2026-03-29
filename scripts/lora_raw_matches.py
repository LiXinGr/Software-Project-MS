"""Eval A: LoRA-DINOv3 raw 1024-dim features — no projection head, no DIFT.

Loads a Stage-1 LoRA checkpoint, merges into DINOv3, extracts block-16 features
at SuperPoint keypoints, L2-normalises, and matches with MNN.

Config key: lora_r{rank}_raw_dinov3_sp_mnn[_rt{thresh}]_mp{max_points}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_matches import (
    canonical_img_size,
    infer_scene_name,
    l2_normalize,
    match_descriptor_sets,
    parse_pairs_file,
    sample_feature_map as fusion_sample,
)
from lora_matches import load_lora_checkpoint, run_lora_dinov3
from train_lora import DINOV3_FEAT_LEVEL
from util import get_superpoint_keypoints, save_matches

_bundle_cache: dict = {}
_MAX_BUNDLE_CACHE = 500


def get_config_key(args: argparse.Namespace) -> str:
    rank = getattr(args, "lora_rank_from_ckpt", args.lora_rank_hint)
    parts = [f"phase2_lora_r{rank}_raw_dinov3", "sp", "mnn" if args.use_mutual else "nn"]
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
    sp_cache_dir: Path,
):
    key = str(img_path)
    if key in _bundle_cache:
        return _bundle_cache[key]

    kpts = get_superpoint_keypoints(
        img_path, device=device, max_keypoints=args.max_points, cache_dir=sp_cache_dir
    ).cpu().float()

    dino_ft, orig_size = run_lora_dinov3(img_path, lora_model, transform, feat_level, dino_cache_dir)
    dino_ft = dino_ft.to(device)
    desc = l2_normalize(fusion_sample(dino_ft, kpts, orig_size, device)).cpu()  # (N, 1024)

    bundle = {"kpts": kpts, "desc": desc}
    if len(_bundle_cache) >= _MAX_BUNDLE_CACHE:
        _bundle_cache.pop(next(iter(_bundle_cache)))
    _bundle_cache[key] = bundle
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval A: raw LoRA-DINOv3 features, no projection")
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--pairs_file", type=str)
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "cache" / "features"))
    parser.add_argument("--lora_cache", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora_rank_hint", type=int, default=4)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--print_config_key", action="store_true")
    args = parser.parse_args()

    if args.print_config_key:
        print(get_config_key(args))
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lora_ckpt_path = Path(args.lora_checkpoint)

    lora_model, transform, _, ckpt_cfg = load_lora_checkpoint(lora_ckpt_path, device)
    args.lora_rank_from_ckpt = int(ckpt_cfg.get("lora_rank", args.lora_rank_hint))
    feat_level = int(ckpt_cfg.get("feat_level", DINOV3_FEAT_LEVEL))
    scene_name = infer_scene_name(args)

    cache_root = Path(args.cache_root)
    config_key = get_config_key(args)
    sp_cache_dir = cache_root / "superpoint_kpts" / scene_name
    sp_cache_dir.mkdir(parents=True, exist_ok=True)
    dino_cache_dir = (
        Path(args.lora_cache) / scene_name if args.lora_cache
        else cache_root / config_key / scene_name
    )

    def process_pair(img1_path: Path, img2_path: Path):
        b1 = _get_bundle(img1_path, args, device, lora_model, transform, feat_level,
                         dino_cache_dir, sp_cache_dir)
        b2 = _get_bundle(img2_path, args, device, lora_model, transform, feat_level,
                         dino_cache_dir, sp_cache_dir)
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

        for img1_name, img2_name in tqdm(pairs, desc="Matching (raw LoRA-DINOv3)"):
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

        print(f"[Eval A] Saved matches to {output_dir}")
    else:
        parser.error("Provide --pairs_file and --output_dir")


if __name__ == "__main__":
    main()
