"""Stage 1.5: Replace DINOv3 portion of sparse training bundles with LoRA-merged DINOv3.

Reads  data/sparse_train/{scene}.pt  — features: (N, 1664) float16
         dims [0:640]    = alpha * l2(DIFT)       — KEPT unchanged
         dims [640:1664] = (1-alpha) * l2(DINOv3) — REPLACED with LoRA-DINOv3

Writes data/sparse_train_lora/{scene}.pt — identical structure, DINOv3 portion updated.

Stages 4 and 5 then load from sparse_train_lora/ and use the fast SparseFeatureDataset
path (~75 pairs/sec in-memory) because no DINOv3 forward pass is needed during training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_sparse_training_data import get_unique_positive_points
from megadepth_pairs import MegaDepthScene
from train_lora import (
    DIFT_DIM,
    DINOV3_FEAT_LEVEL,
    PreprocessCache,
    image_to_tensor,
    load_dinov3_frozen_lora,
    orig_pts_to_preproc,
    sample_feature_map,
)

DEFAULT_SCENES = [
    "0080", "0042", "0380", "0000", "0366",
    "0001", "0005", "0237", "0011", "0148",
]


@torch.no_grad()
def replace_dinov3_in_bundle(
    bundle: dict,
    scene: MegaDepthScene,
    model: torch.nn.Module,
    transform,
    preprocess_cache: PreprocessCache,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    """Return a new bundle dict with dims [DIFT_DIM:] replaced by LoRA-DINOv3 features."""
    alpha = float(bundle["alpha"])
    dino_scale = 1.0 - alpha  # stored as (1-alpha) * l2_dino

    image_ids = bundle["image_ids"]           # (N_images,)
    image_offsets = bundle["image_offsets"]   # (N_images+1,)
    pos_counts = bundle["image_positive_counts"]  # (N_images,)

    features = bundle["features"].float()     # (total, 1664)
    n_images = len(image_ids)

    for i in range(n_images):
        image_id = image_ids[i].item()
        offset = image_offsets[i].item()
        next_offset = image_offsets[i + 1].item()
        pos_count = pos_counts[i].item()
        neg_count = next_offset - offset - pos_count

        if not scene.image_exists.get(image_id, False):
            continue  # keep original DINOv3 features for missing images

        img_path = scene.get_image_path(image_id)
        img_pil, info = preprocess_cache.get(str(img_path))
        final_hw = info.final_size  # (H_pre, W_pre)

        x = image_to_tensor(img_pil, transform, device)
        ft = model(x)[0].squeeze(0)   # (C, H/16, W/16)
        _, fH, fW = ft.shape

        # Positive features — same SfM pixel coords used during extraction
        if pos_count > 0:
            image_obj = scene.images[image_id]
            _, pos_xy = get_unique_positive_points(image_obj)   # (P, 2) original coords
            actual = min(pos_count, len(pos_xy))
            if actual > 0:
                pts_pre = orig_pts_to_preproc(pos_xy[:actual], info)
                pos_feats = sample_feature_map(ft, pts_pre, final_hw)   # (P, C)
                pos_feats = F.normalize(pos_feats, p=2, dim=1) * dino_scale
                features[offset : offset + actual, DIFT_DIM:] = pos_feats.cpu()

        # Negative features — random patches (order doesn't matter for InfoNCE)
        if neg_count > 0:
            n_patches = fH * fW
            idx = rng.choice(n_patches, size=neg_count, replace=(neg_count > n_patches))
            iy = idx // fW
            ix = idx % fW
            neg_feats = ft[:, iy, ix].T   # (neg_count, C)
            neg_feats = F.normalize(neg_feats, p=2, dim=1) * dino_scale
            features[offset + pos_count : next_offset, DIFT_DIM:] = neg_feats.cpu()

        del ft, x

        if (i + 1) % 200 == 0 or i == n_images - 1:
            print(f"    [{i + 1}/{n_images}]  image_id={image_id}")

    out = dict(bundle)
    out["features"] = features.half()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1.5: update sparse bundles with LoRA-DINOv3")
    p.add_argument("--lora_checkpoint", required=True,
                   help="Stage-1 checkpoint (experiments/phase2_lora_dinov3only/best.pt)")
    p.add_argument("--megadepth_root", default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM")
    p.add_argument("--input_dir", default="data/sparse_train")
    p.add_argument("--output_dir", default="data/sparse_train_lora")
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p.add_argument("--feat_level", type=int, default=DINOV3_FEAT_LEVEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache_size", type=int, default=300,
                   help="LRU image cache size (preprocessed PIL images)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LoRA-merged DINOv3 from {args.lora_checkpoint} ...")
    model, transform = load_dinov3_frozen_lora(args.feat_level, args.lora_checkpoint, device)
    model.eval()

    preprocess_cache = PreprocessCache(max_size=args.cache_size)
    rng = np.random.default_rng(args.seed)
    total_start = time.perf_counter()

    for scene_name in args.scenes:
        pt_in = input_dir / f"{scene_name}.pt"
        if not pt_in.exists():
            print(f"[{scene_name}] Missing {pt_in} — skipping")
            continue

        pt_out = output_dir / f"{scene_name}.pt"
        if pt_out.exists():
            print(f"[{scene_name}] Already exists at {pt_out} — skipping")
            continue

        print(f"\n[{scene_name}] Loading bundle from {pt_in} ...")
        bundle = torch.load(pt_in, map_location="cpu", weights_only=False)
        reconstruction = bundle.get("reconstruction", "manhattan/0")
        n_images = len(bundle["image_ids"])
        n_feat = bundle["features"].shape[0]
        n_pairs = bundle["pair_indices"].shape[0]
        print(f"  images={n_images}  features={n_feat:,}  pairs={n_pairs:,}  "
              f"alpha={bundle['alpha']}")

        scene = MegaDepthScene(
            str(Path(args.megadepth_root) / scene_name),
            reconstruction=reconstruction,
        )

        t0 = time.perf_counter()
        bundle = replace_dinov3_in_bundle(
            bundle, scene, model, transform, preprocess_cache, device, rng
        )
        elapsed = time.perf_counter() - t0

        torch.save(bundle, pt_out)
        size_gb = pt_out.stat().st_size / (1024 ** 3)
        print(f"  Saved → {pt_out}  ({size_gb:.2f} GB  {elapsed:.1f}s  "
              f"{n_images / elapsed:.1f} imgs/s)")
        del bundle

    total_elapsed = time.perf_counter() - total_start
    print(f"\nDone: {len(args.scenes)} scene(s) in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
