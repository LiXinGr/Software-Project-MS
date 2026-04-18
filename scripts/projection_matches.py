"""
Projection-head matcher for cached DINOv3 + DIFT features.

This matcher mirrors fusion_matches.py, but after fusing DIFT+DINOv3 descriptors
it applies the trained projection MLP before cosine-similarity MNN matching.

Output format matches the benchmark pipeline: one .npz per image pair containing
mkpts0 and mkpts1 in preprocessed-image pixel coordinates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fusion_matches import (
    DEFAULT_ALPHA,
    canonical_img_size,
    get_or_build_raw_bundle,
    infer_scene_name,
    match_descriptor_sets,
    parse_pairs_file,
    source_cache_dirs,
)
from projection_checkpoint_utils import build_projection_model
from train_projection_head import ProjectionHead
from util import save_matches, visualize_matches


DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase2_projection_v1" / "best.pt"


def get_config_key(args: argparse.Namespace) -> str:
    parts = [args.projection_tag, "sp", "mnn" if args.use_mutual else "nn"]
    if args.ratio_thresh is not None:
        parts.append(f"rt{args.ratio_thresh}")
    parts.append(f"mp{args.max_points}")
    return "_".join(parts)


def final_cache_path(img_path: Path, feature_cache_dir: Path) -> Path:
    return feature_cache_dir / f"{img_path.stem}_projection_desc.pt"


def load_projection_model(checkpoint_path: Path, device: torch.device) -> ProjectionHead:
    model, _ = build_projection_model(checkpoint_path, device=device, eval_mode=True)
    return model


def apply_projection(desc: torch.Tensor, model: ProjectionHead, batch_size: int = 4096) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, desc.shape[0], batch_size):
            chunk = desc[start : start + batch_size].to(next(model.parameters()).device, dtype=torch.float32)
            outputs.append(model(chunk).cpu())
    return torch.cat(outputs, dim=0)


def get_or_build_final_bundle(
    img_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    feature_cache_dir: Path,
    dino_cache_dir: Path,
    dift_cache_dir: Path,
    sp_cache_dir: Path,
    projection_model: ProjectionHead,
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
    projected = apply_projection(desc, projection_model)
    bundle = {"kpts": raw_bundle["kpts"].float(), "desc": projected.cpu()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    return bundle


def prepare_descriptor_cache(
    image_paths,
    args: argparse.Namespace,
    device: torch.device,
    feature_cache_dir: Path,
    dino_cache_dir: Path,
    dift_cache_dir: Path,
    sp_cache_dir: Path,
    projection_model: ProjectionHead,
):
    total = len(image_paths)
    existing = sum(1 for img_path in image_paths if final_cache_path(img_path, feature_cache_dir).exists())
    print(
        f"[PROJECTION] Preparing descriptor cache for {total} images "
        f"({existing} already cached, {total - existing} to build)",
        flush=True,
    )
    built = 0
    reused = 0
    for index, img_path in enumerate(image_paths, start=1):
        cache_path = final_cache_path(img_path, feature_cache_dir)
        if cache_path.exists():
            reused += 1
        else:
            built += 1
        get_or_build_final_bundle(
            img_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
        )
        if index == total or index % 100 == 0:
            print(
                f"[PROJECTION] Cache progress {index}/{total} images "
                f"(reused={reused}, built={built})",
                flush=True,
            )


def process_pair(
    img1_path: Path,
    img2_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    feature_cache_dir: Path,
    dino_cache_dir: Path,
    dift_cache_dir: Path,
    sp_cache_dir: Path,
    projection_model: ProjectionHead,
):
    bundle1 = get_or_build_final_bundle(
        img1_path,
        args,
        device,
        feature_cache_dir,
        dino_cache_dir,
        dift_cache_dir,
        sp_cache_dir,
        projection_model,
    )
    bundle2 = get_or_build_final_bundle(
        img2_path,
        args,
        device,
        feature_cache_dir,
        dino_cache_dir,
        dift_cache_dir,
        sp_cache_dir,
        projection_model,
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


def main():
    parser = argparse.ArgumentParser(description="Match images using projected fused descriptors")
    parser.add_argument("--img1", type=str, help="Path to first image (single-pair mode)")
    parser.add_argument("--img2", type=str, help="Path to second image (single-pair mode)")
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt (batch mode)")
    parser.add_argument("--images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (optional, auto-detected if omitted)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--projection_tag", type=str, default="projection_v1")
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--use_mutual", action="store_true", default=True)
    parser.add_argument("--no_mutual", dest="use_mutual", action="store_false")
    parser.add_argument("--ratio_thresh", type=float, default=None)
    parser.add_argument("--feat_level", type=int, default=-8, help="DINOv3 feature level used by the source cache")
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768], help="DIFT input size used by the source cache")
    parser.add_argument("--t", type=int, default=0, help="DIFT timestep used by the source cache")
    parser.add_argument("--up_ft_index", type=int, choices=[0, 1, 2, 3], default=2, help="DIFT upsampling block used by the source cache")
    parser.add_argument("--ensemble_size", type=int, default=8, help="DIFT ensemble size used by the source cache")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weight applied to DIFT before concatenation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None, help="Directory for projected descriptor cache")
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "cache" / "features"), help="Root directory containing cached matcher features")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to process")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--print_config_key", action="store_true", help="Print the config key for this configuration and exit")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0, 1].")

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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Projection checkpoint not found: {checkpoint_path}")

    sp_cache_dir = Path(args.cache_root) / "superpoint_kpts" / scene_name
    sp_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    projection_model = load_projection_model(checkpoint_path, device)

    if args.pairs_file and args.output_dir:
        images_dir = Path(args.images_dir) if args.images_dir else Path(".")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = parse_pairs_file(args.pairs_file)
        if args.limit:
            pairs = pairs[: args.limit]

        image_paths = sorted({images_dir / p[0] for p in pairs} | {images_dir / p[1] for p in pairs})
        prepare_descriptor_cache(
            image_paths,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
        )

        total_pairs = len(pairs)
        print(f"[PROJECTION] Matching {total_pairs} pairs", flush=True)
        written = 0
        skipped_existing = 0
        skipped_missing = 0
        for pair_index, (img1_name, img2_name) in enumerate(pairs, start=1):
            img1_path = images_dir / img1_name
            img2_path = images_dir / img2_name
            if not img1_path.exists() or not img2_path.exists():
                print(f"[PROJECTION] Skipping missing pair: {img1_name}, {img2_name}")
                skipped_missing += 1
                continue

            pair_name = f"{Path(img1_name).stem}__{Path(img2_name).stem}"
            output_path = output_dir / f"{pair_name}.npz"
            if output_path.exists():
                skipped_existing += 1
                if pair_index == total_pairs or pair_index % 250 == 0:
                    print(
                        f"[PROJECTION] Match progress {pair_index}/{total_pairs} pairs "
                        f"(written={written}, reused={skipped_existing}, missing={skipped_missing})",
                        flush=True,
                    )
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
                projection_model,
            )
            save_matches(output_path, mkpts0, mkpts1)
            written += 1
            if pair_index == total_pairs or pair_index % 250 == 0:
                print(
                    f"[PROJECTION] Match progress {pair_index}/{total_pairs} pairs "
                    f"(written={written}, reused={skipped_existing}, missing={skipped_missing})",
                    flush=True,
                )

        print(
            f"[PROJECTION] Saved matches to {output_dir} "
            f"(written={written}, reused={skipped_existing}, missing={skipped_missing})",
            flush=True,
        )
        return

    if args.img1 and args.img2:
        img1_path = Path(args.img1)
        img2_path = Path(args.img2)
        mkpts0, mkpts1 = process_pair(
            img1_path,
            img2_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
        )
        print(f"[PROJECTION] Found {len(mkpts0)} matches")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_matches(output_dir / f"{img1_path.stem}__{img2_path.stem}.npz", mkpts0, mkpts1)

        if args.visualize or not args.output_dir:
            img1_np = np.array(Image.open(img1_path).convert("RGB"))
            img2_np = np.array(Image.open(img2_path).convert("RGB"))
            vis_path = PROJECT_ROOT / f"datasets/projection_matches_{img1_path.stem}_{img2_path.stem}.png"
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
            print(f"[PROJECTION] Saved visualization to {vis_path}")
        return

    parser.error("Either provide --img1 and --img2, or --pairs_file and --output_dir")


if __name__ == "__main__":
    main()
