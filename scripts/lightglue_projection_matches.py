"""
Zero-shot LightGlue matcher for projected DINOv3 + DIFT descriptors.

This script mirrors the projection-head descriptor pipeline from
projection_matches.py, but replaces cosine-similarity MNN with the pretrained
SuperPoint+LightGlue matcher. The only experimental variable is the descriptor
source:

- Baseline: SuperPoint descriptors -> LightGlue
- Phase 4: projected DINOv3 + DIFT descriptors -> LightGlue

Matches are saved in the standard benchmark format:
    .npz files containing mkpts0 and mkpts1 in preprocessed-image pixel space.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIGHTGLUE_ROOT = PROJECT_ROOT / "external" / "LightGlue"
GF_ROOT = PROJECT_ROOT / "external" / "glue-factory"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_phase4")

sys.path.insert(0, str(LIGHTGLUE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lightglue import LightGlue
from fusion_matches import (
    DEFAULT_ALPHA,
    canonical_img_size,
    infer_scene_name,
    parse_pairs_file,
    source_cache_dirs,
)
from projection_matches import (
    apply_projection,
    get_or_build_raw_bundle,
    load_projection_model,
)
from util import visualize_matches


PHASE4_DEFAULT_CONFIG_KEY = "phase4_lg_zeroshot_proj256"
PHASE4_DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase2_projection_wide" / "best.pt"
PHASE4_DEFAULT_SCENES = ("sacre_coeur", "reichstag", "st_peters_square")

# These defaults mirror the current standalone SP+LightGlue baseline script,
# which constructs `LightGlue(features="superpoint")` without overrides.
BASELINE_FILTER_THRESHOLD = 0.1
BASELINE_DEPTH_CONFIDENCE = 0.95
BASELINE_WIDTH_CONFIDENCE = 0.99
BASELINE_FLASH = True
BASELINE_MP = False
BASELINE_FEATURES = "superpoint"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config_key(args: argparse.Namespace) -> str:
    return args.config_key


def log_prefix(args: argparse.Namespace) -> str:
    if args.depth_confidence < 0 and args.width_confidence < 0:
        return "[PHASE4-NOADAPT]"
    return "[PHASE4]"


def projected_cache_path(img_path: Path, feature_cache_dir: Path) -> Path:
    return feature_cache_dir / f"{img_path.stem}_projection_desc.pt"


def resolve_scene_defaults(args: argparse.Namespace, scene_name: str) -> tuple[Path, Path, Path]:
    dataset_root = PROJECT_ROOT / "datasets" / "phototourism" / scene_name
    default_images_dir = dataset_root / "images_preprocessed"
    if not default_images_dir.exists():
        default_images_dir = dataset_root / "dense" / "images"

    images_dir = Path(args.images_dir) if args.images_dir else default_images_dir
    pairs_file = Path(args.pairs_file) if args.pairs_file else PROJECT_ROOT / "output" / f"pairs_{scene_name}.txt"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / "matches" / args.config_key / scene_name
    return images_dir, pairs_file, output_dir


def clone_args_with_max_points(args: argparse.Namespace, max_points: int) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.max_points = int(max_points)
    return cloned


def resolve_source_cache_dirs(args: argparse.Namespace, scene_name: str) -> tuple[Path, Path, int]:
    candidates: list[int] = []
    if args.source_cache_max_points is not None:
        candidates.append(int(args.source_cache_max_points))
    else:
        candidates.append(int(args.max_points))
        if int(args.max_points) != 2000:
            candidates.append(2000)

    seen: set[int] = set()
    missing_messages = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_args = clone_args_with_max_points(args, candidate)
        dino_cache_dir, dift_cache_dir = source_cache_dirs(candidate_args, scene_name)
        dino_exists = dino_cache_dir.exists()
        dift_exists = dift_cache_dir.exists()
        if dino_exists and dift_exists:
            return dino_cache_dir, dift_cache_dir, candidate
        missing_messages.append(
            f"source_max_points={candidate}: "
            f"DINO={'ok' if dino_exists else 'missing'} at {dino_cache_dir}, "
            f"DIFT={'ok' if dift_exists else 'missing'} at {dift_cache_dir}"
        )

    joined = "\n  ".join(missing_messages)
    raise FileNotFoundError(
        "Could not find compatible source feature caches.\n"
        f"  {joined}"
    )


def get_or_build_projected_bundle(
    img_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    feature_cache_dir: Path,
    dino_cache_dir: Path,
    dift_cache_dir: Path,
    sp_cache_dir: Path,
    projection_model,
):
    cache_path = projected_cache_path(img_path, feature_cache_dir)
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
    bundle = {
        "kpts": raw_bundle["kpts"].float().cpu(),
        "desc": projected.float().cpu(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    return bundle


def load_image_tensor(img_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(img_path) as img:
        img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(img_np).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    return image, (img_np.shape[0], img_np.shape[1])


def build_online_extractor(args: argparse.Namespace, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

    if str(GF_ROOT) not in sys.path:
        sys.path.insert(0, str(GF_ROOT))

    from gluefactory.models.extractors.dinov3_dift_projection import DINOv3DIFTProjection

    extractor_conf = {
        "name": "extractors.dinov3_dift_projection",
        "trainable": False,
        "max_num_keypoints": int(args.max_points),
        "force_num_keypoints": False,
        "feat_level": int(args.feat_level),
        "dift_t": int(args.t),
        "dift_up_ft_index": int(args.up_ft_index),
        "dift_ensemble_size": int(args.ensemble_size),
        "dift_input_size": list(args.img_size),
        "projection_checkpoint": str(Path(args.checkpoint)),
        "alpha_dift": float(args.alpha),
        "alpha_dinov3": float(1.0 - args.alpha),
        "sampling_mode": "bilinear",
        "cache_mode": "online",
        "write_cache": False,
        "include_depth_keypoints": False,
        "cache_mem_size": 0,
        "cache_warp_by_homography": False,
    }
    extractor = DINOv3DIFTProjection(extractor_conf).eval().to(device)
    return extractor


def extract_online_bundle(
    img_path: Path,
    extractor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image, (height, width) = load_image_tensor(img_path, device)
    with torch.no_grad():
        pred = extractor(
            {
                "image": image.unsqueeze(0),
                "image_size": torch.tensor([[width, height]], device=device, dtype=torch.float32),
                "scales": torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32),
            }
        )
    return {
        "kpts": pred["keypoints"][0].detach().cpu().to(dtype=torch.float32),
        "desc": pred["descriptors"][0].detach().cpu().to(dtype=torch.float32),
    }


def get_or_build_online_bundle(
    img_path: Path,
    extractor,
    device: torch.device,
    bundle_cache: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, torch.Tensor]:
    if bundle_cache is None:
        return extract_online_bundle(img_path, extractor, device)

    cache_key = str(img_path)
    cached = bundle_cache.get(cache_key)
    if cached is not None:
        return cached

    cached = extract_online_bundle(img_path, extractor, device)
    bundle_cache[cache_key] = cached
    return cached


@lru_cache(maxsize=4096)
def load_image_size(path_str: str) -> tuple[int, int]:
    with Image.open(path_str) as img:
        return img.width, img.height


def make_lightglue_input(bundle, img_path: Path, device: torch.device) -> dict:
    width, height = load_image_size(str(img_path))
    return {
        "keypoints": bundle["kpts"].unsqueeze(0).to(device=device, dtype=torch.float32).contiguous(),
        "descriptors": bundle["desc"].unsqueeze(0).to(device=device, dtype=torch.float32).contiguous(),
        # LightGlue expects image_size in (width, height) order for (x, y) keypoints.
        "image_size": torch.tensor([[width, height]], device=device, dtype=torch.float32),
    }


def match_with_lightglue(bundle0, bundle1, img0_path: Path, img1_path: Path, matcher, device: torch.device):
    feats0 = make_lightglue_input(bundle0, img0_path, device)
    feats1 = make_lightglue_input(bundle1, img1_path, device)

    with torch.no_grad():
        matches01 = matcher({"image0": feats0, "image1": feats1})

    matches = matches01["matches"][0]
    scores = matches01["scores"][0]
    kpts0 = bundle0["kpts"]
    kpts1 = bundle1["kpts"]

    if matches.numel() == 0:
        mkpts0 = np.zeros((0, 2), dtype=np.float32)
        mkpts1 = np.zeros((0, 2), dtype=np.float32)
        match_scores = np.zeros((0,), dtype=np.float32)
    else:
        mkpts0 = kpts0[matches[:, 0].cpu()].numpy().astype(np.float32, copy=False)
        mkpts1 = kpts1[matches[:, 1].cpu()].numpy().astype(np.float32, copy=False)
        match_scores = scores.detach().cpu().numpy().astype(np.float32, copy=False)

    return mkpts0, mkpts1, match_scores, int(matches01["stop"])


def save_match_archive(
    output_path: Path,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    args: argparse.Namespace,
    match_scores: np.ndarray | None = None,
    stop_layer: int | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "filter_threshold": np.array(args.filter_threshold, dtype=np.float32),
        "depth_confidence": np.array(args.depth_confidence, dtype=np.float32),
        "width_confidence": np.array(args.width_confidence, dtype=np.float32),
        # These scores are after LightGlue's own filter_threshold has been applied.
        "scores_are_post_filter": np.array(1, dtype=np.uint8),
    }
    if args.save_scores and match_scores is not None:
        payload["scores"] = match_scores
    if stop_layer is not None:
        payload["stop"] = np.array(stop_layer, dtype=np.int16)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{output_path.stem}_",
        suffix=output_path.suffix,
        dir=output_path.parent,
    )
    os.close(fd)
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def print_hyperparameter_verification(
    args: argparse.Namespace,
    matcher,
    source_cache_max_points: int | None,
) -> None:
    prefix = log_prefix(args)
    print(f"{prefix} === Hyperparameter Verification ===", flush=True)
    print(f"{prefix} descriptor_dim:    {matcher.conf.descriptor_dim}", flush=True)
    print(f"{prefix} n_layers:          {matcher.conf.n_layers}", flush=True)
    print(f"{prefix} num_heads:         {matcher.conf.num_heads}", flush=True)
    print(f"{prefix} filter_threshold:  {matcher.conf.filter_threshold}", flush=True)
    print(f"{prefix} depth_confidence:  {matcher.conf.depth_confidence}", flush=True)
    print(f"{prefix} width_confidence:  {matcher.conf.width_confidence}", flush=True)
    print(f"{prefix} flash:             {matcher.conf.flash}", flush=True)
    print(f"{prefix} mp:                {matcher.conf.mp}", flush=True)
    print(f"{prefix} max_keypoints:     {args.max_points}", flush=True)
    print(f"{prefix} coord_format:      pixel + image_size[W,H] (LightGlue normalizes internally)", flush=True)
    print(f"{prefix} weights:           {matcher.conf.weights}", flush=True)
    if args.online_extraction:
        print(f"{prefix} extraction_mode:   online (RAM memoization per image, no disk cache)", flush=True)
        print(f"{prefix} projection_head:   {args.checkpoint}", flush=True)
    else:
        print(f"{prefix} source_cache_mp:   {source_cache_max_points}", flush=True)
    if prefix == "[PHASE4-NOADAPT]":
        print(f"{prefix} All other params identical to phase4_lg_zeroshot_proj256", flush=True)
    else:
        print(f"{prefix} === These must match the SP+LG baseline ===", flush=True)


def process_pair(
    img0_path: Path,
    img1_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    feature_cache_dir: Path,
    dino_cache_dir: Path,
    dift_cache_dir: Path,
    sp_cache_dir: Path,
    projection_model,
    matcher,
    online_extractor=None,
    online_bundle_cache=None,
):
    if args.online_extraction:
        bundle0 = get_or_build_online_bundle(img0_path, online_extractor, device, online_bundle_cache)
        bundle1 = get_or_build_online_bundle(img1_path, online_extractor, device, online_bundle_cache)
    else:
        bundle0 = get_or_build_projected_bundle(
            img0_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
        )
        bundle1 = get_or_build_projected_bundle(
            img1_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
        )

    return match_with_lightglue(bundle0, bundle1, img0_path, img1_path, matcher, device)


def maybe_visualize(
    img0_path: Path,
    img1_path: Path,
    mkpts0,
    mkpts1,
    output_dir: Path,
    max_lines: int,
    prefix: str,
) -> None:
    img0_np = np.array(Image.open(img0_path).convert("RGB"))
    img1_np = np.array(Image.open(img1_path).convert("RGB"))
    vis_path = output_dir / f"{img0_path.stem}__{img1_path.stem}.png"
    visualize_matches(
        img0_np,
        img1_np,
        mkpts0[:, 0],
        mkpts0[:, 1],
        mkpts1[:, 0],
        mkpts1[:, 1],
        (img0_np.shape[0], img0_np.shape[1]),
        (img1_np.shape[0], img1_np.shape[1]),
        out_path=str(vis_path),
        max_lines=max_lines,
    )
    print(f"{prefix} Saved visualization to {vis_path}", flush=True)


def build_matcher(args: argparse.Namespace, device: torch.device) -> LightGlue:
    matcher = LightGlue(
        features=BASELINE_FEATURES,
        filter_threshold=args.filter_threshold,
        depth_confidence=args.depth_confidence,
        width_confidence=args.width_confidence,
        flash=args.flash,
        mp=args.mp,
    ).eval().to(device)
    if args.lightglue_checkpoint:
        checkpoint = torch.load(args.lightglue_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        matcher_state = {
            key.removeprefix("matcher."): value
            for key, value in state.items()
            if key.startswith("matcher.")
        }
        if not matcher_state:
            matcher_state = state
        load_result = matcher.load_state_dict(matcher_state, strict=False)
        print(
            f"{log_prefix(args)} Loaded fine-tuned LightGlue checkpoint: {args.lightglue_checkpoint} "
            f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})",
            flush=True,
        )
    return matcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Match projected DINOv3+DIFT descriptors with pretrained LightGlue")
    parser.add_argument("--img1", type=str, help="Path to first image (single-pair mode)")
    parser.add_argument("--img2", type=str, help="Path to second image (single-pair mode)")
    parser.add_argument("--pairs_file", type=str, help="Path to pairs.txt (batch mode)")
    parser.add_argument("--images_dir", type=str, help="Base directory for images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for matches")
    parser.add_argument("--scene", type=str, default=None, choices=PHASE4_DEFAULT_SCENES, help="Scene name for benchmark convenience mode")
    parser.add_argument("--config_key", type=str, default=PHASE4_DEFAULT_CONFIG_KEY)
    parser.add_argument("--checkpoint", type=str, default=str(PHASE4_DEFAULT_CHECKPOINT))
    parser.add_argument("--lightglue_checkpoint", type=str, default=None,
                        help="Optional Glue Factory LightGlue checkpoint (.tar) used to override pretrained matcher weights")
    parser.add_argument("--max_points", type=int, default=2048)
    parser.add_argument("--source_cache_max_points", type=int, default=None,
                        help="Which cached DINOv3/DIFT source namespace to reuse; defaults to max_points with a fallback to 2000")
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--feat_level", type=int, default=-8, help="DINOv3 feature level used by the source cache")
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768], help="DIFT input size used by the source cache")
    parser.add_argument("--t", type=int, default=0, help="DIFT timestep used by the source cache")
    parser.add_argument("--up_ft_index", type=int, choices=[0, 1, 2, 3], default=2, help="DIFT upsampling block used by the source cache")
    parser.add_argument("--ensemble_size", type=int, default=8, help="DIFT ensemble size used by the source cache")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weight applied to DIFT before concatenation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_cache", type=str, default=None, help="Directory for projected descriptor cache")
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "cache" / "features"), help="Root directory containing cached matcher features")
    parser.add_argument("--online_extraction", action="store_true",
                        help="Run DINOv3+DIFT+projection online from images instead of loading cached descriptors")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate matches even if the .npz exists")
    parser.add_argument("--print_config_key", action="store_true", help="Print the config key and exit")
    parser.add_argument("--filter_threshold", type=float, default=BASELINE_FILTER_THRESHOLD)
    parser.add_argument("--depth_confidence", type=float, default=BASELINE_DEPTH_CONFIDENCE)
    parser.add_argument("--width_confidence", type=float, default=BASELINE_WIDTH_CONFIDENCE)
    parser.add_argument("--flash", action="store_true", default=BASELINE_FLASH)
    parser.add_argument("--no_flash", dest="flash", action="store_false")
    parser.add_argument("--mp", action="store_true", default=BASELINE_MP)
    parser.add_argument("--save_scores", action="store_true", default=True,
                        help="Save LightGlue scores and metadata alongside mkpts arrays")
    parser.add_argument("--no_save_scores", dest="save_scores", action="store_false")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0, 1].")

    args.img_size = canonical_img_size(args.img_size)

    if args.print_config_key:
        print(get_config_key(args))
        return

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene_name = infer_scene_name(args)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Projection checkpoint not found: {checkpoint_path}")

    feature_cache_dir = None
    dino_cache_dir = None
    dift_cache_dir = None
    sp_cache_dir = None
    source_cache_max_points = None
    projection_model = None
    online_extractor = None
    online_bundle_cache = None

    if args.online_extraction:
        online_extractor = build_online_extractor(args, device)
        online_bundle_cache = {}
    else:
        if args.feature_cache is None:
            args.feature_cache = PROJECT_ROOT / "cache" / "features" / f"{args.config_key}_mp{args.max_points}" / scene_name
        feature_cache_dir = Path(args.feature_cache)
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        dino_cache_dir, dift_cache_dir, source_cache_max_points = resolve_source_cache_dirs(args, scene_name)
        sp_cache_dir = Path(args.cache_root) / "superpoint_kpts" / scene_name
        sp_cache_dir.mkdir(parents=True, exist_ok=True)
        projection_model = load_projection_model(checkpoint_path, device)

    matcher = build_matcher(args, device)
    print_hyperparameter_verification(args, matcher, source_cache_max_points)
    prefix = log_prefix(args)

    if args.img1 and args.img2:
        img0_path = Path(args.img1)
        img1_path = Path(args.img2)
        mkpts0, mkpts1, match_scores, stop_layer = process_pair(
            img0_path,
            img1_path,
            args,
            device,
            feature_cache_dir,
            dino_cache_dir,
            dift_cache_dir,
            sp_cache_dir,
            projection_model,
            matcher,
            online_extractor=online_extractor,
            online_bundle_cache=online_bundle_cache,
        )
        mean_conf = float(match_scores.mean()) if match_scores.size else 0.0
        print(
            f"{prefix} Found {len(mkpts0)} matches "
            f"(mean_conf={mean_conf:.3f}, stop={stop_layer})",
            flush=True,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{img0_path.stem}__{img1_path.stem}.npz"
            save_match_archive(output_path, mkpts0, mkpts1, args, match_scores=match_scores, stop_layer=stop_layer)

        if args.visualize and len(mkpts0) > 0:
            out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "datasets"
            out_dir.mkdir(parents=True, exist_ok=True)
            maybe_visualize(img0_path, img1_path, mkpts0, mkpts1, out_dir, args.max_lines, prefix)
        return

    if args.pairs_file or args.scene:
        if scene_name == "single_pair":
            parser.error("Batch mode requires --scene or an inferable scene in --pairs_file/--images_dir.")

        images_dir, pairs_file, output_dir = resolve_scene_defaults(args, scene_name)
        if not pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = parse_pairs_file(pairs_file)
        if args.limit is not None:
            pairs = pairs[: args.limit]

        total_pairs = len(pairs)
        print(f"{prefix} Matching {total_pairs} pairs for scene={scene_name}", flush=True)

        total_matches = 0
        total_conf = 0.0
        zero_match_pairs = 0
        written = 0
        skipped_existing = 0

        for pair_index, (img0_name, img1_name) in enumerate(pairs, start=1):
            img0_path = images_dir / img0_name
            img1_path = images_dir / img1_name
            if not img0_path.exists() or not img1_path.exists():
                print(f"{prefix} Skipping missing pair: {img0_name}, {img1_name}", flush=True)
                continue

            output_path = output_dir / f"{Path(img0_name).stem}__{Path(img1_name).stem}.npz"
            if output_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            mkpts0, mkpts1, match_scores, stop_layer = process_pair(
                img0_path,
                img1_path,
                args,
                device,
                feature_cache_dir,
                dino_cache_dir,
                dift_cache_dir,
                sp_cache_dir,
                projection_model,
                matcher,
                online_extractor=online_extractor,
                online_bundle_cache=online_bundle_cache,
            )

            save_match_archive(output_path, mkpts0, mkpts1, args, match_scores=match_scores, stop_layer=stop_layer)
            written += 1

            num_matches = int(len(mkpts0))
            mean_conf = float(match_scores.mean()) if match_scores.size else 0.0
            total_matches += num_matches
            total_conf += mean_conf
            if num_matches == 0:
                zero_match_pairs += 1

            if pair_index == total_pairs or pair_index % 100 == 0:
                cache_entries = len(online_bundle_cache) if online_bundle_cache is not None else 0
                print(
                    f"{prefix} {scene_name} pair {pair_index}/{total_pairs}: "
                    f"matches={num_matches}, conf={mean_conf:.3f}, stop={stop_layer}, "
                    f"online_cache_entries={cache_entries}",
                    flush=True,
                )

        evaluated_pairs = written
        avg_matches = total_matches / evaluated_pairs if evaluated_pairs else 0.0
        avg_conf = total_conf / evaluated_pairs if evaluated_pairs else 0.0
        print(
            f"{prefix} {scene_name} DONE: {evaluated_pairs} pairs, "
            f"avg_matches={avg_matches:.1f}, avg_conf={avg_conf:.3f}, "
            f"zero_match_pairs={zero_match_pairs}, skipped_existing={skipped_existing}, "
            f"online_cache_entries={len(online_bundle_cache) if online_bundle_cache is not None else 0}",
            flush=True,
        )
        return

    parser.error("Provide either --img1/--img2 or batch inputs via --scene or --pairs_file/--output_dir.")


if __name__ == "__main__":
    main()
