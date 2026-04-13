"""Precompute DINOv3 + DIFT dense features for MegaDepth training images."""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import PILToTensor


# Suppress diffusers safety checker warnings to match the existing DIFT script.
warnings.filterwarnings("ignore", message=".*safety checker.*")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIFT_ROOT = PROJECT_ROOT / "external" / "DIFT"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DIFT_ROOT))

from megadepth_pairs import MegaDepthPairSampler, MegaDepthScene
from util import preprocess_image
from src.models.dift_sd import SDFeaturizer


PATCH_SIZE = 16
DINOV3_NUM_BLOCKS = 24
DINOV3_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
DINOV3_FEAT_LEVEL = -8
DIFT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DIFT_PROMPT = ""
DIFT_T = 0
DIFT_UP_FT_INDEX = 2
DIFT_ENSEMBLE_SIZE = 1
DIFT_INPUT_HW = (1120, 1120)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def feat_level_to_out_idx(feat_level: int) -> int:
    """Convert a possibly negative DINOv3 feature level into a block index."""
    if feat_level < 0:
        return DINOV3_NUM_BLOCKS + feat_level
    return feat_level


def canonical_scenes(root: Path, scenes: Optional[List[str]]) -> List[str]:
    if scenes is None:
        return sorted(path.name for path in root.iterdir() if path.is_dir())
    return list(scenes)


def load_dinov3_model(device: torch.device) -> Tuple[torch.nn.Module, T.Compose]:
    """Load DINOv3 ViT-L/16 exactly as in scripts/dinov3_matches.py."""
    out_idx = feat_level_to_out_idx(DINOV3_FEAT_LEVEL)
    model = timm.create_model(
        DINOV3_MODEL_NAME,
        pretrained=True,
        features_only=True,
        out_indices=[out_idx],
        dynamic_img_size=True,
    )
    model.to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ]
    )
    return model, transform


def load_dift_model(device: torch.device) -> SDFeaturizer:
    """Load DIFT exactly as in scripts/dift_matches.py."""
    return SDFeaturizer(sd_id=DIFT_MODEL_ID, device=str(device))


def preprocess_eval_image(
    image_path: Path,
    max_size: int = 1120,
) -> Tuple[Image.Image, Dict[str, np.ndarray | float]]:
    """Apply the shared letterbox preprocessing used by the benchmark pipeline."""
    img = Image.open(image_path).convert("RGB")
    img_preprocessed, info = preprocess_image(
        img,
        target_long_edge=max_size,
        divisibility=16,
        return_info=True,
    )
    pre_h, pre_w = info.final_size
    dift_h, dift_w = DIFT_INPUT_HW
    meta = {
        "image_size": np.asarray(info.orig_size, dtype=np.int32),
        "preprocess_scale": float(info.scale),
        "preprocessed_size": np.asarray(info.final_size, dtype=np.int32),
        "resized_size": np.asarray(info.resized_size, dtype=np.int32),
        "pad_left": np.int32(info.pad_left),
        "pad_top": np.int32(info.pad_top),
        # Match the current eval pipeline: DIFT receives the letterboxed image
        # after an explicit resize to a fixed square input grid.
        "dift_input_size": np.asarray(DIFT_INPUT_HW, dtype=np.int32),
        "dift_pre_to_input_scale_xy": np.asarray(
            [dift_w / pre_w, dift_h / pre_h],
            dtype=np.float32,
        ),
    }
    return img_preprocessed, meta


def extract_dinov3_features(
    model: torch.nn.Module,
    transform: T.Compose,
    image: Image.Image,
) -> np.ndarray:
    """Extract block-16 DINOv3 dense features and return HxWxC float16."""
    width, height = image.size
    x = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

    with torch.inference_mode():
        feats = model(x)

    ft = feats[0].squeeze(0)
    array = ft.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float16, copy=False)
    del ft, feats, x
    return array


def extract_dift_features(
    model: SDFeaturizer,
    image: Image.Image,
) -> np.ndarray:
    """Extract DIFT t=0/up=2 dense features and return HxWxC float16."""
    img_resized = image.resize((DIFT_INPUT_HW[1], DIFT_INPUT_HW[0]))
    img_tensor = (PILToTensor()(img_resized) / 255.0 - 0.5) * 2
    img_tensor = img_tensor.unsqueeze(0)

    with torch.inference_mode():
        ft = model.forward(
            img_tensor,
            prompt=DIFT_PROMPT,
            t=DIFT_T,
            up_ft_index=DIFT_UP_FT_INDEX,
            ensemble_size=DIFT_ENSEMBLE_SIZE,
        )

    ft = ft.squeeze(0)
    array = ft.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float16, copy=False)
    del ft, img_tensor, img_resized
    return array


def find_scene_reconstruction(scene_dir: Path) -> Optional[str]:
    return MegaDepthPairSampler._find_reconstruction(scene_dir)


def iter_scene_images(root: Path, scenes: List[str]) -> Iterable[Tuple[str, str, List[Path], int]]:
    """Yield (scene_name, reconstruction, image_paths, missing_count) per scene."""
    for scene_name in scenes:
        scene_dir = root / scene_name
        if not scene_dir.is_dir():
            print(f"Scene {scene_name}: SKIPPED - directory not found")
            continue

        reconstruction = find_scene_reconstruction(scene_dir)
        if reconstruction is None:
            print(f"Scene {scene_name}: SKIPPED - no valid reconstruction found")
            continue

        resolved = MegaDepthPairSampler._resolve_reconstruction_dir(
            scene_dir, reconstruction
        )
        if resolved is None:
            print(f"Scene {scene_name}: SKIPPED - reconstruction path could not be resolved")
            continue
        recon_dir, reconstruction = resolved
        images = MegaDepthScene._parse_images(recon_dir / "images.txt")
        image_paths: List[Path] = []
        missing_count = 0

        for image_id in sorted(images):
            image_path = scene_dir / "images" / images[image_id].name
            if image_path.suffix not in IMAGE_EXTENSIONS:
                continue
            if image_path.is_file():
                image_paths.append(image_path)
            else:
                missing_count += 1

        yield scene_name, reconstruction, image_paths, missing_count


def validate_feature_array(name: str, array: np.ndarray) -> Dict[str, object]:
    return {
        "name": name,
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "has_nan": bool(np.isnan(array).any()),
        "min": float(array.min()) if array.size else float("nan"),
        "max": float(array.max()) if array.size else float("nan"),
        "mean": float(array.mean()) if array.size else float("nan"),
    }


def npz_contains_keys(path: Path, required_keys: List[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return all(key in data.files for key in required_keys)
    except Exception:
        return False


def load_existing_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_feature_npz(path: Path, payload: Dict[str, np.ndarray | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed .npz is substantially faster on the shared filesystem and
    # preserves the exact stored arrays without extra CPU time spent compressing.
    np.savez(path, **payload)


def format_mem_gb(value_bytes: int) -> float:
    return value_bytes / (1024 ** 3)


def process_single_image(
    image_path: Path,
    scene_name: str,
    output_dir: Path,
    need_dinov3: bool,
    need_dift: bool,
    dinov3_model: Optional[torch.nn.Module],
    dinov3_transform: Optional[T.Compose],
    dift_model: Optional[SDFeaturizer],
    device: torch.device,
    skip_existing: bool,
) -> Tuple[str, Optional[Dict[str, object]]]:
    output_path = output_dir / scene_name / f"{image_path.stem}.npz"
    required_keys = ["image_size", "preprocess_scale"]
    if need_dinov3:
        required_keys.append("dinov3")
    if need_dift:
        required_keys.append("dift")

    if skip_existing and npz_contains_keys(output_path, required_keys):
        return "skipped", None

    start_time = time.perf_counter()
    preprocessed_image, meta = preprocess_eval_image(image_path)
    preprocess_time = time.perf_counter() - start_time

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    payload = load_existing_npz(output_path)
    payload.update(meta)
    stats: Dict[str, object] = {
        "scene": scene_name,
        "image_name": image_path.name,
        "output_path": str(output_path),
        "preprocess_time_sec": preprocess_time,
    }

    dino_time = 0.0
    dift_time = 0.0

    try:
        if need_dinov3:
            if dinov3_model is None or dinov3_transform is None:
                raise RuntimeError("DINOv3 model requested but not loaded.")
            t0 = time.perf_counter()
            dinov3 = extract_dinov3_features(dinov3_model, dinov3_transform, preprocessed_image)
            dino_time = time.perf_counter() - t0
            payload["dinov3"] = dinov3
            stats["dinov3"] = validate_feature_array("dinov3", dinov3)

        if need_dift:
            if dift_model is None:
                raise RuntimeError("DIFT model requested but not loaded.")
            t0 = time.perf_counter()
            dift = extract_dift_features(dift_model, preprocessed_image)
            dift_time = time.perf_counter() - t0
            payload["dift"] = dift
            payload["dift_feature_size"] = np.asarray(dift.shape[:2], dtype=np.int32)
            stats["dift"] = validate_feature_array("dift", dift)
        elif "dift" in payload and "dift_feature_size" not in payload:
            payload["dift_feature_size"] = np.asarray(payload["dift"].shape[:2], dtype=np.int32)

        if need_dinov3:
            payload["dinov3_feature_size"] = np.asarray(payload["dinov3"].shape[:2], dtype=np.int32)
        elif "dinov3" in payload and "dinov3_feature_size" not in payload:
            payload["dinov3_feature_size"] = np.asarray(payload["dinov3"].shape[:2], dtype=np.int32)

        save_t0 = time.perf_counter()
        save_feature_npz(output_path, payload)
        save_time = time.perf_counter() - save_t0
        total_time = time.perf_counter() - start_time

        stats["image_size"] = tuple(int(x) for x in payload["image_size"])
        stats["preprocessed_size"] = tuple(int(x) for x in payload["preprocessed_size"])
        stats["preprocess_scale"] = float(payload["preprocess_scale"])
        stats["dinov3_time_sec"] = dino_time
        stats["dift_time_sec"] = dift_time
        stats["save_time_sec"] = save_time
        stats["total_time_sec"] = total_time
        stats["output_size_mb"] = output_path.stat().st_size / (1024 ** 2)

        if device.type == "cuda":
            stats["gpu_mem_allocated_gb"] = format_mem_gb(torch.cuda.memory_allocated(device))
            stats["gpu_mem_reserved_gb"] = format_mem_gb(torch.cuda.memory_reserved(device))
            stats["gpu_peak_allocated_gb"] = format_mem_gb(torch.cuda.max_memory_allocated(device))
            stats["gpu_peak_reserved_gb"] = format_mem_gb(torch.cuda.max_memory_reserved(device))

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return "extracted", stats
    except Exception:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute DINOv3 + DIFT dense features for MegaDepth training images."
    )
    parser.add_argument(
        "--megadepth_root",
        default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM",
        help="MegaDepth SfM root directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where per-image .npz feature files will be stored",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Scene names to process. Default: all scenes under the MegaDepth root",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Reserved for future batching. Exact eval matching uses batch_size=1.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images whose output .npz already contains the requested keys",
    )
    parser.add_argument(
        "--dinov3_only",
        action="store_true",
        help="Only extract DINOv3 features",
    )
    parser.add_argument(
        "--dift_only",
        action="store_true",
        help="Only extract DIFT features",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Stop after this many images globally. Useful for smoke tests.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N processed images",
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "batch_size > 1 is intentionally unsupported here so feature tensors "
            "match the single-image eval pipeline exactly."
        )
    if args.dinov3_only and args.dift_only:
        raise ValueError("Choose at most one of --dinov3_only and --dift_only.")

    need_dinov3 = not args.dift_only
    need_dift = not args.dinov3_only
    verbose_images = args.max_images is not None and args.max_images <= 20

    root = Path(args.megadepth_root)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"MegaDepth root: {root}")
    print(f"Output dir: {output_dir}")
    print(f"Using device: {device}")
    print(f"DINOv3 enabled: {need_dinov3}")
    print(f"DIFT enabled: {need_dift}")

    scenes = canonical_scenes(root, args.scenes)
    print(f"Scenes to process: {len(scenes)}")
    print(f"Verbose per-image logging: {verbose_images}")

    dinov3_model = None
    dinov3_transform = None
    dift_model = None

    if need_dinov3:
        print(
            f"Loading DINOv3 {DINOV3_MODEL_NAME} at feat_level={DINOV3_FEAT_LEVEL} "
            f"(block {feat_level_to_out_idx(DINOV3_FEAT_LEVEL)})"
        )
        dinov3_model, dinov3_transform = load_dinov3_model(device)

    if need_dift:
        print(
            f"Loading DIFT {DIFT_MODEL_ID} with t={DIFT_T}, "
            f"up_ft_index={DIFT_UP_FT_INDEX}, ensemble_size={DIFT_ENSEMBLE_SIZE}"
        )
        dift_model = load_dift_model(device)

    if device.type == "cuda":
        print(f"CUDA current allocated: {format_mem_gb(torch.cuda.memory_allocated(device)):.3f} GB")
        print(f"CUDA current reserved: {format_mem_gb(torch.cuda.memory_reserved(device)):.3f} GB")

    total_processed = 0
    total_extracted = 0
    total_skipped = 0
    total_failed = 0
    run_start = time.perf_counter()
    scene_entries = list(iter_scene_images(root, scenes))
    total_expected_images = sum(len(image_paths) for _, _, image_paths, _ in scene_entries)
    if args.max_images is not None:
        total_expected_images = min(total_expected_images, args.max_images)
    print(f"Expected images to visit: {total_expected_images}")

    for scene_name, reconstruction, image_paths, missing_count in scene_entries:
        print(
            f"\nScene {scene_name}: reconstruction={reconstruction}, "
            f"images_in_recon={len(image_paths)}, missing_on_disk={missing_count}"
        )
        extracted = 0
        skipped = 0
        failed = 0
        scene_start = time.perf_counter()

        progress = tqdm(image_paths, desc=f"{scene_name}", leave=False)
        for image_path in progress:
            if args.max_images is not None and total_processed >= args.max_images:
                break

            try:
                status, stats = process_single_image(
                    image_path=image_path,
                    scene_name=scene_name,
                    output_dir=output_dir,
                    need_dinov3=need_dinov3,
                    need_dift=need_dift,
                    dinov3_model=dinov3_model,
                    dinov3_transform=dinov3_transform,
                    dift_model=dift_model,
                    device=device,
                    skip_existing=args.skip_existing,
                )
            except Exception as exc:
                failed += 1
                total_failed += 1
                total_processed += 1
                print(f"FAILED {scene_name}/{image_path.name}: {exc}")
                continue

            total_processed += 1
            if status == "skipped":
                skipped += 1
                total_skipped += 1
            else:
                extracted += 1
                total_extracted += 1

                if stats is not None and verbose_images:
                    print(f"\nImage {scene_name}/{stats['image_name']}")
                    print(
                        f"  image_size={stats['image_size']} "
                        f"preprocessed_size={stats['preprocessed_size']} "
                        f"scale={stats['preprocess_scale']:.6f}"
                    )
                    if "dinov3" in stats:
                        dinov3_stats = stats["dinov3"]
                        print(
                            "  DINOv3: "
                            f"shape={dinov3_stats['shape']} "
                            f"dtype={dinov3_stats['dtype']} "
                            f"nan={dinov3_stats['has_nan']} "
                            f"range=[{dinov3_stats['min']:.4f}, {dinov3_stats['max']:.4f}] "
                            f"mean={dinov3_stats['mean']:.4f}"
                        )
                    if "dift" in stats:
                        dift_stats = stats["dift"]
                        print(
                            "  DIFT: "
                            f"shape={dift_stats['shape']} "
                            f"dtype={dift_stats['dtype']} "
                            f"nan={dift_stats['has_nan']} "
                            f"range=[{dift_stats['min']:.4f}, {dift_stats['max']:.4f}] "
                            f"mean={dift_stats['mean']:.4f}"
                        )
                    print(
                        "  timings: "
                        f"pre={stats['preprocess_time_sec']:.3f}s "
                        f"dino={stats['dinov3_time_sec']:.3f}s "
                        f"dift={stats['dift_time_sec']:.3f}s "
                        f"save={stats['save_time_sec']:.3f}s "
                        f"total={stats['total_time_sec']:.3f}s"
                    )
                    print(f"  output: {stats['output_path']} ({stats['output_size_mb']:.2f} MB)")
                    if device.type == "cuda":
                        print(
                            "  gpu: "
                            f"alloc={stats['gpu_mem_allocated_gb']:.3f} GB "
                            f"reserved={stats['gpu_mem_reserved_gb']:.3f} GB "
                            f"peak_alloc={stats['gpu_peak_allocated_gb']:.3f} GB "
                            f"peak_reserved={stats['gpu_peak_reserved_gb']:.3f} GB"
                        )

            if args.progress_every > 0 and total_processed % args.progress_every == 0:
                print(
                    f"Progress: processed={total_processed}, extracted={total_extracted}, "
                    f"skipped={total_skipped}, failed={total_failed}"
                )

        print(
            f"Scene {scene_name}: {len(image_paths)} images, "
            f"{extracted} extracted, {skipped} skipped, {failed} failed"
        )
        scene_elapsed = time.perf_counter() - scene_start
        run_elapsed = time.perf_counter() - run_start
        print(
            f"  Scene time: {scene_elapsed / 60:.2f} min | "
            f"Cumulative: {run_elapsed / 3600:.2f} h"
        )

        if total_processed > 0 and total_expected_images > total_processed:
            images_per_sec = total_processed / run_elapsed
            remaining_images = total_expected_images - total_processed
            eta_seconds = remaining_images / images_per_sec
            print(
                f"  ETA after scene {scene_name}: "
                f"{eta_seconds / 3600:.2f} h remaining "
                f"({remaining_images} images at {images_per_sec:.3f} img/s)"
            )

        if args.max_images is not None and total_processed >= args.max_images:
            break

    print(
        f"\nDone: processed={total_processed}, extracted={total_extracted}, "
        f"skipped={total_skipped}, failed={total_failed}"
    )


if __name__ == "__main__":
    main()
