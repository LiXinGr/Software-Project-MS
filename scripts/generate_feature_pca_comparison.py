#!/usr/bin/env python3
"""
Generate PCA-colored dense feature visualizations for DINOv3 and DIFT.

This script:
1. Lists the first candidate sacre_coeur preprocessed images.
2. Selects one visually suitable image by default (with fallback auto-selection).
3. Extracts DINOv3 block-16 and DIFT t=0/up=2 dense feature maps.
4. Fits independent 3-component PCA projections over spatial tokens.
5. Saves separate panels plus a combined side-by-side figure.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import PILToTensor


warnings.filterwarnings("ignore", message=".*safety checker.*")


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIFT_ROOT = PROJECT_ROOT / "external" / "DIFT"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DIFT_ROOT))

from src.models.dift_sd import SDFeaturizer


PATCH_SIZE = 16
DINO_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
DINO_FEAT_LEVEL = -8
DIFT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DIFT_INPUT_HW = (768, 768)
DIFT_T = 0
DIFT_UP_FT_INDEX = 2
DIFT_ENSEMBLE_SIZE = 1
DIFT_PROMPT = ""

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "phototourism" / "sacre_coeur" / "images_preprocessed"
CURATED_DEFAULT_IMAGE = DEFAULT_IMAGE_DIR / "01012753_375984446.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
RESAMPLING = getattr(Image, "Resampling", Image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing preprocessed sacre_coeur images.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional explicit image path. If omitted, a curated default is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the panel PNGs and combined figure are saved. Default: figures/.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=10,
        help="How many leading filenames to list before selecting an image.",
    )
    return parser.parse_args()


def list_candidate_images(image_dir: Path, candidate_count: int) -> List[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file())
    if not paths:
        raise FileNotFoundError("No images found in {}".format(image_dir))
    return paths[:candidate_count]


def absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def choose_image(candidate_images: Sequence[Path], explicit_image: Optional[Path]) -> Path:
    if explicit_image is not None:
        explicit_image = absolute_path(explicit_image)
        if not explicit_image.exists():
            raise FileNotFoundError("Explicit image not found: {}".format(explicit_image))
        return explicit_image

    if CURATED_DEFAULT_IMAGE.exists():
        return absolute_path(CURATED_DEFAULT_IMAGE)

    for path in candidate_images:
        with Image.open(path) as img:
            width, height = img.size
        aspect_ratio = width / float(height)
        if 1.15 <= aspect_ratio <= 1.35:
            return absolute_path(path)

    return absolute_path(candidate_images[0])


def print_candidate_summary(candidate_images: Iterable[Path]) -> None:
    print("[INFO] First candidate images:")
    for idx, path in enumerate(candidate_images, start=1):
        with Image.open(path) as img:
            width, height = img.size
        aspect_ratio = width / float(height)
        print(
            "  {:02d}. {} ({}x{}, AR={:.3f})".format(
                idx, path.name, width, height, aspect_ratio
            )
        )


def feat_level_to_block(feat_level: int) -> int:
    return feat_level if feat_level >= 0 else 24 + feat_level


def load_dinov3_model(device: torch.device) -> Tuple[torch.nn.Module, T.Compose]:
    block_idx = feat_level_to_block(DINO_FEAT_LEVEL)
    print(
        "[INFO] Loading DINOv3 model {} at feat_level={} (block {})".format(
            DINO_MODEL_NAME, DINO_FEAT_LEVEL, block_idx
        )
    )
    model = timm.create_model(
        DINO_MODEL_NAME,
        pretrained=True,
        features_only=True,
        out_indices=[block_idx],
        dynamic_img_size=True,
    )
    model = model.to(device)
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
    print(
        "[INFO] Loading DIFT model {} with t={}, up_ft_index={}, ensemble={}".format(
            DIFT_MODEL_ID, DIFT_T, DIFT_UP_FT_INDEX, DIFT_ENSEMBLE_SIZE
        )
    )
    return SDFeaturizer(sd_id=DIFT_MODEL_ID, device=str(device))


def extract_dinov3_features(
    image: Image.Image,
    model: torch.nn.Module,
    transform: T.Compose,
) -> np.ndarray:
    width, height = image.size
    x = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    with torch.inference_mode():
        feats = model(x)

    feature_map = feats[0].squeeze(0).permute(1, 2, 0).contiguous()
    array = feature_map.detach().cpu().numpy().astype(np.float32, copy=False)
    return array


def extract_dift_features(image: Image.Image, model: SDFeaturizer) -> np.ndarray:
    image_resized = image.resize(
        (DIFT_INPUT_HW[1], DIFT_INPUT_HW[0]),
        resample=RESAMPLING.BICUBIC,
    )
    image_tensor = (PILToTensor()(image_resized) / 255.0 - 0.5) * 2.0
    image_tensor = image_tensor.unsqueeze(0)

    with torch.inference_mode():
        feats = model.forward(
            image_tensor,
            prompt=DIFT_PROMPT,
            t=DIFT_T,
            up_ft_index=DIFT_UP_FT_INDEX,
            ensemble_size=DIFT_ENSEMBLE_SIZE,
        )

    feature_map = feats.squeeze(0).permute(1, 2, 0).contiguous()
    array = feature_map.detach().cpu().numpy().astype(np.float32, copy=False)
    return array


def pca_rgb(feature_map: np.ndarray) -> np.ndarray:
    height, width, channels = feature_map.shape
    flat = feature_map.reshape(height * width, channels).astype(np.float32, copy=False)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(flat, full_matrices=False)
    projected = flat @ vh[:3].T
    projected = projected.reshape(height, width, 3)

    rgb = np.empty_like(projected, dtype=np.float32)
    for channel_idx in range(3):
        channel = projected[..., channel_idx]
        channel_min = float(channel.min())
        channel_max = float(channel.max())
        if channel_max > channel_min:
            rgb[..., channel_idx] = (channel - channel_min) / (channel_max - channel_min)
        else:
            rgb[..., channel_idx] = 0.0
    return rgb


def resize_rgb_image(rgb: np.ndarray, size_hw: Tuple[int, int], resample: int) -> np.ndarray:
    height, width = size_hw
    pil_image = Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))
    resized = pil_image.resize((width, height), resample=resample)
    return np.asarray(resized).astype(np.float32) / 255.0


def save_panel(image_array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image_array * 255.0, 0.0, 255.0).astype(np.uint8)).save(path)


def save_combined_figure(panels: Sequence[np.ndarray], output_path: Path) -> None:
    titles = ["Input image", "DINOv3 (block 16)", "DIFT (t=0)"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        dpi=300,
        facecolor="white",
        gridspec_kw={"wspace": 0.015},
    )

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="normal", color="#333333", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("[INFO] Using device: cuda")
    else:
        print("[WARN] CUDA unavailable, using CPU.")

    candidate_images = list_candidate_images(args.image_dir, args.candidate_count)
    print_candidate_summary(candidate_images)
    image_path = choose_image(candidate_images, args.image_path)

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    print("[INFO] Selected image: {}".format(image_path))
    print("Image path used: {}".format(image_path))

    dinov3_model, dinov3_transform = load_dinov3_model(device)
    dift_model = load_dift_model(device)

    dinov3_features = extract_dinov3_features(image, dinov3_model, dinov3_transform)
    dift_features = extract_dift_features(image, dift_model)

    print(
        "[DINOV3] Feature shape: {}x{}x{}".format(
            dinov3_features.shape[0], dinov3_features.shape[1], dinov3_features.shape[2]
        )
    )
    print(
        "[DIFT] Feature shape: {}x{}x{}".format(
            dift_features.shape[0], dift_features.shape[1], dift_features.shape[2]
        )
    )

    dinov3_rgb = pca_rgb(dinov3_features)
    dift_rgb = pca_rgb(dift_features)

    display_size_hw = (image_height, image_width)
    original_panel = np.asarray(image).astype(np.float32) / 255.0
    dinov3_panel = resize_rgb_image(dinov3_rgb, display_size_hw, resample=RESAMPLING.NEAREST)
    dift_panel = resize_rgb_image(dift_rgb, display_size_hw, resample=RESAMPLING.NEAREST)

    output_dir = args.output_dir
    original_path = output_dir / "feature_pca_original.png"
    dinov3_path = output_dir / "feature_pca_dinov3.png"
    dift_path = output_dir / "feature_pca_dift.png"
    figure_path = output_dir / "feature_pca_comparison.png"

    save_panel(original_panel, original_path)
    save_panel(dinov3_panel, dinov3_path)
    save_panel(dift_panel, dift_path)
    save_combined_figure([original_panel, dinov3_panel, dift_panel], figure_path)

    print("[DONE] Figures saved to figures/")


if __name__ == "__main__":
    main()
