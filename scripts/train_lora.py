"""LoRA fine-tuning of DINOv3 with joint projection head training (Phase 2b).

Architecture per training step:
  1. Load image pair from MegaDepth
  2. Run both images through DINOv3 + LoRA adapters -> dense feature maps (grad)
  3. Sample DINOv3 features at SfM correspondence points -> (N, 1024)
  4. Load precomputed DIFT features for the same points from sparse bundles -> (N, 640)
     [Split confirmed: features[:, :640] = DIFT, features[:, 640:] = DINOv3
      matching concatenation order in fusion_matches.py and extract_sparse_training_data.py]
  5. L2-normalize each component, concatenate [alpha*DIFT, (1-alpha)*DINOv3] -> (N, 1664)
  6. Projection head (wide: 1664->1024->256) + L2-normalize -> (N, 256)
  7. Symmetric InfoNCE loss
  8. Backprop through projection head + LoRA params (DINOv3 base weights frozen)

peft is not used due to transformers version conflict; LoRA is implemented manually.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import resource
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from megadepth_pairs import MegaDepthScene
from train_projection_head import ProjectionHead, pair_loss_and_accuracy, SparseFeatureDataset
from util import preprocess_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFT_DIM = 640
DINOV3_DIM = 1024
FUSED_DIM = DIFT_DIM + DINOV3_DIM  # 1664
PATCH_SIZE = 16
DINOV3_NUM_BLOCKS = 24
DINOV3_FEAT_LEVEL = -8  # block 16 of 24
DINOV3_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
MAX_IMG_SIZE = 1120


# ---------------------------------------------------------------------------
# Manual LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adapters.

    Base weight and bias are frozen (requires_grad=False).
    Only lora_A and lora_B are trained.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = alpha / rank

        # Frozen base weights
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA parameters
        # lora_A: (rank, in_features)  – down-projection, init random
        # lora_B: (out_features, rank) – up-projection, init zero -> delta = 0 at start
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path
        out = F.linear(x, self.weight, self.bias)
        # LoRA delta: x -> (*, rank) -> (*, out_features)
        lora = F.linear(self.dropout(x), self.lora_A)   # (*, rank)
        lora = F.linear(lora, self.lora_B)               # (*, out_features)
        return out + self.scaling * lora

    def merged_weight(self) -> torch.Tensor:
        """Return base weight with LoRA delta merged in (for inference)."""
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)


class LoRALinearSplitQKV(nn.Module):
    """LoRA for fused QKV projection with separate adapters for Q and V.

    Applies rank-r adapters independently to the Q slice (rows 0:D) and V slice
    (rows 2D:3D) of the output, where D = out_features // 3.  K is left unchanged,
    which is standard practice (Q and V carry most of the representational benefit).

    Param count with rank r, D=head_dim:
      Q: A_q (r×in) + B_q (D×r)  =  r*(in + D)
      V: A_v (r×in) + B_v (D×r)  =  r*(in + D)
      For ViT-L/16 (in=1024, D=1024, r=4): 4*(1024+1024)*2 = 16 384 — same as the
      old single-LoRA approach (which had B of shape 3072×r = 3*D*r), so the total
      parameter budget is identical but now split correctly across Q and V.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert base.out_features % 3 == 0, (
            f"out_features ({base.out_features}) must be divisible by 3 for fused QKV"
        )
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.head_dim = base.out_features // 3  # D (= model dim for ViT)
        self.scaling = alpha / rank

        # Frozen base weights
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Q adapter: lora_A_q (rank, in_features), lora_B_q (head_dim, rank)
        self.lora_A_q = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(self.head_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))

        # V adapter: lora_A_v (rank, in_features), lora_B_v (head_dim, rank)
        self.lora_A_v = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(self.head_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        xd = self.dropout(x)
        # Q delta: (*, head_dim)
        delta_q = F.linear(F.linear(xd, self.lora_A_q), self.lora_B_q)
        # V delta: (*, head_dim)
        delta_v = F.linear(F.linear(xd, self.lora_A_v), self.lora_B_v)
        sc = self.scaling
        D = self.head_dim
        return torch.cat([
            out[..., :D] + sc * delta_q,          # Q
            out[..., D:2 * D],                     # K  (unchanged)
            out[..., 2 * D:] + sc * delta_v,       # V
        ], dim=-1)

    def merged_weight(self) -> torch.Tensor:
        """Return base weight with Q and V LoRA deltas merged in (for inference)."""
        w = self.weight.clone()
        D = self.head_dim
        w[:D] += self.scaling * (self.lora_B_q @ self.lora_A_q)
        w[2 * D:] += self.scaling * (self.lora_B_v @ self.lora_A_v)
        return w


def inject_lora(
    model: nn.Module,
    target_suffixes: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    qkv_suffixes: Optional[List[str]] = None,
) -> int:
    """Replace Linear layers whose name ends with any target suffix with a LoRA wrapper.

    Layers whose name also ends with a suffix in *qkv_suffixes* use
    LoRALinearSplitQKV (separate Q and V adapters); all others use LoRALinear.

    Returns the number of layers replaced.
    """
    if qkv_suffixes is None:
        qkv_suffixes = []
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        for suffix in target_suffixes:
            if name.endswith(suffix):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                use_split = any(name.endswith(s) for s in qkv_suffixes)
                if use_split:
                    wrapper = LoRALinearSplitQKV(module, rank=rank, alpha=alpha, dropout=dropout)
                else:
                    wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, parts[-1], wrapper)
                replaced += 1
                break
    return replaced


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora_into_model(model: nn.Module) -> int:
    """Replace LoRALinear / LoRALinearSplitQKV with plain nn.Linear (merged weights).

    Operates in-place.  Returns the number of layers replaced.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, (LoRALinear, LoRALinearSplitQKV)):
            continue
        merged_w = module.merged_weight()
        linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        linear.weight = nn.Parameter(merged_w.detach())
        if module.bias is not None:
            linear.bias = nn.Parameter(module.bias.detach().clone())
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)
        merged += 1
    return merged


# ---------------------------------------------------------------------------
# DINOv3 loading
# ---------------------------------------------------------------------------

def load_dinov3_with_lora(
    feat_level: int,
    rank: int,
    lora_alpha: float,
    dropout: float,
    device: torch.device,
) -> Tuple[nn.Module, T.Compose]:
    """Load DINOv3 ViT-L/16, freeze base weights, inject LoRA into qkv layers."""
    out_idx = DINOV3_NUM_BLOCKS + feat_level if feat_level < 0 else feat_level
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

    # Freeze all base weights
    for param in model.parameters():
        param.requires_grad_(False)

    # Inject LoRA into combined QKV layers of all transformer blocks.
    # Use LoRALinearSplitQKV so Q and V each get their own rank-r adapter;
    # K is left unchanged (standard practice).
    n_replaced = inject_lora(
        model,
        target_suffixes=["attn.qkv"],
        rank=rank,
        alpha=lora_alpha,
        dropout=dropout,
        qkv_suffixes=["attn.qkv"],
    )
    model.to(device)

    n_trainable = count_trainable(model)
    print(f"[LoRA] DINOv3 feat_level={feat_level} (block {out_idx})")
    print(f"[LoRA] Replaced {n_replaced} qkv layers with split-QV rank-{rank} LoRA (alpha={lora_alpha})")
    print(f"[LoRA] Trainable parameters: {n_trainable:,}")

    return model, transform


def load_dinov3_frozen_lora(
    feat_level: int,
    lora_checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, T.Compose]:
    """Load DINOv3, inject LoRA from Stage-1 checkpoint, merge, freeze all weights.

    Used for Stages 4 and 5 where the backbone is fixed and only a new
    projection head is trained.
    """
    ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    rank = int(ckpt_cfg.get("lora_rank", 4))
    alpha = float(ckpt_cfg.get("lora_alpha", 8.0))
    dropout = float(ckpt_cfg.get("lora_dropout", 0.0))
    lora_state = ckpt["lora_state_dict"]

    out_idx = DINOV3_NUM_BLOCKS + feat_level if feat_level < 0 else feat_level
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

    # Freeze base weights, inject LoRA structure to match checkpoint shape
    for param in model.parameters():
        param.requires_grad_(False)
    inject_lora(model, target_suffixes=["attn.qkv"], rank=rank, alpha=alpha,
                dropout=dropout, qkv_suffixes=["attn.qkv"])

    # Load LoRA weights then merge into base weights
    model.load_state_dict(lora_state, strict=False)
    n_merged = merge_lora_into_model(model)

    # Re-freeze (merge replaced LoRALinear with plain nn.Linear)
    for param in model.parameters():
        param.requires_grad_(False)

    model.to(device).eval()
    print(f"[LoRA] Frozen backbone loaded from {lora_checkpoint_path}")
    print(f"[LoRA] Merged {n_merged} LoRA layers; all weights frozen")
    return model, transform


def init_trainable_lora_from_checkpoint(
    model: nn.Module,
    lora_checkpoint_path: str,
) -> Dict:
    """Load Stage-1 LoRA weights into an injected trainable LoRA model."""
    ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
    lora_state = ckpt["lora_state_dict"]

    try:
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load trainable LoRA init from {lora_checkpoint_path}. "
            "This usually means the checkpoint rank/alpha layout does not match "
            "the currently injected LoRA structure."
        ) from exc

    missing_lora = [k for k in missing if "lora_" in k]
    unexpected_lora = [k for k in unexpected if "lora_" in k]
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "LoRA init checkpoint did not load cleanly.\n"
            f"  missing LoRA keys: {missing_lora[:5]}\n"
            f"  unexpected LoRA keys: {unexpected_lora[:5]}"
        )

    ckpt_cfg = ckpt.get("config", {})
    print(f"[LoRA] Trainable adapters initialised from {lora_checkpoint_path}")
    print(
        f"[LoRA] Init config: rank={ckpt_cfg.get('lora_rank', 'unknown')} "
        f"alpha={ckpt_cfg.get('lora_alpha', 'unknown')} "
        f"dropout={ckpt_cfg.get('lora_dropout', 'unknown')}"
    )
    return ckpt_cfg


# ---------------------------------------------------------------------------
# Feature map sampling (differentiable)
# ---------------------------------------------------------------------------

def sample_feature_map(
    ft: torch.Tensor,
    pts_pre: np.ndarray,
    final_hw: Tuple[int, int],
) -> torch.Tensor:
    """Bilinear-sample a [C, H_feat, W_feat] feature map at preprocessed-space pixel coords.

    Args:
        ft: [C, H_feat, W_feat] on device (may carry grad)
        pts_pre: (N, 2) float32 array of (x, y) coords in preprocessed image space
        final_hw: (H_pre, W_pre) — size of the preprocessed image

    Returns:
        (N, C) tensor on the same device as ft, gradients preserved.
    """
    C, H_feat, W_feat = ft.shape
    H_pre, W_pre = final_hw
    N = pts_pre.shape[0]

    if N == 0:
        return torch.empty((0, C), dtype=ft.dtype, device=ft.device)

    kpts = torch.from_numpy(pts_pre).float().to(ft.device)  # (N, 2)
    # Normalize to [-1, 1] using preprocessed image dimensions
    gx = (kpts[:, 0] / (W_pre - 1)) * 2.0 - 1.0
    gy = (kpts[:, 1] / (H_pre - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=1).view(1, N, 1, 2)  # (1, N, 1, 2)

    # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
    desc = F.grid_sample(
        ft.unsqueeze(0),  # (1, C, H_feat, W_feat)
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    # desc: (1, C, N, 1) -> (N, C)
    return desc.squeeze(0).squeeze(-1).t()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LoRADataset:
    """Provides (DIFT features, pixel coords, image paths) for LoRA training.

    Uses precomputed sparse bundles for DIFT features and COLMAP reconstructions
    for ground-truth pixel-level correspondences. DINOv3 features are computed
    online by the training loop using the LoRA-adapted model.
    """

    def __init__(
        self,
        sparse_dir: str,
        megadepth_root: str,
        scenes: List[str],
        num_correspondences: int = 256,
        min_correspondences: int = 50,
    ) -> None:
        self.sparse_dir = Path(sparse_dir)
        self.megadepth_root = Path(megadepth_root)
        self.num_correspondences = num_correspondences
        self.min_correspondences = min_correspondences

        self.scene_data: Dict[str, Dict] = {}
        self.valid_pairs: List[Tuple[str, int, int]] = []

        for scene_name in scenes:
            self._load_scene(scene_name)

        print(f"[LoRADataset] {len(self.valid_pairs):,} valid pairs across {len(self.scene_data)} scenes")

    def _load_scene(self, scene_name: str) -> None:
        bundle_path = self.sparse_dir / f"{scene_name}.pt"
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Missing sparse bundle: {bundle_path}")

        raw = torch.load(bundle_path, map_location="cpu", weights_only=False)

        def to_np(v):
            return v.numpy() if torch.is_tensor(v) else np.asarray(v)

        features = to_np(raw["features"])                          # (N, 1664) float16
        point3D_ids = to_np(raw["point3D_ids"])                    # (N,) int64
        image_offsets = to_np(raw["image_offsets"])                # (n_imgs+1,) int64
        pos_counts = to_np(raw["image_positive_counts"])           # (n_imgs,) int32
        pair_indices = to_np(raw["pair_indices"])                  # (n_pairs, 2) int32
        image_names: List[str] = list(raw.get("image_names", []))
        reconstruction: str = raw.get("reconstruction", "manhattan/0")

        del raw
        gc.collect()

        # Load COLMAP data for pixel coordinates
        scene_dir = self.megadepth_root / scene_name
        try:
            colmap_scene = MegaDepthScene(str(scene_dir), reconstruction=reconstruction, min_covis=0)
        except Exception as exc:
            print(f"[LoRADataset] WARNING: COLMAP load failed for {scene_name}: {exc}")
            colmap_scene = None

        # Index: image_name (filename only) -> COLMAP image object
        name_to_colmap = {}
        if colmap_scene is not None:
            for img_obj in colmap_scene.images.values():
                # Store under both full name and basename
                name_to_colmap[img_obj.name] = img_obj
                name_to_colmap[Path(img_obj.name).name] = img_obj

        self.scene_data[scene_name] = {
            "features": features,
            "point3D_ids": point3D_ids,
            "image_offsets": image_offsets,
            "pos_counts": pos_counts,
            "image_names": image_names,
            "name_to_colmap": name_to_colmap,
        }

        for img_idx_1, img_idx_2 in pair_indices:
            self.valid_pairs.append((scene_name, int(img_idx_1), int(img_idx_2)))

        print(f"  {scene_name}: {pair_indices.shape[0]:,} pairs, {len(image_names)} images")

    def _colmap_pid_to_xy(self, scene_name: str, img_idx: int) -> Optional[Dict[int, np.ndarray]]:
        """Build {point3D_id: (x, y)} map for one image using COLMAP data."""
        sd = self.scene_data[scene_name]
        name = sd["image_names"][img_idx]
        img_obj = sd["name_to_colmap"].get(name) or sd["name_to_colmap"].get(Path(name).name)
        if img_obj is None:
            return None
        valid = img_obj.point2D_ids >= 0
        pids = img_obj.point2D_ids[valid]
        xys = img_obj.point2D_xy[valid]
        return {int(p): xy for p, xy in zip(pids, xys)}

    def get_pair_batch(
        self,
        pair_idx: int,
        rng: np.random.Generator,
        fallback_pool: Optional[List[int]] = None,
        max_attempts: int = 20,
    ) -> Optional[Dict]:
        pool = list(fallback_pool) if fallback_pool else None

        for attempt in range(max_attempts):
            if attempt == 0:
                actual_idx = pair_idx
            elif pool:
                actual_idx = int(pool[rng.integers(0, len(pool))])
            else:
                actual_idx = int(rng.integers(0, len(self.valid_pairs)))

            scene_name, img_idx_1, img_idx_2 = self.valid_pairs[actual_idx]
            sd = self.scene_data[scene_name]

            start_1 = int(sd["image_offsets"][img_idx_1])
            pos_1 = int(sd["pos_counts"][img_idx_1])
            start_2 = int(sd["image_offsets"][img_idx_2])
            pos_2 = int(sd["pos_counts"][img_idx_2])

            if pos_1 == 0 or pos_2 == 0:
                continue

            ids_1 = sd["point3D_ids"][start_1 : start_1 + pos_1]
            ids_2 = sd["point3D_ids"][start_2 : start_2 + pos_2]

            _, idx_1, idx_2 = np.intersect1d(ids_1, ids_2, assume_unique=True, return_indices=True)
            if idx_1.size < self.min_correspondences:
                continue

            # Subsample if necessary
            if idx_1.size > self.num_correspondences:
                sel = rng.choice(idx_1.size, size=self.num_correspondences, replace=False)
                idx_1 = idx_1[sel]
                idx_2 = idx_2[sel]

            # Get pixel coordinates from COLMAP
            shared_ids = ids_1[idx_1]
            pid_xy_1 = self._colmap_pid_to_xy(scene_name, img_idx_1)
            pid_xy_2 = self._colmap_pid_to_xy(scene_name, img_idx_2)
            if pid_xy_1 is None or pid_xy_2 is None:
                continue

            pts_1_list, pts_2_list = [], []
            valid = True
            for pid in shared_ids:
                xy1 = pid_xy_1.get(int(pid))
                xy2 = pid_xy_2.get(int(pid))
                if xy1 is None or xy2 is None:
                    valid = False
                    break
                pts_1_list.append(xy1)
                pts_2_list.append(xy2)
            if not valid:
                continue

            pts_1 = np.array(pts_1_list, dtype=np.float32)  # (N, 2) x,y original coords
            pts_2 = np.array(pts_2_list, dtype=np.float32)

            # DIFT features (first 640 dims of fused features)
            dift_1 = sd["features"][start_1 + idx_1, :DIFT_DIM].astype(np.float32, copy=True)
            dift_2 = sd["features"][start_2 + idx_2, :DIFT_DIM].astype(np.float32, copy=True)

            # Image paths
            name_1 = sd["image_names"][img_idx_1]
            name_2 = sd["image_names"][img_idx_2]
            img_path_1 = self.megadepth_root / scene_name / "images" / name_1
            img_path_2 = self.megadepth_root / scene_name / "images" / name_2
            if not img_path_1.exists() or not img_path_2.exists():
                continue

            return {
                "dift_1": torch.from_numpy(dift_1),   # (N, 640) float32
                "dift_2": torch.from_numpy(dift_2),
                "pts_1": pts_1,                        # (N, 2) float32, original image coords
                "pts_2": pts_2,
                "img_path_1": str(img_path_1),
                "img_path_2": str(img_path_2),
                "n": idx_1.size,
            }

        return None


# ---------------------------------------------------------------------------
# Image preprocessing cache
# ---------------------------------------------------------------------------

class PreprocessCache:
    """LRU cache: path -> (preprocessed PIL image, PreprocessInfo)."""

    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, img_path: str):
        if img_path in self._cache:
            self._cache.move_to_end(img_path)
            return self._cache[img_path]
        img_pre, info = self._load(img_path)
        self._cache[img_path] = (img_pre, info)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return img_pre, info

    @staticmethod
    def _load(img_path: str):
        img = Image.open(img_path).convert("RGB")
        img_pre, info = preprocess_image(
            img,
            target_long_edge=MAX_IMG_SIZE,
            divisibility=PATCH_SIZE,
            return_info=True,
        )
        return img_pre, info


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def orig_pts_to_preproc(pts: np.ndarray, info) -> np.ndarray:
    """Transform original-image pixel coords to preprocessed-image coords."""
    out = pts.astype(np.float32, copy=True)
    out *= float(info.scale)
    out[:, 0] += float(info.pad_left)
    out[:, 1] += float(info.pad_top)
    return out


def image_to_tensor(img_pil: Image.Image, transform: T.Compose, device: torch.device) -> torch.Tensor:
    """Apply transform and move to device. Image must already be preprocessed."""
    return transform(img_pil).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Derived dims / architecture
    proj_input_dim: int = DINOV3_DIM if args.dinov3_only else FUSED_DIM  # 1024 or 1664
    default_hidden: List[int] = [512] if args.dinov3_only else [1024]
    hidden_dims: List[int] = list(args.hidden_dims) if args.hidden_dims else default_hidden

    # fast_sparse: pre-extracted LoRA features → no DINOv3 forward pass at all
    fast_path = args.freeze_backbone and getattr(args, "fast_sparse", False)

    # ---- Dataset ----
    print("\nLoading dataset...")
    if fast_path:
        dataset = SparseFeatureDataset(
            sparse_dir=args.sparse_dir,
            scenes=args.scenes,
            num_correspondences=args.num_correspondences,
            min_correspondences=args.min_correspondences,
            scene_cache_size=len(args.scenes),   # keep all scenes in RAM
        )
    else:
        dataset = LoRADataset(
            sparse_dir=args.sparse_dir,
            megadepth_root=args.megadepth_root,
            scenes=args.scenes,
            num_correspondences=args.num_correspondences,
            min_correspondences=args.min_correspondences,
        )

    # ---- DINOv3 backbone ----
    # original_lora_state is kept when freeze_backbone so we can re-save it in checkpoints
    # (the eval pipeline needs it to reconstruct the merged backbone).
    original_lora_state: Optional[Dict] = None
    dinov3 = None
    transform = None

    if fast_path:
        # No DINOv3 model needed; load lora_state only for output checkpoint
        if args.lora_checkpoint:
            _src = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
            original_lora_state = _src["lora_state_dict"]
            print(f"\n[fast_sparse] Using precomputed LoRA features from {args.sparse_dir}")
            print(f"  LoRA state loaded from {args.lora_checkpoint} (for checkpoint saving only)")
        else:
            original_lora_state = {}  # no LoRA — raw DINOv3 features (sanity check mode)
            print(f"\n[fast_sparse] Using precomputed features from {args.sparse_dir}")
            print("  No --lora_checkpoint: lora_state will be empty in output checkpoint.")
        lora_params: List[nn.Parameter] = []
    elif args.freeze_backbone:
        if not args.lora_checkpoint:
            raise ValueError("--lora_checkpoint required with --freeze_backbone")
        print(f"\nLoading frozen LoRA-merged DINOv3 from {args.lora_checkpoint}...")
        dinov3, transform = load_dinov3_frozen_lora(
            feat_level=DINOV3_FEAT_LEVEL,
            lora_checkpoint_path=args.lora_checkpoint,
            device=device,
        )
        _src = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
        original_lora_state = _src["lora_state_dict"]
        lora_params: List[nn.Parameter] = []
    else:
        print("\nLoading DINOv3 with LoRA...")
        dinov3, transform = load_dinov3_with_lora(
            feat_level=DINOV3_FEAT_LEVEL,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            device=device,
        )
        if args.lora_checkpoint:
            init_trainable_lora_from_checkpoint(dinov3, args.lora_checkpoint)
        # Keep base model in eval mode (freezes dropout/BN); LoRA params still train
        dinov3.eval()
        lora_params = get_lora_params(dinov3)

    # ---- Projection head ----
    if args.projection_checkpoint:
        print(f"\nLoading projection head from {args.projection_checkpoint}...")
        ckpt = torch.load(args.projection_checkpoint, map_location="cpu", weights_only=False)
        ph_cfg = ckpt.get("config", {})
        ph_hidden = ph_cfg.get("hidden_dims") or [int(ph_cfg.get("hidden_dim", 1024))]
        ph_input = int(ph_cfg.get("input_dim", FUSED_DIM))
        ph_output = int(ph_cfg.get("output_dim", 256))
        proj_head = ProjectionHead(input_dim=ph_input, hidden_dims=ph_hidden, output_dim=ph_output)
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "proj_state_dict"
        proj_head.load_state_dict(ckpt[state_key])
        print(f"  Loaded: {ph_input} -> {ph_hidden} -> {ph_output}")
    else:
        print(f"\nFresh projection head: {proj_input_dim} -> {hidden_dims} -> 256")
        proj_head = ProjectionHead(input_dim=proj_input_dim, hidden_dims=hidden_dims, output_dim=256)

    proj_head.to(device).train()
    print(f"  Proj params: {sum(p.numel() for p in proj_head.parameters()):,}")

    # ---- Optimizer ----
    n_lora = sum(p.numel() for p in lora_params)
    n_proj = sum(p.numel() for p in proj_head.parameters())

    param_groups: List[Dict] = [{"params": list(proj_head.parameters()), "lr": args.lr_proj}]
    if lora_params:
        param_groups.insert(0, {"params": lora_params, "lr": args.lr_lora})

    if args.freeze_backbone:
        print(f"\nOptimizer: {n_proj:,} proj params (lr={args.lr_proj}) [backbone frozen]")
        eta_min = args.lr_proj * 0.01
    else:
        print(f"\nOptimizer: {n_lora:,} LoRA params (lr={args.lr_lora}), "
              f"{n_proj:,} proj params (lr={args.lr_proj})")
        eta_min = min(args.lr_lora, args.lr_proj) * 0.01

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"  AMP (fp16): {'enabled' if amp_enabled else 'disabled'}")

    pairs_per_epoch = min(args.pairs_per_epoch, len(dataset.valid_pairs))
    total_steps = args.epochs * pairs_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=eta_min,
    )

    # ---- Output dir + config ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    config["n_valid_pairs"] = len(dataset.valid_pairs)
    config["pairs_per_epoch_effective"] = pairs_per_epoch
    config["input_dim"] = proj_input_dim
    config["output_dim"] = 256
    config["hidden_dims"] = hidden_dims
    config["feat_level"] = DINOV3_FEAT_LEVEL
    if not args.dinov3_only:
        config["feat_split_verified"] = {
            "dift_dims": f"[:, 0:{DIFT_DIM}]",
            "dinov3_dims": f"[:, {DIFT_DIM}:{FUSED_DIM}]",
            "source": "fusion_matches.py + extract_sparse_training_data.py",
        }
    with (output_dir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    # ---- Training loop ----
    preprocess_cache = PreprocessCache(max_size=200) if not fast_path else None
    rng = np.random.default_rng(args.seed)
    train_log: List[Dict] = []
    best_loss = float("inf")

    mode_str = "DINOv3-only" if args.dinov3_only else "DIFT+DINOv3"
    bb_str = "frozen (LoRA merged)" if args.freeze_backbone else "LoRA-trainable"
    print(f"\nStarting training: {args.epochs} epochs × {pairs_per_epoch} pairs/epoch")
    print(f"  mode={mode_str}  backbone={bb_str}")
    print(f"  temperature={args.temperature}  num_correspondences={args.num_correspondences}")
    if not args.dinov3_only:
        print(f"  fusion_alpha={args.fusion_alpha}")

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        pair_indices = rng.choice(
            len(dataset.valid_pairs), size=pairs_per_epoch, replace=False
        ).tolist()

        total_loss = 0.0
        total_acc = 0.0
        count = 0
        skipped = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for step, pair_idx in enumerate(pair_indices):
            # ------------------------------------------------------------------
            # Fast path: precomputed LoRA features, no DINOv3 forward pass
            # ------------------------------------------------------------------
            if fast_path:
                try:
                    batch = dataset.get_pair_batch(pair_idx, rng, fallback_pool=pair_indices)
                except RuntimeError:
                    skipped += 1
                    continue

                feat1 = batch["features_1"].to(device).float()   # (N1, 1664)
                feat2 = batch["features_2"].to(device).float()
                idx1 = batch["pos_indices_1"]                     # (N_pos,) indices into pos rows
                idx2 = batch["pos_indices_2"]

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    if args.dinov3_only:
                        # dims [DIFT_DIM:] = (1-alpha)*l2_dino → re-norm → pure l2_dino
                        # Pass ALL features (positives + hard negatives) through projection
                        proj_in_1 = F.normalize(feat1[:, DIFT_DIM:], dim=1)   # (N_all, 1024)
                        proj_in_2 = F.normalize(feat2[:, DIFT_DIM:], dim=1)
                    else:
                        # Use bundle features as-is: [alpha*l2_dift, (1-alpha)*l2_dino]
                        # Matches lora_matches.py eval format exactly (alpha=0.5, norm≈0.5 per half).
                        # DO NOT re-normalize — that changes scale to unit-norm which mismatches eval.
                        proj_in_1 = feat1.float()  # (N_all, 1664)
                        proj_in_2 = feat2.float()

                    proj_1 = proj_head(proj_in_1)
                    proj_2 = proj_head(proj_in_2)
                    N = idx1.shape[0]  # number of positive correspondences (for logging)
                    # idx1/idx2 index into the positive rows of feat1/feat2 respectively.
                    # These are also valid indices into proj_1/proj_2 (same row order).
                    loss, acc = pair_loss_and_accuracy(
                        proj_1=proj_1, proj_2=proj_2,
                        pos_indices_1=idx1.to(device), pos_indices_2=idx2.to(device),
                        temperature=args.temperature,
                    )

            # ------------------------------------------------------------------
            # Online path: DINOv3 forward pass (trainable or frozen backbone)
            # ------------------------------------------------------------------
            else:
                batch = dataset.get_pair_batch(pair_idx, rng, fallback_pool=pair_indices)
                if batch is None:
                    skipped += 1
                    continue

                try:
                    img_pre_1, info_1 = preprocess_cache.get(batch["img_path_1"])
                    img_pre_2, info_2 = preprocess_cache.get(batch["img_path_2"])
                except Exception as exc:
                    skipped += 1
                    if step % args.log_interval == 0:
                        print(f"  [skip img] {exc}")
                    continue

                H1, W1 = info_1.final_size
                H2, W2 = info_2.final_size

                x1 = image_to_tensor(img_pre_1, transform, device)
                x2 = image_to_tensor(img_pre_2, transform, device)

                if not args.dinov3_only:
                    dift_1 = batch["dift_1"].to(device)  # (N, 640)
                    dift_2 = batch["dift_2"].to(device)

                pts1_pre = orig_pts_to_preproc(batch["pts_1"], info_1)
                pts2_pre = orig_pts_to_preproc(batch["pts_2"], info_2)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    if args.freeze_backbone:
                        with torch.no_grad():
                            ft1 = dinov3(x1)[0].squeeze(0)
                            ft2 = dinov3(x2)[0].squeeze(0)
                            dinov3_1 = F.normalize(sample_feature_map(ft1, pts1_pre, (H1, W1)), dim=1)
                            dinov3_2 = F.normalize(sample_feature_map(ft2, pts2_pre, (H2, W2)), dim=1)
                    else:
                        ft1 = dinov3(x1)[0].squeeze(0)
                        ft2 = dinov3(x2)[0].squeeze(0)
                        dinov3_1 = F.normalize(sample_feature_map(ft1, pts1_pre, (H1, W1)), dim=1)
                        dinov3_2 = F.normalize(sample_feature_map(ft2, pts2_pre, (H2, W2)), dim=1)

                    if args.dinov3_only:
                        proj_in_1 = dinov3_1  # (N, 1024)
                        proj_in_2 = dinov3_2
                    else:
                        dift_1n = F.normalize(dift_1.to(dinov3_1.dtype), dim=1)
                        dift_2n = F.normalize(dift_2.to(dinov3_2.dtype), dim=1)
                        proj_in_1 = torch.cat(
                            [args.fusion_alpha * dift_1n, (1.0 - args.fusion_alpha) * dinov3_1],
                            dim=1,
                        )
                        proj_in_2 = torch.cat(
                            [args.fusion_alpha * dift_2n, (1.0 - args.fusion_alpha) * dinov3_2],
                            dim=1,
                        )

                    proj_1 = proj_head(proj_in_1)
                    proj_2 = proj_head(proj_in_2)
                    N = proj_1.shape[0]
                    pos_idx = torch.arange(N, device=device)
                    loss, acc = pair_loss_and_accuracy(
                        proj_1=proj_1, proj_2=proj_2,
                        pos_indices_1=pos_idx, pos_indices_2=pos_idx,
                        temperature=args.temperature,
                    )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    lora_params + list(proj_head.parameters()), args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())
            total_acc += float(acc)
            count += 1

            if step % args.log_interval == 0:
                lrs = scheduler.get_last_lr()
                lr_str = "  ".join(
                    f"lr_{'lora' if i == 0 and not args.freeze_backbone else 'proj'}={lr:.2e}"
                    for i, lr in enumerate(lrs)
                )
                print(
                    f"  [ep{epoch} {step}/{pairs_per_epoch}] "
                    f"loss={loss.item():.4f} acc={acc:.4f} N={N}  {lr_str}"
                )

        elapsed = time.perf_counter() - epoch_start
        avg_loss = total_loss / max(1, count)
        avg_acc = total_acc / max(1, count)
        pairs_per_sec = count / max(elapsed, 1e-6)
        lrs = scheduler.get_last_lr()

        entry: Dict = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "time_sec": elapsed,
            "pairs_per_second": pairs_per_sec,
            "skipped": skipped,
            "ram_maxrss_gb": maxrss_gb(),
        }
        for i, lr in enumerate(lrs):
            entry[f"lr_{i}"] = float(lr)
        if device.type == "cuda":
            entry["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            entry["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

        train_log.append(entry)

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}: "
            f"loss={avg_loss:.4f} acc={avg_acc:.4f} "
            f"time={elapsed:.0f}s ({pairs_per_sec:.3f} pairs/s) skipped={skipped}"
        )
        if device.type == "cuda":
            print(
                f"  GPU: alloc={entry['gpu_peak_allocated_gb']:.2f} GB  "
                f"reserved={entry['gpu_peak_reserved_gb']:.2f} GB"
            )
        print(f"  RAM: {entry['ram_maxrss_gb']:.2f} GB")

        # --- Save checkpoint ---
        if args.freeze_backbone:
            # Re-save original Stage-1 LoRA state so the eval pipeline can reconstruct the backbone
            lora_state = original_lora_state
        else:
            lora_state = {n: p.detach().cpu() for n, p in dinov3.named_parameters() if "lora_" in n}

        ckpt_data = {
            "epoch": epoch,
            "lora_state_dict": lora_state,
            "proj_state_dict": proj_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_log": train_log,
            "config": config,
            "best_loss": best_loss,
        }
        torch.save(ckpt_data, output_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data["best_loss"] = best_loss
            torch.save(ckpt_data, output_dir / "best.pt")
            print(f"  -> Saved best (loss={best_loss:.4f})")

        with (output_dir / "train_log.json").open("w") as fh:
            json.dump(train_log, fh, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SCENES = [
    "0080", "0042", "0380", "0000", "0366",
    "0001", "0005", "0237", "0011", "0148",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of DINOv3 + projection head.\n\n"
        "Modes:\n"
        "  default              joint LoRA + DIFT+DINOv3 projection head\n"
        "  --dinov3_only        LoRA on DINOv3-only (throwaway head, no DIFT)\n"
        "  --freeze_backbone    frozen LoRA-merged backbone, train fresh projection head\n"
        "  --freeze_backbone --dinov3_only  frozen backbone, DINOv3-only head\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--megadepth_root", default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM")
    p.add_argument("--sparse_dir", default="data/sparse_train")
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # Mode flags
    p.add_argument("--dinov3_only", action="store_true", default=False,
                   help="Use DINOv3 features only (no DIFT). Projection head dim: 1024->512->256.")
    p.add_argument("--freeze_backbone", action="store_true", default=False,
                   help="Merge LoRA from --lora_checkpoint and freeze backbone. "
                        "Only trains a fresh projection head.")
    p.add_argument("--fast_sparse", action="store_true", default=False,
                   help="Fast path: load pre-extracted LoRA features directly from sparse bundles "
                        "(requires --freeze_backbone + LoRA-updated --sparse_dir). "
                        "Skips DINOv3 forward pass entirely (~75 pairs/sec).")
    p.add_argument("--lora_checkpoint", default=None,
                   help="Stage-1 LoRA checkpoint to initialise trainable adapters from, "
                        "or to initialise a frozen merged backbone when --freeze_backbone "
                        "(required when --freeze_backbone).")

    # Projection head
    p.add_argument("--projection_checkpoint", default=None,
                   help="Optional: pre-trained projection head to initialise from. "
                        "If omitted, a fresh head is created.")
    p.add_argument("--hidden_dims", nargs="+", type=int, default=None,
                   help="Projection head hidden layer sizes. "
                        "Default: [512] when --dinov3_only, [1024] otherwise.")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pairs_per_epoch", type=int, default=10000)
    p.add_argument("--num_correspondences", type=int, default=256)
    p.add_argument("--min_correspondences", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--fusion_alpha", type=float, default=0.5,
                   help="Fusion weighting for online DIFT+DINOv3 training: "
                        "[alpha * DIFT, (1-alpha) * DINOv3]. Ignored for --dinov3_only.")
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--lr_proj", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use AMP (fp16) for the forward pass (default: True)")

    # Misc
    p.add_argument("--output_dir", default="experiments/phase2_lora_r4_v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_interval", type=int, default=50)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
