"""Train a projection head on MegaDepth fused features with InfoNCE."""

from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import sys
import time
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from megadepth_pairs import MegaDepthPairSampler


DEFAULT_SCENES = [
    "0080",
    "0042",
    "0380",
    "0000",
    "0366",
    "0001",
    "0005",
    "0237",
    "0011",
    "0148",
]


def normalize_hidden_dims(
    hidden_dims: Optional[Sequence[int]],
    fallback_hidden_dim: int = 512,
) -> List[int]:
    if hidden_dims is None:
        dims = [int(fallback_hidden_dim)]
    else:
        dims = [int(dim) for dim in hidden_dims]

    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All hidden dims must be positive, got {dims}")
    return dims


class ProjectionHead(nn.Module):
    """MLP projection head with configurable hidden stack and L2-normalized output."""

    def __init__(
        self,
        input_dim: int = 1664,
        hidden_dims: Optional[Sequence[int]] = None,
        output_dim: int = 256,
    ):
        super().__init__()
        dims = [input_dim, *normalize_hidden_dims(hidden_dims), output_dim]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


def npz_has_feature_keys(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        return "dinov3.npy" in names and "dift.npy" in names
    except Exception:
        return False


def vector_l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, eps, None)


def maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def to_cpu_byte_tensor(value: object) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def pair_loss_and_accuracy(
    proj_1: torch.Tensor,
    proj_2: torch.Tensor,
    pos_indices_1: torch.Tensor,
    pos_indices_2: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    """InfoNCE where positives are indexed into full candidate banks."""
    logits_1_to_2 = torch.mm(proj_1.index_select(0, pos_indices_1), proj_2.t()) / temperature
    logits_2_to_1 = torch.mm(proj_2.index_select(0, pos_indices_2), proj_1.t()) / temperature

    loss_1_to_2 = F.cross_entropy(logits_1_to_2, pos_indices_2)
    loss_2_to_1 = F.cross_entropy(logits_2_to_1, pos_indices_1)
    loss = 0.5 * (loss_1_to_2 + loss_2_to_1)

    with torch.no_grad():
        pred_1_to_2 = logits_1_to_2.argmax(dim=1)
        pred_2_to_1 = logits_2_to_1.argmax(dim=1)
        acc_1_to_2 = (pred_1_to_2 == pos_indices_2).float().mean().item()
        acc_2_to_1 = (pred_2_to_1 == pos_indices_1).float().mean().item()
        acc = 0.5 * (acc_1_to_2 + acc_2_to_1)

    return loss, acc


class DenseFeatureDataset:
    """Fallback dataset that samples fused descriptors from dense per-image .npz files."""

    def __init__(
        self,
        pair_sampler: MegaDepthPairSampler,
        feature_dir: str,
        num_correspondences: int = 512,
        min_correspondences: int = 50,
        alpha: float = 0.5,
        feature_cache_size: int = 1000,
    ):
        self.pair_sampler = pair_sampler
        self.feature_dir = Path(feature_dir)
        self.num_correspondences = num_correspondences
        self.min_correspondences = min_correspondences
        self.alpha = float(alpha)
        self.feature_cache_size = max(0, int(feature_cache_size))

        self._feature_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._image_point_cache: "OrderedDict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

        print("Scanning feature directory for complete DINOv3+DIFT files...")
        self.available_features = self._build_available_feature_index()
        print("Filtering MegaDepth pairs to those with complete feature caches...")
        self.valid_pairs = self._build_valid_pairs()

        if not self.valid_pairs:
            raise RuntimeError("No valid dense training pairs found with complete features.")

        print(
            f"Valid dense pairs with complete features: {len(self.valid_pairs)} "
            f"(from {len(self.pair_sampler.all_pairs)} total)"
        )
        print(f"Dense feature cache size: {self.feature_cache_size} images")

    def _build_available_feature_index(self) -> Dict[str, set[str]]:
        available: Dict[str, set[str]] = {}
        for scene_name in self.pair_sampler.scenes:
            scene_dir = self.feature_dir / scene_name
            if not scene_dir.is_dir():
                available[scene_name] = set()
                continue

            ready = set()
            for npz_path in scene_dir.glob("*.npz"):
                if npz_has_feature_keys(npz_path):
                    ready.add(npz_path.stem)
            available[scene_name] = ready
        return available

    def _build_valid_pairs(self) -> List[Tuple[str, int, int, int]]:
        valid_pairs: List[Tuple[str, int, int, int]] = []
        for scene_name, id1, id2, covis in self.pair_sampler.all_pairs:
            scene = self.pair_sampler.scenes[scene_name]
            stem1 = Path(scene.images[id1].name).stem
            stem2 = Path(scene.images[id2].name).stem
            available = self.available_features.get(scene_name, set())
            if stem1 in available and stem2 in available:
                valid_pairs.append((scene_name, id1, id2, covis))
        return valid_pairs

    def reset_cache_stats(self) -> None:
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total else 0.0
        return {
            "cache_hits": float(self._cache_hits),
            "cache_misses": float(self._cache_misses),
            "cache_hit_rate": hit_rate,
            "cache_entries": float(len(self._feature_cache)),
        }

    def sample_pair_indices(
        self,
        num_pairs: int,
        rng: np.random.Generator,
        exclude: Optional[Sequence[int]] = None,
    ) -> List[int]:
        total = len(self.valid_pairs)
        if total == 0:
            return []

        if exclude is None or len(exclude) == 0:
            return rng.choice(total, size=min(num_pairs, total), replace=False).tolist()

        exclude_set = set(int(idx) for idx in exclude)
        available = total - len(exclude_set)
        if available <= 0:
            return []

        target = min(num_pairs, available)
        chosen: List[int] = []
        seen = set()
        while len(chosen) < target:
            batch_size = max(4096, (target - len(chosen)) * 3)
            for idx in rng.integers(0, total, size=batch_size):
                idx_int = int(idx)
                if idx_int in exclude_set or idx_int in seen:
                    continue
                chosen.append(idx_int)
                seen.add(idx_int)
                if len(chosen) >= target:
                    break
        return chosen

    def order_pair_indices_for_locality(self, pair_indices: Sequence[int]) -> List[int]:
        return sorted(
            pair_indices,
            key=lambda idx: (
                self.valid_pairs[idx][0],
                min(self.valid_pairs[idx][1], self.valid_pairs[idx][2]),
                max(self.valid_pairs[idx][1], self.valid_pairs[idx][2]),
            ),
        )

    def _load_image_points(self, scene_name: str, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = (scene_name, image_id)
        cached = self._image_point_cache.get(cache_key)
        if cached is not None:
            self._image_point_cache.move_to_end(cache_key)
            return cached

        scene = self.pair_sampler.scenes[scene_name]
        image = scene.images[image_id]
        valid = image.point2D_ids >= 0
        point_ids = image.point2D_ids[valid].astype(np.int64, copy=False)
        xy = image.point2D_xy[valid].astype(np.float32, copy=False)
        if point_ids.size > 1:
            order = np.argsort(point_ids, kind="mergesort")
            point_ids = point_ids[order]
            xy = xy[order]
            unique_ids, unique_idx = np.unique(point_ids, return_index=True)
            point_ids = unique_ids
            xy = xy[unique_idx]

        self._image_point_cache[cache_key] = (point_ids, xy)
        self._image_point_cache.move_to_end(cache_key)
        while len(self._image_point_cache) > max(1, self.feature_cache_size * 4):
            self._image_point_cache.popitem(last=False)
        return point_ids, xy

    def _get_correspondences(self, scene_name: str, id1: int, id2: int) -> Tuple[np.ndarray, np.ndarray]:
        point_ids_1, xy_1 = self._load_image_points(scene_name, id1)
        point_ids_2, xy_2 = self._load_image_points(scene_name, id2)
        if point_ids_1.size == 0 or point_ids_2.size == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        _, idx_1, idx_2 = np.intersect1d(
            point_ids_1,
            point_ids_2,
            assume_unique=True,
            return_indices=True,
        )
        return xy_1[idx_1], xy_2[idx_2]

    def _load_features(self, npz_path: Path) -> Dict[str, np.ndarray]:
        cache_key = str(npz_path)
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            self._feature_cache.move_to_end(cache_key)
            return cached

        self._cache_misses += 1
        with np.load(npz_path, allow_pickle=False) as data:
            loaded = {
                "dinov3": data["dinov3"].astype(np.float32, copy=False),
                "dift": data["dift"].astype(np.float32, copy=False),
                "preprocess_scale": np.float32(data["preprocess_scale"]),
                "preprocessed_size": data["preprocessed_size"].astype(np.float32, copy=False),
                "pad_left": np.float32(data["pad_left"]) if "pad_left" in data else np.float32(0.0),
                "pad_top": np.float32(data["pad_top"]) if "pad_top" in data else np.float32(0.0),
                "dift_pre_to_input_scale_xy": data["dift_pre_to_input_scale_xy"].astype(np.float32, copy=False),
                "dift_input_size": data["dift_input_size"].astype(np.float32, copy=False),
            }

        if self.feature_cache_size > 0:
            self._feature_cache[cache_key] = loaded
            self._feature_cache.move_to_end(cache_key)
            while len(self._feature_cache) > self.feature_cache_size:
                self._feature_cache.popitem(last=False)

        return loaded

    def _sample_fused_features(self, feat_data: Dict[str, np.ndarray], pts: np.ndarray) -> np.ndarray:
        dinov3_feat = feat_data["dinov3"]
        dift_feat = feat_data["dift"]

        preprocess_scale = float(feat_data["preprocess_scale"])
        preprocessed_size = feat_data["preprocessed_size"]
        pad_left = float(feat_data["pad_left"])
        pad_top = float(feat_data["pad_top"])
        dift_scale_xy = feat_data["dift_pre_to_input_scale_xy"]
        dift_input_size = feat_data["dift_input_size"]

        pts_pre = pts.astype(np.float32, copy=True)
        pts_pre *= preprocess_scale
        pts_pre[:, 0] += pad_left
        pts_pre[:, 1] += pad_top

        h_dino, w_dino = dinov3_feat.shape[:2]
        h_pre = float(preprocessed_size[0])
        w_pre = float(preprocessed_size[1])
        x_dino = pts_pre[:, 0] * (w_dino / w_pre)
        y_dino = pts_pre[:, 1] * (h_dino / h_pre)
        ix_dino = np.clip(np.rint(x_dino).astype(np.int64), 0, w_dino - 1)
        iy_dino = np.clip(np.rint(y_dino).astype(np.int64), 0, h_dino - 1)
        dinov3_vecs = dinov3_feat[iy_dino, ix_dino]

        h_dift, w_dift = dift_feat.shape[:2]
        pts_dift_x = pts_pre[:, 0] * dift_scale_xy[0]
        pts_dift_y = pts_pre[:, 1] * dift_scale_xy[1]
        stride_x = float(dift_input_size[1]) / w_dift
        stride_y = float(dift_input_size[0]) / h_dift
        ix_dift = np.clip(np.rint(pts_dift_x / stride_x).astype(np.int64), 0, w_dift - 1)
        iy_dift = np.clip(np.rint(pts_dift_y / stride_y).astype(np.int64), 0, h_dift - 1)
        dift_vecs = dift_feat[iy_dift, ix_dift]

        dinov3_vecs = vector_l2_normalize(dinov3_vecs)
        dift_vecs = vector_l2_normalize(dift_vecs)
        fused = np.concatenate(
            [self.alpha * dift_vecs, (1.0 - self.alpha) * dinov3_vecs],
            axis=1,
        )
        return fused.astype(np.float32, copy=False)

    def get_pair_batch(
        self,
        pair_idx: int,
        rng: np.random.Generator,
        fallback_pool: Optional[Sequence[int]] = None,
        max_attempts: int = 20,
    ) -> Dict[str, torch.Tensor]:
        pool = list(fallback_pool) if fallback_pool is not None else None

        for attempt in range(max_attempts):
            actual_pair_idx = pair_idx
            if attempt > 0:
                if pool:
                    actual_pair_idx = int(pool[rng.integers(0, len(pool))])
                else:
                    actual_pair_idx = int(rng.integers(0, len(self.valid_pairs)))

            scene_name, id1, id2, _ = self.valid_pairs[actual_pair_idx]
            pts1, pts2 = self._get_correspondences(scene_name, id1, id2)
            if len(pts1) < self.min_correspondences:
                continue

            if len(pts1) > self.num_correspondences:
                indices = rng.choice(len(pts1), size=self.num_correspondences, replace=False)
                pts1 = pts1[indices]
                pts2 = pts2[indices]

            scene = self.pair_sampler.scenes[scene_name]
            stem1 = Path(scene.images[id1].name).stem
            stem2 = Path(scene.images[id2].name).stem
            feat_path_1 = self.feature_dir / scene_name / f"{stem1}.npz"
            feat_path_2 = self.feature_dir / scene_name / f"{stem2}.npz"

            feat_data_1 = self._load_features(feat_path_1)
            feat_data_2 = self._load_features(feat_path_2)
            features_1 = self._sample_fused_features(feat_data_1, pts1)
            features_2 = self._sample_fused_features(feat_data_2, pts2)

            pos_indices = torch.arange(features_1.shape[0], dtype=torch.long)
            return {
                "features_1": torch.from_numpy(features_1),
                "features_2": torch.from_numpy(features_2),
                "pos_indices_1": pos_indices.clone(),
                "pos_indices_2": pos_indices.clone(),
            }

        raise RuntimeError("Failed to sample a dense pair with enough correspondences.")


class SparseFeatureDataset:
    """Loads compact per-scene sparse feature bundles entirely into RAM."""

    def __init__(
        self,
        sparse_dir: str,
        scenes: Sequence[str],
        num_correspondences: int = 512,
        min_correspondences: int = 50,
        scene_cache_size: int = 1,
    ):
        self.sparse_dir = Path(sparse_dir)
        self.num_correspondences = num_correspondences
        self.min_correspondences = min_correspondences
        self.scene_cache_size = max(1, int(scene_cache_size))
        self.scene_meta: Dict[str, Dict[str, object]] = {}
        self.scene_data: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self.valid_pairs: List[Tuple[str, int, int, int]] = []
        self._cache_hits = 0
        self._cache_misses = 0
        total_bytes = 0

        for idx, scene_name in enumerate(scenes, start=1):
            path = self.sparse_dir / f"{scene_name}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing sparse training bundle: {path}")
            size_gb = path.stat().st_size / (1024 ** 3)
            total_bytes += path.stat().st_size
            print(
                f"[sparse meta {idx}/{len(scenes)}] {scene_name}: "
                f"reading pair metadata from {path} ({size_gb:.2f} GB bundle)"
            )

            raw = torch.load(path, map_location="cpu")
            pair_indices = raw["pair_indices"]
            pair_covis = raw["pair_covis"]

            scene = {
                "path": path,
                "pair_indices": pair_indices.numpy() if isinstance(pair_indices, torch.Tensor) else np.asarray(pair_indices),
                "pair_covis": pair_covis.numpy() if isinstance(pair_covis, torch.Tensor) else np.asarray(pair_covis),
                "image_names": list(raw.get("image_names", [])),
            }
            self.scene_meta[scene_name] = scene

            for (img_idx_1, img_idx_2), covis in zip(scene["pair_indices"], scene["pair_covis"]):
                self.valid_pairs.append((scene_name, int(img_idx_1), int(img_idx_2), int(covis)))
            print(
                f"[sparse meta {idx}/{len(scenes)}] {scene_name}: "
                f"pairs={scene['pair_indices'].shape[0]:,}"
            )
            del raw
            gc.collect()

        if not self.valid_pairs:
            raise RuntimeError("No valid sparse training pairs found.")

        print(
            f"Loaded sparse training metadata for {len(self.scene_meta)} scenes, "
            f"{len(self.valid_pairs)} pairs "
            f"({total_bytes / (1024 ** 3):.2f} GB across bundles, cache_size={self.scene_cache_size})"
        )

        if self.scene_cache_size >= len(self.scene_meta):
            print(
                "Preloading all sparse scene bundles into RAM "
                f"({len(self.scene_meta)} scenes, cache_size={self.scene_cache_size})"
            )
            for scene_name in scenes:
                self._load_scene(scene_name)
            print("Finished preloading sparse scene bundles")

    def reset_cache_stats(self) -> None:
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total else 1.0
        return {
            "cache_hits": float(self._cache_hits),
            "cache_misses": float(self._cache_misses),
            "cache_hit_rate": hit_rate,
            "cache_entries": float(len(self.scene_data)),
        }

    def _load_scene(self, scene_name: str) -> Dict[str, np.ndarray]:
        cached = self.scene_data.get(scene_name)
        if cached is not None:
            self._cache_hits += 1
            self.scene_data.move_to_end(scene_name)
            return cached

        self._cache_misses += 1
        scene_path = self.scene_meta[scene_name]["path"]
        size_gb = scene_path.stat().st_size / (1024 ** 3)
        print(f"[sparse cache miss] {scene_name}: loading full bundle ({size_gb:.2f} GB)")
        raw = torch.load(scene_path, map_location="cpu")
        scene = {
            "features": raw["features"].numpy()
            if isinstance(raw["features"], torch.Tensor)
            else np.asarray(raw["features"]),
            "point3D_ids": raw["point3D_ids"].numpy()
            if isinstance(raw["point3D_ids"], torch.Tensor)
            else np.asarray(raw["point3D_ids"]),
            "image_offsets": raw["image_offsets"].numpy()
            if isinstance(raw["image_offsets"], torch.Tensor)
            else np.asarray(raw["image_offsets"]),
            "image_positive_counts": raw["image_positive_counts"].numpy()
            if isinstance(raw["image_positive_counts"], torch.Tensor)
            else np.asarray(raw["image_positive_counts"]),
        }
        del raw
        self.scene_data[scene_name] = scene
        self.scene_data.move_to_end(scene_name)

        while len(self.scene_data) > self.scene_cache_size:
            evicted_name, evicted = self.scene_data.popitem(last=False)
            print(f"[sparse cache evict] {evicted_name}")
            del evicted
            gc.collect()

        return scene

    def sample_pair_indices(
        self,
        num_pairs: int,
        rng: np.random.Generator,
        exclude: Optional[Sequence[int]] = None,
    ) -> List[int]:
        total = len(self.valid_pairs)
        if total == 0:
            return []

        if exclude is None or len(exclude) == 0:
            return rng.choice(total, size=min(num_pairs, total), replace=False).tolist()

        exclude_set = set(int(idx) for idx in exclude)
        available = total - len(exclude_set)
        if available <= 0:
            return []

        target = min(num_pairs, available)
        chosen: List[int] = []
        seen = set()
        while len(chosen) < target:
            batch_size = max(4096, (target - len(chosen)) * 3)
            for idx in rng.integers(0, total, size=batch_size):
                idx_int = int(idx)
                if idx_int in exclude_set or idx_int in seen:
                    continue
                chosen.append(idx_int)
                seen.add(idx_int)
                if len(chosen) >= target:
                    break
        return chosen

    def order_pair_indices_for_locality(self, pair_indices: Sequence[int]) -> List[int]:
        return sorted(
            pair_indices,
            key=lambda idx: (
                self.valid_pairs[idx][0],
                min(self.valid_pairs[idx][1], self.valid_pairs[idx][2]),
                max(self.valid_pairs[idx][1], self.valid_pairs[idx][2]),
            ),
        )

    def get_pair_batch(
        self,
        pair_idx: int,
        rng: np.random.Generator,
        fallback_pool: Optional[Sequence[int]] = None,
        max_attempts: int = 20,
    ) -> Dict[str, torch.Tensor]:
        pool = list(fallback_pool) if fallback_pool is not None else None

        for attempt in range(max_attempts):
            actual_pair_idx = pair_idx
            if attempt > 0:
                if pool:
                    actual_pair_idx = int(pool[rng.integers(0, len(pool))])
                else:
                    actual_pair_idx = int(rng.integers(0, len(self.valid_pairs)))

            scene_name, img_idx_1, img_idx_2, _ = self.valid_pairs[actual_pair_idx]
            scene = self._load_scene(scene_name)

            offsets = scene["image_offsets"]
            positive_counts = scene["image_positive_counts"]
            all_features = scene["features"]
            all_point_ids = scene["point3D_ids"]

            start_1 = int(offsets[img_idx_1])
            end_1 = int(offsets[img_idx_1 + 1])
            start_2 = int(offsets[img_idx_2])
            end_2 = int(offsets[img_idx_2 + 1])

            features_1 = all_features[start_1:end_1]
            features_2 = all_features[start_2:end_2]
            positive_count_1 = int(positive_counts[img_idx_1])
            positive_count_2 = int(positive_counts[img_idx_2])
            point_ids_1 = all_point_ids[start_1 : start_1 + positive_count_1]
            point_ids_2 = all_point_ids[start_2 : start_2 + positive_count_2]

            if point_ids_1.size == 0 or point_ids_2.size == 0:
                continue

            _, idx_1, idx_2 = np.intersect1d(
                point_ids_1,
                point_ids_2,
                assume_unique=True,
                return_indices=True,
            )
            if idx_1.size < self.min_correspondences:
                continue

            if idx_1.size > self.num_correspondences:
                selection = rng.choice(idx_1.size, size=self.num_correspondences, replace=False)
                idx_1 = idx_1[selection]
                idx_2 = idx_2[selection]

            return {
                "features_1": torch.from_numpy(features_1),
                "features_2": torch.from_numpy(features_2),
                "pos_indices_1": torch.from_numpy(idx_1.astype(np.int64, copy=False)),
                "pos_indices_2": torch.from_numpy(idx_2.astype(np.int64, copy=False)),
            }

        raise RuntimeError("Failed to sample a sparse pair with enough correspondences.")


def validate_one_pair(dataset: object, rng: np.random.Generator) -> Dict[str, object]:
    batch = dataset.get_pair_batch(0, rng)
    pos_1 = batch["pos_indices_1"]
    pos_2 = batch["pos_indices_2"]
    features_1 = batch["features_1"]
    features_2 = batch["features_2"]
    return {
        "features_1_shape": tuple(features_1.shape),
        "features_2_shape": tuple(features_2.shape),
        "dtype_1": str(features_1.dtype),
        "dtype_2": str(features_2.dtype),
        "num_positives": int(pos_1.numel()),
        "nan_1": bool(torch.isnan(features_1.float()).any().item()),
        "nan_2": bool(torch.isnan(features_2.float()).any().item()),
        "positive_norm_mean_1": float(features_1.index_select(0, pos_1).float().norm(dim=1).mean().item()),
        "positive_norm_mean_2": float(features_2.index_select(0, pos_2).float().norm(dim=1).mean().item()),
    }


def build_dataset(args: argparse.Namespace) -> object:
    if args.sparse_dir:
        return SparseFeatureDataset(
            sparse_dir=args.sparse_dir,
            scenes=args.scenes,
            num_correspondences=args.num_correspondences,
            min_correspondences=args.min_correspondences,
            scene_cache_size=args.sparse_scene_cache_size,
        )

    if not args.feature_dir:
        raise ValueError("Either --sparse_dir or --feature_dir must be provided.")

    sampler = MegaDepthPairSampler(
        root=args.megadepth_root,
        min_covis=args.min_covis,
        max_covis=args.max_covis,
        scenes=args.scenes,
    )
    return DenseFeatureDataset(
        pair_sampler=sampler,
        feature_dir=args.feature_dir,
        num_correspondences=args.num_correspondences,
        min_correspondences=args.min_correspondences,
        alpha=args.alpha,
        feature_cache_size=args.feature_cache_size,
    )


def run_pairs(
    model: ProjectionHead,
    dataset: object,
    pair_indices: Sequence[int],
    rng: np.random.Generator,
    device: torch.device,
    temperature: float,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    log_interval: int,
    phase_name: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    dataset.reset_cache_stats()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.set_grad_enabled(is_train):
        for batch_idx, pair_idx in enumerate(pair_indices):
            batch = dataset.get_pair_batch(pair_idx, rng, fallback_pool=pair_indices)
            features_1 = batch["features_1"].to(device=device, dtype=torch.float32, non_blocking=True)
            features_2 = batch["features_2"].to(device=device, dtype=torch.float32, non_blocking=True)
            pos_indices_1 = batch["pos_indices_1"].to(device=device, dtype=torch.long, non_blocking=True)
            pos_indices_2 = batch["pos_indices_2"].to(device=device, dtype=torch.long, non_blocking=True)

            proj_1 = model(features_1)
            proj_2 = model(features_2)
            loss, acc = pair_loss_and_accuracy(
                proj_1=proj_1,
                proj_2=proj_2,
                pos_indices_1=pos_indices_1,
                pos_indices_2=pos_indices_2,
                temperature=temperature,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item())
            total_acc += acc
            count += 1

            if batch_idx % log_interval == 0:
                lr_str = ""
                if scheduler is not None:
                    lr_str = f" lr={scheduler.get_last_lr()[0]:.2e}"
                print(
                    f"  [{phase_name} {batch_idx}/{len(pair_indices)}] "
                    f"loss={loss.item():.4f} acc={acc:.4f} "
                    f"positives={pos_indices_1.numel()} "
                    f"bank1={features_1.shape[0]} bank2={features_2.shape[0]}{lr_str}"
                )

    elapsed = time.perf_counter() - start
    stats = dataset.cache_stats()
    result: Dict[str, float] = {
        "loss": total_loss / max(1, count),
        "accuracy": total_acc / max(1, count),
        "time_sec": elapsed,
        "pairs_per_second": count / max(elapsed, 1e-6),
        **stats,
        "ram_maxrss_gb": maxrss_gb(),
    }

    if device.type == "cuda":
        result["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        result["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return result


def make_checkpoint(
    epoch: int,
    model: ProjectionHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_loss: float,
    train_log: List[Dict[str, float]],
    config: Dict[str, object],
    pair_rng: np.random.Generator,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "train_log": train_log,
        "config": config,
        "pair_rng_state": pair_rng.bit_generator.state,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device}, but CUDA is not available.")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if args.num_workers != 0:
        print("Warning: --num_workers is ignored; training uses manual pair sampling.")

    device = torch.device(args.device)
    pair_rng = np.random.default_rng(args.seed)

    dataset = build_dataset(args)
    sanity = validate_one_pair(dataset, pair_rng)
    print("Feature sampling sanity check:")
    print(json.dumps(sanity, indent=2))

    model = ProjectionHead(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
    ).to(device)

    num_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Training pairs available: {len(dataset.valid_pairs):,}")

    pairs_per_epoch = min(args.pairs_per_epoch, len(dataset.valid_pairs))
    remaining = max(0, len(dataset.valid_pairs) - pairs_per_epoch)
    val_pairs_per_epoch = min(args.val_pairs_per_epoch, remaining)
    print(f"Pairs per epoch: {pairs_per_epoch:,}")
    print(f"Validation pairs per epoch: {val_pairs_per_epoch:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * max(1, pairs_per_epoch)),
        eta_min=args.lr * 0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["num_valid_pairs"] = len(dataset.valid_pairs)
    config["pairs_per_epoch_effective"] = pairs_per_epoch
    config["val_pairs_per_epoch_effective"] = val_pairs_per_epoch
    config["timestamp"] = datetime.now().isoformat()
    config["fusion_matches_exact"] = {
        "component_normalization": "separate_l2",
        "component_order": ["dift", "dinov3"],
        "weighting": {"dift": args.alpha, "dinov3": 1.0 - args.alpha},
        "input_concatenation_l2": False,
        "projection_output_l2": True,
        "sampling": "nearest_neighbor",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    start_epoch = 0
    best_loss = float("inf")
    train_log: List[Dict[str, float]] = []

    if args.resume:
        # Resume checkpoints are our own experiment artifacts and include
        # optimizer/scheduler/random-state objects, so they require full pickle load.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", float("inf"))))
        train_log = list(checkpoint.get("train_log", []))
        start_epoch = int(checkpoint["epoch"]) + 1

        if checkpoint.get("pair_rng_state") is not None:
            pair_rng.bit_generator.state = checkpoint["pair_rng_state"]
        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("torch_random_state") is not None:
            torch.random.set_rng_state(to_cpu_byte_tensor(checkpoint["torch_random_state"]))
        if device.type == "cuda" and checkpoint.get("torch_cuda_random_state_all") is not None:
            cuda_states = [to_cpu_byte_tensor(state) for state in checkpoint["torch_cuda_random_state_all"]]
            current_cuda_devices = torch.cuda.device_count()
            if len(cuda_states) == current_cuda_devices:
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                print(
                    "Warning: skipping CUDA RNG restore because checkpoint has "
                    f"{len(cuda_states)} CUDA state(s) but current host has "
                    f"{current_cuda_devices} visible CUDA device(s)."
                )

        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_pair_indices = dataset.sample_pair_indices(pairs_per_epoch, pair_rng)
        train_pair_indices = dataset.order_pair_indices_for_locality(train_pair_indices)
        val_pair_indices = dataset.sample_pair_indices(
            val_pairs_per_epoch,
            pair_rng,
            exclude=train_pair_indices,
        )
        val_pair_indices = dataset.order_pair_indices_for_locality(val_pair_indices)

        train_metrics = run_pairs(
            model=model,
            dataset=dataset,
            pair_indices=train_pair_indices,
            rng=pair_rng,
            device=device,
            temperature=args.temperature,
            optimizer=optimizer,
            scheduler=scheduler,
            log_interval=args.log_interval,
            phase_name="train",
        )

        if val_pair_indices:
            val_metrics = run_pairs(
                model=model,
                dataset=dataset,
                pair_indices=val_pair_indices,
                rng=pair_rng,
                device=device,
                temperature=args.temperature,
                optimizer=None,
                scheduler=None,
                log_interval=max(1, args.log_interval),
                phase_name="val",
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "accuracy": float("nan"),
                "time_sec": 0.0,
                "pairs_per_second": 0.0,
                "cache_hits": 0.0,
                "cache_misses": 0.0,
                "cache_hit_rate": 0.0,
                "cache_entries": 0.0,
                "ram_maxrss_gb": maxrss_gb(),
            }

        log_entry: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_time_sec": train_metrics["time_sec"],
            "train_pairs_per_second": train_metrics["pairs_per_second"],
            "train_cache_hit_rate": train_metrics["cache_hit_rate"],
            "train_cache_hits": train_metrics["cache_hits"],
            "train_cache_misses": train_metrics["cache_misses"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_time_sec": val_metrics["time_sec"],
            "val_pairs_per_second": val_metrics["pairs_per_second"],
            "val_cache_hit_rate": val_metrics["cache_hit_rate"],
            "val_cache_hits": val_metrics["cache_hits"],
            "val_cache_misses": val_metrics["cache_misses"],
            "lr": float(scheduler.get_last_lr()[0]),
            "ram_maxrss_gb": max(train_metrics["ram_maxrss_gb"], val_metrics["ram_maxrss_gb"]),
        }

        if device.type == "cuda":
            log_entry["gpu_peak_allocated_gb"] = max(
                train_metrics.get("gpu_peak_allocated_gb", 0.0),
                val_metrics.get("gpu_peak_allocated_gb", 0.0),
            )
            log_entry["gpu_peak_reserved_gb"] = max(
                train_metrics.get("gpu_peak_reserved_gb", 0.0),
                val_metrics.get("gpu_peak_reserved_gb", 0.0),
            )

        train_log.append(log_entry)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        print(
            f"  train: time={train_metrics['time_sec']:.1f}s pairs/s={train_metrics['pairs_per_second']:.2f} "
            f"cache_hit_rate={train_metrics['cache_hit_rate']:.3f}"
        )
        print(
            f"  val:   time={val_metrics['time_sec']:.1f}s pairs/s={val_metrics['pairs_per_second']:.2f} "
            f"cache_hit_rate={val_metrics['cache_hit_rate']:.3f}"
        )
        print(f"  RAM maxrss={log_entry['ram_maxrss_gb']:.2f} GB")
        if device.type == "cuda":
            print(
                f"  GPU peak: alloc={log_entry['gpu_peak_allocated_gb']:.3f} GB "
                f"reserved={log_entry['gpu_peak_reserved_gb']:.3f} GB"
            )

        checkpoint = make_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            train_log=train_log,
            config=config,
            pair_rng=pair_rng,
        )
        torch.save(checkpoint, output_dir / "latest.pt")

        monitor_loss = val_metrics["loss"] if not np.isnan(val_metrics["loss"]) else train_metrics["loss"]
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  -> Saved best model (monitor_loss={best_loss:.4f})")

        with (output_dir / "train_log.json").open("w", encoding="utf-8") as handle:
            json.dump(train_log, handle, indent=2)

    print(f"\nTraining complete. Best monitored loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train projection head with InfoNCE")

    parser.add_argument("--megadepth_root", default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM")
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--sparse_dir", default=None)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)

    parser.add_argument("--input_dim", type=int, default=1664)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs_per_epoch", type=int, default=50000)
    parser.add_argument("--val_pairs_per_epoch", type=int, default=1000)
    parser.add_argument("--feature_cache_size", type=int, default=1000)
    parser.add_argument("--sparse_scene_cache_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_correspondences", type=int, default=512)
    parser.add_argument("--min_correspondences", type=int, default=50)
    parser.add_argument("--min_covis", type=int, default=50)
    parser.add_argument("--max_covis", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", default="experiments/projection_head_v1")

    args = parser.parse_args()
    args.hidden_dims = normalize_hidden_dims(args.hidden_dims, args.hidden_dim)
    return args


if __name__ == "__main__":
    train(parse_args())
