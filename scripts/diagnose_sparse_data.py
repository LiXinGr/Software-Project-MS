"""Diagnose sparse MegaDepth training bundles versus the dense feature path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from megadepth_pairs import MegaDepthScene
from train_projection_head import SparseFeatureDataset, vector_l2_normalize


def load_dense_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "dinov3": data["dinov3"],
            "dift": data["dift"],
            "preprocess_scale": np.float32(data["preprocess_scale"]),
            "preprocessed_size": data["preprocessed_size"].astype(np.float32, copy=False),
            "pad_left": np.float32(data["pad_left"]) if "pad_left" in data else np.float32(0.0),
            "pad_top": np.float32(data["pad_top"]) if "pad_top" in data else np.float32(0.0),
            "dift_pre_to_input_scale_xy": data["dift_pre_to_input_scale_xy"].astype(np.float32, copy=False),
            "dift_input_size": data["dift_input_size"].astype(np.float32, copy=False),
        }


def sample_fused_features(
    feat_data: Dict[str, np.ndarray],
    pts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if pts.size == 0:
        return np.empty((0, 1664), dtype=np.float32)

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
    dinov3_vecs = dinov3_feat[iy_dino, ix_dino].astype(np.float32, copy=False)

    h_dift, w_dift = dift_feat.shape[:2]
    pts_dift_x = pts_pre[:, 0] * dift_scale_xy[0]
    pts_dift_y = pts_pre[:, 1] * dift_scale_xy[1]
    stride_x = float(dift_input_size[1]) / w_dift
    stride_y = float(dift_input_size[0]) / h_dift
    ix_dift = np.clip(np.rint(pts_dift_x / stride_x).astype(np.int64), 0, w_dift - 1)
    iy_dift = np.clip(np.rint(pts_dift_y / stride_y).astype(np.int64), 0, h_dift - 1)
    dift_vecs = dift_feat[iy_dift, ix_dift].astype(np.float32, copy=False)

    dinov3_vecs = vector_l2_normalize(dinov3_vecs)
    dift_vecs = vector_l2_normalize(dift_vecs)
    return np.concatenate([alpha * dift_vecs, (1.0 - alpha) * dinov3_vecs], axis=1)


def get_unique_positive_points(image) -> Tuple[np.ndarray, np.ndarray]:
    valid = image.point2D_ids >= 0
    point_ids = image.point2D_ids[valid].astype(np.int64, copy=False)
    points_xy = image.point2D_xy[valid].astype(np.float32, copy=False)
    if point_ids.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    order = np.argsort(point_ids, kind="mergesort")
    point_ids = point_ids[order]
    points_xy = points_xy[order]
    unique_ids, unique_idx = np.unique(point_ids, return_index=True)
    return unique_ids, points_xy[unique_idx]


def normalize_rows(array: np.ndarray) -> np.ndarray:
    return vector_l2_normalize(array.astype(np.float32, copy=False))


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_n = normalize_rows(a)
    b_n = normalize_rows(b)
    return float(np.sum(a_n * b_n, axis=1).mean())


def mean_negative_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] <= 1 or b.shape[0] <= 1:
        return float("nan")
    count = min(a.shape[0], b.shape[0])
    a_n = normalize_rows(a[:count])
    b_n = normalize_rows(b[:count])
    perm = np.roll(np.arange(count), 1)
    return float(np.sum(a_n * b_n[perm], axis=1).mean())


def diag_offdiag_cosine(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan"), float("nan")
    count = min(a.shape[0], b.shape[0])
    a_n = normalize_rows(a[:count])
    b_n = normalize_rows(b[:count])
    sim = a_n @ b_n.T
    diag_mean = float(np.diag(sim).mean())
    if count == 1:
        return diag_mean, float("nan")
    offdiag_mask = ~np.eye(count, dtype=bool)
    offdiag_mean = float(sim[offdiag_mask].mean())
    return diag_mean, offdiag_mean


def bundle_image_slices(bundle: Dict[str, object], local_image_idx: int) -> Dict[str, np.ndarray]:
    offsets = bundle["image_offsets"]
    positive_counts = bundle["image_positive_counts"]
    features = bundle["features"]
    point_ids = bundle["point3D_ids"]

    start = int(offsets[local_image_idx])
    end = int(offsets[local_image_idx + 1])
    pos_count = int(positive_counts[local_image_idx])
    return {
        "all_features": features[start:end],
        "all_point_ids": point_ids[start:end],
        "positive_features": features[start : start + pos_count],
        "positive_point_ids": point_ids[start : start + pos_count],
        "positive_count": np.asarray(pos_count),
    }


def load_sparse_bundle(path: Path) -> Dict[str, object]:
    raw = torch.load(path, map_location="cpu")
    converted = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            converted[key] = value.numpy()
        else:
            converted[key] = value
    return converted


def format_shared_ids(ids: np.ndarray, limit: int = 10) -> str:
    if ids.size == 0:
        return "[]"
    shown = ", ".join(str(int(x)) for x in ids[:limit])
    suffix = "" if ids.size <= limit else ", ..."
    return f"[{shown}{suffix}]"


def compare_configs(debug_path: Path, full_path: Path) -> Dict[str, object]:
    debug = json.loads(debug_path.read_text())
    full = json.loads(full_path.read_text())
    differing = {}
    for key in sorted(set(debug) | set(full)):
        if debug.get(key) != full.get(key):
            differing[key] = {"debug": debug.get(key), "full": full.get(key)}
    return {
        "debug_num_valid_pairs": debug.get("num_valid_pairs"),
        "full_num_valid_pairs": full.get("num_valid_pairs"),
        "debug_num_correspondences": debug.get("num_correspondences"),
        "full_num_correspondences": full.get("num_correspondences"),
        "debug_scenes": debug.get("scenes"),
        "full_scenes": full.get("scenes"),
        "differing_fields": differing,
    }


def diagnose_pairs(args: argparse.Namespace) -> None:
    bundle_path = Path(args.bundle_path)
    bundle = load_sparse_bundle(bundle_path)

    scene_name = str(bundle["scene_name"])
    reconstruction = str(bundle["reconstruction"])
    alpha = float(bundle["alpha"])
    image_ids = bundle["image_ids"].astype(np.int64, copy=False)
    image_names = list(bundle["image_names"])
    pair_indices = bundle["pair_indices"].astype(np.int64, copy=False)

    scene_dir = Path(args.megadepth_root) / scene_name
    scene = MegaDepthScene(str(scene_dir), reconstruction=reconstruction)

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(pair_indices.shape[0], size=min(args.num_pairs, pair_indices.shape[0]), replace=False)

    print("=== Sparse Bundle Summary ===")
    print(f"bundle: {bundle_path}")
    print(f"scene: {scene_name}")
    print(f"reconstruction: {reconstruction}")
    print(f"images: {len(image_ids)}")
    print(f"pair_count: {pair_indices.shape[0]}")
    print(f"feature_vectors: {bundle['features'].shape[0]}")
    print(f"alpha: {alpha}")

    print("\n=== Hypothesis 1: Sparse Correspondence Integrity ===")
    for bundle_pair_idx in chosen:
        local_i1, local_i2 = pair_indices[bundle_pair_idx]
        actual_i1 = int(image_ids[local_i1])
        actual_i2 = int(image_ids[local_i2])
        name1 = image_names[local_i1]
        name2 = image_names[local_i2]

        slice_1 = bundle_image_slices(bundle, int(local_i1))
        slice_2 = bundle_image_slices(bundle, int(local_i2))

        ids_1 = slice_1["positive_point_ids"].astype(np.int64, copy=False)
        ids_2 = slice_2["positive_point_ids"].astype(np.int64, copy=False)
        shared_ids, idx_1, idx_2 = np.intersect1d(ids_1, ids_2, assume_unique=True, return_indices=True)
        sparse_pos_1 = slice_1["positive_features"][idx_1]
        sparse_pos_2 = slice_2["positive_features"][idx_2]

        sparse_pos_cos = mean_cosine(sparse_pos_1, sparse_pos_2)
        sparse_neg_cos = mean_negative_cosine(sparse_pos_1, sparse_pos_2)

        dense_image_1 = scene.images[actual_i1]
        dense_image_2 = scene.images[actual_i2]
        dense_ids_1, dense_xy_1 = get_unique_positive_points(dense_image_1)
        dense_ids_2, dense_xy_2 = get_unique_positive_points(dense_image_2)
        dense_shared_ids, dense_idx_1, dense_idx_2 = np.intersect1d(
            dense_ids_1,
            dense_ids_2,
            assume_unique=True,
            return_indices=True,
        )
        dense_xy_shared_1 = dense_xy_1[dense_idx_1]
        dense_xy_shared_2 = dense_xy_2[dense_idx_2]

        dense_bundle_1 = load_dense_bundle(Path(args.feature_dir) / scene_name / f"{Path(name1).stem}.npz")
        dense_bundle_2 = load_dense_bundle(Path(args.feature_dir) / scene_name / f"{Path(name2).stem}.npz")
        dense_feat_1 = sample_fused_features(dense_bundle_1, dense_xy_shared_1, alpha)
        dense_feat_2 = sample_fused_features(dense_bundle_2, dense_xy_shared_2, alpha)

        dense_pos_cos = mean_cosine(dense_feat_1, dense_feat_2)
        dense_neg_cos = mean_negative_cosine(dense_feat_1, dense_feat_2)

        align_ids, sparse_align_idx, dense_align_idx = np.intersect1d(
            shared_ids,
            dense_shared_ids,
            assume_unique=True,
            return_indices=True,
        )
        sparse_dense_cos_1 = mean_cosine(sparse_pos_1[sparse_align_idx], dense_feat_1[dense_align_idx])
        sparse_dense_cos_2 = mean_cosine(sparse_pos_2[sparse_align_idx], dense_feat_2[dense_align_idx])

        print(f"\nPair index {int(bundle_pair_idx)}")
        print(f"  image1: local={int(local_i1)} actual={actual_i1} name={name1}")
        print(f"  image2: local={int(local_i2)} actual={actual_i2} name={name2}")
        print(
            f"  features image1: total={slice_1['all_features'].shape[0]} "
            f"positive={ids_1.shape[0]} negatives={slice_1['all_features'].shape[0] - ids_1.shape[0]}"
        )
        print(
            f"  features image2: total={slice_2['all_features'].shape[0]} "
            f"positive={ids_2.shape[0]} negatives={slice_2['all_features'].shape[0] - ids_2.shape[0]}"
        )
        print(f"  shared point3D ids: {shared_ids.shape[0]}")
        print(f"  first shared ids: {format_shared_ids(shared_ids)}")
        print(
            f"  sparse cosine: positives={sparse_pos_cos:.4f} negatives={sparse_neg_cos:.4f}"
        )
        print(
            f"  dense cosine:  positives={dense_pos_cos:.4f} negatives={dense_neg_cos:.4f}"
        )
        print(
            f"  sparse-vs-dense alignment cosine: image1={sparse_dense_cos_1:.4f} "
            f"image2={sparse_dense_cos_2:.4f} shared_aligned={align_ids.shape[0]}"
        )

    print("\n=== Hypothesis 2: SparseFeatureDataset Batch Semantics ===")
    dataset = SparseFeatureDataset(
        sparse_dir=str(Path(args.bundle_path).parent),
        scenes=[scene_name],
        num_correspondences=args.num_correspondences,
        min_correspondences=args.min_correspondences,
    )
    print("SparseFeatureDataset does not implement __getitem__; diagnosing get_pair_batch(), which train() uses.")
    batch = dataset.get_pair_batch(0, np.random.default_rng(args.seed))
    feats_1 = batch["features_1"].numpy().astype(np.float32, copy=False)
    feats_2 = batch["features_2"].numpy().astype(np.float32, copy=False)
    pos_1 = batch["pos_indices_1"].numpy().astype(np.int64, copy=False)
    pos_2 = batch["pos_indices_2"].numpy().astype(np.int64, copy=False)
    pos_feats_1 = feats_1[pos_1]
    pos_feats_2 = feats_2[pos_2]
    diag_mean, offdiag_mean = diag_offdiag_cosine(pos_feats_1, pos_feats_2)
    print(f"  features_1 shape: {tuple(feats_1.shape)}")
    print(f"  features_2 shape: {tuple(feats_2.shape)}")
    print(f"  positives in batch: {pos_1.shape[0]}")
    print(f"  mean diagonal cosine: {diag_mean:.4f}")
    print(f"  mean off-diagonal cosine: {offdiag_mean:.4f}")

    print("\n=== Hypothesis 3: Debug Run vs Full Run Config Differences ===")
    comparison = compare_configs(Path(args.debug_config), Path(args.full_config))
    print(
        f"debug num_valid_pairs={comparison['debug_num_valid_pairs']} "
        f"full num_valid_pairs={comparison['full_num_valid_pairs']}"
    )
    print(
        f"debug num_correspondences={comparison['debug_num_correspondences']} "
        f"full num_correspondences={comparison['full_num_correspondences']}"
    )
    print("Differing config fields:")
    print(json.dumps(comparison["differing_fields"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose sparse MegaDepth training data")
    parser.add_argument("--bundle_path", default="data/sparse_train/0148.pt")
    parser.add_argument("--feature_dir", default="/mnt/datagrid/personal/gorbuden/megadepth_features")
    parser.add_argument("--megadepth_root", default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM")
    parser.add_argument("--debug_config", default="experiments/projection_head_debug/config.json")
    parser.add_argument("--full_config", default="experiments/p2_projection_v1/config.json")
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--num_correspondences", type=int, default=512)
    parser.add_argument("--min_correspondences", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    diagnose_pairs(parse_args())
