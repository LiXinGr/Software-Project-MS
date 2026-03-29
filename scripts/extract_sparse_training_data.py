"""Extract compact sparse MegaDepth training bundles from dense feature caches."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

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


def load_dense_feature_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "dinov3": data["dinov3"],
            "dift": data["dift"],
            "image_size": data["image_size"].astype(np.float32, copy=False),
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
        return np.empty((0, 1664), dtype=np.float16)

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
    fused = np.concatenate([alpha * dift_vecs, (1.0 - alpha) * dinov3_vecs], axis=1)
    return fused.astype(np.float16, copy=False)


def get_unique_positive_points(image) -> Tuple[np.ndarray, np.ndarray]:
    valid = image.point2D_ids >= 0
    point_ids = image.point2D_ids[valid].astype(np.int64, copy=False)
    points_xy = image.point2D_xy[valid].astype(np.float32, copy=False)
    if point_ids.size == 0:
        return point_ids, np.empty((0, 2), dtype=np.float32)

    order = np.argsort(point_ids, kind="mergesort")
    point_ids = point_ids[order]
    points_xy = points_xy[order]
    unique_ids, unique_idx = np.unique(point_ids, return_index=True)
    return unique_ids, points_xy[unique_idx]


def sample_random_negative_points(
    feat_data: Dict[str, np.ndarray],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_samples <= 0:
        return np.empty((0, 2), dtype=np.float32)

    h_dino, w_dino = feat_data["dinov3"].shape[:2]
    pre_h, pre_w = feat_data["preprocessed_size"]
    orig_h, orig_w = feat_data["image_size"]
    preprocess_scale = float(feat_data["preprocess_scale"])
    pad_left = float(feat_data["pad_left"])
    pad_top = float(feat_data["pad_top"])

    grid_x = (np.arange(w_dino, dtype=np.float32) + 0.5) * (float(pre_w) / w_dino)
    grid_y = (np.arange(h_dino, dtype=np.float32) + 0.5) * (float(pre_h) / h_dino)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    orig_x = (mesh_x - pad_left) / preprocess_scale
    orig_y = (mesh_y - pad_top) / preprocess_scale
    valid = (
        (orig_x >= 0.0)
        & (orig_x < float(orig_w))
        & (orig_y >= 0.0)
        & (orig_y < float(orig_h))
    )

    coords = np.stack([orig_x[valid], orig_y[valid]], axis=1)
    if coords.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    sample_count = min(num_samples, coords.shape[0])
    choice = rng.choice(coords.shape[0], size=sample_count, replace=False)
    return coords[choice].astype(np.float32, copy=False)


def extract_scene(scene_name: str, args: argparse.Namespace, seed_offset: int) -> Dict[str, object]:
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed + seed_offset)

    sampler = MegaDepthPairSampler(
        root=args.megadepth_root,
        min_covis=args.min_covis,
        max_covis=args.max_covis,
        scenes=[scene_name],
    )
    if scene_name not in sampler.scenes:
        raise RuntimeError(f"Scene {scene_name} could not be loaded.")

    scene = sampler.scenes[scene_name]
    valid_pairs = [(id1, id2, covis) for _, id1, id2, covis in sampler.all_pairs]
    candidate_image_ids = sorted({image_id for pair in valid_pairs for image_id in pair[:2]})

    processed_images: Dict[int, Dict[str, object]] = {}
    missing_features: List[int] = []
    failed_images: List[Tuple[int, str]] = []

    print(
        f"Scene {scene_name}: extracting sparse features for {len(candidate_image_ids)} candidate images"
    )
    for image_index, image_id in enumerate(candidate_image_ids, start=1):
        image = scene.images[image_id]
        stem = Path(image.name).stem
        npz_path = Path(args.feature_dir) / scene_name / f"{stem}.npz"
        if not npz_path.is_file() or not npz_has_feature_keys(npz_path):
            missing_features.append(image_id)
            continue

        try:
            feat_data = load_dense_feature_bundle(npz_path)
            positive_ids, positive_points = get_unique_positive_points(image)
            positive_features = sample_fused_features(feat_data, positive_points, args.alpha)

            negative_points = sample_random_negative_points(feat_data, args.num_negatives, rng)
            negative_features = sample_fused_features(feat_data, negative_points, args.alpha)
            negative_ids = np.full(negative_features.shape[0], -1, dtype=np.int64)

            features = np.concatenate([positive_features, negative_features], axis=0)
            point_ids = np.concatenate([positive_ids, negative_ids], axis=0)
            processed_images[image_id] = {
                "name": image.name,
                "features": features,
                "point3D_ids": point_ids,
                "positive_count": int(positive_features.shape[0]),
            }
        except Exception as exc:  # noqa: BLE001
            failed_images.append((image_id, str(exc)))

        if image_index % args.progress_every == 0 or image_index == len(candidate_image_ids):
            extracted = len(processed_images)
            print(
                f"  [{scene_name} {image_index}/{len(candidate_image_ids)}] "
                f"extracted={extracted} missing={len(missing_features)} failed={len(failed_images)}"
            )

    final_pairs = [
        (id1, id2, covis)
        for id1, id2, covis in valid_pairs
        if id1 in processed_images and id2 in processed_images
    ]
    if not final_pairs:
        raise RuntimeError(f"Scene {scene_name}: no valid pairs remain after feature filtering.")

    final_image_ids = sorted({image_id for pair in final_pairs for image_id in pair[:2]})
    local_image_index = {image_id: idx for idx, image_id in enumerate(final_image_ids)}

    feature_chunks: List[np.ndarray] = []
    point_id_chunks: List[np.ndarray] = []
    image_offsets = [0]
    image_positive_counts = []
    image_names = []

    for image_id in final_image_ids:
        image_entry = processed_images[image_id]
        features = image_entry["features"]
        point_ids = image_entry["point3D_ids"]
        feature_chunks.append(features)
        point_id_chunks.append(point_ids)
        image_offsets.append(image_offsets[-1] + int(features.shape[0]))
        image_positive_counts.append(int(image_entry["positive_count"]))
        image_names.append(str(image_entry["name"]))

    all_features = np.concatenate(feature_chunks, axis=0)
    all_point_ids = np.concatenate(point_id_chunks, axis=0)
    pair_indices = np.asarray(
        [[local_image_index[id1], local_image_index[id2]] for id1, id2, _ in final_pairs],
        dtype=np.int32,
    )
    pair_covis = np.asarray([covis for _, _, covis in final_pairs], dtype=np.int32)

    bundle = {
        "scene_name": scene_name,
        "reconstruction": scene.reconstruction,
        "image_ids": torch.as_tensor(final_image_ids, dtype=torch.int32),
        "image_names": image_names,
        "image_offsets": torch.as_tensor(image_offsets, dtype=torch.int64),
        "image_positive_counts": torch.as_tensor(image_positive_counts, dtype=torch.int32),
        "features": torch.from_numpy(all_features),
        "point3D_ids": torch.from_numpy(all_point_ids),
        "pair_indices": torch.from_numpy(pair_indices),
        "pair_covis": torch.from_numpy(pair_covis),
        "alpha": float(args.alpha),
        "num_negatives_per_image": int(args.num_negatives),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_name}.pt"
    torch.save(bundle, output_path)

    elapsed = time.perf_counter() - start
    summary = {
        "scene": scene_name,
        "reconstruction": scene.reconstruction,
        "image_count": len(final_image_ids),
        "feature_vector_count": int(all_features.shape[0]),
        "pair_count": int(pair_indices.shape[0]),
        "file_size_gb": output_path.stat().st_size / (1024 ** 3),
        "time_sec": elapsed,
        "missing_features": len(missing_features),
        "failed_images": len(failed_images),
    }
    print(
        f"Scene {scene_name}: images={summary['image_count']} "
        f"features={summary['feature_vector_count']} "
        f"pairs={summary['pair_count']} "
        f"size={summary['file_size_gb']:.2f} GB "
        f"time={summary['time_sec']:.1f}s "
        f"missing={summary['missing_features']} failed={summary['failed_images']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact sparse MegaDepth training bundles")
    parser.add_argument("--megadepth_root", default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM")
    parser.add_argument("--feature_dir", default="/mnt/datagrid/personal/gorbuden/megadepth_features")
    parser.add_argument("--output_dir", default="data/sparse_train")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min_covis", type=int, default=50)
    parser.add_argument("--max_covis", type=int, default=5000)
    parser.add_argument("--num_negatives", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    total_start = time.perf_counter()
    for index, scene_name in enumerate(args.scenes):
        summaries.append(extract_scene(scene_name, args, seed_offset=index))

    total_elapsed = time.perf_counter() - total_start
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenes": summaries,
                "total_file_size_gb": sum(entry["file_size_gb"] for entry in summaries),
                "total_time_sec": total_elapsed,
            },
            handle,
            indent=2,
        )
    print(
        f"Done: scenes={len(summaries)} total_size="
        f"{sum(entry['file_size_gb'] for entry in summaries):.2f} GB "
        f"total_time={total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
