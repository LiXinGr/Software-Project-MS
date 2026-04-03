from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from megadepth_pairs import MegaDepthPairSampler, MegaDepthScene


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GF_ROOT = PROJECT_ROOT / "external" / "glue-factory"
GF_DATA_ROOT = GF_ROOT / "data"
GF_SCENE_LIST_ROOT = GF_ROOT / "gluefactory" / "datasets" / "megadepth_scene_lists"


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_to_calibration(camera) -> np.ndarray:
    model = camera.model.upper()
    params = camera.params.astype(np.float64)
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def pose_from_image(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = qvec2rotmat(image.qvec).astype(np.float32)
    pose[:3, 3] = image.tvec.astype(np.float32)
    return pose


def load_scene_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ensure_scene_symlink(gf_data_root: Path, raw_root: Path, scene_name: str) -> None:
    target = raw_root / scene_name
    link_path = gf_data_root / "Undistorted_SfM" / scene_name
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        return
    os.symlink(target, link_path)


def overlap_score(shared: int, n0: int, n1: int, denom: str) -> float:
    if shared <= 0 or n0 <= 0 or n1 <= 0:
        return 0.0
    if denom == "min":
        base = min(n0, n1)
    elif denom == "max":
        base = max(n0, n1)
    elif denom == "union":
        base = n0 + n1 - shared
    else:
        raise ValueError(f"Unknown overlap denominator: {denom}")
    if base <= 0:
        return 0.0
    return float(shared) / float(base)


def build_scene_info(scene: MegaDepthScene, scene_name: str, min_shared_points: int, denom: str) -> tuple[dict, dict]:
    valid_image_ids = [
        image_id
        for image_id in sorted(scene.images)
        if scene.image_exists.get(image_id, False)
    ]
    image_index = {image_id: idx for idx, image_id in enumerate(valid_image_ids)}

    image_paths = []
    depth_paths = []
    intrinsics = []
    poses = []
    visible_counts = []

    for image_id in valid_image_ids:
        image = scene.images[image_id]
        camera = scene.cameras[image.camera_id]
        visible_count = 0
        for point_id in image.point2D_ids:
            if point_id < 0 or point_id not in scene.points3D:
                continue
            visible_count += 1
        image_paths.append(f"Undistorted_SfM/{scene_name}/images/{image.name}")
        depth_paths.append(f"depth_undistorted/{scene_name}/{Path(image.name).stem}.h5")
        intrinsics.append(camera_to_calibration(camera))
        poses.append(pose_from_image(image))
        visible_counts.append(visible_count)

    n_images = len(valid_image_ids)
    overlap_matrix = np.zeros((n_images, n_images), dtype=np.float32)
    np.fill_diagonal(overlap_matrix, 1.0)

    kept_pairs = 0
    score_bins = Counter()
    for (img0, img1), shared in scene.pair_covis.items():
        if shared < min_shared_points:
            continue
        if img0 not in image_index or img1 not in image_index:
            continue
        idx0 = image_index[img0]
        idx1 = image_index[img1]
        score = overlap_score(shared, visible_counts[idx0], visible_counts[idx1], denom)
        overlap_matrix[idx0, idx1] = score
        overlap_matrix[idx1, idx0] = score
        if 0.1 < score <= 0.3:
            score_bins["0.1-0.3"] += 1
        elif 0.3 < score <= 0.5:
            score_bins["0.3-0.5"] += 1
        elif 0.5 < score <= 0.7:
            score_bins["0.5-0.7"] += 1
        if 0.1 < score <= 0.7:
            kept_pairs += 1

    payload = {
        "image_paths": np.asarray(image_paths, dtype=object),
        "depth_paths": np.asarray(depth_paths, dtype=object),
        "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32, copy=False),
        "poses": np.stack(poses, axis=0).astype(np.float32, copy=False),
        "overlap_matrix": overlap_matrix,
    }
    stats = {
        "scene": scene_name,
        "images": n_images,
        "pairs_overlap_0.1_0.7": kept_pairs,
        "bin_0.1_0.3": int(score_bins["0.1-0.3"]),
        "bin_0.3_0.5": int(score_bins["0.3-0.5"]),
        "bin_0.5_0.7": int(score_bins["0.5-0.7"]),
    }
    return payload, stats


def prepare_scene(
    raw_root: Path,
    gf_data_root: Path,
    scene_name: str,
    min_shared_points: int,
    denom: str,
) -> dict:
    scene_dir = raw_root / scene_name
    reconstruction = MegaDepthPairSampler._find_reconstruction(scene_dir)
    if reconstruction is None:
        raise FileNotFoundError(f"No valid reconstruction found for {scene_name}")

    scene = MegaDepthScene(str(scene_dir), reconstruction=reconstruction, min_covis=1)
    ensure_scene_symlink(gf_data_root, raw_root, scene_name)

    scene_info_dir = gf_data_root / "scene_info"
    scene_info_dir.mkdir(parents=True, exist_ok=True)
    (gf_data_root / "depth_undistorted" / scene_name).mkdir(parents=True, exist_ok=True)

    payload, stats = build_scene_info(scene, scene_name, min_shared_points, denom)
    np.savez(scene_info_dir / f"{scene_name}.npz", **payload)
    return stats


def resolve_scenes(raw_root: Path, split_path: Path, scenes: list[str] | None) -> list[str]:
    if scenes:
        return sorted(set(scenes))
    available = {path.name for path in raw_root.iterdir() if path.is_dir()}
    split_scenes = set(load_scene_list(split_path))
    return sorted(available & split_scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw MegaDepth scenes for Glue Factory.")
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/mnt/datasets/MegaDepth/MegaDepth_v1_SfM"),
    )
    parser.add_argument(
        "--gf_data_root",
        type=Path,
        default=GF_DATA_ROOT / "megadepth_phase4",
    )
    parser.add_argument(
        "--clean_train_split",
        type=Path,
        default=GF_SCENE_LIST_ROOT / "train_scenes_clean.txt",
    )
    parser.add_argument(
        "--split_output_name",
        type=str,
        default="train_scenes_datagrid_overlap67.txt",
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument("--min_shared_points", type=int, default=50)
    parser.add_argument("--overlap_denominator", choices=["min", "max", "union"], default="min")
    parser.add_argument("--write_summary_json", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    scenes = resolve_scenes(args.raw_root, args.clean_train_split, args.scenes)
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]

    if not scenes:
        raise RuntimeError("No scenes selected for preparation.")

    t0 = time.perf_counter()
    per_scene = []
    skipped = []
    for idx, scene_name in enumerate(scenes, start=1):
        start = time.perf_counter()
        try:
            stats = prepare_scene(
                args.raw_root,
                args.gf_data_root,
                scene_name,
                args.min_shared_points,
                args.overlap_denominator,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            if not args.skip_errors:
                raise
            skipped.append(
                {
                    "scene": scene_name,
                    "prepare_time_sec": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"[PREP] {idx}/{len(scenes)} {scene_name}: SKIPPED "
                f"({type(exc).__name__}: {exc}) after {elapsed:.2f}s",
                flush=True,
            )
            continue

        elapsed = time.perf_counter() - start
        stats["prepare_time_sec"] = elapsed
        per_scene.append(stats)
        print(
            f"[PREP] {idx}/{len(scenes)} {scene_name}: "
            f"images={stats['images']}, pairs_0.1_0.7={stats['pairs_overlap_0.1_0.7']}, "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    split_path = GF_SCENE_LIST_ROOT / args.split_output_name
    prepared_scenes = [item["scene"] for item in per_scene]
    split_path.write_text("\n".join(prepared_scenes) + "\n")

    total_time = time.perf_counter() - t0
    total_pairs = sum(item["pairs_overlap_0.1_0.7"] for item in per_scene)
    summary = {
        "raw_root": str(args.raw_root),
        "gf_data_root": str(args.gf_data_root),
        "num_requested_scenes": len(scenes),
        "num_scenes": len(prepared_scenes),
        "num_skipped_scenes": len(skipped),
        "total_pairs_overlap_0.1_0.7": total_pairs,
        "total_prepare_time_sec": total_time,
        "mean_prepare_time_sec": total_time / max(len(prepared_scenes), 1),
        "min_shared_points": args.min_shared_points,
        "overlap_denominator": args.overlap_denominator,
        "scene_list_path": str(split_path),
        "scenes": per_scene,
        "skipped_scenes": skipped,
    }
    print(
        f"[PREP] DONE: prepared={len(prepared_scenes)}, skipped={len(skipped)}, "
        f"total_pairs_0.1_0.7={total_pairs}, total_time={total_time:.2f}s, "
        f"mean_time={summary['mean_prepare_time_sec']:.2f}s",
        flush=True,
    )
    if args.write_summary_json:
        out_path = args.gf_data_root / "prepare_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[PREP] Summary written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
