"""MegaDepth correspondence pair sampler for contrastive training."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ColmapImage:
    id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    point2D_xy: np.ndarray
    point2D_ids: np.ndarray


@dataclass
class ColmapPoint3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]


class MegaDepthScene:
    """Loads a single MegaDepth scene from a COLMAP text reconstruction."""

    def __init__(
        self,
        scene_dir: str,
        reconstruction: str = "manhattan/0",
        min_covis: int = 50,
    ):
        """
        Args:
            scene_dir: e.g. /mnt/datasets/MegaDepth/MegaDepth_v1_SfM/0015
            reconstruction: sub-path under sparse/, e.g. "manhattan/0"
            min_covis: minimum shared 3D points required for a valid pair
        """
        self.scene_dir = Path(scene_dir)
        self.image_dir = self.scene_dir / "images"
        self.reconstruction = reconstruction
        self.min_covis = min_covis

        recon_dir = self.scene_dir / "sparse" / reconstruction
        if not recon_dir.exists():
            raise FileNotFoundError(f"Reconstruction not found: {recon_dir}")

        self.cameras = self._parse_cameras(recon_dir / "cameras.txt")
        self.images = self._parse_images(recon_dir / "images.txt")
        self.points3D = self._parse_points3D(recon_dir / "points3D.txt")
        self.image_exists = {
            image_id: (self.image_dir / image.name).is_file()
            for image_id, image in self.images.items()
        }

        self._build_covisibility()

    @staticmethod
    def _parse_cameras(path: Path) -> Dict[int, ColmapCamera]:
        """Parse COLMAP cameras.txt."""
        cameras: Dict[int, ColmapCamera] = {}

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Malformed camera line in {path}: {line}")

                camera_id = int(parts[0])
                cameras[camera_id] = ColmapCamera(
                    id=camera_id,
                    model=parts[1],
                    width=int(parts[2]),
                    height=int(parts[3]),
                    params=np.asarray(parts[4:], dtype=np.float64),
                )

        return cameras

    @staticmethod
    def _parse_images(path: Path) -> Dict[int, ColmapImage]:
        """Parse COLMAP images.txt with two lines per image."""
        images: Dict[int, ColmapImage] = {}
        pending_header: Optional[str] = None

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.startswith("#"):
                    continue

                line = raw_line.strip()

                if pending_header is None:
                    if not line:
                        continue
                    pending_header = line
                    continue

                header_parts = pending_header.split()
                if len(header_parts) < 10:
                    raise ValueError(
                        f"Malformed image header in {path}: {pending_header}"
                    )

                image_id = int(header_parts[0])
                qvec = np.asarray(header_parts[1:5], dtype=np.float64)
                tvec = np.asarray(header_parts[5:8], dtype=np.float64)
                camera_id = int(header_parts[8])
                name = " ".join(header_parts[9:])

                if line:
                    point_data = np.fromstring(line, sep=" ", dtype=np.float64)
                    if point_data.size % 3 != 0:
                        raise ValueError(
                            f"Malformed image points line in {path}: {line[:120]}"
                        )
                    point_data = point_data.reshape(-1, 3)
                    point2D_xy = point_data[:, :2].astype(np.float64, copy=False)
                    point2D_ids = point_data[:, 2].astype(np.int64, copy=False)
                else:
                    point2D_xy = np.empty((0, 2), dtype=np.float64)
                    point2D_ids = np.empty((0,), dtype=np.int64)

                images[image_id] = ColmapImage(
                    id=image_id,
                    name=name,
                    camera_id=camera_id,
                    qvec=qvec,
                    tvec=tvec,
                    point2D_xy=point2D_xy,
                    point2D_ids=point2D_ids,
                )
                pending_header = None

        if pending_header is not None:
            raise ValueError(f"Dangling image header at end of {path}: {pending_header}")

        return images

    @staticmethod
    def _parse_points3D(path: Path) -> Dict[int, ColmapPoint3D]:
        """Parse COLMAP points3D.txt."""
        points3D: Dict[int, ColmapPoint3D] = {}

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    raise ValueError(f"Malformed 3D point line in {path}: {line}")

                point_id = int(parts[0])
                xyz = np.asarray(parts[1:4], dtype=np.float64)
                rgb = np.asarray(parts[4:7], dtype=np.uint8)
                error = float(parts[7])

                track_tokens = [int(token) for token in parts[8:]]
                if len(track_tokens) % 2 != 0:
                    raise ValueError(f"Malformed 3D point track in {path}: {line[:120]}")
                track = [
                    (track_tokens[idx], track_tokens[idx + 1])
                    for idx in range(0, len(track_tokens), 2)
                ]

                points3D[point_id] = ColmapPoint3D(
                    id=point_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    track=track,
                )

        return points3D

    def _build_covisibility(self) -> None:
        """Build image-pair co-visibility counts from COLMAP tracks."""
        self.pair_covis: Counter[Tuple[int, int]] = Counter()

        for point in self.points3D.values():
            image_ids = sorted(
                {
                    image_id
                    for image_id, _ in point.track
                    if image_id in self.images and self.image_exists.get(image_id, False)
                }
            )
            if len(image_ids) < 2:
                continue

            for image_id_1, image_id_2 in combinations(image_ids, 2):
                self.pair_covis[(image_id_1, image_id_2)] += 1

        valid_pairs = []
        for (image_id_1, image_id_2), covis_count in self.pair_covis.items():
            if covis_count < self.min_covis:
                continue
            if not self.image_exists.get(image_id_1, False):
                continue
            if not self.image_exists.get(image_id_2, False):
                continue
            valid_pairs.append((image_id_1, image_id_2, covis_count))

        valid_pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
        self.valid_pairs = valid_pairs

    def get_correspondences(
        self,
        img_id_1: int,
        img_id_2: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return pixel correspondences for shared 3D points in two images."""
        img1 = self.images[img_id_1]
        img2 = self.images[img_id_2]

        pts1_by_point_id: Dict[int, np.ndarray] = {}
        for index, point3D_id in enumerate(img1.point2D_ids):
            if point3D_id < 0:
                continue
            if point3D_id not in pts1_by_point_id:
                pts1_by_point_id[point3D_id] = img1.point2D_xy[index]

        pts1_list: List[np.ndarray] = []
        pts2_list: List[np.ndarray] = []
        used_point_ids = set()

        for index, point3D_id in enumerate(img2.point2D_ids):
            if point3D_id < 0 or point3D_id in used_point_ids:
                continue
            if point3D_id not in pts1_by_point_id:
                continue

            pts1_list.append(pts1_by_point_id[point3D_id])
            pts2_list.append(img2.point2D_xy[index])
            used_point_ids.add(point3D_id)

        if not pts1_list:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty.copy()

        return np.vstack(pts1_list), np.vstack(pts2_list)

    def get_image_path(self, img_id: int) -> Path:
        return self.image_dir / self.images[img_id].name


class MegaDepthPairSampler:
    """Samples MegaDepth training pairs from sparse COLMAP reconstructions."""

    def __init__(
        self,
        root: str,
        min_covis: int = 50,
        max_covis: int = 5000,
        scenes: Optional[List[str]] = None,
    ):
        """
        Args:
            root: e.g. /mnt/datasets/MegaDepth/MegaDepth_v1_SfM
            min_covis: minimum shared 3D points for a valid pair
            max_covis: maximum shared 3D points to keep
            scenes: list of scene names to use, or None for all scenes
        """
        self.root = Path(root)
        self.min_covis = min_covis
        self.max_covis = max_covis
        self.scenes: Dict[str, MegaDepthScene] = {}
        self.all_pairs: List[Tuple[str, int, int, int]] = []

        if scenes is None:
            scenes = sorted(path.name for path in self.root.iterdir() if path.is_dir())

        for scene_name in scenes:
            scene_dir = self.root / scene_name
            if not scene_dir.is_dir():
                print(f"Scene {scene_name}: SKIPPED - directory not found")
                continue

            reconstruction = self._find_reconstruction(scene_dir)
            if reconstruction is None:
                print(f"Scene {scene_name}: SKIPPED - no valid reconstruction found")
                continue

            try:
                scene = MegaDepthScene(
                    str(scene_dir),
                    reconstruction=reconstruction,
                    min_covis=min_covis,
                )
            except Exception as exc:
                print(f"Scene {scene_name}: FAILED - {exc}")
                continue

            self.scenes[scene_name] = scene
            kept_pairs = 0
            for image_id_1, image_id_2, covis_count in scene.valid_pairs:
                if covis_count > max_covis:
                    continue
                self.all_pairs.append((scene_name, image_id_1, image_id_2, covis_count))
                kept_pairs += 1

            print(
                f"Scene {scene_name}: recon={reconstruction}, "
                f"{len(scene.images)} images, {len(scene.points3D)} points3D, "
                f"{kept_pairs} kept pairs"
            )

        print(f"\nTotal: {len(self.scenes)} scenes, {len(self.all_pairs)} training pairs")

    @staticmethod
    def _find_reconstruction(scene_dir: Path) -> Optional[str]:
        sparse_dir = scene_dir / "sparse"
        if not sparse_dir.is_dir():
            return None

        preferred = sparse_dir / "manhattan" / "0"
        if MegaDepthPairSampler._is_valid_reconstruction_dir(preferred):
            return str(preferred.relative_to(sparse_dir))

        candidates = sorted(
            path.parent
            for path in sparse_dir.rglob("cameras.txt")
            if MegaDepthPairSampler._is_valid_reconstruction_dir(path.parent)
        )
        if not candidates:
            return None

        return str(candidates[0].relative_to(sparse_dir))

    @staticmethod
    def _is_valid_reconstruction_dir(path: Path) -> bool:
        return (
            path / "cameras.txt"
        ).is_file() and (path / "images.txt").is_file() and (path / "points3D.txt").is_file()

    def sample_pair(self) -> dict:
        """Sample a random training pair and return paths plus correspondences."""
        if not self.all_pairs:
            raise RuntimeError("No valid MegaDepth pairs available for sampling.")

        scene_name, image_id_1, image_id_2, covis_count = random.choice(self.all_pairs)
        scene = self.scenes[scene_name]
        pts1, pts2 = scene.get_correspondences(image_id_1, image_id_2)

        return {
            "scene": scene_name,
            "img_path_1": str(scene.get_image_path(image_id_1)),
            "img_path_2": str(scene.get_image_path(image_id_2)),
            "pts1": pts1,
            "pts2": pts2,
            "covis": covis_count,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample MegaDepth correspondence pairs from COLMAP text models."
    )
    parser.add_argument(
        "--root",
        default="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM",
        help="MegaDepth SfM root directory",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["0015", "0025"],
        help="Scene names to load",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sampled pairs to print",
    )
    parser.add_argument(
        "--min_covis",
        type=int,
        default=50,
        help="Minimum shared 3D points for a valid pair",
    )
    parser.add_argument(
        "--max_covis",
        type=int,
        default=5000,
        help="Maximum shared 3D points to keep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    sampler = MegaDepthPairSampler(
        args.root,
        min_covis=args.min_covis,
        max_covis=args.max_covis,
        scenes=args.scenes,
    )

    print(f"\n--- Sampling {args.num_samples} pairs ---")
    for sample_idx in range(args.num_samples):
        pair = sampler.sample_pair()
        image_path_1 = Path(pair["img_path_1"])
        image_path_2 = Path(pair["img_path_2"])

        print(f"\nPair {sample_idx + 1}:")
        print(f"  Scene: {pair['scene']}")
        print(f"  Image 1: {image_path_1.name}")
        print(f"  Image 2: {image_path_2.name}")
        print(f"  Correspondences: {len(pair['pts1'])}")
        print(f"  Co-visibility: {pair['covis']}")
        print(f"  Image 1 exists: {image_path_1.exists()}")
        print(f"  Image 2 exists: {image_path_2.exists()}")
        if len(pair["pts1"]) > 0:
            print(
                "  pts1 range: "
                f"x=[{pair['pts1'][:, 0].min():.0f}, {pair['pts1'][:, 0].max():.0f}], "
                f"y=[{pair['pts1'][:, 1].min():.0f}, {pair['pts1'][:, 1].max():.0f}]"
            )


if __name__ == "__main__":
    main()
