from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GF_ROOT = PROJECT_ROOT / "external" / "glue-factory"
SCENE_LIST_ROOT = GF_ROOT / "gluefactory" / "datasets" / "megadepth_scene_lists"
SCENE_INFO_ROOT = GF_ROOT / "data" / "megadepth_phase4" / "scene_info"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "lightglue_training"


@dataclass
class SceneStats:
    scene: str
    num_images: int
    num_pairs: int


def load_scene_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def collect_scene_stats(scene_list: list[str]) -> list[SceneStats]:
    stats: list[SceneStats] = []
    for scene in scene_list:
        path = SCENE_INFO_ROOT / f"{scene}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared scene info: {path}")
        info = np.load(path, allow_pickle=True)
        mat = info["overlap_matrix"]
        mask = (mat > 0.1) & (mat <= 0.7)
        num_pairs = int(np.triu(mask, 1).sum())
        num_images = int(len(info["image_paths"]))
        stats.append(SceneStats(scene=scene, num_images=num_images, num_pairs=num_pairs))
    return stats


def balance_scene_shards(stats: list[SceneStats], num_shards: int) -> list[list[SceneStats]]:
    shards: list[list[SceneStats]] = [[] for _ in range(num_shards)]
    shard_loads = [0 for _ in range(num_shards)]
    for stat in sorted(stats, key=lambda item: item.num_images, reverse=True):
        shard_idx = min(range(num_shards), key=lambda idx: shard_loads[idx])
        shards[shard_idx].append(stat)
        shard_loads[shard_idx] += stat.num_images
    for shard in shards:
        shard.sort(key=lambda item: item.scene)
    return shards


def write_scene_shards(shards: list[list[SceneStats]], output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, shard in enumerate(shards):
        path = output_root / f"scenes_shard_{idx}.txt"
        path.write_text("".join(f"{item.scene}\n" for item in shard))
        written.append(path)
    return written


def build_validation_pairs(
    scene_list: list[str],
    output_path: Path,
    pairs_per_scene: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    all_lines: list[str] = []
    for scene in scene_list:
        info = np.load(SCENE_INFO_ROOT / f"{scene}.npz", allow_pickle=True)
        mat = info["overlap_matrix"]
        image_paths = info["image_paths"]
        candidates = np.argwhere(np.triu((mat > 0.1) & (mat <= 0.7), 1))
        if len(candidates) == 0:
            continue
        if len(candidates) > pairs_per_scene:
            selected_idx = rng.choice(len(candidates), size=pairs_per_scene, replace=False)
            candidates = candidates[selected_idx]
        for i, j in candidates:
            path0 = str(image_paths[int(i)]).replace("Undistorted_SfM/", "", 1)
            path1 = str(image_paths[int(j)]).replace("Undistorted_SfM/", "", 1)
            all_lines.append(f"{path0} {path1}")
    output_path.write_text("\n".join(all_lines) + "\n")
    return len(all_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shard files and fixed validation pairs for LightGlue training.")
    parser.add_argument(
        "--scene_list",
        type=Path,
        default=SCENE_LIST_ROOT / "train_scenes_datagrid_accessible60.txt",
    )
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--val_pairs_per_scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scene_list = load_scene_list(args.scene_list)
    stats = collect_scene_stats(scene_list)
    shards = balance_scene_shards(stats, args.num_shards)
    shard_paths = write_scene_shards(shards, OUTPUT_ROOT)

    val_pairs_path = SCENE_LIST_ROOT / "valid_pairs_phase4_60.txt"
    num_val_pairs = build_validation_pairs(
        scene_list,
        val_pairs_path,
        pairs_per_scene=args.val_pairs_per_scene,
        seed=args.seed,
    )

    summary = {
        "scene_list": str(args.scene_list),
        "num_scenes": len(scene_list),
        "num_shards": args.num_shards,
        "val_pairs_per_scene": args.val_pairs_per_scene,
        "num_val_pairs": num_val_pairs,
        "shards": [
            {
                "index": idx,
                "num_scenes": len(shard),
                "num_images": int(sum(item.num_images for item in shard)),
                "num_pairs": int(sum(item.num_pairs for item in shard)),
                "scenes": [item.scene for item in shard],
            }
            for idx, shard in enumerate(shards)
        ],
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[ASSETS] scenes={len(scene_list)} val_pairs={num_val_pairs}")
    for idx, shard in enumerate(summary["shards"]):
        print(
            f"[ASSETS] shard {idx}: scenes={shard['num_scenes']} "
            f"images={shard['num_images']} pairs={shard['num_pairs']} "
            f"path={shard_paths[idx]}",
            flush=True,
        )
    print(f"[ASSETS] validation_pairs={val_pairs_path}", flush=True)
    print(f"[ASSETS] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
