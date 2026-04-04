from __future__ import annotations

import argparse
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMLIST = Path("/mnt/datagrid/public_datasets/revisitop1m/imlist.txt")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "revisitop1m_stage1"


def load_image_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in items))


def prefix_root(items: list[str]) -> list[str]:
    prefixed = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.startswith("jpg/"):
            prefixed.append(item)
        else:
            prefixed.append(f"jpg/{item}")
    return prefixed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic revisitop1m train/val/test manifests for Stage 1 homography pretraining."
    )
    parser.add_argument("--imlist", type=Path, default=DEFAULT_IMLIST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_size", type=int, default=150_000)
    parser.add_argument("--val_size", type=int, default=10_000)
    parser.add_argument("--test_size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = load_image_list(args.imlist)
    required = args.train_size + args.val_size + args.test_size
    if len(images) < required:
        raise RuntimeError(
            f"Not enough images in {args.imlist}: need {required}, found {len(images)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(images)

    train_images = images[: args.train_size]
    val_start = args.train_size
    val_end = val_start + args.val_size
    val_images = images[val_start:val_end]
    test_images = images[val_end : val_end + args.test_size]

    write_list(args.output_dir / "train_150k.txt", train_images)
    write_list(args.output_dir / "val_10k.txt", val_images)
    write_list(args.output_dir / "test_10k.txt", test_images)
    write_list(args.output_dir / "train.txt", prefix_root(train_images))
    write_list(args.output_dir / "val.txt", prefix_root(val_images))
    write_list(args.output_dir / "test.txt", prefix_root(test_images))
    write_list(
        args.output_dir / "all_170k.txt",
        prefix_root(train_images + val_images + test_images),
    )

    print(
        f"[SPLIT] seed={args.seed} train={len(train_images)} val={len(val_images)} "
        f"test={len(test_images)} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
