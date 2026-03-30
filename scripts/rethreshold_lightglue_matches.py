#!/usr/bin/env python3
"""Re-threshold saved LightGlue matches when score arrays are available."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-threshold saved LightGlue match archives")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--scene", default=None, help="Optional scene label for logging")
    return parser.parse_args()


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main() -> None:
    args = parse_args()
    scene = args.scene or args.input_dir.name

    files = sorted(args.input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {args.input_dir}")

    with np.load(files[0]) as first:
        keys = set(first.files)
        if "scores" not in keys:
            raise RuntimeError(
                f"Saved matches in {args.input_dir} do not contain scores. "
                "Threshold sweep is not possible for this run."
            )

        stored_filter = float(first["filter_threshold"]) if "filter_threshold" in keys else None
        scores_are_post_filter = bool(first["scores_are_post_filter"]) if "scores_are_post_filter" in keys else True

    if scores_are_post_filter and stored_filter is not None and args.threshold < stored_filter - 1e-9:
        raise RuntimeError(
            f"Requested threshold {args.threshold} is below the stored source filter {stored_filter}. "
            "These saved scores are already post-filter, so only more aggressive thresholds are valid."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    zero_match_pairs = 0
    written = 0

    for path in files:
        with np.load(path) as data:
            mkpts0 = data["mkpts0"]
            mkpts1 = data["mkpts1"]
            scores = data["scores"]
            keep = scores >= args.threshold

            payload: dict[str, np.ndarray] = {
                "mkpts0": mkpts0[keep],
                "mkpts1": mkpts1[keep],
                "scores": scores[keep],
                "filter_threshold": np.array(args.threshold, dtype=np.float32),
                "scores_are_post_filter": np.array(1, dtype=np.uint8),
            }
            for key in ("depth_confidence", "width_confidence", "stop"):
                if key in data.files:
                    payload[key] = data[key]

        num_matches = int(keep.sum())
        total_matches += num_matches
        zero_match_pairs += int(num_matches == 0)
        written += 1
        save_npz_atomic(args.output_dir / path.name, payload)

    avg_matches = total_matches / written if written else 0.0
    print(
        f"[SWEEP] {scene} threshold={args.threshold:.3f}: "
        f"pairs={written}, avg_matches={avg_matches:.1f}, zero_match_pairs={zero_match_pairs}",
        flush=True,
    )


if __name__ == "__main__":
    main()
