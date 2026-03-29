#!/bin/bash
# Check MegaDepth feature precomputation progress at the feature-key level.
# Usage: ./scripts/check_precompute_progress.sh [--scenes 0080 0042 ...]

set -euo pipefail

REQUESTED_OUTPUT_DIR="/mnt/datagrid/gorbuden/megadepth_features"
FALLBACK_OUTPUT_DIR="/mnt/datagrid/personal/gorbuden/megadepth_features"
OUTPUT_DIR="${MEGADEPTH_OUTPUT_DIR:-$REQUESTED_OUTPUT_DIR}"
PYTHON_BIN="${MEGADEPTH_PROGRESS_PYTHON:-/home.stud/gorbuden/.conda/envs/train/bin/python}"
SCENE_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --scenes)
            shift
            while [ "$#" -gt 0 ]; do
                SCENE_ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: ./scripts/check_precompute_progress.sh [--scenes 0080 0042 ...]" >&2
            exit 1
            ;;
    esac
done

if [ ! -d "$OUTPUT_DIR" ] && [ -d "$FALLBACK_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$FALLBACK_OUTPUT_DIR"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

exec "$PYTHON_BIN" - "$OUTPUT_DIR" "${SCENE_ARGS[@]}" <<'PY'
from __future__ import annotations

import sys
import zipfile
from datetime import datetime
from pathlib import Path

output_dir = Path(sys.argv[1])

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

default_scenes = ["0080", "0042", "0380", "0000", "0366", "0001", "0005", "0237", "0011", "0148"]
expected_recon = {
    "0080": 6494,
    "0042": 4079,
    "0380": 3898,
    "0000": 3803,
    "0366": 3735,
    "0001": 3409,
    "0005": 3167,
    "0237": 3054,
    "0011": 2966,
    "0148": 2682,
}
expected_on_disk = {
    "0080": 5136,
    "0042": 4079,
    "0380": 3898,
    "0000": 3803,
    "0366": 3735,
    "0001": 3409,
    "0005": 3167,
    "0237": 3054,
    "0011": 2966,
    "0148": 2682,
}

requested_scenes = sys.argv[2:]
if requested_scenes:
    unknown = [scene for scene in requested_scenes if scene not in expected_on_disk]
    if unknown:
        print(f"Unknown scenes: {' '.join(unknown)}", file=sys.stderr)
        print(f"Known scenes: {' '.join(default_scenes)}", file=sys.stderr)
        raise SystemExit(1)
    scenes = requested_scenes
else:
    scenes = default_scenes

print("=== Precomputation Progress ===")
print(f"Output dir: {output_dir}")
print(f"Scenes: {' '.join(scenes)}")
print("Legend: both=dinov3+dift, dino_only=only dinov3, dift_only=only dift, bad=unreadable npz")

total_expected = 0
total_files = 0
total_both = 0
total_dino_only = 0
total_dift_only = 0
total_other = 0
total_bad = 0
complete_both = []
complete_any = []
mtime_values = []

for scene in scenes:
    scene_dir = output_dir / scene
    expected = expected_on_disk[scene]
    total_expected += expected

    total = both = dino_only = dift_only = other = bad = 0
    newest = None

    if scene_dir.is_dir():
        for path in scene_dir.glob("*.npz"):
            total += 1
            try:
                with zipfile.ZipFile(path) as zf:
                    names = set(zf.namelist())
                has_dino = "dinov3.npy" in names
                has_dift = "dift.npy" in names
                if has_dino and has_dift:
                    both += 1
                elif has_dino:
                    dino_only += 1
                elif has_dift:
                    dift_only += 1
                else:
                    other += 1
            except Exception:
                bad += 1

            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None:
                mtime_values.append(mtime)
                if newest is None or mtime > newest:
                    newest = mtime

    total_files += total
    total_both += both
    total_dino_only += dino_only
    total_dift_only += dift_only
    total_other += other
    total_bad += bad

    if both >= expected and dino_only == 0 and dift_only == 0 and other == 0 and bad == 0:
        status = "COMPLETE_BOTH"
        complete_both.append(scene)
    elif total >= expected:
        status = "FILES_COMPLETE"
        complete_any.append(scene)
    elif total > 0:
        status = "IN_PROGRESS"
    else:
        status = "PENDING"

    pct_files = 100.0 * total / expected if expected else 0.0
    pct_both = 100.0 * both / expected if expected else 0.0
    newest_str = "-" if newest is None else datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  {scene}: files={total}/{expected} ({pct_files:.1f}%), "
        f"both={both} ({pct_both:.1f}%), dino_only={dino_only}, dift_only={dift_only}, "
        f"other={other}, bad={bad}, recon={expected_recon[scene]} {status}"
    , flush=True)
    print(f"    newest={newest_str}", flush=True)

print()
print(
    f"Totals: files={total_files}/{total_expected}, both={total_both}/{total_expected}, "
    f"dino_only={total_dino_only}, dift_only={total_dift_only}, other={total_other}, bad={total_bad}"
, flush=True)

if complete_both:
    print("Scenes complete with both features:", " ".join(complete_both), flush=True)
else:
    print("Scenes complete with both features: none", flush=True)

if complete_any:
    print("Scenes with all files present but not fully complete:", " ".join(complete_any), flush=True)
else:
    print("Scenes with all files present but not fully complete: none", flush=True)

if mtime_values:
    first_ts = min(mtime_values)
    last_ts = max(mtime_values)
    print(f"First file: {datetime.fromtimestamp(first_ts):%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"Last file:  {datetime.fromtimestamp(last_ts):%Y-%m-%d %H:%M:%S}", flush=True)

    elapsed_sec = max(1.0, last_ts - first_ts)
    if total_both > 0:
        rate_per_hour = total_both * 3600.0 / elapsed_sec
        remaining_both = max(0, total_expected - total_both)
        eta_sec = remaining_both * elapsed_sec / total_both
        finish_ts = last_ts + eta_sec
        print(f"Observed full-feature throughput: {rate_per_hour:.1f} images/hour", flush=True)
        print(f"Estimated remaining for both features: {eta_sec / 3600.0:.2f} hours", flush=True)
        print(f"Estimated finish: {datetime.fromtimestamp(finish_ts):%Y-%m-%d %H:%M:%S}", flush=True)
    else:
        print("Observed full-feature throughput: not enough complete files yet", flush=True)
else:
    print("No npz files found yet", flush=True)
PY
