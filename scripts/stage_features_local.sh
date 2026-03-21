#!/bin/bash
# Stage precomputed features to local NVMe for fast training.
# Run BEFORE training. /tmp is wiped on reboot.

set -e

SRC="/mnt/datagrid/personal/gorbuden/megadepth_features"
DST="/tmp/megadepth_features"
SFM_SRC="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM"
SFM_DST="/tmp/megadepth_sfm"

SCENES="0080 0042 0380 0000 0366 0001 0005 0237 0011 0148"

echo "Staging features to local NVMe..."
echo "Source: $SRC"
echo "Destination: $DST"
echo "Start: $(date)"

mkdir -p "$DST" "$SFM_DST"

for scene in $SCENES; do
    if [ -d "$DST/$scene" ]; then
        echo "  $scene: already staged, skipping"
    else
        echo "  $scene: copying features..."
        cp -r "$SRC/$scene" "$DST/$scene"
    fi

    echo "  $scene: staging SfM data..."
    mkdir -p "$SFM_DST/$scene"
    if [ ! -d "$SFM_DST/$scene/sparse" ]; then
        cp -r "$SFM_SRC/$scene/sparse" "$SFM_DST/$scene/"
    fi
    ln -sfn "$SFM_SRC/$scene/images" "$SFM_DST/$scene/images"

    echo "  $scene: done ($(du -sh "$DST/$scene" | cut -f1))"
done

echo "Staging complete: $(date)"
echo "Total staged: $(du -sh "$DST" | cut -f1)"
echo "Free space remaining: $(df -h /tmp | tail -1 | awk '{print $4}')"
