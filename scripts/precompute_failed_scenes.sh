#!/bin/bash
# Retry only the MegaDepth scenes that previously failed due to CUDA OOM.
# Launch: screen -S precompute_failed ./scripts/precompute_failed_scenes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

module load Anaconda3/2020.07
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate train
set -u

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/mpl
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCENES=(0080 0042 0380)
REQUESTED_OUTPUT_DIR="/mnt/datagrid/gorbuden/megadepth_features"
FALLBACK_OUTPUT_DIR="/mnt/datagrid/personal/gorbuden/megadepth_features"
OUTPUT_DIR="${MEGADEPTH_OUTPUT_DIR:-$REQUESTED_OUTPUT_DIR}"
MEGADEPTH_ROOT="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM"

if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
    OUTPUT_DIR="$FALLBACK_OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

LOG="$OUTPUT_DIR/precompute_failed_$(date +%Y%m%d_%H%M%S).txt"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$LOG"
}

run_pass() {
    local scene="$1"
    local mode="$2"
    shift 2

    log "Scene $scene | Mode $mode | Starting"
    set +e
    python -u "$PROJECT_ROOT/scripts/precompute_megadepth_features.py" \
        --megadepth_root "$MEGADEPTH_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --scenes "$scene" \
        --device cuda:0 \
        --skip_existing \
        --progress_every 100 \
        "$@" \
        2>&1 | while IFS= read -r line; do
            log "$line"
        done
    local status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
        log "Scene $scene | Mode $mode | FAILED with exit code $status"
        exit "$status"
    fi

    local count=0
    if [ -d "$OUTPUT_DIR/$scene" ]; then
        count="$(find "$OUTPUT_DIR/$scene" -maxdepth 1 -name '*.npz' | wc -l)"
    fi
    log "Scene $scene | Mode $mode | Done | npz_count=$count"
}

log "Starting failed-scene precomputation retry"
log "Project root: $PROJECT_ROOT"
log "MegaDepth root: $MEGADEPTH_ROOT"
log "Output dir: $OUTPUT_DIR"
if [ "$OUTPUT_DIR" != "$REQUESTED_OUTPUT_DIR" ]; then
    log "Requested output dir was not writable; using fallback $OUTPUT_DIR"
fi
log "Scenes: ${SCENES[*]}"
log "Allocator config: $PYTORCH_CUDA_ALLOC_CONF"
log "Strategy: fresh Python process per scene and per feature family"
log "Pass order: DINOv3 first, then DIFT"
log "Known permanent failure in prior run: 0080/13231930014_ed13cf616e_b.jpg was corrupt"
log "GPU snapshot:"
nvidia-smi | tee -a "$LOG"

for scene in "${SCENES[@]}"; do
    run_pass "$scene" "dinov3_only" --dinov3_only
    run_pass "$scene" "dift_only" --dift_only
done

log "Finished failed-scene retry"
du -sh "$OUTPUT_DIR" | tee -a "$LOG"
