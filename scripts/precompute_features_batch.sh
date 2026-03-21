#!/bin/bash
# Feature precomputation for Phase 2 training.
# Launch: screen -S precompute ./scripts/precompute_features_batch.sh
# Monitor: ./scripts/check_precompute_progress.sh
# Expected: ~31 hours, ~1 TB on datagrid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

module load Anaconda3/2020.07
CONDA_BASE="$(conda info --base)"
# conda.sh assumes PS1 exists; temporarily disable nounset for non-interactive shells.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate train
set -u

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/mpl

SCENES="0080 0042 0380 0000 0366 0001 0005 0237 0011 0148"
REQUESTED_OUTPUT_DIR="/mnt/datagrid/gorbuden/megadepth_features"
FALLBACK_OUTPUT_DIR="/mnt/datagrid/personal/gorbuden/megadepth_features"
OUTPUT_DIR="${MEGADEPTH_OUTPUT_DIR:-$REQUESTED_OUTPUT_DIR}"
MEGADEPTH_ROOT="/mnt/datasets/MegaDepth/MegaDepth_v1_SfM"

if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
    OUTPUT_DIR="$FALLBACK_OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

LOG="$OUTPUT_DIR/precompute_log_$(date +%Y%m%d_%H%M%S).txt"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$LOG"
}

log "Starting precomputation"
log "Project root: $PROJECT_ROOT"
log "MegaDepth root: $MEGADEPTH_ROOT"
log "Output dir: $OUTPUT_DIR"
if [ "$OUTPUT_DIR" != "$REQUESTED_OUTPUT_DIR" ]; then
    log "Requested output dir was not writable; using fallback $OUTPUT_DIR"
fi
log "Scenes: $SCENES"
log "Mode: one pass with DINOv3 + DIFT"

set +e
python -u "$PROJECT_ROOT/scripts/precompute_megadepth_features.py" \
    --megadepth_root "$MEGADEPTH_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --scenes $SCENES \
    --device cuda:0 \
    --skip_existing \
    --progress_every 100 \
    2>&1 | while IFS= read -r line; do
        log "$line"
    done
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
    log "Precomputation failed with exit code $status"
    exit "$status"
fi

log "Finished precomputation"
log "Disk usage:"
du -sh "$OUTPUT_DIR" | tee -a "$LOG"
