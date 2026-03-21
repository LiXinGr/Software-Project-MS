#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/experiments/phase2_projection_v1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/projection_eval_${TIMESTAMP}.txt"
LATEST_LOG="$LOG_DIR/projection_eval_latest.log"
SCENES=("sacre_coeur" "reichstag" "st_peters_square")
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$LOG_DIR"
ln -sfn "$LOG_FILE" "$LATEST_LOG"

load_anaconda_module() {
    local module_name
    for module_name in Anaconda3/2020.07 Anaconda3/2022.10 Anaconda3/2024.02-1; do
        if module load "$module_name" >/dev/null 2>&1; then
            echo "Using $module_name"
            return 0
        fi
    done
    echo "Failed to load any supported Anaconda3 module" >&2
    return 1
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

{
    cd "$PROJECT_ROOT"
    load_anaconda_module
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_projection_eval}"

    log "Starting projection-head PhotoTourism benchmark"
    log "Project root: $PROJECT_ROOT"
    log "Device: $DEVICE"
    log "Checkpoint: $PROJECT_ROOT/experiments/phase2_projection_v1/best.pt"

    for scene in "${SCENES[@]}"; do
        run_id="phase2_projection_v1"
        log "=== Scene: $scene ==="
        ./run_thesis_benchmark.sh projection \
            --run_id "$run_id" \
            --scene "$scene" \
            --skip-depth \
            --device "$DEVICE"
        log "=== Scene complete: $scene ==="
    done

    log "Projection benchmark run finished"
} 2>&1 | tee "$LOG_FILE"
