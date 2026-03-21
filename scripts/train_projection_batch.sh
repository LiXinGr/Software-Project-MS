#!/bin/bash
# Full projection-head training from sparse scene bundles.
# Launch: screen -S train ./scripts/train_projection_batch.sh

set -euo pipefail

load_anaconda() {
    local module_name
    for module_name in Anaconda3/2020.07 Anaconda3/2022.10 Anaconda3/2024.02-1; do
        if module load "$module_name" >/dev/null 2>&1; then
            echo "Using $module_name"
            return 0
        fi
    done
    echo "Failed to load a supported Anaconda3 module" >&2
    return 1
}

load_anaconda

CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate train
set -u

cd /home.stud/gorbuden/datagrid/Software-Project-MS

OUTPUT_DIR="experiments/p2_projection_v1"
LOG_DIR="$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/train_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] Starting sparse projection training" | tee "$LOG_FILE"
echo "[$(date '+%F %T')] Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"

python -u scripts/train_projection_head.py \
    --sparse_dir data/sparse_train \
    --scenes 0080 0042 0380 0000 0366 0001 0005 0237 0011 0148 \
    --epochs 10 \
    --pairs_per_epoch 50000 \
    --val_pairs_per_epoch 1000 \
    --lr 1e-3 \
    --temperature 0.07 \
    --num_correspondences 512 \
    --output_dir "$OUTPUT_DIR" \
    --log_interval 500 \
    --device cuda:0 \
    --num_workers 0 \
    --seed 42 \
    2>&1 | while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%F %T')" "$line"
    done | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] Training complete" | tee -a "$LOG_FILE"
