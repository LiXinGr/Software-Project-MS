#!/bin/bash
# Phase 2A end-to-end pipeline: sparse extraction if needed, then training.
# Launch: screen -S phase2a ./scripts/run_phase2a_full.sh

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

SCENES=(0080 0042 0380 0000 0366 0001 0005 0237 0011 0148)
SPARSE_DIR="data/sparse_train"
OUTPUT_DIR="experiments/p2_projection_v1"
LOG_FILE="$OUTPUT_DIR/phase2a_$(date +%Y%m%d_%H%M%S).txt"
LATEST_LOG="$OUTPUT_DIR/phase2a_latest.log"
mkdir -p "$SPARSE_DIR" "$OUTPUT_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"

echo "[$(date '+%F %T')] Starting Phase 2A pipeline" | tee "$LOG_FILE"

missing=0
for scene in "${SCENES[@]}"; do
    if [ ! -f "$SPARSE_DIR/$scene.pt" ]; then
        missing=1
        break
    fi
done

if [ "$missing" -eq 1 ]; then
    echo "[$(date '+%F %T')] Sparse bundles missing; running extraction first" | tee -a "$LOG_FILE"
    python -u scripts/extract_sparse_training_data.py \
        --output_dir "$SPARSE_DIR" \
        --scenes "${SCENES[@]}" \
        --progress_every 100 \
        2>&1 | while IFS= read -r line; do
            printf '[%s] %s\n' "$(date '+%F %T')" "$line"
        done | tee -a "$LOG_FILE"
else
    echo "[$(date '+%F %T')] Sparse bundles already present; skipping extraction" | tee -a "$LOG_FILE"
fi

echo "[$(date '+%F %T')] Starting training" | tee -a "$LOG_FILE"
python -u scripts/train_projection_head.py \
    --sparse_dir "$SPARSE_DIR" \
    --scenes "${SCENES[@]}" \
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

echo "[$(date '+%F %T')] Phase 2A pipeline complete" | tee -a "$LOG_FILE"
