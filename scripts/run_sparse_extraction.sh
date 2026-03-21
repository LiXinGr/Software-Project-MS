#!/bin/bash
# Build sparse MegaDepth training bundles for the 10 training scenes.
# Launch: screen -S sparse_extract ./scripts/run_sparse_extraction.sh

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

OUTPUT_DIR="data/sparse_train"
LOG_FILE="$OUTPUT_DIR/extract_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$OUTPUT_DIR"

echo "[$(date '+%F %T')] Starting sparse extraction" | tee "$LOG_FILE"
echo "[$(date '+%F %T')] Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"

python -u scripts/extract_sparse_training_data.py \
    --output_dir "$OUTPUT_DIR" \
    --scenes 0080 0042 0380 0000 0366 0001 0005 0237 0011 0148 \
    --progress_every 100 \
    2>&1 | while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%F %T')" "$line"
    done | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] Sparse extraction complete" | tee -a "$LOG_FILE"
