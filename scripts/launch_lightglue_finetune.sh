#!/bin/bash
set -euo pipefail

ROOT="/home.stud/gorbuden/datagrid/Software-Project-MS"
GF_ROOT="$ROOT/external/glue-factory"
PYTHON_BIN="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
EXPERIMENT_NAME="${1:-phase4_dinov3_lg_full_v1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/phase4_training_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

if [ ! -f "$ROOT/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/valid_pairs_phase4_60.txt" ]; then
    echo "[TRAIN-LAUNCH] Missing validation pair list. Run scripts/setup_lightglue_training_assets.py first." >&2
    exit 1
fi

if [ "$(find "$ROOT/data/gf_cache" -mindepth 1 -maxdepth 1 -type d | wc -l)" -lt 60 ]; then
    echo "[TRAIN-LAUNCH] Cache appears incomplete under $ROOT/data/gf_cache" >&2
    exit 1
fi

module load Anaconda3/2020.07
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dinov3
set -u

export CUDA_VISIBLE_DEVICES=0,2,3
export MPLCONFIGDIR=/tmp/mpl_phase4

cd "$GF_ROOT"
echo "[TRAIN-LAUNCH] Effective per-rank batch will be floor(32/3)=10, global batch=30 with current Glue Factory DDP division." | tee "$LOG_FILE"
exec "$PYTHON_BIN" -m gluefactory.train "$EXPERIMENT_NAME" \
    --conf gluefactory/configs/dinov3_lightglue_megadepth.yaml \
    --distributed \
    --mixed_precision float16 \
    2>&1 | tee -a "$LOG_FILE"
