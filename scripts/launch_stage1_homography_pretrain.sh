#!/bin/bash
set -euo pipefail

ROOT="/home.stud/gorbuden/datagrid/Software-Project-MS"
GF_ROOT="$ROOT/external/glue-factory"
PYTHON_BIN="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
EXPERIMENT_NAME="${1:-phase4_dinov3_lg_homography_stage1_v1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/phase4_stage1_training_${TIMESTAMP}.log"
MANIFEST="$ROOT/data/revisitop1m_stage1/train_150k.txt"
CACHE_ROOT="$ROOT/data/gf_cache_stage1_revisitop1m"
mkdir -p "$LOG_DIR"

if [ ! -f "$MANIFEST" ]; then
    echo "[TRAIN-STAGE1] Missing Stage 1 train manifest. Run scripts/create_revisitop1m_stage1_splits.py first." >&2
    exit 1
fi

if [ ! -d "$CACHE_ROOT" ]; then
    echo "[TRAIN-STAGE1] Missing Stage 1 cache root at $CACHE_ROOT" >&2
    exit 1
fi

module load Anaconda3/2020.07
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dinov3
set -u

export CUDA_VISIBLE_DEVICES=0,2,3
export MPLCONFIGDIR=/tmp/mpl_phase4_stage1
export PYTHONNOUSERSITE=1

cd "$GF_ROOT"
echo "[TRAIN-STAGE1] Using batch_size=96 for an exact 32 samples per rank on 3 GPUs." | tee "$LOG_FILE"
exec "$PYTHON_BIN" -m gluefactory.train "$EXPERIMENT_NAME" \
    --conf gluefactory/configs/dinov3_lightglue_homography.yaml \
    --distributed \
    --mixed_precision float16 \
    2>&1 | tee -a "$LOG_FILE"
