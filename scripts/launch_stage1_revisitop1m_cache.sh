#!/bin/bash
set -euo pipefail

ROOT="/home.stud/gorbuden/datagrid/Software-Project-MS"
PYTHON_BIN="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
MANIFEST="${1:-$ROOT/data/revisitop1m_stage1/all_170k.txt}"
CACHE_ROOT="${2:-$ROOT/data/gf_cache_stage1_revisitop1m}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

module load Anaconda3/2020.07
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dinov3
set -u
export MPLCONFIGDIR=/tmp/mpl_phase4_stage1
export PYTHONNOUSERSITE=1

cd "$ROOT"
if [ ! -f "$MANIFEST" ]; then
    "$PYTHON_BIN" scripts/create_revisitop1m_stage1_splits.py
fi

NUM_WORKERS=3
for entry in "0:0" "1:2" "2:3"; do
    WORKER_IDX="${entry%%:*}"
    GPU_ID="${entry##*:}"
    SESSION="phase4_stage1_cache_g${GPU_ID}"
    LOG_FILE="$LOG_DIR/phase4_stage1_cache_w${WORKER_IDX}_${TIMESTAMP}.log"
    screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
    screen -dmS "$SESSION" bash -lc "
        module load Anaconda3/2020.07
        set +u
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate dinov3
        set -u
        export MPLCONFIGDIR=/tmp/mpl_phase4_stage1
        export PYTHONNOUSERSITE=1
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        cd '$ROOT'
        '$PYTHON_BIN' scripts/cache_revisitop1m_features.py \
            --image_list '$MANIFEST' \
            --image_root '/mnt/datagrid/public_datasets/revisitop1m' \
            --cache_root '$CACHE_ROOT' \
            --max_num_keypoints 2048 \
            --gpu_id ${GPU_ID} \
            --device cuda \
            --skip_existing \
            --shard ${WORKER_IDX}/${NUM_WORKERS} \
            > '$LOG_FILE' 2>&1
    "
    echo "[CACHE-STAGE1-LAUNCH] worker=${WORKER_IDX} gpu=${GPU_ID} session=${SESSION} log=${LOG_FILE}"
done

echo "[CACHE-STAGE1-LAUNCH] screen sessions:"
screen -ls | grep phase4_stage1_cache_ || true
