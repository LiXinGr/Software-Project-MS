#!/bin/bash
set -euo pipefail

ROOT="/home.stud/gorbuden/datagrid/Software-Project-MS"
PYTHON_BIN="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

module load Anaconda3/2020.07
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dinov3
set -u
export MPLCONFIGDIR=/tmp/mpl_phase4

cd "$ROOT"
"$PYTHON_BIN" scripts/setup_lightglue_training_assets.py

for entry in "0:0" "1:2" "2:3"; do
    SHARD_IDX="${entry%%:*}"
    GPU_ID="${entry##*:}"
    SESSION="phase4_cache_g${GPU_ID}"
    LOG_FILE="$LOG_DIR/phase4_cache_shard${SHARD_IDX}_${TIMESTAMP}.log"
    screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
    screen -dmS "$SESSION" bash -lc "
        module load Anaconda3/2020.07
        set +u
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate dinov3
        set -u
        export MPLCONFIGDIR=/tmp/mpl_phase4
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        cd '$ROOT'
        '$PYTHON_BIN' scripts/cache_gluefactory_features.py \
            --scenes_file '$ROOT/data/lightglue_training/scenes_shard_${SHARD_IDX}.txt' \
            --cache_root '$ROOT/data/gf_cache' \
            --max_num_keypoints 2048 \
            --device cuda \
            --skip_existing \
            > '$LOG_FILE' 2>&1
    "
    echo "[CACHE-LAUNCH] shard=${SHARD_IDX} gpu=${GPU_ID} session=${SESSION} log=${LOG_FILE}"
done

echo "[CACHE-LAUNCH] screen sessions:"
screen -ls | grep phase4_cache_ || true
