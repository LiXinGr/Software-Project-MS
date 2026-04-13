#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
RAW_ROOT="/home.stud/gorbuden/datagrid/MegaDepth_full/Undistorted_SfM"
CACHE_ROOT="$ROOT/data/gf_cache_stage2_151"
LOG_DIR="$ROOT/logs"
MPL_DIR="/tmp/mpl_stage2_151"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SESSION_PREFIX="stage2_151_e1_g"

SCENE_FILES=(
  "$ROOT/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_151_shard0.txt"
  "$ROOT/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_151_shard1.txt"
  "$ROOT/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_151_shard2.txt"
  "$ROOT/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_151_shard3.txt"
)
GPUS=(0 1 2 3)

mkdir -p "$LOG_DIR" "$MPL_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[LAUNCH] Missing python env at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "[LAUNCH] Missing MegaDepth root at $RAW_ROOT" >&2
  exit 1
fi

for file in "${SCENE_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[LAUNCH] Missing shard file: $file" >&2
    exit 1
  fi
done

if ! command -v screen >/dev/null 2>&1; then
  echo "[LAUNCH] screen is not available in PATH" >&2
  exit 1
fi

if ps -eo cmd | grep -F "scripts/cache_gluefactory_features.py" | grep -v grep >/dev/null 2>&1; then
  echo "[LAUNCH] cache_gluefactory_features.py is already running. Refusing to launch duplicate jobs." >&2
  ps -eo pid,etimes,%cpu,%mem,cmd | grep -F "scripts/cache_gluefactory_features.py" | grep -v grep >&2 || true
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  session="${SESSION_PREFIX}${gpu}"
  if screen -ls | grep -F ".$session" >/dev/null 2>&1; then
    echo "[LAUNCH] Screen session $session already exists. Refusing to reuse it." >&2
    exit 1
  fi
done

echo "[LAUNCH] GPU sanity check"
PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count()); [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

echo "[LAUNCH] Existing cache sessions outside this launcher:"
screen -ls | grep -E 'cache151_g[0-3]' || true

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  session="${SESSION_PREFIX}${gpu}"
  scene_file="${SCENE_FILES[$idx]}"
  log_file="$LOG_DIR/cache_stage2_151_shard${idx}_${RUN_TS}.log"

  screen -dmS "$session" bash -lc "
    set -euo pipefail
    cd '$ROOT'
    export PYTHONNOUSERSITE=1
    export HF_HOME=/home.stud/gorbuden/.cache/huggingface
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export DIFFUSERS_OFFLINE=1
    export MPLCONFIGDIR='$MPL_DIR'
    export CUDA_VISIBLE_DEVICES=${gpu}
    {
      echo '[CACHE] Started shard ${idx} on GPU ${gpu}: '\"\$(date)\"
      '$PYTHON_BIN' scripts/cache_gluefactory_features.py \
        --raw_root '$RAW_ROOT' \
        --scenes_file '$scene_file' \
        --device cuda:0 \
        --cache_root '$CACHE_ROOT' \
        --max_num_keypoints 2048 \
        --dift_ensemble_size 1 \
        --skip_existing
      echo '[CACHE] Finished shard ${idx} on GPU ${gpu}: '\"\$(date)\"
    } 2>&1 | tee '$log_file'
  "

  echo "[LAUNCH] shard=${idx} gpu=${gpu} session=${session} log=${log_file}"
done

echo "[LAUNCH] Active sessions:"
screen -ls | grep -F "$SESSION_PREFIX" || true
echo "[LAUNCH] Attach with: screen -r ${SESSION_PREFIX}0"
