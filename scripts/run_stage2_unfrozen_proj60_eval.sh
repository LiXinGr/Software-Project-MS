#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_stage2_unfrozen_proj_60_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

ensure_module_cmd() {
    if type module >/dev/null 2>&1; then
        return 0
    fi
    if [ -f /etc/profile.d/lmod.sh ]; then
        # Lmod is not guaranteed to be initialized in a fresh non-interactive shell.
        # Source it explicitly so `module load ...` works when this script is launched via screen.
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    type module >/dev/null 2>&1
}

load_anaconda_module() {
    local module_name
    for module_name in Anaconda3/2020.07 Anaconda3/2022.10 Anaconda3/2024.02-1; do
        if module load "$module_name" >/dev/null 2>&1; then
            return 0
        fi
    done
    echo "Failed to load any supported Anaconda3 module" >&2
    return 1
}

ensure_module_cmd
load_anaconda_module
export PS1="${PS1-}"
# conda.sh assumes PS1 exists; disable nounset just while sourcing it.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate dinov3

export PYTHONNOUSERSITE=1
export HF_HOME=/home.stud/gorbuden/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

DINOV3_PY="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
REPOSED_PY="/home.stud/gorbuden/.conda/envs/reposed/bin/python"
TRAIN_CKPT="$PROJECT_ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar"

if [ ! -f "$TRAIN_CKPT" ]; then
    echo "Missing checkpoint: $TRAIN_CKPT" >&2
    exit 1
fi

if [ "${EVAL_DRY_RUN:-0}" = "1" ]; then
    echo "[$(date '+%F %T')] Dry-run startup OK"
    echo "log_file=$LOG_FILE"
    echo "dinov3_python=$DINOV3_PY"
    echo "reposed_python=$REPOSED_PY"
    exit 0
fi

run_scene() {
    local gpu="$1"
    local scene="$2"
    local config_key="$3"
    local threshold="$4"

    echo "[$(date '+%F %T')] scene=$scene gpu=$gpu config_key=$config_key threshold=$threshold"

    CUDA_VISIBLE_DEVICES="$gpu" "$DINOV3_PY" "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
        --scene "$scene" \
        --config_key "$config_key" \
        --checkpoint "$TRAIN_CKPT" \
        --lightglue_checkpoint "$TRAIN_CKPT" \
        --filter_threshold "$threshold" \
        --seed 42 \
        --max_points 2048 \
        --source_cache_max_points 2000 \
        --feat_level -8 \
        --img_size 768 768 \
        --t 0 \
        --up_ft_index 2 \
        --ensemble_size 8 \
        --device cuda

    "$REPOSED_PY" "$PROJECT_ROOT/scripts/pack_benchmark.py" \
        --matches_dir "$PROJECT_ROOT/output/matches/$config_key/$scene" \
        --depth_dir "$PROJECT_ROOT/datasets/phototourism/$scene/depth_unidepth" \
        --sparse_dir "$PROJECT_ROOT/datasets/phototourism/$scene/dense/sparse" \
        --pairs_file "$PROJECT_ROOT/output/pairs_${scene}.txt" \
        --output "$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5"

    (
        cd "$PROJECT_ROOT/external/RePoseD"
        "$REPOSED_PY" eval.py "$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5" \
            -nw 8 --thesis \
            --output_dir "$PROJECT_ROOT/output/results/$config_key/$scene" \
            --preprocess_info "$PROJECT_ROOT/datasets/phototourism/$scene/images_preprocessed/preprocess_info.json"

        "$REPOSED_PY" eval_shared_f.py "$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5" \
            -nw 8 --thesis \
            --output_dir "$PROJECT_ROOT/output/results/$config_key/$scene" \
            --preprocess_info "$PROJECT_ROOT/datasets/phototourism/$scene/images_preprocessed/preprocess_info.json" || true

        "$REPOSED_PY" eval_varying_f.py "$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5" \
            -nw 8 --thesis \
            --output_dir "$PROJECT_ROOT/output/results/$config_key/$scene" \
            --preprocess_info "$PROJECT_ROOT/datasets/phototourism/$scene/images_preprocessed/preprocess_info.json" || true
    )

    CONFIG_KEY="$config_key" SCENE="$scene" PROJECT_ROOT="$PROJECT_ROOT" "$DINOV3_PY" - <<'PY'
import json
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
config_key = os.environ["CONFIG_KEY"]
scene = os.environ["SCENE"]
results_dir = project_root / "output" / "results" / config_key / scene

paths = [
    (results_dir / f"calibrated-{config_key}_{scene}.json", "calibrated"),
    (results_dir / f"shared_focal-{config_key}_{scene}.json", "shared_f"),
    (results_dir / f"varying_focal-{config_key}_{scene}.json", "varying_f"),
]

combined = []
for path, exp_type in paths:
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    for row in data:
        if isinstance(row, dict):
            row["exp_type"] = exp_type
    combined.extend(data)

output_path = results_dir / f"results_{config_key}_{scene}.json"
output_path.write_text(json.dumps(combined))
print(output_path)
PY

    "$DINOV3_PY" "$PROJECT_ROOT/scripts/reconstruct_results_csv.py" \
        --input-json "$PROJECT_ROOT/output/results/$config_key/$scene/results_${config_key}_${scene}.json" \
        --output-csv "$PROJECT_ROOT/output/results/$config_key/$scene/results_${config_key}_${scene}.csv" \
        --matcher "$config_key" \
        --depth UniDepth \
        --max-points 2048 \
        --img-size 768 \
        --feat-level -8 \
        --up-ft-index 2 \
        --dift-t 0

    echo "[$(date '+%F %T')] done scene=$scene config_key=$config_key"
}

run_threshold() {
    local config_key="$1"
    local threshold="$2"

    run_scene 0 sacre_coeur "$config_key" "$threshold" &
    local pid0=$!
    run_scene 1 reichstag "$config_key" "$threshold" &
    local pid1=$!
    run_scene 2 st_peters_square "$config_key" "$threshold" &
    local pid2=$!

    wait "$pid0"
    wait "$pid1"
    wait "$pid2"
}

echo "[$(date '+%F %T')] Starting evaluation with checkpoint $TRAIN_CKPT"
run_threshold stage2_unfrozen_proj_60_v2_ft010 0.10
run_threshold stage2_unfrozen_proj_60_v2_ft015 0.15
echo "[$(date '+%F %T')] Evaluation complete"
