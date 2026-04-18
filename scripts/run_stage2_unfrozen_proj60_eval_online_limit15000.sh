#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_stage2_unfrozen_proj_60_v2_online_limit15000_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

ensure_module_cmd() {
    if type module >/dev/null 2>&1; then
        return 0
    fi
    if [ -f /etc/profile.d/lmod.sh ]; then
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

clear_old_outputs() {
    local suffix
    local config_key
    for suffix in ft010 ft015; do
        config_key="stage2_unfrozen_proj_60_v2_${suffix}"
        echo "[$(date '+%F %T')] Clearing $config_key"
        rm -rf "$PROJECT_ROOT/output/matches/$config_key"
        rm -f "$PROJECT_ROOT"/output/benchmarks/"${config_key}"_*.h5
        rm -rf "$PROJECT_ROOT/output/results/$config_key"
        rm -rf "$PROJECT_ROOT/cache/features/${config_key}_proj_cache"
    done
}

ensure_module_cmd
load_anaconda_module
export PS1="${PS1-}"
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
    echo "checkpoint=$TRAIN_CKPT"
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
        --online_extraction \
        --limit 15000 \
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
    )

    "$REPOSED_PY" "$PROJECT_ROOT/scripts/reconstruct_results_csv.py" \
        --input-json "$PROJECT_ROOT/output/results/$config_key/$scene/calibrated-${config_key}_${scene}.json" \
        --output-csv "$PROJECT_ROOT/output/results/$config_key/$scene/results_${config_key}_${scene}.csv" \
        --matcher "$config_key" \
        --depth UniDepth \
        --exp-type calibrated \
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

print_results() {
    PROJECT_ROOT="$PROJECT_ROOT" "$DINOV3_PY" - <<'PY'
import json
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
prefix = "stage2_unfrozen_proj_60_v2"

print()
print("==========================================")
print("  UNFROZEN PROJECTION HEAD 60sc RESULTS")
print("==========================================")

summary = {}
for suffix in ("ft010", "ft015"):
    config_key = f"{prefix}_{suffix}"
    summary[config_key] = {}
    print()
    print(f"--- Threshold: {suffix} ---")
    for scene in ("sacre_coeur", "reichstag", "st_peters_square"):
        result_path = project_root / "output" / "results" / config_key / scene / f"calibrated-{config_key}_{scene}.json"
        try:
            data = json.loads(result_path.read_text())
            maa = data["3p_ours_shift_scale+12"]["mAA_10"]
            summary[config_key][scene] = maa
            print(f"  {scene}: {maa:.1f}%")
        except Exception as exc:
            summary[config_key][scene] = None
            print(f"  {scene}: FAILED ({exc})")

print()
print("=== COMPARISON TABLE ===")
print("| Config | sacre | reichstag | st_peters | avg |")
print("|---|---|---|---|---|")
print("| Frozen 60sc from-scratch (4c) | 85.3 | 82.9 | 74.9 | 81.0 |")
print("| Frozen 151sc (4d)             | 85.5 | 83.6 | 75.0 | 81.4 |")

for config_key, results in summary.items():
    vals = [v for v in results.values() if v is not None]
    avg = sum(vals) / len(vals) if len(vals) == 3 else None
    def fmt(v):
        return f"{v:.1f}" if v is not None else "N/A"
    label = "ft010" if config_key.endswith("ft010") else "ft015"
    print(
        f"| Unfrozen 60sc ({label})        | "
        f"{fmt(results.get('sacre_coeur'))} | {fmt(results.get('reichstag'))} | "
        f"{fmt(results.get('st_peters_square'))} | {fmt(avg)} |"
    )

print("| SP+LG verified                | 86.0 | 83.8 | 75.2 | 81.7 |")
PY
}

echo "[$(date '+%F %T')] Starting comparable online evaluation with checkpoint $TRAIN_CKPT"
clear_old_outputs
run_threshold stage2_unfrozen_proj_60_v2_ft010 0.10
run_threshold stage2_unfrozen_proj_60_v2_ft015 0.15
print_results
echo "[$(date '+%F %T')] Evaluation complete"
