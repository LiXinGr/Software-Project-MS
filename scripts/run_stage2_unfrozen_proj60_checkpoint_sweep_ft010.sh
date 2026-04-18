#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_stage2_unfrozen_proj_60_v2_checkpoint_sweep_ft010_$(date +%Y%m%d_%H%M%S).log"
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
CKPT_DIR="$PROJECT_ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu"
ORIG_PROJ_CKPT="$PROJECT_ROOT/experiments/phase2_projection_wide/best.pt"

declare -a SWEEP_EPOCHS=(41 44 49)
declare -A SWEEP_CKPTS=(
    [41]="checkpoint_41_92900.tar"
    [44]="checkpoint_44_99536.tar"
    [49]="checkpoint_49_110596.tar"
)
declare -A SWEEP_GPUS=(
    [41]=0
    [44]=1
    [49]=2
)

if [ "${EVAL_DRY_RUN:-0}" = "1" ]; then
    echo "[$(date '+%F %T')] Dry-run startup OK"
    echo "log_file=$LOG_FILE"
    for epoch in "${SWEEP_EPOCHS[@]}"; do
        echo "epoch=$epoch gpu=${SWEEP_GPUS[$epoch]} ckpt=${SWEEP_CKPTS[$epoch]}"
    done
    exit 0
fi

run_scene() {
    local gpu="$1"
    local train_ckpt="$2"
    local config_key="$3"
    local scene="$4"

    echo "[$(date '+%F %T')] scene=$scene gpu=$gpu config_key=$config_key checkpoint=$(basename "$train_ckpt")"

    CUDA_VISIBLE_DEVICES="$gpu" "$DINOV3_PY" "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
        --scene "$scene" \
        --config_key "$config_key" \
        --checkpoint "$train_ckpt" \
        --lightglue_checkpoint "$train_ckpt" \
        --filter_threshold 0.10 \
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

run_checkpoint() {
    local epoch="$1"
    local ckpt_name="$2"
    local gpu="$3"
    local train_ckpt="$CKPT_DIR/$ckpt_name"
    local config_key="stage2_unfrozen_proj_60_v2_ep${epoch}_ft010"

    if [ ! -f "$train_ckpt" ]; then
        echo "Missing checkpoint: $train_ckpt" >&2
        return 1
    fi

    echo "[$(date '+%F %T')] Starting checkpoint sweep eval for epoch=$epoch gpu=$gpu config_key=$config_key"
    rm -rf "$PROJECT_ROOT/output/matches/$config_key"
    rm -f "$PROJECT_ROOT"/output/benchmarks/"${config_key}"_*.h5
    rm -rf "$PROJECT_ROOT/output/results/$config_key"
    rm -rf "$PROJECT_ROOT/cache/features/${config_key}_proj_cache"

    for scene in sacre_coeur reichstag st_peters_square; do
        run_scene "$gpu" "$train_ckpt" "$config_key" "$scene"
    done

    echo "[$(date '+%F %T')] Finished checkpoint sweep eval for epoch=$epoch"
}

print_summary() {
    PROJECT_ROOT="$PROJECT_ROOT" CKPT_DIR="$CKPT_DIR" ORIG_PROJ_CKPT="$ORIG_PROJ_CKPT" "$DINOV3_PY" - <<'PY'
import json
import os
from pathlib import Path

import torch

project_root = Path(os.environ["PROJECT_ROOT"])
ckpt_dir = Path(os.environ["CKPT_DIR"])
orig = torch.load(os.environ["ORIG_PROJ_CKPT"], map_location="cpu")

rows = [
    ("checkpoint_41_92900.tar", 41, "stage2_unfrozen_proj_60_v2_ep41_ft010"),
    ("checkpoint_44_99536.tar", 44, "stage2_unfrozen_proj_60_v2_ep44_ft010"),
    ("checkpoint_best.tar", 47, "stage2_unfrozen_proj_60_v2_ft010"),
    ("checkpoint_49_110596.tar", 49, "stage2_unfrozen_proj_60_v2_ep49_ft010"),
]

def read_maa(config_key, scene):
    path = project_root / "output" / "results" / config_key / scene / f"calibrated-{config_key}_{scene}.json"
    data = json.loads(path.read_text())
    return data["3p_ours_shift_scale+12"]["mAA_10"]

def avg_drift(checkpoint_name):
    ckpt = torch.load(ckpt_dir / checkpoint_name, map_location="cpu")
    model_state = ckpt["model"]
    drifts = []
    for orig_key in sorted(orig.keys()):
        tensor = orig[orig_key]
        if not isinstance(tensor, torch.Tensor):
            continue
        ckpt_key = f"extractor.projection.{orig_key}"
        if ckpt_key not in model_state:
            continue
        diff = (tensor.float() - model_state[ckpt_key].float()).norm()
        orig_norm = tensor.float().norm()
        drifts.append((diff / orig_norm).item())
    return sum(drifts) / len(drifts)

print()
print("==========================================================")
print("  UNFROZEN PROJ HEAD — mAA@10 ACROSS TRAINING EPOCHS")
print("==========================================================")
print("| Checkpoint | Epoch | sacre | reichstag | st_peters | avg | proj_drift |")
print("|---|---|---|---|---|---|---|")

for checkpoint_name, epoch, config_key in rows:
    sacre = read_maa(config_key, "sacre_coeur")
    reich = read_maa(config_key, "reichstag")
    stpet = read_maa(config_key, "st_peters_square")
    avg = (sacre + reich + stpet) / 3.0
    drift = avg_drift(checkpoint_name)
    label = "checkpoint_best" if checkpoint_name == "checkpoint_best.tar" else checkpoint_name
    print(f"| {label} | {epoch} | {sacre:.1f} | {reich:.1f} | {stpet:.1f} | {avg:.1f} | {drift:.4f} |")

print()
print("=== BASELINES FOR REFERENCE ===")
print("| Frozen 60sc (4c) | — | 85.3 | 82.9 | 74.9 | 81.0 | 0.0 |")
print("| Frozen 151sc (4d) | — | 85.5 | 83.6 | 75.0 | 81.4 | 0.0 |")
PY
}

echo "[$(date '+%F %T')] Starting checkpoint sweep ft010"
echo "[$(date '+%F %T')] Log file: $LOG_FILE"
echo "[$(date '+%F %T')] Sweep epochs: ${SWEEP_EPOCHS[*]}"

declare -a PIDS=()
for epoch in "${SWEEP_EPOCHS[@]}"; do
    run_checkpoint "$epoch" "${SWEEP_CKPTS[$epoch]}" "${SWEEP_GPUS[$epoch]}" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

print_summary
echo "[$(date '+%F %T')] Checkpoint sweep complete"
