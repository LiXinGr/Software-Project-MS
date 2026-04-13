#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DINOV3_PY="/home.stud/gorbuden/.conda/envs/dinov3/bin/python"
REPOSED_PY="/home.stud/gorbuden/.conda/envs/reposed/bin/python"

GPU_ID="${1:-0}"
PAIR_LIMIT="${PAIR_LIMIT:-15000}"
NUM_WORKERS="${REPOSED_NUM_WORKERS:-8}"

SCENES=("sacre_coeur" "reichstag" "st_peters_square")
THRESHOLDS=("0.10" "0.15")

LIGHTGLUE_CKPT="$PROJECT_ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
PROJECTION_CKPT="$PROJECT_ROOT/experiments/phase2_projection_wide/best.pt"
SHARED_FEATURE_CACHE_ROOT="$PROJECT_ROOT/cache/features/stage2_151scenes_eval_proj256_mp2048"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/home.stud/gorbuden/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_stage2_151_eval}"

mkdir -p "$PROJECT_ROOT/logs" "$MPLCONFIGDIR" "$SHARED_FEATURE_CACHE_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

threshold_to_key() {
    case "$1" in
        0.10) echo "stage2_151scenes_lg_ft010" ;;
        0.15) echo "stage2_151scenes_lg_ft015" ;;
        *)
            echo "Unsupported threshold: $1" >&2
            exit 1
            ;;
    esac
}

ensure_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

run_scene_eval() {
    local config_key="$1"
    local threshold="$2"
    local scene="$3"

    local dataset_root="$PROJECT_ROOT/datasets/phototourism/$scene"
    local preprocess_info="$dataset_root/images_preprocessed/preprocess_info.json"
    local sparse_dir="$dataset_root/dense/sparse"
    local depth_dir="$dataset_root/depth_unidepth"
    local pairs_file="$PROJECT_ROOT/output/pairs_${scene}.txt"
    local matches_dir="$PROJECT_ROOT/output/matches/$config_key/$scene"
    local results_dir="$PROJECT_ROOT/output/results/$config_key/$scene"
    local benchmark_file="$PROJECT_ROOT/output/benchmarks/${config_key}_${scene}.h5"
    local feature_cache_dir="$SHARED_FEATURE_CACHE_ROOT/$scene"

    mkdir -p "$matches_dir" "$results_dir" "$feature_cache_dir"

    log "[EVAL-151] Checkpoint: ${LIGHTGLUE_CKPT#$PROJECT_ROOT/}"
    log "[EVAL-151] filter_threshold: $threshold"
    log "[EVAL-151] Config key: $config_key"
    log "[EVAL-151] Scene: $scene"

    local limit_args=()
    if [ -n "$PAIR_LIMIT" ]; then
        limit_args=(--limit "$PAIR_LIMIT")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$DINOV3_PY" \
        "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
        --scene "$scene" \
        --config_key "$config_key" \
        --checkpoint "$PROJECTION_CKPT" \
        --lightglue_checkpoint "$LIGHTGLUE_CKPT" \
        --filter_threshold "$threshold" \
        --seed 42 \
        --max_points 2048 \
        --source_cache_max_points 2000 \
        --feat_level -8 \
        --img_size 768 768 \
        --t 0 \
        --up_ft_index 2 \
        --ensemble_size 8 \
        --feature_cache "$feature_cache_dir" \
        --device cuda \
        "${limit_args[@]}"

    "$REPOSED_PY" "$PROJECT_ROOT/scripts/pack_benchmark.py" \
        --matches_dir "$matches_dir" \
        --depth_dir "$depth_dir" \
        --sparse_dir "$sparse_dir" \
        --pairs_file "$pairs_file" \
        --output "$benchmark_file" \
        "${limit_args[@]}"

    pushd "$PROJECT_ROOT/external/RePoseD" >/dev/null
    "$REPOSED_PY" eval.py \
        "$benchmark_file" \
        -nw "$NUM_WORKERS" \
        --thesis \
        --output_dir "$results_dir" \
        --preprocess_info "$preprocess_info"
    popd >/dev/null

    "$REPOSED_PY" "$PROJECT_ROOT/scripts/reconstruct_results_csv.py" \
        --input-json "$results_dir/calibrated-${config_key}_${scene}.json" \
        --output-csv "$results_dir/results_${config_key}_${scene}.csv" \
        --matcher "$config_key" \
        --depth UniDepth \
        --exp-type calibrated \
        --max-points 2048 \
        --img-size 768 \
        --feat-level -8 \
        --up-ft-index 2 \
        --dift-t 0
}

ensure_file "$LIGHTGLUE_CKPT"
ensure_file "$PROJECTION_CKPT"
ensure_file "$DINOV3_PY"
ensure_file "$REPOSED_PY"

cd "$PROJECT_ROOT"

log "Stage 2 151-scene evaluation"
log "Project root: $PROJECT_ROOT"
log "GPU id: $GPU_ID"
log "Pair limit: $PAIR_LIMIT"

for threshold in "${THRESHOLDS[@]}"; do
    config_key="$(threshold_to_key "$threshold")"
    log "Resetting outputs for $config_key"
    rm -rf "$PROJECT_ROOT/output/matches/$config_key"
    rm -rf "$PROJECT_ROOT/output/results/$config_key"
    rm -f "$PROJECT_ROOT/output/benchmarks/${config_key}_"*.h5

    for scene in "${SCENES[@]}"; do
        run_scene_eval "$config_key" "$threshold" "$scene"
    done
done

"$DINOV3_PY" "$PROJECT_ROOT/scripts/generate_stage2_151_eval_report.py"

log "Evaluation complete"
log "Report: $PROJECT_ROOT/docs/reports/stage2_151scenes_training_report.md"
