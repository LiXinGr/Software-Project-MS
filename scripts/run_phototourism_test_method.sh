#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/phototourism_test_common.sh"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <method_key> <gpu_id> [--dry-run]" >&2
    exit 1
fi

METHOD_KEY="$1"
GPU_ID="$2"
shift 2
DRY_RUN=0
SCENE_SPEC=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --scenes)
            SCENE_SPEC="${2:?missing value for --scenes}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

prepare_runtime_env
require_runtime_prereqs

case "$METHOD_KEY" in
    test_splg|test_dinov3_mnn|test_dift_mnn|test_ours_151sc_ft010|test_roma|test_romav2)
        ;;
    *)
        echo "Unsupported method key: $METHOD_KEY" >&2
        exit 1
        ;;
esac

runtime_meta_for_scene() {
    printf '%s\n' "$PROJECT_ROOT/output/results/$METHOD_KEY/$1/runtime_summary.json"
}

write_runtime_meta() {
    local scene="$1"
    local started_at="$2"
    local finished_at="$3"
    local wall_seconds="$4"
    local limit="$5"
    local match_seconds="$6"
    local pack_seconds="$7"
    local eval_seconds="$8"
    local csv_seconds="$9"
    local output_file
    output_file="$(runtime_meta_for_scene "$scene")"
    mkdir -p "$(dirname "$output_file")"
    "$DINOV3_PY" - "$output_file" "$METHOD_KEY" "$scene" "$GPU_ID" "$started_at" "$finished_at" "$wall_seconds" "$limit" "$match_seconds" "$pack_seconds" "$eval_seconds" "$csv_seconds" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "method_key": sys.argv[2],
    "scene": sys.argv[3],
    "gpu_id": sys.argv[4],
    "started_at": sys.argv[5],
    "finished_at": sys.argv[6],
    "wall_seconds": float(sys.argv[7]),
    "pair_limit": int(sys.argv[8]),
    "match_seconds": float(sys.argv[9]),
    "pack_seconds": float(sys.argv[10]),
    "eval_seconds": float(sys.argv[11]),
    "csv_seconds": float(sys.argv[12]),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_match_stage() {
    local scene="$1"
    local limit="$2"
    local images_dir pairs_file matches_dir cache_dir romav2_pythonpath romav2_ld_library_path romav2_library_path
    images_dir="$(scene_preprocessed_dir "$scene")"
    pairs_file="$(scene_pairs_file "$scene")"
    matches_dir="$PROJECT_ROOT/output/matches/$METHOD_KEY/$scene"
    cache_dir="$PROJECT_ROOT/cache/features/$METHOD_KEY/$scene"
    mkdir -p "$matches_dir" "$cache_dir"

    romav2_pythonpath="${PYTHONPATH:-}"
    if [ -n "${ROMAV2_EXTRA_PYTHONPATH:-}" ]; then
        if [ -n "$romav2_pythonpath" ]; then
            romav2_pythonpath="${ROMAV2_EXTRA_PYTHONPATH}:$romav2_pythonpath"
        else
            romav2_pythonpath="${ROMAV2_EXTRA_PYTHONPATH}"
        fi
    fi

    romav2_ld_library_path="${LD_LIBRARY_PATH:-}"
    romav2_library_path="${LIBRARY_PATH:-}"
    if [ -n "${ROMAV2_CUDA_SHIM_DIR:-}" ]; then
        if [ -n "$romav2_ld_library_path" ]; then
            romav2_ld_library_path="${ROMAV2_CUDA_SHIM_DIR}:$romav2_ld_library_path"
        else
            romav2_ld_library_path="${ROMAV2_CUDA_SHIM_DIR}"
        fi
        if [ -n "$romav2_library_path" ]; then
            romav2_library_path="${ROMAV2_CUDA_SHIM_DIR}:$romav2_library_path"
        else
            romav2_library_path="${ROMAV2_CUDA_SHIM_DIR}"
        fi
    fi

    case "$METHOD_KEY" in
        test_splg)
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$LIGHTGLUE_PY" \
                "$PROJECT_ROOT/scripts/superpoint_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --matcher lightglue \
                --max_points 2048 \
                --feature_cache "$cache_dir" \
                --device cuda \
                --limit "$limit"
            ;;
        test_dinov3_mnn)
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$DINOV3_PY" \
                "$PROJECT_ROOT/scripts/dinov3_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --feat_level -8 \
                --img_size 768 \
                --max_points 2048 \
                --use_sp_keypoints \
                --feature_cache "$cache_dir" \
                --device cuda \
                --limit "$limit"
            ;;
        test_dift_mnn)
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$DIFT_PY" \
                "$PROJECT_ROOT/scripts/dift_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --max_points 2048 \
                --use_sp_keypoints \
                --img_size 768 768 \
                --t 0 \
                --up_ft_index 2 \
                --ensemble_size 8 \
                --feature_cache "$cache_dir" \
                --device cuda \
                --limit "$limit"
            ;;
        test_ours_151sc_ft010)
            log "[MATCH][$METHOD_KEY][$scene] note: --online_extraction uses in-process RAM memoization; on-disk --feature_cache is ignored by the current script."
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$DINOV3_PY" \
                "$PROJECT_ROOT/scripts/lightglue_projection_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --config_key "$METHOD_KEY" \
                --checkpoint "$PROJECTION_CKPT" \
                --lightglue_checkpoint "$OURS151_LIGHTGLUE_CKPT" \
                --filter_threshold 0.10 \
                --seed 42 \
                --max_points 2048 \
                --source_cache_max_points 2000 \
                --feat_level -8 \
                --img_size 768 768 \
                --t 0 \
                --up_ft_index 2 \
                --ensemble_size 8 \
                --feature_cache "$cache_dir" \
                --online_extraction \
                --device cuda \
                --limit "$limit"
            ;;
        test_roma)
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$ROMA_PY" \
                "$PROJECT_ROOT/scripts/roma_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --max_points 2048 \
                --device cuda \
                --limit "$limit"
            ;;
        test_romav2)
            CUDA_VISIBLE_DEVICES="$GPU_ID" \
                PYTHONPATH="$romav2_pythonpath" \
                LD_LIBRARY_PATH="$romav2_ld_library_path" \
                LIBRARY_PATH="$romav2_library_path" \
                "$ROMAV2_PY" \
                "$PROJECT_ROOT/scripts/romav2_matches.py" \
                --pairs_file "$pairs_file" \
                --images_dir "$images_dir" \
                --output_dir "$matches_dir" \
                --max_points 2048 \
                --setting precise \
                --device cuda \
                --limit "$limit"
            ;;
    esac
}

csv_args_for_method() {
    case "$METHOD_KEY" in
        test_dinov3_mnn)
            printf '%s\n' "--max-points 2048 --img-size 768 --feat-level -8 --up-ft-index '' --dift-t ''"
            ;;
        test_dift_mnn)
            printf '%s\n' "--max-points 2048 --img-size 768 --feat-level '' --up-ft-index 2 --dift-t 0"
            ;;
        test_ours_151sc_ft010)
            printf '%s\n' "--max-points 2048 --img-size 768 --feat-level -8 --up-ft-index 2 --dift-t 0"
            ;;
        *)
            printf '%s\n' "--max-points 2048 --img-size 768 --feat-level '' --up-ft-index '' --dift-t ''"
            ;;
    esac
}

run_scene_pipeline() {
    local scene="$1"
    local limit="$2"
    local matches_dir benchmark_file results_dir preprocess_info pairs_file sparse_dir depth_dir result_json result_csv
    matches_dir="$PROJECT_ROOT/output/matches/$METHOD_KEY/$scene"
    benchmark_file="$PROJECT_ROOT/output/benchmarks/${METHOD_KEY}_${scene}.h5"
    results_dir="$PROJECT_ROOT/output/results/$METHOD_KEY/$scene"
    preprocess_info="$(scene_preprocess_info "$scene")"
    pairs_file="$(scene_pairs_file "$scene")"
    sparse_dir="$(scene_sparse_dir "$scene")"
    depth_dir="$(scene_depth_dir "$scene")"
    result_json="$results_dir/calibrated-${METHOD_KEY}_${scene}.json"
    result_csv="$results_dir/results_${METHOD_KEY}_${scene}.csv"

    mkdir -p "$matches_dir" "$results_dir" "$(dirname "$benchmark_file")"

    if [ -f "$result_json" ] && [ -f "$result_csv" ]; then
        log "[SKIP][$METHOD_KEY][$scene] calibrated JSON and CSV already exist"
        return 0
    fi

    started_at="$(date --iso-8601=seconds)"
    start_epoch="$(date +%s)"
    log "[START][$METHOD_KEY][$scene] gpu=$GPU_ID limit=$limit"

    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN][$METHOD_KEY][$scene] matching -> $matches_dir"
        log "[DRY-RUN][$METHOD_KEY][$scene] pack -> $benchmark_file"
        log "[DRY-RUN][$METHOD_KEY][$scene] eval -> $results_dir"
        return 0
    fi

    local match_start match_end pack_start pack_end eval_start eval_end csv_start csv_end
    local match_seconds pack_seconds eval_seconds csv_seconds

    match_start="$(date +%s)"
    run_match_stage "$scene" "$limit"
    match_end="$(date +%s)"
    match_seconds="$((match_end - match_start))"

    pack_start="$(date +%s)"
    "$REPOSED_PY" "$PROJECT_ROOT/scripts/pack_benchmark.py" \
        --matches_dir "$matches_dir" \
        --depth_dir "$depth_dir" \
        --sparse_dir "$sparse_dir" \
        --pairs_file "$pairs_file" \
        --output "$benchmark_file" \
        --limit "$limit"
    pack_end="$(date +%s)"
    pack_seconds="$((pack_end - pack_start))"

    setup_mkl_openmp_env
    eval_start="$(date +%s)"
    (
        cd "$PROJECT_ROOT/external/RePoseD"
        "$REPOSED_PY" eval.py "$benchmark_file" \
            -nw "$REPOSED_NUM_WORKERS" \
            --thesis \
            --output_dir "$results_dir" \
            --preprocess_info "$preprocess_info"
    )
    eval_end="$(date +%s)"
    eval_seconds="$((eval_end - eval_start))"

    # shellcheck disable=SC2206
    local extra_csv_args=( $(csv_args_for_method) )
    csv_start="$(date +%s)"
    "$REPOSED_PY" "$PROJECT_ROOT/scripts/reconstruct_results_csv.py" \
        --input-json "$result_json" \
        --output-csv "$result_csv" \
        --matcher "$METHOD_KEY" \
        --depth UniDepth \
        --exp-type calibrated \
        "${extra_csv_args[@]}"
    csv_end="$(date +%s)"
    csv_seconds="$((csv_end - csv_start))"

    end_epoch="$(date +%s)"
    finished_at="$(date --iso-8601=seconds)"
    wall_seconds="$((end_epoch - start_epoch))"
    write_runtime_meta \
        "$scene" "$started_at" "$finished_at" "$wall_seconds" "$limit" \
        "$match_seconds" "$pack_seconds" "$eval_seconds" "$csv_seconds"
    log "[DONE][$METHOD_KEY][$scene] wall=${wall_seconds}s match=${match_seconds}s pack=${pack_seconds}s eval=${eval_seconds}s csv=${csv_seconds}s"
}

if [ "$DRY_RUN" -eq 0 ] && ! print_readiness_table >/dev/null; then
    echo "Preprocessing is incomplete. Run scripts/run_phototourism_test_preprocess.sh first." >&2
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ] && ! print_readiness_table >/dev/null; then
    log "[DRY-RUN] preprocessing is incomplete; showing planned commands anyway."
fi

log "Method runner"
log "Method: $METHOD_KEY ($(method_label "$METHOD_KEY"))"
log "GPU: $GPU_ID"
log "Dry run: $DRY_RUN"

ACTIVE_SCENES=("${TEST_SCENES[@]}")
if [ -n "$SCENE_SPEC" ]; then
    IFS=',' read -r -a requested_scenes <<<"$SCENE_SPEC"
    ACTIVE_SCENES=()
    for requested_scene in "${requested_scenes[@]}"; do
        found=0
        for scene in "${TEST_SCENES[@]}"; do
            if [ "$scene" = "$requested_scene" ]; then
                ACTIVE_SCENES+=("$scene")
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            echo "Unknown scene in --scenes: $requested_scene" >&2
            exit 1
        fi
    done
fi
log "Scenes: ${ACTIVE_SCENES[*]}"

for scene in "${ACTIVE_SCENES[@]}"; do
    run_scene_pipeline "$scene" "$(scene_pair_limit "$scene")"
done

if [ "$DRY_RUN" -eq 1 ]; then
    log "Dry-run complete for $METHOD_KEY."
else
    log "Method complete: $METHOD_KEY"
fi
