#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="final_selected_expanded151_lg_proj_dinov3_dift_ft002_mp2048"
MATCHER_CKPT="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
PROJECTION_CKPT="$ROOT/experiments/phase2_projection_wide/best.pt"

DINOV3_PY="${DINOV3_PY:-/home.stud/gorbuden/.conda/envs/dinov3/bin/python}"
REPOSED_PY="${REPOSED_PY:-/home.stud/gorbuden/.conda/envs/reposed/bin/python}"

SCENES=(
    british_museum
    florence_cathedral_side
    lincoln_memorial_statue
    milan_cathedral
    mount_rushmore
    piazza_san_marco
    sagrada_familia
    st_pauls_cathedral
    taj_mahal
    temple_nara_japan
)
MODES=(calibrated shared_focal varying_focal)

THESIS_PAIR_LIMIT="${THESIS_PAIR_LIMIT:-15000}"
REPOSED_NUM_WORKERS="${REPOSED_NUM_WORKERS:-8}"
SCREEN_NAME="${SCREEN_NAME:-final_selected_lg_test}"
RUNTIME_ESTIMATE="${RUNTIME_ESTIMATE:-overnight run; online DINOv3+DIFT extraction plus 10 held-out scenes x 3 solver modes}"

LOG_ROOT="$ROOT/output_v2/logs/final_selected_expanded151_lg_test"
STATUS_DIR="$LOG_ROOT/status"
FAILURES_TSV="$LOG_ROOT/failures.tsv"
MATCHES_ROOT="$ROOT/output_v2/matches_v2/$CONFIG"
BENCH_ROOT="$ROOT/output_v2/benchmarks_v2"
RESULTS_ROOT="$ROOT/output_v2/results_v2/$CONFIG"
REPORT_PATH="$ROOT/output_v2/reports/final_selected_expanded151_lg_test_report.md"
PLAN_PATH="$ROOT/output_v2/reports/final_selected_expanded151_lg_test_launch_plan.md"
REPOSED_DIR="$ROOT/external/RePoseD"

GPUS_SPEC="auto"
GPUS_CSV=""
NO_SCREEN=0
PLAN_ONLY=0
AGGREGATE_ONLY=0
WORKER=""
GPU=""
SHARD=0
NUM_SHARDS=1
LAUNCH_REQUESTED=0

usage() {
    cat <<EOF
Usage:
  scripts/run_final_selected_lg_test.sh --launch [--gpus 0,1,3] [--no-screen]
  scripts/run_final_selected_lg_test.sh --plan-only [--gpus 0,1,3]
  scripts/run_final_selected_lg_test.sh --aggregate-only
  scripts/run_final_selected_lg_test.sh --worker supervisor --gpus 0,1,3
  scripts/run_final_selected_lg_test.sh --worker scene --gpu GPU --shard I --num-shards N

Evaluation only. No training and no checkpoint modification.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) LAUNCH_REQUESTED=1; shift ;;
        --gpus) GPUS_SPEC="${2:?missing value for --gpus}"; shift 2 ;;
        --no-screen) NO_SCREEN=1; shift ;;
        --plan-only) PLAN_ONLY=1; shift ;;
        --aggregate-only) AGGREGATE_ONLY=1; shift ;;
        --worker) WORKER="${2:?missing value for --worker}"; shift 2 ;;
        --gpu) GPU="${2:?missing value for --gpu}"; shift 2 ;;
        --shard) SHARD="${2:?missing value for --shard}"; shift 2 ;;
        --num-shards) NUM_SHARDS="${2:?missing value for --num-shards}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

setup_env() {
    export PYTHONNOUSERSITE=1
    export HF_HOME="${HF_HOME:-/home.stud/gorbuden/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
    export TOKENIZERS_PARALLELISM=false
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_final_selected_lg_test}"
    mkdir -p "$MPLCONFIGDIR"
    setup_mkl_openmp_env
}

setup_mkl_openmp_env() {
    local mkl_pkg omp_pkg extra_libs
    mkl_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libmkl_intel_lp64.so.2" 2>/dev/null | head -1 | xargs -r dirname || true)"
    omp_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libiomp5.so" 2>/dev/null | head -1 | xargs -r dirname || true)"
    extra_libs=""
    [[ -n "$mkl_pkg" ]] && extra_libs="$mkl_pkg"
    [[ -n "$omp_pkg" ]] && extra_libs="${extra_libs:+$extra_libs:}$omp_pkg"
    [[ -n "$extra_libs" ]] && export LD_LIBRARY_PATH="${extra_libs}:${LD_LIBRARY_PATH:-}"
}

init_dirs() {
    mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$MATCHES_ROOT" "$BENCH_ROOT" "$RESULTS_ROOT" \
        "$ROOT/output_v2/csv" "$ROOT/output_v2/reports"
}

init_failures() {
    if [[ ! -f "$FAILURES_TSV" ]]; then
        printf "time\tstage\tconfig\tscene\tsolver_mode\tstatus\treason\tlog_path\tnotes\n" > "$FAILURES_TSV"
    fi
}

record_failure() {
    init_failures
    local stage="$1" scene="$2" mode="$3" reason="$4" log_path="$5" notes="${6:-}"
    printf "%s\t%s\t%s\t%s\t%s\tfailed\t%s\t%s\t%s\n" \
        "$(date --iso-8601=seconds)" "$stage" "$CONFIG" "$scene" "$mode" "$reason" "$log_path" "$notes" >> "$FAILURES_TSV"
    log "[FAIL][$scene][$mode][$stage] $reason"
}

run_logged() {
    local log_path="$1"
    shift
    mkdir -p "$(dirname "$log_path")"
    {
        echo
        echo "[$(date --iso-8601=seconds)] RUN $*"
    } >> "$log_path"
    "$@" >> "$log_path" 2>&1
    local code=$?
    echo "[$(date --iso-8601=seconds)] EXIT $code" >> "$log_path"
    return "$code"
}

run_logged_cwd() {
    local log_path="$1" cwd="$2"
    shift 2
    mkdir -p "$(dirname "$log_path")"
    {
        echo
        echo "[$(date --iso-8601=seconds)] CWD $cwd"
        echo "[$(date --iso-8601=seconds)] RUN $*"
    } >> "$log_path"
    (cd "$cwd" && "$@") >> "$log_path" 2>&1
    local code=$?
    echo "[$(date --iso-8601=seconds)] EXIT $code" >> "$log_path"
    return "$code"
}

detect_gpus() {
    if [[ "$GPUS_SPEC" != "auto" ]]; then
        echo "$GPUS_SPEC" | tr ',' ' '
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '
        return
    fi
    local free
    free="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '$2 + 0 < 2048 {gsub(/ /, "", $1); print $1}' \
        | paste -sd' ' -)"
    if [[ -n "$free" ]]; then
        echo "$free"
        return
    fi
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

scene_images_dir() {
    echo "$ROOT/datasets/phototourism/$1/images_preprocessed"
}

scene_preprocess_info() {
    echo "$ROOT/datasets/phototourism/$1/images_preprocessed/preprocess_info.json"
}

scene_sparse_dir() {
    echo "$ROOT/datasets/phototourism/$1/dense/sparse"
}

scene_depth_dir() {
    echo "$ROOT/datasets/phototourism/$1/depth_unidepth"
}

scene_pairs_file() {
    echo "$ROOT/output/pairs_$1.txt"
}

scene_pair_count() {
    local file
    file="$(scene_pairs_file "$1")"
    [[ -f "$file" ]] || { echo 0; return; }
    awk 'NF {count++} END {print count+0}' "$file"
}

scene_pair_limit() {
    local count
    count="$(scene_pair_count "$1")"
    if [[ "$count" -le 0 ]]; then
        echo 0
    elif [[ "$count" -lt "$THESIS_PAIR_LIMIT" ]]; then
        echo "$count"
    else
        echo "$THESIS_PAIR_LIMIT"
    fi
}

matches_dir() {
    echo "$MATCHES_ROOT/$1"
}

benchmark_path() {
    echo "$BENCH_ROOT/${CONFIG}_$1.h5"
}

results_dir() {
    echo "$RESULTS_ROOT/$1"
}

summary_path() {
    local scene="$1" mode="$2"
    echo "$(results_dir "$scene")/${mode}-${CONFIG}_${scene}-2.0t_summary.json"
}

raw_json_path() {
    local scene="$1" mode="$2"
    echo "$(results_dir "$scene")/${mode}-${CONFIG}_${scene}-2.0t.json"
}

matches_done_path() {
    echo "$LOG_ROOT/status/$1.matches.done"
}

scene_complete() {
    local scene="$1" mode
    [[ -f "$(benchmark_path "$scene")" ]] || return 1
    for mode in "${MODES[@]}"; do
        [[ -f "$(summary_path "$scene" "$mode")" ]] || return 1
    done
    return 0
}

all_complete() {
    local scene
    for scene in "${SCENES[@]}"; do
        scene_complete "$scene" || return 1
    done
    return 0
}

match_count() {
    local scene="$1"
    find "$(matches_dir "$scene")" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l
}

preflight() {
    local fatal=0 scene
    [[ -x "$DINOV3_PY" ]] || { echo "Missing executable: $DINOV3_PY" >&2; fatal=1; }
    [[ -x "$REPOSED_PY" ]] || { echo "Missing executable: $REPOSED_PY" >&2; fatal=1; }
    [[ -f "$MATCHER_CKPT" ]] || { echo "Missing matcher checkpoint: $MATCHER_CKPT" >&2; fatal=1; }
    [[ -f "$PROJECTION_CKPT" ]] || { echo "Missing projection checkpoint: $PROJECTION_CKPT" >&2; fatal=1; }
    [[ -f "$ROOT/scripts/lightglue_projection_matches.py" ]] || { echo "Missing matcher script" >&2; fatal=1; }
    [[ -f "$ROOT/scripts/pack_benchmark.py" ]] || { echo "Missing pack_benchmark.py" >&2; fatal=1; }
    [[ -f "$ROOT/scripts/aggregate_final_selected_lg_test.py" ]] || { echo "Missing aggregate_final_selected_lg_test.py" >&2; fatal=1; }
    for scene in "${SCENES[@]}"; do
        [[ -d "$(scene_images_dir "$scene")" ]] || { echo "Missing images_preprocessed for $scene" >&2; fatal=1; }
        [[ -f "$(scene_preprocess_info "$scene")" ]] || { echo "Missing preprocess info for $scene" >&2; fatal=1; }
        [[ -d "$(scene_sparse_dir "$scene")" ]] || { echo "Missing sparse dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_depth_dir "$scene")" ]] || { echo "Missing depth dir for $scene" >&2; fatal=1; }
        [[ -f "$(scene_pairs_file "$scene")" ]] || { echo "Missing pairs file for $scene" >&2; fatal=1; }
    done
    return "$fatal"
}

write_plan() {
    local gpus_csv="$1" scene_count="$2" missing_scene_count="$3"
    cat > "$PLAN_PATH" <<EOF
# Final Selected Expanded-151 LightGlue Test Launch Plan

Generated: $(date --iso-8601=seconds)

## Exact Command

\`\`\`bash
cd $ROOT
CUDA_VISIBLE_DEVICES=$gpus_csv scripts/run_final_selected_lg_test.sh --launch --gpus $gpus_csv
\`\`\`

## Screen

- Screen session: \`$SCREEN_NAME\`
- Attach: \`screen -r $SCREEN_NAME\`
- Detach: \`Ctrl-a d\`

## Workload

- Config: \`$CONFIG\`
- Scenes: $scene_count
- Incomplete scenes at launch: $missing_scene_count
- Solver modes: calibrated, shared_focal, varying_focal
- Pair cap per scene: $THESIS_PAIR_LIMIT
- GPU allocation requested: \`$gpus_csv\`
- Runtime estimate: $RUNTIME_ESTIMATE

## Outputs

- \`output_v2/results_v2/$CONFIG\`
- \`output_v2/matches_v2/$CONFIG\`
- \`output_v2/benchmarks_v2/${CONFIG}_<scene>.h5\`
- \`output_v2/csv/final_selected_expanded151_lg_test_summary.csv\`
- \`output_v2/csv/final_selected_expanded151_lg_test_per_scene.csv\`
- \`output_v2/reports/final_selected_expanded151_lg_test_report.md\`
EOF
}

aggregate() {
    "$REPOSED_PY" "$ROOT/scripts/aggregate_final_selected_lg_test.py" \
        --write \
        --command "CUDA_VISIBLE_DEVICES=${GPUS_CSV:-} scripts/run_final_selected_lg_test.sh --launch --gpus ${GPUS_CSV:-auto}" \
        --gpus "${GPUS_CSV:-auto}" \
        --runtime-estimate "$RUNTIME_ESTIMATE"
}

run_match_stage() {
    local scene="$1" limit="$2"
    local done_path log_path count
    done_path="$(matches_done_path "$scene")"
    count="$(match_count "$scene")"
    if [[ -f "$done_path" || "$count" -ge "$limit" ]]; then
        log "[SKIP][$scene][match] existing matches count=$count limit=$limit"
        mkdir -p "$(dirname "$done_path")"
        date +%s > "$done_path"
        return 0
    fi

    log_path="$LOG_ROOT/${scene}_match.log"
    mkdir -p "$(matches_dir "$scene")"
    log "[RUN][$scene][match] gpu=$GPU limit=$limit"
    run_logged "$log_path" \
        env CUDA_VISIBLE_DEVICES="$GPU" \
        "$DINOV3_PY" "$ROOT/scripts/lightglue_projection_matches.py" \
            --pairs_file "$(scene_pairs_file "$scene")" \
            --images_dir "$(scene_images_dir "$scene")" \
            --output_dir "$(matches_dir "$scene")" \
            --config_key "$CONFIG" \
            --checkpoint "$PROJECTION_CKPT" \
            --lightglue_checkpoint "$MATCHER_CKPT" \
            --filter_threshold 0.02 \
            --seed 42 \
            --max_points 2048 \
            --source_cache_max_points 2000 \
            --feat_level -8 \
            --img_size 768 768 \
            --t 0 \
            --up_ft_index 2 \
            --ensemble_size 2 \
            --feature_cache "$ROOT/output_v2/cache/final_selected_expanded151_lg_test/$scene" \
            --online_extraction \
            --device cuda \
            --limit "$limit" \
        || { record_failure "match" "$scene" "" "match command failed" "$log_path"; return 1; }
    date +%s > "$done_path"
}

run_pack_stage() {
    local scene="$1" limit="$2" bench log_path
    bench="$(benchmark_path "$scene")"
    if [[ -f "$bench" ]]; then
        log "[SKIP][$scene][pack] $bench exists"
        return 0
    fi
    log_path="$LOG_ROOT/${scene}_pack.log"
    log "[RUN][$scene][pack] limit=$limit"
    run_logged "$log_path" \
        "$REPOSED_PY" "$ROOT/scripts/pack_benchmark.py" \
            --matches_dir "$(matches_dir "$scene")" \
            --depth_dir "$(scene_depth_dir "$scene")" \
            --sparse_dir "$(scene_sparse_dir "$scene")" \
            --pairs_file "$(scene_pairs_file "$scene")" \
            --output "$bench" \
            --limit "$limit" \
        || { record_failure "pack" "$scene" "" "pack command failed" "$log_path"; return 1; }
}

eval_script_for_mode() {
    case "$1" in
        calibrated) echo "$REPOSED_DIR/eval.py" ;;
        shared_focal) echo "$REPOSED_DIR/eval_shared_f.py" ;;
        varying_focal) echo "$REPOSED_DIR/eval_varying_f.py" ;;
        *) return 1 ;;
    esac
}

run_eval_mode() {
    local scene="$1" mode="$2" bench summary raw log_path script load_arg=()
    summary="$(summary_path "$scene" "$mode")"
    if [[ -f "$summary" ]]; then
        log "[SKIP][$scene][$mode] summary exists"
        return 0
    fi
    bench="$(benchmark_path "$scene")"
    raw="$(raw_json_path "$scene" "$mode")"
    script="$(eval_script_for_mode "$mode")"
    log_path="$LOG_ROOT/${scene}_${mode}.log"
    [[ -f "$bench" ]] || { record_failure "eval" "$scene" "$mode" "missing benchmark" "$log_path"; return 1; }
    [[ -f "$raw" ]] && load_arg=(--load)

    log "[RUN][$scene][$mode] ${load_arg[*]:-fresh}"
    run_logged_cwd "$log_path" "$REPOSED_DIR" \
        "$REPOSED_PY" "$script" "$bench" \
            -nw "$REPOSED_NUM_WORKERS" \
            --thesis \
            --output_dir "$(results_dir "$scene")" \
            --preprocess_info "$(scene_preprocess_info "$scene")" \
            --max_epipolar_error 2.0 \
            --reproj_threshold 16.0 \
            "${load_arg[@]}" \
        || { record_failure "eval" "$scene" "$mode" "eval command failed" "$log_path"; return 1; }

    if [[ ! -f "$summary" ]]; then
        record_failure "eval" "$scene" "$mode" "summary missing after eval" "$log_path"
        return 1
    fi
}

write_scene_runtime() {
    local scene="$1" started="$2" finished="$3" wall="$4" match_s="$5" pack_s="$6" eval_s="$7"
    local out
    out="$(results_dir "$scene")/runtime_summary.json"
    mkdir -p "$(dirname "$out")"
    cat > "$out" <<EOF
{
  "config": "$CONFIG",
  "scene": "$scene",
  "gpu_id": "$GPU",
  "started_at": "$started",
  "finished_at": "$finished",
  "wall_seconds": $wall,
  "match_seconds": $match_s,
  "pack_seconds": $pack_s,
  "eval_seconds": $eval_s,
  "pair_limit": $(scene_pair_limit "$scene")
}
EOF
}

run_scene_pipeline() {
    local scene="$1" limit started_at start_epoch end_epoch finished_at
    local match_start match_end pack_start pack_end eval_start eval_end
    local match_seconds=0 pack_seconds=0 eval_seconds=0 mode scene_failed=0
    limit="$(scene_pair_limit "$scene")"
    mkdir -p "$(matches_dir "$scene")" "$(results_dir "$scene")"

    if scene_complete "$scene"; then
        log "[SKIP][$scene] complete"
        return 0
    fi

    started_at="$(date --iso-8601=seconds)"
    start_epoch="$(date +%s)"
    log "[START][$scene] gpu=$GPU limit=$limit"

    match_start="$(date +%s)"
    run_match_stage "$scene" "$limit" || scene_failed=1
    match_end="$(date +%s)"
    match_seconds=$((match_end - match_start))

    if [[ "$scene_failed" -eq 0 ]]; then
        pack_start="$(date +%s)"
        run_pack_stage "$scene" "$limit" || scene_failed=1
        pack_end="$(date +%s)"
        pack_seconds=$((pack_end - pack_start))
    fi

    if [[ "$scene_failed" -eq 0 ]]; then
        eval_start="$(date +%s)"
        for mode in "${MODES[@]}"; do
            run_eval_mode "$scene" "$mode" || scene_failed=1
        done
        eval_end="$(date +%s)"
        eval_seconds=$((eval_end - eval_start))
    fi

    end_epoch="$(date +%s)"
    finished_at="$(date --iso-8601=seconds)"
    write_scene_runtime "$scene" "$started_at" "$finished_at" "$((end_epoch - start_epoch))" "$match_seconds" "$pack_seconds" "$eval_seconds"
    if [[ "$scene_failed" -eq 0 ]]; then
        log "[DONE][$scene] wall=$((end_epoch - start_epoch))s match=${match_seconds}s pack=${pack_seconds}s eval=${eval_seconds}s"
    else
        log "[DONE_WITH_FAILURES][$scene] wall=$((end_epoch - start_epoch))s"
    fi
    return "$scene_failed"
}

worker_scene() {
    setup_env
    init_dirs
    init_failures
    local idx=0 scene worker_failed=0
    for scene in "${SCENES[@]}"; do
        if (( idx % NUM_SHARDS == SHARD )); then
            run_scene_pipeline "$scene" || worker_failed=1
        fi
        idx=$((idx + 1))
    done
    return "$worker_failed"
}

supervisor() {
    setup_env
    init_dirs
    init_failures
    preflight
    date +%s > "$STATUS_DIR/supervisor.started"

    if all_complete; then
        log "All final selected test outputs already exist; aggregating only."
        aggregate
        date +%s > "$STATUS_DIR/supervisor.done"
        return 0
    fi

    IFS=',' read -r -a gpu_ids <<<"$GPUS_CSV"
    local scene_count="${#SCENES[@]}" missing=0 scene
    for scene in "${SCENES[@]}"; do
        scene_complete "$scene" || missing=$((missing + 1))
    done
    write_plan "$GPUS_CSV" "$scene_count" "$missing"

    log "Final selected LightGlue test supervisor"
    log "Config: $CONFIG"
    log "GPUs: $GPUS_CSV"
    log "Scenes: ${SCENES[*]}"

    local pids=() tags=() idx worker_failed=0
    for idx in "${!gpu_ids[@]}"; do
        (
            GPU="${gpu_ids[$idx]}"
            SHARD="$idx"
            NUM_SHARDS="${#gpu_ids[@]}"
            worker_scene
        ) > >(tee -a "$LOG_ROOT/worker_${idx}_gpu${gpu_ids[$idx]}.log") 2>&1 &
        pids+=("$!")
        tags+=("gpu${gpu_ids[$idx]}")
    done

    for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
            log "[SUPERVISOR] worker ${tags[$idx]} failed"
            worker_failed=1
        fi
    done

    log "Workers finished; aggregating final selected test report"
    aggregate

    if [[ "$worker_failed" -eq 0 && -s "$FAILURES_TSV" && "$(wc -l < "$FAILURES_TSV")" -gt 1 ]]; then
        worker_failed=1
    fi

    if [[ "$worker_failed" -eq 0 ]]; then
        date +%s > "$STATUS_DIR/supervisor.done"
        log "Final selected test complete. Report: $REPORT_PATH"
    else
        date +%s > "$STATUS_DIR/supervisor.failed"
        log "Final selected test completed with failures. Report: $REPORT_PATH"
        return 1
    fi
}

main_launch() {
    setup_env
    init_dirs
    preflight
    mapfile -t gpu_list < <(detect_gpus | tr ' ' '\n' | sed '/^$/d')
    if [[ "${#gpu_list[@]}" -eq 0 ]]; then
        echo "No GPUs detected. Pass --gpus explicitly." >&2
        exit 1
    fi
    GPUS_CSV="$(join_by_comma "${gpu_list[@]}")"

    local missing=0 scene
    for scene in "${SCENES[@]}"; do
        scene_complete "$scene" || missing=$((missing + 1))
    done
    write_plan "$GPUS_CSV" "${#SCENES[@]}" "$missing"

    if all_complete; then
        log "Complete outputs already exist; aggregating without relaunch."
        aggregate
        echo "Report: $REPORT_PATH"
        return 0
    fi

    if screen -ls 2>/dev/null | grep -q "[.]$SCREEN_NAME[[:space:]]"; then
        echo "Screen session '$SCREEN_NAME' is already running." >&2
        echo "Attach with: screen -r $SCREEN_NAME" >&2
        exit 1
    fi

    local cmd
    cmd="cd '$ROOT' && GPUS_CSV='$GPUS_CSV' RUNTIME_ESTIMATE='$RUNTIME_ESTIMATE' '$ROOT/scripts/run_final_selected_lg_test.sh' --worker supervisor --gpus '$GPUS_CSV' > '$LOG_ROOT/final_selected_lg_test.screen.log' 2>&1"
    echo "Exact launch command:"
    echo "screen -S $SCREEN_NAME -dm bash -lc \"$cmd\""
    if [[ "$PLAN_ONLY" -eq 1 ]]; then
        echo "Plan saved to: $PLAN_PATH"
        return 0
    fi
    if [[ "$NO_SCREEN" -eq 1 ]]; then
        bash -lc "$cmd"
    else
        screen -S "$SCREEN_NAME" -dm bash -lc "$cmd"
        echo "Launched screen session: $SCREEN_NAME"
        echo "Attach with: screen -r $SCREEN_NAME"
        echo "Plan: $PLAN_PATH"
        echo "Report will be saved to: $REPORT_PATH"
    fi
}

if [[ "$AGGREGATE_ONLY" -eq 1 ]]; then
    setup_env
    init_dirs
    aggregate
    echo "Report: $REPORT_PATH"
    exit 0
fi

if [[ -n "$WORKER" ]]; then
    case "$WORKER" in
        supervisor)
            [[ -n "$GPUS_SPEC" ]] && GPUS_CSV="$GPUS_SPEC"
            [[ -n "$GPUS_CSV" ]] || GPUS_CSV="${CUDA_VISIBLE_DEVICES:-}"
            [[ -n "$GPUS_CSV" ]] || { echo "--worker supervisor requires --gpus or CUDA_VISIBLE_DEVICES" >&2; exit 2; }
            supervisor
            ;;
        scene)
            [[ -n "$GPU" ]] || { echo "--worker scene requires --gpu" >&2; exit 2; }
            worker_scene
            ;;
        *)
            echo "Unknown worker: $WORKER" >&2
            exit 2
            ;;
    esac
    exit $?
fi

if [[ "$PLAN_ONLY" -eq 1 || "$LAUNCH_REQUESTED" -eq 1 ]]; then
    main_launch
else
    usage
fi
