#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REPOSED_PY="${REPOSED_PY:-/home.stud/gorbuden/.conda/envs/reposed/bin/python}"
DINOV3_PY="${DINOV3_PY:-/home.stud/gorbuden/.conda/envs/dinov3/bin/python}"

SCENES=(sacre_coeur reichstag st_peters_square)
MODES=(shared_focal varying_focal)
PAIR_LIMIT="${PAIR_LIMIT:-15000}"
REPOSED_NUM_WORKERS="${REPOSED_NUM_WORKERS:-16}"
SCREEN_NAME="${SCREEN_NAME:-ch6_selected_all_modes}"
RUNTIME_ESTIMATE="${RUNTIME_ESTIMATE:-20-60 minutes; RePoseD-only pass reusing existing benchmarks}"

LOG_ROOT="$ROOT/output_v2/logs/ch6_selected_all_modes"
STATUS_DIR="$LOG_ROOT/status"
FAILURES_TSV="$LOG_ROOT/failures.tsv"
TASKS_TSV="$LOG_ROOT/tasks.tsv"
MATCHES_ROOT="$ROOT/output_v2/matches_v2"
BENCH_ROOT="$ROOT/output_v2/benchmarks_v2"
RESULTS_ROOT="$ROOT/output_v2/results_v2"
REPOSED_DIR="$ROOT/external/RePoseD"
PLAN_PATH="$ROOT/output_v2/reports/chapter6_selected_all_modes_launch_plan.md"

FORCE=0
NO_SCREEN=0
PLAN_ONLY=0
WORKER=""
GPU=""
GPUS_CSV=""
SHARD=0
NUM_SHARDS=1

usage() {
    cat <<EOF
Usage:
  scripts/run_ch6_selected_all_modes.sh --launch [--force] [--no-screen]
  scripts/run_ch6_selected_all_modes.sh --plan-only
  scripts/run_ch6_selected_all_modes.sh --worker supervisor --gpus 0,1,3 [--force]
  scripts/run_ch6_selected_all_modes.sh --worker eval --gpu GPU --shard I --num-shards N [--force]

Evaluation only. Reuses selected-config matches/benchmarks and runs missing
shared_focal/varying_focal RePoseD summaries. Calibrated summaries are reused.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) shift ;;
        --force) FORCE=1; shift ;;
        --no-screen) NO_SCREEN=1; shift ;;
        --plan-only) PLAN_ONLY=1; shift ;;
        --worker) WORKER="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --gpus) GPUS_CSV="$2"; shift 2 ;;
        --shard) SHARD="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

export GPUS_CSV RUNTIME_ESTIMATE

setup_env() {
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_ch6_selected_all_modes}"
    mkdir -p "$MPLCONFIGDIR"

    local mkl_pkg omp_pkg extra_libs
    mkl_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libmkl_intel_lp64.so.2" 2>/dev/null | head -1 | xargs -r dirname || true)"
    omp_pkg="$(find "$HOME/.conda/pkgs" -maxdepth 3 -name "libiomp5.so" 2>/dev/null | head -1 | xargs -r dirname || true)"
    extra_libs=""
    [[ -n "$mkl_pkg" ]] && extra_libs="$mkl_pkg"
    [[ -n "$omp_pkg" ]] && extra_libs="${extra_libs:+$extra_libs:}$omp_pkg"
    [[ -n "$extra_libs" ]] && export LD_LIBRARY_PATH="${extra_libs}:${LD_LIBRARY_PATH:-}"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

init_dirs() {
    mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$ROOT/output_v2/reports" "$ROOT/output_v2/csv" \
        "$MATCHES_ROOT" "$BENCH_ROOT" "$RESULTS_ROOT"
}

init_failures() {
    if [[ ! -f "$FAILURES_TSV" ]]; then
        printf "time\tstage\tconfig_key\tscene\tsolver_mode\tstatus\treason\tlog_path\tnotes\n" > "$FAILURES_TSV"
    fi
}

record_failure() {
    init_failures
    local stage="$1" key="$2" scene="$3" mode="$4" status="$5" reason="$6" log_path="$7" notes="${8:-}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date --iso-8601=seconds)" "$stage" "$key" "$scene" "$mode" "$status" "$reason" "$log_path" "$notes" >> "$FAILURES_TSV"
    log "[FAIL] $stage $key $scene $mode: $reason"
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
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '
        return
    fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '$2 + 0 < 2048 {gsub(/ /, "", $1); print $1}' \
        | tr '\n' ' '
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

scene_sparse_dir() {
    echo "$ROOT/datasets/phototourism/$1/dense/sparse"
}

scene_depth_dir() {
    echo "$ROOT/output_v2/depth_raw/$1"
}

scene_pairs_file() {
    echo "$ROOT/output/pairs_$1.txt"
}

benchmark_path() {
    echo "$BENCH_ROOT/${1}_${2}.h5"
}

summary_glob() {
    local key="$1" scene="$2" mode="$3"
    printf "%s/%s/%s/%s-%s_%s-2.0t_summary.json" "$RESULTS_ROOT" "$key" "$scene" "$mode" "$key" "$scene"
}

summary_exists() {
    compgen -G "$(summary_glob "$1" "$2" "$3")" >/dev/null
}

task_lines() {
    local key scene mode
    local keys=(
        ch6_oldproj_warmstart_lg_full_v1_ft002_sp_mnn_mp2048
        ch6_oldproj_scratch_stage2_lg_v1_ft002_sp_mnn_mp2048
        ch6_oldproj_expanded151_lg_ft002_sp_mnn_mp2048
        ch6_oldproj_joint_unfrozen_proj60_ft000_sp_mnn_mp2048
    )
    for key in "${keys[@]}"; do
        for scene in "${SCENES[@]}"; do
            for mode in "${MODES[@]}"; do
                printf "%s\t%s\t%s\n" "$key" "$scene" "$mode"
            done
        done
    done
}

write_tasks_file() {
    printf "config_key\tscene\tsolver_mode\n" > "$TASKS_TSV"
    task_lines >> "$TASKS_TSV"
}

preflight() {
    local fatal=0 scene key
    [[ -x "$REPOSED_PY" ]] || { echo "Missing executable: $REPOSED_PY" >&2; fatal=1; }
    [[ -x "$DINOV3_PY" ]] || { echo "Missing executable: $DINOV3_PY" >&2; fatal=1; }
    for path in "$ROOT/scripts/pack_benchmark.py" "$ROOT/scripts/aggregate_ch6_selected_all_modes.py"; do
        [[ -f "$path" ]] || { echo "Missing script: $path" >&2; fatal=1; }
    done
    for path in "$REPOSED_DIR/eval_shared_f.py" "$REPOSED_DIR/eval_varying_f.py"; do
        [[ -f "$path" ]] || { echo "Missing RePoseD script: $path" >&2; fatal=1; }
    done
    for scene in "${SCENES[@]}"; do
        [[ -d "$(scene_sparse_dir "$scene")" ]] || { echo "Missing sparse dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_depth_dir "$scene")" ]] || { echo "Missing raw depth dir for $scene" >&2; fatal=1; }
        [[ -f "$(scene_pairs_file "$scene")" ]] || { echo "Missing pairs file for $scene" >&2; fatal=1; }
    done
    while IFS=$'\t' read -r key scene mode; do
        [[ "$key" == "config_key" ]] && continue
        if [[ ! -f "$(benchmark_path "$key" "$scene")" && ! -d "$MATCHES_ROOT/$key/$scene" ]]; then
            echo "Missing both benchmark and matches for $key $scene" >&2
            fatal=1
        fi
    done < <(printf "config_key\tscene\tsolver_mode\n"; task_lines)
    return "$fatal"
}

write_plan() {
    local gpus_csv="$1" gpu_count="$2" task_count="$3" missing_count="$4"
    cat > "$PLAN_PATH" <<EOF
# Chapter 6 Selected All-Modes Launch Plan

Generated: $(date --iso-8601=seconds)

## Exact Command

\`\`\`bash
cd $ROOT
CUDA_VISIBLE_DEVICES=$gpus_csv scripts/run_ch6_selected_all_modes.sh --launch
\`\`\`

## Screen

- Screen session: \`$SCREEN_NAME\`
- Attach: \`screen -r $SCREEN_NAME\`
- Detach: \`Ctrl-a d\`

## Workload

- GPU allocation requested: \`$gpus_csv\`
- Worker shards: $gpu_count
- Planned RePoseD tasks: $task_count
- Missing RePoseD tasks at launch: $missing_count
- Solver modes: shared_focal, varying_focal
- Calibrated summaries are reused.

## Runtime Estimate

- Estimate: $RUNTIME_ESTIMATE

## Outputs

- \`output_v2/results_v2/<selected-config>/<scene>/shared_focal-*.json\`
- \`output_v2/results_v2/<selected-config>/<scene>/varying_focal-*.json\`
- \`output_v2/csv/chapter6_selected_all_modes.csv\`
- \`output_v2/csv/chapter6_selected_per_scene.csv\`
- \`output_v2/reports/chapter6_selected_all_modes_report.md\`
EOF
}

aggregate_report() {
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch6_selected_all_modes.py" \
        --write \
        --command "CUDA_VISIBLE_DEVICES=$GPUS_CSV scripts/run_ch6_selected_all_modes.sh --launch" \
        --gpus "$GPUS_CSV" \
        --runtime-estimate "$RUNTIME_ESTIMATE" || true
}

pack_if_missing() {
    local key="$1" scene="$2" log_path="$3"
    local bench
    bench="$(benchmark_path "$key" "$scene")"
    if [[ -f "$bench" && "$FORCE" != "1" ]]; then
        return 0
    fi
    if [[ ! -d "$MATCHES_ROOT/$key/$scene" ]]; then
        record_failure "pack" "$key" "$scene" "" "missing_matches" "Cannot pack missing matches directory" "$log_path"
        return 1
    fi
    run_logged "$log_path" "$REPOSED_PY" "$ROOT/scripts/pack_benchmark.py" \
        --matches_dir "$MATCHES_ROOT/$key/$scene" \
        --depth_dir "$(scene_depth_dir "$scene")" \
        --sparse_dir "$(scene_sparse_dir "$scene")" \
        --pairs_file "$(scene_pairs_file "$scene")" \
        --output "$bench" \
        --limit "$PAIR_LIMIT" || {
            record_failure "pack" "$key" "$scene" "" "failed" "Benchmark packing failed" "$log_path"
            return 1
        }
}

run_eval_task() {
    local key="$1" scene="$2" mode="$3"
    local log_path="$LOG_ROOT/${key}_${scene}_${mode}.log"
    if summary_exists "$key" "$scene" "$mode" && [[ "$FORCE" != "1" ]]; then
        log "Skip complete $key $scene $mode"
        return 0
    fi
    pack_if_missing "$key" "$scene" "$log_path" || return 1
    local eval_script
    case "$mode" in
        shared_focal) eval_script="eval_shared_f.py" ;;
        varying_focal) eval_script="eval_varying_f.py" ;;
        *) record_failure "eval" "$key" "$scene" "$mode" "bad_mode" "Unknown solver mode" "$log_path"; return 1 ;;
    esac
    mkdir -p "$RESULTS_ROOT/$key/$scene"
    run_logged_cwd "$log_path" "$REPOSED_DIR" "$REPOSED_PY" "$eval_script" "$(benchmark_path "$key" "$scene")" \
        -nw "$REPOSED_NUM_WORKERS" \
        --thesis \
        --output_dir "$RESULTS_ROOT/$key/$scene" \
        --max_epipolar_error 2.0 \
        --reproj_threshold 16.0 || {
            record_failure "eval" "$key" "$scene" "$mode" "failed" "RePoseD evaluation failed" "$log_path"
            return 1
        }
}

run_eval_worker() {
    setup_env
    init_dirs
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.started"
    local idx=0 key scene mode
    while IFS=$'\t' read -r key scene mode; do
        [[ "$key" == "config_key" ]] && continue
        if (( idx % NUM_SHARDS == SHARD )); then
            log "Shard $SHARD GPU $GPU running $key $scene $mode"
            run_eval_task "$key" "$scene" "$mode" || true
        fi
        idx=$((idx + 1))
    done < "$TASKS_TSV"
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.done"
}

run_supervisor() {
    setup_env
    init_dirs
    init_failures
    preflight
    write_tasks_file
    echo "$(date +%s)" > "$STATUS_DIR/supervisor.started"

    IFS=',' read -r -a gpus <<< "$GPUS_CSV"
    local gpu_count="${#gpus[@]}"
    local pids=()
    local shard gpu log_path
    local force_args=()
    [[ "$FORCE" == "1" ]] && force_args=(--force)
    for shard in "${!gpus[@]}"; do
        gpu="${gpus[$shard]}"
        log_path="$LOG_ROOT/worker_${shard}_gpu${gpu}.log"
        log "Starting worker shard $shard/$gpu_count on GPU $gpu"
        (
            cd "$ROOT"
            "$0" --worker eval --gpu "$gpu" --shard "$shard" --num-shards "$gpu_count" "${force_args[@]}"
        ) > "$log_path" 2>&1 &
        pids+=("$!")
    done

    local code=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            code=1
            record_failure "worker" "ch6_selected_all_modes" "" "" "failed" "A worker exited non-zero" "$LOG_ROOT/${SCREEN_NAME}.screen.log"
        fi
    done

    if [[ "$code" == "0" ]]; then
        echo "$(date +%s)" > "$STATUS_DIR/supervisor.done"
    else
        echo "$(date +%s)" > "$STATUS_DIR/supervisor.failed"
    fi
    log "Workers finished, aggregating selected all-mode report"
    aggregate_report
    return "$code"
}

count_missing_tasks() {
    local missing=0 key scene mode
    while IFS=$'\t' read -r key scene mode; do
        [[ "$key" == "config_key" ]] && continue
        if ! summary_exists "$key" "$scene" "$mode" || [[ "$FORCE" == "1" ]]; then
            missing=$((missing + 1))
        fi
    done < <(printf "config_key\tscene\tsolver_mode\n"; task_lines)
    echo "$missing"
}

launch() {
    setup_env
    init_dirs
    init_failures
    local gpus
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        gpus=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '))
    else
        gpus=($(detect_gpus))
    fi
    if [[ "${#gpus[@]}" -eq 0 ]]; then
        gpus=(0)
    fi
    GPUS_CSV="$(join_by_comma "${gpus[@]}")"
    local task_count missing_count
    task_count="$(task_lines | wc -l | tr -d ' ')"
    missing_count="$(count_missing_tasks)"
    write_plan "$GPUS_CSV" "${#gpus[@]}" "$task_count" "$missing_count"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        cat "$PLAN_PATH"
        exit 0
    fi
    preflight
    write_tasks_file
    aggregate_report
    log "Plan written to $PLAN_PATH"
    log "Launching selected all-mode eval on shards: $GPUS_CSV"
    if [[ "$NO_SCREEN" == "1" ]]; then
        GPUS_CSV="$GPUS_CSV" run_supervisor
    else
        if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
            echo "Screen session already exists: $SCREEN_NAME" >&2
            echo "Attach with: screen -r $SCREEN_NAME" >&2
            exit 1
        fi
        local force_text=""
        [[ "$FORCE" == "1" ]] && force_text=" --force"
        screen -dmS "$SCREEN_NAME" bash -lc "cd '$ROOT' && RUNTIME_ESTIMATE='$RUNTIME_ESTIMATE' GPUS_CSV='$GPUS_CSV' '$0' --worker supervisor --gpus '$GPUS_CSV'$force_text > '$LOG_ROOT/${SCREEN_NAME}.screen.log' 2>&1"
        log "Started screen session: $SCREEN_NAME"
        log "Attach: screen -r $SCREEN_NAME"
        log "Progress log: $LOG_ROOT/${SCREEN_NAME}.screen.log"
    fi
}

case "$WORKER" in
    eval) run_eval_worker ;;
    supervisor) run_supervisor ;;
    "")
        if [[ "$PLAN_ONLY" == "1" ]]; then
            init_dirs
            gpus=($(detect_gpus))
            [[ "${#gpus[@]}" -gt 0 ]] || gpus=(0)
            GPUS_CSV="$(join_by_comma "${gpus[@]}")"
            write_plan "$GPUS_CSV" "${#gpus[@]}" "$(task_lines | wc -l | tr -d ' ')" "$(count_missing_tasks)"
            cat "$PLAN_PATH"
        else
            launch
        fi
        ;;
    *) echo "Unknown worker: $WORKER" >&2; usage >&2; exit 2 ;;
esac
