#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DINOV3_PY="${DINOV3_PY:-/home.stud/gorbuden/.conda/envs/dinov3/bin/python}"
REPOSED_PY="${REPOSED_PY:-/home.stud/gorbuden/.conda/envs/reposed/bin/python}"

SCENES=(sacre_coeur reichstag st_peters_square)
MODES_CSV="calibrated,shared_focal,varying_focal"

OLD_PROJ_CKPT="$ROOT/experiments/phase2_projection_wide/best.pt"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-8}"
SOURCE_CACHE_MAX_POINTS="${SOURCE_CACHE_MAX_POINTS:-2048}"
PAIR_LIMIT="${PAIR_LIMIT:-15000}"
REPOSED_NUM_WORKERS="${REPOSED_NUM_WORKERS:-16}"
SCREEN_NAME="${SCREEN_NAME:-ch6_oldproj_eval}"
SKIP_MNN_BASELINE="${SKIP_MNN_BASELINE:-0}"
PRIORITIZE_EXPANDED151="${PRIORITIZE_EXPANDED151:-0}"
AGGREGATE_AFTER_TASK="${AGGREGATE_AFTER_TASK:-1}"

LOG_ROOT="$ROOT/output_v2/logs/ch6_oldproj_eval"
STATUS_DIR="$LOG_ROOT/status"
FAILURES_TSV="$LOG_ROOT/failures.tsv"
TASKS_TSV="$LOG_ROOT/tasks.tsv"
MATCHES_ROOT="$ROOT/output_v2/matches_v2"
BENCH_ROOT="$ROOT/output_v2/benchmarks_v2"
RESULTS_ROOT="$ROOT/output_v2/results_v2"
FEATURE_ROOT="$ROOT/output_v2/feature_cache_raw"
SP_ROOT="$ROOT/output_v2/sp_cache_raw"
TIMING_ROOT="$ROOT/output_v2/timing"
REPOSED_DIR="$ROOT/external/RePoseD"
PLAN_PATH="$ROOT/output_v2/reports/chapter6_oldproj_eval_launch_plan.md"
MANIFEST_PATH="$ROOT/output_v2/reports/chapter6_oldproj_eval_manifest.json"

if [[ "$ENSEMBLE_SIZE" == "8" ]]; then
    SHARED_OLDPROJ_CACHE_KEY="${SHARED_OLDPROJ_CACHE_KEY:-ch5_diag_proj_wide_b16_ens8_sp_mnn_mp2048}"
elif [[ "$ENSEMBLE_SIZE" == "2" ]]; then
    SHARED_OLDPROJ_CACHE_KEY="${SHARED_OLDPROJ_CACHE_KEY:-ch5_diag_proj_wide_b16_ens2_sp_mnn_mp2048}"
else
    SHARED_OLDPROJ_CACHE_KEY="${SHARED_OLDPROJ_CACHE_KEY:-}"
fi

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
  scripts/run_ch6_oldproj_eval.sh --launch [--force] [--no-screen]
  scripts/run_ch6_oldproj_eval.sh --plan-only
  scripts/run_ch6_oldproj_eval.sh --worker supervisor --gpus 0,1,3 [--force]
  scripts/run_ch6_oldproj_eval.sh --worker eval --gpu GPU --shard I --num-shards N [--force]

Evaluation only. Uses the old matcher-compatible projection checkpoint:
  experiments/phase2_projection_wide/best.pt

Default DIFT ensemble is ${ENSEMBLE_SIZE}. Set ENSEMBLE_SIZE=2 or ENSEMBLE_SIZE=8 to override.
Set SKIP_MNN_BASELINE=1 to omit the descriptor-only baseline.
Set PRIORITIZE_EXPANDED151=1 to queue expanded-151 learned matcher tasks first.
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

export ENSEMBLE_SIZE SKIP_MNN_BASELINE PRIORITIZE_EXPANDED151 AGGREGATE_AFTER_TASK GPUS_CSV

setup_env() {
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HOME="${HF_HOME:-/home.stud/gorbuden/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_ch6_oldproj_eval}"
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
        "$MATCHES_ROOT" "$BENCH_ROOT" "$RESULTS_ROOT" "$FEATURE_ROOT" "$SP_ROOT" "$TIMING_ROOT"
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

scene_images_dir() {
    echo "$ROOT/datasets/phototourism/$1/dense/images"
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

oldproj_feature_cache_dir() {
    local scene="$1"
    if [[ -n "$SHARED_OLDPROJ_CACHE_KEY" && -d "$FEATURE_ROOT/$SHARED_OLDPROJ_CACHE_KEY/$scene" ]]; then
        echo "$FEATURE_ROOT/$SHARED_OLDPROJ_CACHE_KEY/$scene"
    else
        echo "$FEATURE_ROOT/ch6_oldproj_shared_ens${ENSEMBLE_SIZE}/$scene"
    fi
}

summary_glob() {
    local key="$1" scene="$2" mode="$3"
    printf "%s/%s/%s/%s-%s_%s-2.0t_summary.json" "$RESULTS_ROOT" "$key" "$scene" "$mode" "$key" "$scene"
}

summary_exists() {
    compgen -G "$(summary_glob "$1" "$2" "$3")" >/dev/null
}

all_summaries_exist() {
    local key="$1" scene="$2" modes_csv="$3" mode
    IFS=',' read -r -a modes <<< "$modes_csv"
    for mode in "${modes[@]}"; do
        summary_exists "$key" "$scene" "$mode" || return 1
    done
    return 0
}

task_lines() {
    local scene th suffix key
    local expanded="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"

    if [[ "$PRIORITIZE_EXPANDED151" == "1" ]]; then
        for th in 0.05 0.10 0.15 0.20; do
            case "$th" in
                0.05) suffix=ft005 ;;
                0.10) suffix=ft010 ;;
                0.15) suffix=ft015 ;;
                0.20) suffix=ft020 ;;
            esac
            key="ch6_oldproj_expanded151_lg_${suffix}_sp_mnn_mp2048"
            for scene in "${SCENES[@]}"; do
                printf "lg|%s|%s|%s|%s|%s|adapt|expanded 151 old-projection threshold sweep\n" \
                    "$key" "$scene" "$MODES_CSV" "$th" "$expanded"
            done
        done
    fi

    if [[ "$SKIP_MNN_BASELINE" != "1" ]]; then
        for scene in "${SCENES[@]}"; do
            printf "mnn|%s|%s|%s|none|none|adapt|old projection MNN baseline\n" \
                "ch6_input_oldproj_wide_t007_mnn_sp_mnn_mp2048" "$scene" "$MODES_CSV"
        done
    fi

    for th in 0.00 0.02 0.05 0.10 0.15 0.20; do
        case "$th" in
            0.00) suffix=ft000 ;;
            0.02) suffix=ft002 ;;
            0.05) suffix=ft005 ;;
            0.10) suffix=ft010 ;;
            0.15) suffix=ft015 ;;
            0.20) suffix=ft020 ;;
        esac
        key="ch6_oldproj_zeroshot_lg_${suffix}_sp_mnn_mp2048"
        for scene in "${SCENES[@]}"; do
            printf "lg|%s|%s|%s|%s|none|adapt|zero-shot old-projection threshold sweep\n" \
                "$key" "$scene" "$MODES_CSV" "$th"
        done
    done

    for scene in "${SCENES[@]}"; do
        printf "lg|%s|%s|%s|0.10|none|noadapt|zero-shot old-projection noadapt diagnostic\n" \
            "ch6_oldproj_zeroshot_lg_noadapt_ft010_sp_mnn_mp2048" "$scene" "$MODES_CSV"
    done

    local warm="$ROOT/external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar"
    local scratch="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar"
    local joint="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar"

    for th in 0.10 0.15; do
        case "$th" in
            0.10) suffix=ft010 ;;
            0.15) suffix=ft015 ;;
        esac
        for scene in "${SCENES[@]}"; do
            printf "lg|%s|%s|%s|%s|%s|adapt|warm-start old-projection evaluation\n" \
                "ch6_oldproj_warmstart_lg_full_v1_${suffix}_sp_mnn_mp2048" "$scene" "$MODES_CSV" "$th" "$warm"
            printf "lg|%s|%s|%s|%s|%s|adapt|scratch stage2 old-projection evaluation\n" \
                "ch6_oldproj_scratch_stage2_lg_v1_${suffix}_sp_mnn_mp2048" "$scene" "$MODES_CSV" "$th" "$scratch"
        done
    done

    if [[ "$PRIORITIZE_EXPANDED151" != "1" ]]; then
        for th in 0.05 0.10 0.15 0.20; do
            case "$th" in
                0.05) suffix=ft005 ;;
                0.10) suffix=ft010 ;;
                0.15) suffix=ft015 ;;
                0.20) suffix=ft020 ;;
            esac
            key="ch6_oldproj_expanded151_lg_${suffix}_sp_mnn_mp2048"
            for scene in "${SCENES[@]}"; do
                printf "lg|%s|%s|%s|%s|%s|adapt|expanded 151 old-projection threshold sweep\n" \
                    "$key" "$scene" "$MODES_CSV" "$th" "$expanded"
            done
        done
    fi

    for scene in "${SCENES[@]}"; do
        printf "lg|%s|%s|%s|0.10|%s|adapt|joint checkpoint diagnostic with old projection\n" \
            "ch6_oldproj_joint_unfrozen_proj60_ft010_sp_mnn_mp2048" "$scene" "$MODES_CSV" "$joint"
    done
}

all_config_keys() {
    task_lines | awk -F'|' '{print $2}' | sort -u
}

preflight() {
    local fatal=0 ckpt scene count
    for exe in "$DINOV3_PY" "$REPOSED_PY"; do
        [[ -x "$exe" ]] || { echo "Missing executable: $exe" >&2; fatal=1; }
    done
    for path in "$ROOT/scripts/projection_matches.py" "$ROOT/scripts/lightglue_projection_matches.py" \
        "$ROOT/scripts/pack_benchmark.py" "$ROOT/scripts/aggregate_ch6_oldproj_eval.py"; do
        [[ -f "$path" ]] || { echo "Missing script: $path" >&2; fatal=1; }
    done
    for path in "$REPOSED_DIR/eval.py" "$REPOSED_DIR/eval_shared_f.py" "$REPOSED_DIR/eval_varying_f.py"; do
        [[ -f "$path" ]] || { echo "Missing RePoseD script: $path" >&2; fatal=1; }
    done
    [[ -f "$OLD_PROJ_CKPT" ]] || { echo "Missing old projection checkpoint: $OLD_PROJ_CKPT" >&2; fatal=1; }
    [[ -f "$ROOT/experiments/phase2_projection_wide/config.json" ]] || { echo "Missing old projection config.json" >&2; fatal=1; }
    for scene in "${SCENES[@]}"; do
        [[ -d "$(scene_images_dir "$scene")" ]] || { echo "Missing images dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_sparse_dir "$scene")" ]] || { echo "Missing sparse dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_depth_dir "$scene")" ]] || { echo "Missing raw depth dir for $scene" >&2; fatal=1; }
        [[ -f "$(scene_pairs_file "$scene")" ]] || { echo "Missing pairs file for $scene" >&2; fatal=1; }
        [[ -d "$FEATURE_ROOT/dinov3_l-8_sp_mnn_mp${SOURCE_CACHE_MAX_POINTS}/$scene" ]] || {
            echo "Missing DINOv3 source cache for $scene" >&2; fatal=1;
        }
        [[ -d "$FEATURE_ROOT/dift_t0_up2_ens${ENSEMBLE_SIZE}_sp_mnn_mp${SOURCE_CACHE_MAX_POINTS}/$scene" ]] || {
            echo "Missing DIFT ens${ENSEMBLE_SIZE} source cache for $scene" >&2; fatal=1;
        }
        count="$(find "$(oldproj_feature_cache_dir "$scene")" -maxdepth 1 -type f -name '*_projection_desc.pt' 2>/dev/null | wc -l || true)"
        log "Preflight cache $scene: $(oldproj_feature_cache_dir "$scene") has $count projected descriptor files"
    done
    for ckpt in \
        "$ROOT/external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar"; do
        [[ -f "$ckpt" ]] || { echo "Missing matcher checkpoint: $ckpt" >&2; fatal=1; }
    done
    return "$fatal"
}

write_plan() {
    local gpus_csv="$1" gpu_count="$2" task_count="$3"
    cat > "$PLAN_PATH" <<EOF
# Chapter 6 Old-Projection Evaluation Launch Plan

Generated: $(date --iso-8601=seconds)

## Exact Command

\`\`\`bash
cd $ROOT
ENSEMBLE_SIZE=$ENSEMBLE_SIZE SKIP_MNN_BASELINE=$SKIP_MNN_BASELINE PRIORITIZE_EXPANDED151=$PRIORITIZE_EXPANDED151 CUDA_VISIBLE_DEVICES=$gpus_csv scripts/run_ch6_oldproj_eval.sh --launch
\`\`\`

## Screen

- Screen session: \`$SCREEN_NAME\`
- Attach: \`screen -r $SCREEN_NAME\`
- Detach: \`Ctrl-a d\`

## Progress

\`\`\`bash
tail -f output_v2/logs/ch6_oldproj_eval/${SCREEN_NAME}.screen.log
ls output_v2/logs/ch6_oldproj_eval/status
tail -f output_v2/logs/ch6_oldproj_eval/failures.tsv
nvidia-smi
\`\`\`

## Projection And Ensemble

- Projection checkpoint: \`experiments/phase2_projection_wide/best.pt\`
- Projection architecture: \`1664 -> 1024 -> 256\`
- Projection temperature: \`0.07\`
- DIFT ensemble used for this run: \`$ENSEMBLE_SIZE\`
- MNN baseline skipped: \`$SKIP_MNN_BASELINE\`
- Expanded-151 queued first: \`$PRIORITIZE_EXPANDED151\`
- Shared projected descriptor cache: \`output_v2/feature_cache_raw/$SHARED_OLDPROJ_CACHE_KEY/<scene>\`

Training configs for the learned matchers record \`dift_ensemble_size: 1\`, while the historical Chapter 7 evaluation script used \`ensemble_size=8\`. This launch uses ensemble $ENSEMBLE_SIZE to reproduce the historical evaluation descriptor as closely as possible under the final raw-image protocol.

## Workload

- GPU allocation: \`$gpus_csv\`
- Number of GPU shards: $gpu_count
- Number of scene/config matching tasks: $task_count
- Scenes: ${SCENES[*]}
- Solver modes for every produced benchmark: calibrated, shared_focal, varying_focal

## Runtime Estimate

- Descriptor recache: expected near-zero for ensemble $ENSEMBLE_SIZE because projected old-projection caches already exist.
- With $gpu_count free GPUs: about 4-8 hours.
- On one GPU: about 12-22 hours.
- If the shared projected cache is ignored or incomplete: add roughly 3-8 hours for ensemble 8 feature/projection rebuild.

## Expected Outputs

- \`output_v2/results_v2/ch6_oldproj_*/\`
- \`output_v2/matches_v2/ch6_oldproj_*/\`
- \`output_v2/benchmarks_v2/ch6_oldproj_*_<scene>.h5\`
- \`output_v2/csv/chapter6_oldproj_learned_matcher_main.csv\`
- \`output_v2/reports/chapter6_oldproj_learned_matcher_eval_report.md\`
- \`output_v2/reports/chapter6_oldproj_eval_manifest.json\`
EOF
}

aggregate_snapshot() {
    [[ "$AGGREGATE_AFTER_TASK" == "1" ]] || return 0
    local lock_path="$LOG_ROOT/aggregate.lock"
    local log_path="$LOG_ROOT/aggregate_snapshot.log"
    (
        flock -w 120 9 || exit 0
        "$DINOV3_PY" "$ROOT/scripts/aggregate_ch6_oldproj_eval.py" \
            --write \
            --command "ENSEMBLE_SIZE=$ENSEMBLE_SIZE SKIP_MNN_BASELINE=$SKIP_MNN_BASELINE PRIORITIZE_EXPANDED151=$PRIORITIZE_EXPANDED151 CUDA_VISIBLE_DEVICES=$GPUS_CSV scripts/run_ch6_oldproj_eval.sh --launch" \
            --gpus "$GPUS_CSV" \
            --ensemble-size "$ENSEMBLE_SIZE" >> "$log_path" 2>&1 || true
    ) 9>"$lock_path"
}

write_tasks_file() {
    task_lines > "$TASKS_TSV"
}

write_manifest_initial() {
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch6_oldproj_eval.py" \
        --manifest-only \
        --command "ENSEMBLE_SIZE=$ENSEMBLE_SIZE SKIP_MNN_BASELINE=$SKIP_MNN_BASELINE PRIORITIZE_EXPANDED151=$PRIORITIZE_EXPANDED151 CUDA_VISIBLE_DEVICES=$GPUS_CSV scripts/run_ch6_oldproj_eval.sh --launch" \
        --gpus "$GPUS_CSV" \
        --ensemble-size "$ENSEMBLE_SIZE" || true
}

run_pack_and_eval() {
    local key="$1" scene="$2" modes_csv="$3" log_path="$4"
    local bench matches out_dir mode eval_script
    bench="$(benchmark_path "$key" "$scene")"
    matches="$MATCHES_ROOT/$key/$scene"
    out_dir="$RESULTS_ROOT/$key/$scene"
    mkdir -p "$out_dir"
    if [[ ! -f "$bench" || "$FORCE" == "1" ]]; then
        run_logged "$log_path" "$REPOSED_PY" "$ROOT/scripts/pack_benchmark.py" \
            --matches_dir "$matches" \
            --depth_dir "$(scene_depth_dir "$scene")" \
            --sparse_dir "$(scene_sparse_dir "$scene")" \
            --pairs_file "$(scene_pairs_file "$scene")" \
            --output "$bench" \
            --limit "$PAIR_LIMIT" || {
                record_failure "pack" "$key" "$scene" "" "failed" "Benchmark packing failed" "$log_path"
                return 1
            }
    fi

    IFS=',' read -r -a modes <<< "$modes_csv"
    for mode in "${modes[@]}"; do
        if summary_exists "$key" "$scene" "$mode" && [[ "$FORCE" != "1" ]]; then
            continue
        fi
        case "$mode" in
            calibrated) eval_script="eval.py" ;;
            shared_focal) eval_script="eval_shared_f.py" ;;
            varying_focal) eval_script="eval_varying_f.py" ;;
            *) record_failure "eval" "$key" "$scene" "$mode" "bad_mode" "Unknown solver mode" "$log_path"; continue ;;
        esac
        run_logged_cwd "$log_path" "$REPOSED_DIR" "$REPOSED_PY" "$eval_script" "$bench" \
            -nw "$REPOSED_NUM_WORKERS" \
            --thesis \
            --output_dir "$out_dir" \
            --max_epipolar_error 2.0 \
            --reproj_threshold 16.0 || {
                record_failure "eval" "$key" "$scene" "$mode" "failed" "RePoseD evaluation failed" "$log_path"
                continue
            }
    done
}

run_mnn_task() {
    local key="$1" scene="$2" modes_csv="$3" log_path
    log_path="$LOG_ROOT/${key}_${scene}.log"
    if all_summaries_exist "$key" "$scene" "$modes_csv" && [[ "$FORCE" != "1" ]]; then
        log "Skip complete $key $scene $modes_csv"
        return
    fi
    if [[ ! -f "$(benchmark_path "$key" "$scene")" || "$FORCE" == "1" ]]; then
        mkdir -p "$MATCHES_ROOT/$key/$scene"
        CUDA_VISIBLE_DEVICES="$GPU" run_logged "$log_path" "$DINOV3_PY" "$ROOT/scripts/projection_matches.py" \
            --pairs_file "$(scene_pairs_file "$scene")" \
            --images_dir "$(scene_images_dir "$scene")" \
            --output_dir "$MATCHES_ROOT/$key/$scene" \
            --scene "$scene" \
            --checkpoint "$OLD_PROJ_CKPT" \
            --projection_tag "${key%_sp_mnn_mp2048}" \
            --max_points 2048 \
            --feat_level -8 \
            --img_size 768 768 \
            --t 0 \
            --up_ft_index 2 \
            --ensemble_size "$ENSEMBLE_SIZE" \
            --alpha 0.5 \
            --feature_cache "$(oldproj_feature_cache_dir "$scene")" \
            --cache_root "$FEATURE_ROOT" \
            --sp_cache_dir "$SP_ROOT/$scene" \
            --device cuda \
            --limit "$PAIR_LIMIT" \
            --timing_output "$TIMING_ROOT/${key}_${scene}_timing.json" || {
                record_failure "match_mnn" "$key" "$scene" "" "failed" "Projection MNN matching failed" "$log_path"
                return
            }
    fi
    run_pack_and_eval "$key" "$scene" "$modes_csv" "$log_path"
}

run_lg_task() {
    local key="$1" scene="$2" modes_csv="$3" threshold="$4" lg_ckpt="$5" adaptivity="$6" method_note="$7"
    local log_path="$LOG_ROOT/${key}_${scene}.log"
    [[ "$lg_ckpt" == "none" ]] && lg_ckpt=""
    if all_summaries_exist "$key" "$scene" "$modes_csv" && [[ "$FORCE" != "1" ]]; then
        log "Skip complete $key $scene $modes_csv"
        return
    fi
    if [[ ! -f "$(benchmark_path "$key" "$scene")" || "$FORCE" == "1" ]]; then
        mkdir -p "$MATCHES_ROOT/$key/$scene"
        local cmd=(
            "$DINOV3_PY" "$ROOT/scripts/lightglue_projection_matches.py"
            --pairs_file "$(scene_pairs_file "$scene")"
            --images_dir "$(scene_images_dir "$scene")"
            --output_dir "$MATCHES_ROOT/$key/$scene"
            --scene "$scene"
            --config_key "$key"
            --checkpoint "$OLD_PROJ_CKPT"
            --filter_threshold "$threshold"
            --seed 42
            --max_points 2048
            --source_cache_max_points "$SOURCE_CACHE_MAX_POINTS"
            --feat_level -8
            --img_size 768 768
            --t 0
            --up_ft_index 2
            --ensemble_size "$ENSEMBLE_SIZE"
            --alpha 0.5
            --feature_cache "$(oldproj_feature_cache_dir "$scene")"
            --cache_root "$FEATURE_ROOT"
            --sp_cache_dir "$SP_ROOT/$scene"
            --device cuda
            --limit "$PAIR_LIMIT"
            --raw_images
            --timing_output "$TIMING_ROOT/${key}_${scene}_timing.json"
        )
        if [[ -n "$lg_ckpt" ]]; then
            if [[ ! -f "$lg_ckpt" ]]; then
                record_failure "match_lg" "$key" "$scene" "" "checkpoint_missing" "Missing LightGlue checkpoint: $lg_ckpt" "$log_path" "$method_note"
                return
            fi
            cmd+=(--lightglue_checkpoint "$lg_ckpt")
        fi
        if [[ "$adaptivity" == "noadapt" ]]; then
            cmd+=(--depth_confidence -1 --width_confidence -1)
        fi
        CUDA_VISIBLE_DEVICES="$GPU" run_logged "$log_path" "${cmd[@]}" || {
            record_failure "match_lg" "$key" "$scene" "" "failed_or_incompatible" "LightGlue matching failed or checkpoint incompatible" "$log_path" "$method_note"
            return
        }
    fi
    run_pack_and_eval "$key" "$scene" "$modes_csv" "$log_path"
}

run_eval_worker() {
    setup_env
    init_dirs
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.started"
    local idx=0 kind key scene modes threshold ckpt adapt note
    while IFS='|' read -r kind key scene modes threshold ckpt adapt note; do
        if (( idx % NUM_SHARDS == SHARD )); then
            log "Shard $SHARD GPU $GPU running $kind $key $scene $modes"
            case "$kind" in
                mnn) run_mnn_task "$key" "$scene" "$modes" || true ;;
                lg) run_lg_task "$key" "$scene" "$modes" "$threshold" "$ckpt" "$adapt" "$note" || true ;;
                *) record_failure "task" "$key" "$scene" "$modes" "bad_kind" "Unknown task kind $kind" "$FAILURES_TSV" ;;
            esac
            aggregate_snapshot || true
        fi
        idx=$((idx + 1))
    done < <(task_lines)
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch6_oldproj_eval.py" --write || true
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.done"
}

run_supervisor() {
    setup_env
    init_dirs
    init_failures
    preflight
    write_tasks_file
    write_manifest_initial
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
            record_failure "worker" "ch6_oldproj_eval" "" "" "failed" "A worker exited non-zero" "$LOG_ROOT/${SCREEN_NAME}.screen.log"
        fi
    done

    log "Workers finished, aggregating"
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch6_oldproj_eval.py" \
        --write \
        --command "ENSEMBLE_SIZE=$ENSEMBLE_SIZE SKIP_MNN_BASELINE=$SKIP_MNN_BASELINE PRIORITIZE_EXPANDED151=$PRIORITIZE_EXPANDED151 CUDA_VISIBLE_DEVICES=$GPUS_CSV scripts/run_ch6_oldproj_eval.sh --launch" \
        --gpus "$GPUS_CSV" \
        --ensemble-size "$ENSEMBLE_SIZE" || code=1

    if [[ "$code" == "0" ]]; then
        echo "$(date +%s)" > "$STATUS_DIR/supervisor.done"
    else
        echo "$(date +%s)" > "$STATUS_DIR/supervisor.failed"
    fi
    return "$code"
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
        echo "No free GPUs detected. Set CUDA_VISIBLE_DEVICES explicitly." >&2
        exit 1
    fi
    GPUS_CSV="$(join_by_comma "${gpus[@]}")"
    local task_count
    task_count="$(task_lines | wc -l | tr -d ' ')"
    write_plan "$GPUS_CSV" "${#gpus[@]}" "$task_count"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        cat "$PLAN_PATH"
        exit 0
    fi
    preflight
    log "Plan written to $PLAN_PATH"
    log "Launching old-projection Chapter 6 eval on GPUs: $GPUS_CSV"
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
        screen -dmS "$SCREEN_NAME" bash -lc "cd '$ROOT' && ENSEMBLE_SIZE='$ENSEMBLE_SIZE' SKIP_MNN_BASELINE='$SKIP_MNN_BASELINE' PRIORITIZE_EXPANDED151='$PRIORITIZE_EXPANDED151' AGGREGATE_AFTER_TASK='$AGGREGATE_AFTER_TASK' GPUS_CSV='$GPUS_CSV' '$0' --worker supervisor --gpus '$GPUS_CSV'$force_text > '$LOG_ROOT/${SCREEN_NAME}.screen.log' 2>&1"
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
            write_plan "$GPUS_CSV" "${#gpus[@]}" "$(task_lines | wc -l | tr -d ' ')"
            cat "$PLAN_PATH"
        else
            launch
        fi
        ;;
    *) echo "Unknown worker: $WORKER" >&2; usage >&2; exit 2 ;;
esac
