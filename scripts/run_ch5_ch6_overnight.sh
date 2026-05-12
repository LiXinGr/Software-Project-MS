#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DINOV3_PY="${DINOV3_PY:-/home.stud/gorbuden/.conda/envs/dinov3/bin/python}"
REPOSED_PY="${REPOSED_PY:-/home.stud/gorbuden/.conda/envs/reposed/bin/python}"

SCENES=(sacre_coeur reichstag st_peters_square)
MODES_ALL=(calibrated shared_focal varying_focal)
TRAIN_SCENES=(0080 0042 0380 0000 0366 0001 0005 0237 0011 0148)

NEW_TRAIN_KEY="ch5_b16_train_proj_temp005_h1024_d256"
NEW_EVAL_KEY="ch5_b16_eval_proj_temp005_h1024_d256_sp_mnn_mp2048"
NEW_PROJ_DIR="$ROOT/experiments/$NEW_TRAIN_KEY"
NEW_PROJ_CKPT="$NEW_PROJ_DIR/best.pt"

LOG_ROOT="$ROOT/output_v2/logs/ch5_ch6_overnight"
STATUS_DIR="$LOG_ROOT/status"
FAILURES_TSV="$LOG_ROOT/failures.tsv"
PLAN_PATH="$ROOT/output_v2/reports/ch5_ch6_overnight_launch_plan.md"
MATCHES_ROOT="$ROOT/output_v2/matches_v2"
BENCH_ROOT="$ROOT/output_v2/benchmarks_v2"
RESULTS_ROOT="$ROOT/output_v2/results_v2"
FEATURE_ROOT="$ROOT/output_v2/feature_cache_raw"
SP_ROOT="$ROOT/output_v2/sp_cache_raw"
TIMING_ROOT="$ROOT/output_v2/timing"

PAIR_LIMIT="${PAIR_LIMIT:-15000}"
REPOSED_NUM_WORKERS="${REPOSED_NUM_WORKERS:-16}"
POLL_SECONDS="${POLL_SECONDS:-300}"
FORCE=0
PLAN_ONLY=0
NO_SCREEN=0
WORKER=""
GPU=""
SHARD=0
NUM_SHARDS=1

usage() {
    cat <<EOF
Usage:
  scripts/run_ch5_ch6_overnight.sh [--launch] [--force] [--no-screen]
  scripts/run_ch5_ch6_overnight.sh --plan-only
  scripts/run_ch5_ch6_overnight.sh --worker train --gpu GPU [--force]
  scripts/run_ch5_ch6_overnight.sh --worker eval --gpu GPU --shard I --num-shards N [--force]
  scripts/run_ch5_ch6_overnight.sh --worker finalize --gpu GPU --num-shards N [--force]

The default mode is --launch.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) shift ;;
        --force) FORCE=1; shift ;;
        --plan-only) PLAN_ONLY=1; shift ;;
        --no-screen) NO_SCREEN=1; shift ;;
        --worker) WORKER="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --shard) SHARD="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$ROOT/output_v2/reports" "$ROOT/output_v2/csv" \
    "$MATCHES_ROOT" "$BENCH_ROOT" "$RESULTS_ROOT" "$FEATURE_ROOT" "$SP_ROOT" "$TIMING_ROOT"

setup_env() {
    if [[ "${CH5CH6_LOAD_MODULE:-0}" == "1" ]] && command -v module >/dev/null 2>&1; then
        module load Anaconda3/2020.07 >/dev/null 2>&1 || true
    fi
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HOME="${HF_HOME:-/home.stud/gorbuden/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_ch5_ch6_overnight}"
    mkdir -p "$MPLCONFIGDIR"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
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

join_by_comma() {
    local IFS=,
    echo "$*"
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

summary_glob() {
    local key="$1" scene="$2" mode="$3"
    printf "%s/%s/%s/%s-*%s_%s-2.0t_summary.json" "$RESULTS_ROOT" "$key" "$scene" "$mode" "$key" "$scene"
}

summary_exists() {
    compgen -G "$(summary_glob "$1" "$2" "$3")" >/dev/null
}

all_summaries_exist() {
    local key="$1" scene="$2" modes_csv="$3"
    local mode
    IFS=',' read -r -a modes <<< "$modes_csv"
    for mode in "${modes[@]}"; do
        summary_exists "$key" "$scene" "$mode" || return 1
    done
    return 0
}

safe_clean_config_scene() {
    local key="$1" scene="$2"
    if [[ "$FORCE" != "1" ]]; then
        return
    fi
    if [[ "$key" != ch6_* && "$key" != "$NEW_EVAL_KEY" ]]; then
        echo "Refusing to force-clean non-overnight key: $key" >&2
        exit 3
    fi
    rm -rf "$MATCHES_ROOT/$key/$scene" "$RESULTS_ROOT/$key/$scene" "$FEATURE_ROOT/$key/$scene"
    rm -f "$BENCH_ROOT/${key}_${scene}.h5" "$TIMING_ROOT/${key}_${scene}_timing.json"
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

preflight() {
    local fatal=0
    for exe in "$DINOV3_PY" "$REPOSED_PY"; do
        if [[ ! -x "$exe" ]]; then
            echo "Missing executable: $exe" >&2
            fatal=1
        fi
    done
    for script in train_projection_head.py projection_matches.py lightglue_projection_matches.py pack_benchmark.py aggregate_ch5_ch6_overnight.py; do
        if [[ ! -f "$ROOT/scripts/$script" ]]; then
            echo "Missing script: scripts/$script" >&2
            fatal=1
        fi
    done
    for scene in "${TRAIN_SCENES[@]}"; do
        [[ -f "$ROOT/data/sparse_train/${scene}.pt" ]] || { echo "Missing sparse train bundle: data/sparse_train/${scene}.pt" >&2; fatal=1; }
    done
    for scene in "${SCENES[@]}"; do
        [[ -d "$(scene_images_dir "$scene")" ]] || { echo "Missing images dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_sparse_dir "$scene")" ]] || { echo "Missing sparse dir for $scene" >&2; fatal=1; }
        [[ -d "$(scene_depth_dir "$scene")" ]] || { echo "Missing raw depth dir for $scene" >&2; fatal=1; }
        [[ -f "$(scene_pairs_file "$scene")" ]] || { echo "Missing pairs file for $scene" >&2; fatal=1; }
        [[ -d "$FEATURE_ROOT/dinov3_l-8_sp_mnn_mp2048/$scene" ]] || { echo "Missing DINOv3 block16 source cache for $scene" >&2; fatal=1; }
        [[ -d "$FEATURE_ROOT/dift_t0_up2_ens2_sp_mnn_mp2048/$scene" ]] || { echo "Missing DIFT ens2 source cache for $scene" >&2; fatal=1; }
    done
    local ckpt
    for ckpt in \
        "$ROOT/external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar" \
        "$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar"; do
        if [[ ! -f "$ckpt" ]]; then
            record_failure "preflight" "$(basename "$(dirname "$ckpt")")" "" "" "checkpoint_missing" "Missing matcher checkpoint: $ckpt" "$FAILURES_TSV"
        fi
    done
    return "$fatal"
}

write_launch_plan() {
    local gpus_csv="$1"
    local gpu_count="$2"
    local train_gpu="$3"
    local force_note="no"
    [[ "$FORCE" == "1" ]] && force_note="yes"
    cat > "$PLAN_PATH" <<EOF
# Chapter 5/6 Overnight Launch Plan

Generated: $(date --iso-8601=seconds)

## Exact Command

\`\`\`bash
cd $ROOT
CUDA_VISIBLE_DEVICES=$gpus_csv scripts/run_ch5_ch6_overnight.sh --launch
\`\`\`

Force rerun of only the new overnight keys:

\`\`\`bash
cd $ROOT
CUDA_VISIBLE_DEVICES=$gpus_csv scripts/run_ch5_ch6_overnight.sh --launch --force
\`\`\`

Current launch uses force: \`$force_note\`.

## GPU Allocation

- Detected/selected GPUs: \`$gpus_csv\`.
- Training GPU: \`$train_gpu\`.
- Eval workers: one worker per selected GPU, each waiting for \`$NEW_PROJ_CKPT\`.
- GPU 2 was not selected if occupied by \`nvidia-smi\`.

## Screen Sessions

- \`ch5ch6_train_gpu${train_gpu}\`: trains \`$NEW_TRAIN_KEY\`.
EOF
    local idx=0
    IFS=',' read -r -a plan_gpus <<< "$gpus_csv"
    for gpu in "${plan_gpus[@]}"; do
        cat >> "$PLAN_PATH" <<EOF
- \`ch5ch6_eval_${idx}_gpu${gpu}\`: evaluation shard $idx/$gpu_count on GPU $gpu.
EOF
        idx=$((idx + 1))
    done
    cat >> "$PLAN_PATH" <<EOF
- \`ch5ch6_finalize\`: waits for all workers, runs threshold follow-ups, aggregation, reports, and figures.

Attach:

\`\`\`bash
screen -r ch5ch6_train_gpu${train_gpu}
screen -r ch5ch6_finalize
\`\`\`

Detach from a screen session with \`Ctrl-a d\`.

Check running sessions:

\`\`\`bash
screen -ls
\`\`\`

Check progress:

\`\`\`bash
tail -f output_v2/logs/ch5_ch6_overnight/ch5ch6_finalize.screen.log
tail -f output_v2/logs/ch5_ch6_overnight/ch5_train_${NEW_TRAIN_KEY}.log
find output_v2/results_v2 -path '*ch6_*summary.json' | wc -l
nvidia-smi
\`\`\`

## Expected Output Files

- \`experiments/$NEW_TRAIN_KEY/best.pt\`
- \`experiments/$NEW_TRAIN_KEY/config.json\`
- \`experiments/$NEW_TRAIN_KEY/train_log.json\`
- \`experiments/$NEW_TRAIN_KEY/README.md\`
- \`output_v2/matches_v2/$NEW_EVAL_KEY/<scene>/\`
- \`output_v2/benchmarks_v2/${NEW_EVAL_KEY}_<scene>.h5\`
- \`output_v2/results_v2/$NEW_EVAL_KEY/<scene>/*summary.json\`
- \`output_v2/csv/chapter5_projection_final_wide_temp005.csv\`
- \`output_v2/csv/chapter6_learned_matcher_summary.csv\`
- \`output_v2/csv/chapter6_learned_matcher_per_scene.csv\`
- \`output_v2/csv/chapter6_zeroshot_threshold_sweep.csv\`
- \`output_v2/csv/chapter6_expanded_threshold_sweep.csv\`
- \`output_v2/csv/chapter6_diagnostics.csv\`
- \`output_v2/reports/ch5_projection_final_wide_temp005_report.md\`
- \`output_v2/reports/chapter6_learned_matcher_eval_report.md\`
- \`output_v2/reports/ch5_ch6_overnight_final_report.md\`
- \`Pictures/fig_ch5_supervised_adaptation_comparison.png\`
- \`Pictures/fig_ch5_supervised_adaptation_comparison.pdf\`

## Runtime Estimates

- Chapter 5 training: about 2-3 hours for a 10-epoch projection-head run on one A5000, based on existing projection logs.
- Chapter 5 MNN evaluation: about 1-2 hours if caches stay warm; more if projection descriptors must be rebuilt.
- Chapter 6 learned-matcher evaluations: about 8-14 GPU-hours with current final descriptor caches, plus CPU RePoseD packing/eval.
- Estimated total wall-clock with $gpu_count selected GPU(s): about 8-12 hours.
- Estimated total wall-clock on one GPU: about 18-30 hours.

## Protocol Guardrails

- Raw image coordinates.
- No shared 1120px preprocessed image directory.
- No old \`mp2000\` config keys.
- No UniDepth intrinsics as solver input; packed raw-depth values and COLMAP intrinsics are used.
- Sampson threshold 2.0 px.
- Reprojection threshold 16.0 px.
- 1000 RANSAC iterations and 25 LO iterations through RePoseD defaults/flags.
- 2048 correspondences per pair.
- New result keys only: \`$NEW_EVAL_KEY\` and \`ch6_...\`.
EOF
}

projection_train_complete() {
    [[ -f "$NEW_PROJ_CKPT" && -f "$NEW_PROJ_DIR/config.json" && -f "$NEW_PROJ_DIR/train_log.json" ]]
}

write_projection_readme() {
    cat > "$NEW_PROJ_DIR/README.md" <<EOF
# $NEW_TRAIN_KEY

Projection-head training run for Chapter 5/6 overnight batch.

- Architecture: 1664 -> 1024 -> 256.
- Temperature: tau=0.05.
- Input descriptor: DINOv3+DIFT fused descriptor from sparse training bundles.
- DINOv3 target setup: ViT-L/16 block 16, feat_level=-8.
- DIFT target setup: t=0, up_ft_index=2, ensemble size 2.
- Fusion: independent branch L2 normalization, equal weights.
- Output: L2-normalized 256D descriptor.
- Training scenes: ${TRAIN_SCENES[*]}.
- Output checkpoint: best.pt.

Command and detailed metrics are in config.json and train_log.json.
EOF
}

run_train_worker() {
    setup_env
    echo "$(date +%s)" > "$STATUS_DIR/train.started"
    local log_path="$LOG_ROOT/ch5_train_${NEW_TRAIN_KEY}.log"
    if projection_train_complete && [[ "$FORCE" != "1" ]]; then
        log "Training already complete: $NEW_PROJ_CKPT"
        write_projection_readme
        echo "$(date +%s)" > "$STATUS_DIR/train.done"
        return
    fi
    if [[ -d "$NEW_PROJ_DIR" && "$FORCE" != "1" && ! -f "$NEW_PROJ_CKPT" ]]; then
        record_failure "train" "$NEW_TRAIN_KEY" "" "" "partial_output_exists" "Partial output dir exists; rerun with --force after inspection: $NEW_PROJ_DIR" "$log_path"
        echo "$(date +%s)" > "$STATUS_DIR/train.failed"
        return 1
    fi
    if [[ "$FORCE" == "1" && -d "$NEW_PROJ_DIR" ]]; then
        rm -rf "$NEW_PROJ_DIR"
    fi
    mkdir -p "$NEW_PROJ_DIR"
    local cmd=(
        "$DINOV3_PY" -u "$ROOT/scripts/train_projection_head.py"
        --sparse_dir "$ROOT/data/sparse_train"
        --scenes "${TRAIN_SCENES[@]}"
        --input_dim 1664
        --hidden_dims 1024
        --output_dim 256
        --epochs 10
        --pairs_per_epoch 50000
        --val_pairs_per_epoch 1000
        --sparse_scene_cache_size 10
        --lr 1e-3
        --weight_decay 1e-4
        --temperature 0.05
        --num_correspondences 512
        --min_correspondences 50
        --seed 42
        --device cuda:0
        --num_workers 0
        --log_interval 500
        --output_dir "$NEW_PROJ_DIR"
    )
    if CUDA_VISIBLE_DEVICES="$GPU" run_logged "$log_path" "${cmd[@]}"; then
        if [[ -f "$NEW_PROJ_CKPT" ]]; then
            write_projection_readme
            echo "$(date +%s)" > "$STATUS_DIR/train.done"
            log "Training complete: $NEW_PROJ_CKPT"
        else
            record_failure "train" "$NEW_TRAIN_KEY" "" "" "checkpoint_missing_after_train" "Training exited but best.pt is missing" "$log_path"
            echo "$(date +%s)" > "$STATUS_DIR/train.failed"
            return 1
        fi
    else
        record_failure "train" "$NEW_TRAIN_KEY" "" "" "failed" "Projection training command failed" "$log_path"
        echo "$(date +%s)" > "$STATUS_DIR/train.failed"
        return 1
    fi
}

wait_for_projection_checkpoint() {
    while true; do
        if projection_train_complete; then
            return 0
        fi
        if [[ -f "$STATUS_DIR/train.failed" ]]; then
            return 1
        fi
        log "Waiting for $NEW_PROJ_CKPT"
        sleep "$POLL_SECONDS"
    done
}

run_pack_and_eval() {
    local key="$1" scene="$2" modes_csv="$3" log_path="$4"
    local bench matches out_dir depth_dir sparse_dir pairs_file mode eval_script
    bench="$(benchmark_path "$key" "$scene")"
    matches="$MATCHES_ROOT/$key/$scene"
    out_dir="$RESULTS_ROOT/$key/$scene"
    depth_dir="$(scene_depth_dir "$scene")"
    sparse_dir="$(scene_sparse_dir "$scene")"
    pairs_file="$(scene_pairs_file "$scene")"
    mkdir -p "$out_dir"
    if [[ ! -f "$bench" || "$FORCE" == "1" ]]; then
        run_logged "$log_path" "$REPOSED_PY" "$ROOT/scripts/pack_benchmark.py" \
            --matches_dir "$matches" \
            --depth_dir "$depth_dir" \
            --sparse_dir "$sparse_dir" \
            --pairs_file "$pairs_file" \
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
        run_logged "$log_path" "$REPOSED_PY" "$eval_script" "$bench" \
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
    local scene="$1" modes_csv="$2"
    local key="$NEW_EVAL_KEY"
    local log_path="$LOG_ROOT/${key}_${scene}.log"
    if all_summaries_exist "$key" "$scene" "$modes_csv" && [[ "$FORCE" != "1" ]]; then
        log "Skip complete $key $scene $modes_csv"
        return
    fi
    safe_clean_config_scene "$key" "$scene"
    local bench
    bench="$(benchmark_path "$key" "$scene")"
    if [[ ! -f "$bench" || "$FORCE" == "1" ]]; then
        mkdir -p "$MATCHES_ROOT/$key/$scene"
        CUDA_VISIBLE_DEVICES="$GPU" run_logged "$log_path" "$DINOV3_PY" "$ROOT/scripts/projection_matches.py" \
            --pairs_file "$(scene_pairs_file "$scene")" \
            --images_dir "$(scene_images_dir "$scene")" \
            --output_dir "$MATCHES_ROOT/$key/$scene" \
            --scene "$scene" \
            --checkpoint "$NEW_PROJ_CKPT" \
            --projection_tag "${key%_sp_mnn_mp2048}" \
            --max_points 2048 \
            --feat_level -8 \
            --img_size 768 768 \
            --t 0 \
            --up_ft_index 2 \
            --ensemble_size 2 \
            --alpha 0.5 \
            --feature_cache "$FEATURE_ROOT/$key/$scene" \
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
    if all_summaries_exist "$key" "$scene" "$modes_csv" && [[ "$FORCE" != "1" ]]; then
        log "Skip complete $key $scene $modes_csv"
        return
    fi
    safe_clean_config_scene "$key" "$scene"
    local bench
    bench="$(benchmark_path "$key" "$scene")"
    if [[ ! -f "$bench" || "$FORCE" == "1" ]]; then
        mkdir -p "$MATCHES_ROOT/$key/$scene"
        local cmd=(
            "$DINOV3_PY" "$ROOT/scripts/lightglue_projection_matches.py"
            --pairs_file "$(scene_pairs_file "$scene")"
            --images_dir "$(scene_images_dir "$scene")"
            --output_dir "$MATCHES_ROOT/$key/$scene"
            --scene "$scene"
            --config_key "$key"
            --checkpoint "$NEW_PROJ_CKPT"
            --filter_threshold "$threshold"
            --seed 42
            --max_points 2048
            --source_cache_max_points 2048
            --feat_level -8
            --img_size 768 768
            --t 0
            --up_ft_index 2
            --ensemble_size 2
            --alpha 0.5
            --feature_cache "$FEATURE_ROOT/$key/$scene"
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
        if [[ "$FORCE" == "1" ]]; then
            cmd+=(--overwrite)
        fi
        CUDA_VISIBLE_DEVICES="$GPU" run_logged "$log_path" "${cmd[@]}" || {
            record_failure "match_lg" "$key" "$scene" "" "failed_or_incompatible" "LightGlue matching failed or checkpoint incompatible" "$log_path" "$method_note"
            return
        }
    fi
    run_pack_and_eval "$key" "$scene" "$modes_csv" "$log_path"
}

task_lines() {
    local scene
    for scene in "${SCENES[@]}"; do
        printf "mnn\t%s\t%s\t%s\t\t\tbaseline\n" "$NEW_EVAL_KEY" "$scene" "calibrated,shared_focal,varying_focal"
    done

    local th key suffix
    for th in 0.00 0.02 0.05 0.10 0.15 0.20; do
        case "$th" in
            0.00) suffix=ft000 ;;
            0.02) suffix=ft002 ;;
            0.05) suffix=ft005 ;;
            0.10) suffix=ft010 ;;
            0.15) suffix=ft015 ;;
            0.20) suffix=ft020 ;;
        esac
        key="ch6_zeroshot_lg_proj_h1024_temp005_${suffix}_sp_mnn_mp2048"
        for scene in "${SCENES[@]}"; do
            if [[ "$th" == "0.10" ]]; then
                printf "lg\t%s\t%s\t%s\t%s\t\tadapt\tzero-shot final descriptor\n" "$key" "$scene" "calibrated,shared_focal,varying_focal" "$th"
            else
                printf "lg\t%s\t%s\t%s\t%s\t\tadapt\tzero-shot threshold sweep\n" "$key" "$scene" "calibrated" "$th"
            fi
        done
    done

    for scene in "${SCENES[@]}"; do
        printf "lg\t%s\t%s\t%s\t%s\t\tnoadapt\tzero-shot noadapt diagnostic\n" \
            "ch6_zeroshot_lg_proj_h1024_temp005_noadapt_ft010_sp_mnn_mp2048" "$scene" "calibrated" "0.10"
    done

    local warm="$ROOT/external/glue-factory/outputs/training/phase4_dinov3_lg_full_v1/checkpoint_best.tar"
    local scratch="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_v1/checkpoint_best.tar"
    local expanded="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
    local joint="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_unfrozen_proj_60_v2_4gpu/checkpoint_best.tar"

    for scene in "${SCENES[@]}"; do
        printf "lg\t%s\t%s\t%s\t%s\t%s\tadapt\ttrained with older descriptor; evaluated with final projection if compatible\n" \
            "ch6_warmstart_lg_full_v1_proj_h1024_temp005_ft010_sp_mnn_mp2048" "$scene" "calibrated,shared_focal,varying_focal" "0.10" "$warm"
        printf "lg\t%s\t%s\t%s\t%s\t%s\tadapt\ttrained with older descriptor; evaluated with final projection if compatible\n" \
            "ch6_scratch_stage2_lg_v1_proj_h1024_temp005_ft010_sp_mnn_mp2048" "$scene" "calibrated,shared_focal,varying_focal" "0.10" "$scratch"
        printf "lg\t%s\t%s\t%s\t%s\t%s\tadapt\ttrained with older descriptor; evaluated with final projection if compatible\n" \
            "ch6_expanded151_lg_proj_h1024_temp005_ft010_sp_mnn_mp2048" "$scene" "calibrated,shared_focal,varying_focal" "0.10" "$expanded"
    done

    for th in 0.05 0.10 0.15 0.20; do
        case "$th" in
            0.05) suffix=ft005 ;;
            0.10) suffix=ft010 ;;
            0.15) suffix=ft015 ;;
            0.20) suffix=ft020 ;;
        esac
        key="ch6_expanded151_lg_proj_h1024_temp005_${suffix}_sp_mnn_mp2048"
        for scene in "${SCENES[@]}"; do
            if [[ "$th" == "0.10" ]]; then
                continue
            fi
            printf "lg\t%s\t%s\t%s\t%s\t%s\tadapt\texpanded 151 threshold sweep\n" "$key" "$scene" "calibrated" "$th" "$expanded"
        done
    done

    for scene in "${SCENES[@]}"; do
        printf "lg\t%s\t%s\t%s\t%s\t%s\tadapt\tjoint checkpoint may be incompatible or native-projection diagnostic\n" \
            "ch6_joint_unfrozen_proj60_proj_h1024_temp005_ft010_sp_mnn_mp2048" "$scene" "calibrated,shared_focal,varying_focal" "0.10" "$joint"
    done
}

run_eval_worker() {
    setup_env
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.started"
    if ! wait_for_projection_checkpoint; then
        record_failure "eval_wait" "chapter6" "" "" "blocked" "Part A projection checkpoint missing; Chapter 6 not run" "$LOG_ROOT/ch5ch6_eval_${SHARD}_gpu${GPU}.screen.log"
        echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.failed"
        return 1
    fi
    local idx=0 kind key scene modes threshold ckpt adapt note
    while IFS=$'\t' read -r kind key scene modes threshold ckpt adapt note; do
        if (( idx % NUM_SHARDS == SHARD )); then
            log "Shard $SHARD GPU $GPU running $kind $key $scene $modes"
            case "$kind" in
                mnn) run_mnn_task "$scene" "$modes" || true ;;
                lg) run_lg_task "$key" "$scene" "$modes" "$threshold" "$ckpt" "$adapt" "$note" || true ;;
                *) record_failure "task" "$key" "$scene" "$modes" "bad_kind" "Unknown task kind $kind" "$FAILURES_TSV" ;;
            esac
        fi
        idx=$((idx + 1))
    done < <(task_lines)
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch5_ch6_overnight.py" --write || true
    echo "$(date +%s)" > "$STATUS_DIR/eval_shard_${SHARD}.done"
}

threshold_for_key() {
    case "$1" in
        *ft000*) echo "0.00" ;;
        *ft002*) echo "0.02" ;;
        *ft005*) echo "0.05" ;;
        *ft010*) echo "0.10" ;;
        *ft015*) echo "0.15" ;;
        *ft020*) echo "0.20" ;;
        *) echo "0.10" ;;
    esac
}

run_threshold_followup() {
    local family="$1" key ckpt threshold scene
    key="$("$DINOV3_PY" "$ROOT/scripts/aggregate_ch5_ch6_overnight.py" --best-threshold-family "$family" || true)"
    if [[ -z "$key" ]]; then
        log "No threshold follow-up needed for $family"
        return
    fi
    threshold="$(threshold_for_key "$key")"
    ckpt=""
    if [[ "$family" == "expanded" ]]; then
        ckpt="$ROOT/external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar"
    fi
    log "Running shared/varying follow-up for best $family threshold: $key"
    for scene in "${SCENES[@]}"; do
        run_lg_task "$key" "$scene" "shared_focal,varying_focal" "$threshold" "$ckpt" "adapt" "best threshold follow-up" || true
    done
}

run_finalize_worker() {
    setup_env
    echo "$(date +%s)" > "$STATUS_DIR/finalize.started"
    while true; do
        if [[ -f "$STATUS_DIR/train.failed" ]]; then
            "$DINOV3_PY" "$ROOT/scripts/aggregate_ch5_ch6_overnight.py" --write || true
            echo "$(date +%s)" > "$STATUS_DIR/finalize.failed"
            return 1
        fi
        local done_count
        done_count="$(find "$STATUS_DIR" -maxdepth 1 -name 'eval_shard_*.done' | wc -l)"
        if [[ "$done_count" -ge "$NUM_SHARDS" ]]; then
            break
        fi
        "$DINOV3_PY" "$ROOT/scripts/aggregate_ch5_ch6_overnight.py" --write || true
        log "Finalize waiting for eval shards ($done_count/$NUM_SHARDS done)"
        sleep "$POLL_SECONDS"
    done
    run_threshold_followup zeroshot
    run_threshold_followup expanded
    "$DINOV3_PY" "$ROOT/scripts/aggregate_ch5_ch6_overnight.py" --write --final
    echo "$(date +%s)" > "$STATUS_DIR/finalize.done"
    log "Final aggregation complete"
}

launch_screens() {
    setup_env
    local gpus_raw gpus_csv gpu_count train_gpu force_arg
    gpus_raw="$(detect_gpus)"
    read -r -a gpus <<< "$gpus_raw"
    if [[ "${#gpus[@]}" -eq 0 ]]; then
        echo "No free GPU detected and CUDA_VISIBLE_DEVICES is empty." >&2
        exit 4
    fi
    gpu_count="${#gpus[@]}"
    gpus_csv="$(join_by_comma "${gpus[@]}")"
    train_gpu="${gpus[0]}"
    force_arg=""
    [[ "$FORCE" == "1" ]] && force_arg="--force"

    preflight || exit 5
    echo "$(date +%s)" > "$STATUS_DIR/launch.started"
    echo "$gpus_csv" > "$STATUS_DIR/gpus.csv"
    echo "$gpu_count" > "$STATUS_DIR/num_shards.txt"
    write_launch_plan "$gpus_csv" "$gpu_count" "$train_gpu"
    log "Launch plan: $PLAN_PATH"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        return
    fi

    if [[ "$NO_SCREEN" == "1" ]]; then
        GPU="$train_gpu" run_train_worker
        local shard=0
        for gpu in "${gpus[@]}"; do
            GPU="$gpu" SHARD="$shard" NUM_SHARDS="$gpu_count" run_eval_worker
            shard=$((shard + 1))
        done
        GPU="$train_gpu" NUM_SHARDS="$gpu_count" run_finalize_worker
        return
    fi

    if ! command -v screen >/dev/null 2>&1; then
        echo "screen is required for launch mode but was not found." >&2
        exit 6
    fi

    local train_name="ch5ch6_train_gpu${train_gpu}"
    if screen -ls | grep -q "[.]${train_name}[[:space:]]"; then
        echo "Screen session already exists: $train_name" >&2
        exit 7
    fi
    screen -dmS "$train_name" bash -lc "cd '$ROOT' && '$ROOT/scripts/run_ch5_ch6_overnight.sh' --worker train --gpu '$train_gpu' $force_arg >> '$LOG_ROOT/${train_name}.screen.log' 2>&1"

    local shard=0
    for gpu in "${gpus[@]}"; do
        local eval_name="ch5ch6_eval_${shard}_gpu${gpu}"
        if screen -ls | grep -q "[.]${eval_name}[[:space:]]"; then
            echo "Screen session already exists: $eval_name" >&2
            exit 7
        fi
        screen -dmS "$eval_name" bash -lc "cd '$ROOT' && '$ROOT/scripts/run_ch5_ch6_overnight.sh' --worker eval --gpu '$gpu' --shard '$shard' --num-shards '$gpu_count' $force_arg >> '$LOG_ROOT/${eval_name}.screen.log' 2>&1"
        shard=$((shard + 1))
    done

    local finalize_name="ch5ch6_finalize"
    if screen -ls | grep -q "[.]${finalize_name}[[:space:]]"; then
        echo "Screen session already exists: $finalize_name" >&2
        exit 7
    fi
    screen -dmS "$finalize_name" bash -lc "cd '$ROOT' && '$ROOT/scripts/run_ch5_ch6_overnight.sh' --worker finalize --gpu '$train_gpu' --num-shards '$gpu_count' $force_arg >> '$LOG_ROOT/${finalize_name}.screen.log' 2>&1"

    log "Started screen sessions. Use: screen -ls"
}

main() {
    init_failures
    case "$WORKER" in
        "")
            launch_screens
            ;;
        train)
            [[ -n "$GPU" ]] || { echo "--gpu required for train worker" >&2; exit 2; }
            run_train_worker
            ;;
        eval)
            [[ -n "$GPU" ]] || { echo "--gpu required for eval worker" >&2; exit 2; }
            run_eval_worker
            ;;
        finalize)
            [[ -n "$GPU" ]] || { echo "--gpu required for finalize worker" >&2; exit 2; }
            run_finalize_worker
            ;;
        *)
            echo "Unknown worker: $WORKER" >&2
            exit 2
            ;;
    esac
}

main
