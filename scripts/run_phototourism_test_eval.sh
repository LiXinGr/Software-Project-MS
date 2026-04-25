#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/phototourism_test_common.sh"

GPU_SPEC="auto"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpus)
            GPU_SPEC="${2:?missing value for --gpus}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

prepare_runtime_env
require_runtime_prereqs

LOG_FILE="$DEFAULT_LOG_DIR/eval_master_$(date +%Y%m%d_%H%M%S).log"
if [ "$DRY_RUN" -eq 0 ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [ "$DRY_RUN" -eq 0 ] && ! print_readiness_table >/dev/null; then
    echo "Phase 1 is incomplete. Refusing to start Phase 2." >&2
    print_readiness_table
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ] && ! print_readiness_table >/dev/null; then
    log "[DRY-RUN] Phase 1 is incomplete; proceeding with planning output only."
fi

mapfile -t GPU_IDS < <(resolve_gpu_ids "$GPU_SPEC")
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "No GPUs available for evaluation." >&2
    exit 1
fi

worker_for_gpu() {
    local gpu="$1"
    shift
    local method method_log
    for method in "$@"; do
        method_log="$DEFAULT_LOG_DIR/${method}_gpu${gpu}.log"
        log "[WORKER gpu=$gpu] starting $method ($(method_label "$method"))"
        if [ "$DRY_RUN" -eq 1 ]; then
            "$SCRIPT_DIR/run_phototourism_test_method.sh" "$method" "$gpu" --dry-run
        else
            "$SCRIPT_DIR/run_phototourism_test_method.sh" "$method" "$gpu" \
                > >(tee -a "$method_log") 2>&1
        fi
        log "[WORKER gpu=$gpu] finished $method"
    done
}

declare -a ASSIGNED_METHODS=()
for _ in "${GPU_IDS[@]}"; do
    ASSIGNED_METHODS+=("")
done

for idx in "${!METHOD_ORDER[@]}"; do
    worker_idx=$((idx % ${#GPU_IDS[@]}))
    if [ -n "${ASSIGNED_METHODS[$worker_idx]}" ]; then
        ASSIGNED_METHODS[$worker_idx]+=" "
    fi
    ASSIGNED_METHODS[$worker_idx]+="${METHOD_ORDER[$idx]}"
done

log "Evaluation master"
log "Dry run: $DRY_RUN"
log "GPUs: ${GPU_IDS[*]}"
for idx in "${!GPU_IDS[@]}"; do
    log "Assignment gpu=${GPU_IDS[$idx]} -> ${ASSIGNED_METHODS[$idx]:-(none)}"
done

declare -a WORKER_PIDS=()
declare -a WORKER_TAGS=()
for idx in "${!GPU_IDS[@]}"; do
    [ -n "${ASSIGNED_METHODS[$idx]}" ] || continue
    read -r -a method_batch <<<"${ASSIGNED_METHODS[$idx]}"
    (
        worker_for_gpu "${GPU_IDS[$idx]}" "${method_batch[@]}"
    ) &
    WORKER_PIDS+=("$!")
    WORKER_TAGS+=("gpu${GPU_IDS[$idx]}")
done

worker_fail=0
for idx in "${!WORKER_PIDS[@]}"; do
    if ! wait "${WORKER_PIDS[$idx]}"; then
        log "[EVAL] worker ${WORKER_TAGS[$idx]} failed"
        worker_fail=1
    fi
done

log "Generating summary tables"
"$DINOV3_PY" "$PROJECT_ROOT/scripts/summarize_phototourism_test_results.py"

if [ "$worker_fail" -ne 0 ]; then
    log "Evaluation completed with failures."
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    log "Dry-run complete. No evaluation jobs were started."
else
    log "Evaluation complete."
    log "Master log: $LOG_FILE"
fi
