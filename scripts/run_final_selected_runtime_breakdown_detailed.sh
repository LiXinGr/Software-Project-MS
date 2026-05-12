#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/phototourism_test_common.sh
source "$ROOT/scripts/phototourism_test_common.sh"

SCREEN_NAME="${SCREEN_NAME:-final_selected_runtime_breakdown_detailed}"
GPU_ID="${GPU_ID:-0}"
FORCE=0
NO_SCREEN=0
RUN_WORKER=0
COMMAND_FOR_REPORT=""

usage() {
    cat <<EOF
Usage:
  scripts/run_final_selected_runtime_breakdown_detailed.sh --launch [--gpu 0] [--force] [--no-screen]
  scripts/run_final_selected_runtime_breakdown_detailed.sh --run [--gpu 0] [--force]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) RUN_WORKER=0; shift ;;
        --run) RUN_WORKER=1; shift ;;
        --gpu) GPU_ID="${2:?missing value for --gpu}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --no-screen) NO_SCREEN=1; shift ;;
        --command-for-report) COMMAND_FOR_REPORT="${2:?missing value for --command-for-report}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

force_arg=()
if [[ "$FORCE" -eq 1 ]]; then
    force_arg=(--force)
fi

runtime_cmd_for_report() {
    if [[ -n "$COMMAND_FOR_REPORT" ]]; then
        printf '%s\n' "$COMMAND_FOR_REPORT"
    else
        printf 'CUDA_VISIBLE_DEVICES=%s scripts/run_final_selected_runtime_breakdown_detailed.sh --launch --gpu %s%s\n' \
            "$GPU_ID" "$GPU_ID" "$([[ "$FORCE" -eq 1 ]] && printf ' --force')"
    fi
}

run_worker() {
    prepare_runtime_env
    setup_mkl_openmp_env
    mkdir -p "$ROOT/output_v2/logs/final_runtime_benchmark"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false

    log "[runtime-detailed] running final selected detailed breakdown"
    "$DINOV3_PY" "$ROOT/scripts/final_selected_runtime_breakdown_detailed.py" \
        "${force_arg[@]}" \
        --command "$(runtime_cmd_for_report)"

    log "[runtime-detailed] complete: $ROOT/output_v2/reports/final_selected_runtime_breakdown_detailed.md"
}

if [[ "$RUN_WORKER" -eq 1 ]]; then
    run_worker
    exit 0
fi

mkdir -p "$ROOT/output_v2/logs/final_runtime_benchmark"
screen_log="$ROOT/output_v2/logs/final_runtime_benchmark/final_selected_runtime_breakdown_detailed.screen.log"
cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU_ID' GPU_ID='$GPU_ID' SCREEN_NAME='$SCREEN_NAME' '$ROOT/scripts/run_final_selected_runtime_breakdown_detailed.sh' --run --gpu '$GPU_ID' $([[ "$FORCE" -eq 1 ]] && printf -- '--force') --command-for-report 'CUDA_VISIBLE_DEVICES=$GPU_ID scripts/run_final_selected_runtime_breakdown_detailed.sh --launch --gpu $GPU_ID$([[ "$FORCE" -eq 1 ]] && printf ' --force')' > '$screen_log' 2>&1"

echo "Exact launch command:"
echo "screen -S $SCREEN_NAME -dm bash -lc \"$cmd\""
if [[ "$NO_SCREEN" -eq 1 ]]; then
    bash -lc "$cmd"
else
    if screen -ls 2>/dev/null | grep -q "[.]$SCREEN_NAME[[:space:]]"; then
        echo "Screen session '$SCREEN_NAME' is already running." >&2
        echo "Attach with: screen -r $SCREEN_NAME" >&2
        exit 1
    fi
    screen -S "$SCREEN_NAME" -dm bash -lc "$cmd"
    echo "Launched screen session: $SCREEN_NAME"
    echo "Attach with: screen -r $SCREEN_NAME"
    echo "Log: $screen_log"
fi
