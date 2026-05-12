#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/phototourism_test_common.sh
source "$ROOT/scripts/phototourism_test_common.sh"

SCREEN_NAME="${SCREEN_NAME:-final_runtime_benchmark}"
GPU_ID="${GPU_ID:-0}"
FORCE=0
NO_SCREEN=0
RUN_WORKER=0
AGGREGATE_ONLY=0
PREPARE_ONLY=0
COMMAND_FOR_REPORT=""

usage() {
    cat <<EOF
Usage:
  scripts/run_final_runtime_benchmark.sh --launch [--gpu 0] [--force] [--no-screen]
  scripts/run_final_runtime_benchmark.sh --run [--gpu 0] [--force]
  scripts/run_final_runtime_benchmark.sh --aggregate-only
  scripts/run_final_runtime_benchmark.sh --prepare-only [--force]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) RUN_WORKER=0; shift ;;
        --run) RUN_WORKER=1; shift ;;
        --gpu) GPU_ID="${2:?missing value for --gpu}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --no-screen) NO_SCREEN=1; shift ;;
        --aggregate-only) AGGREGATE_ONLY=1; shift ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
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
        printf 'CUDA_VISIBLE_DEVICES=%s scripts/run_final_runtime_benchmark.sh --launch --gpu %s%s\n' \
            "$GPU_ID" "$GPU_ID" "$([[ "$FORCE" -eq 1 ]] && printf ' --force')"
    fi
}

run_all() {
    prepare_runtime_env
    setup_mkl_openmp_env
    mkdir -p "$ROOT/output_v2/logs/final_runtime_benchmark"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false

    log "[runtime] preparing fixed subset"
    "$REPOSED_PY" "$ROOT/scripts/final_runtime_benchmark.py" --prepare-subset "${force_arg[@]}"

    log "[runtime] running SuperPoint+LightGlue"
    "$LIGHTGLUE_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
        --run-method superpoint_lg_mp2048 "${force_arg[@]}"

    log "[runtime] running final selected expanded-151 LightGlue"
    "$DINOV3_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
        --run-method final_selected_expanded151_lg_proj_dinov3_dift_ft002_mp2048 "${force_arg[@]}"

    log "[runtime] running RoMa"
    "$ROMA_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
        --run-method roma_outdoor_mp2048 "${force_arg[@]}"

    local romav2_pythonpath romav2_ld_library_path romav2_library_path
    romav2_pythonpath="${PYTHONPATH:-}"
    if [[ -n "${ROMAV2_EXTRA_PYTHONPATH:-}" ]]; then
        if [[ -n "$romav2_pythonpath" ]]; then
            romav2_pythonpath="${ROMAV2_EXTRA_PYTHONPATH}:$romav2_pythonpath"
        else
            romav2_pythonpath="${ROMAV2_EXTRA_PYTHONPATH}"
        fi
    fi
    romav2_ld_library_path="${LD_LIBRARY_PATH:-}"
    romav2_library_path="${LIBRARY_PATH:-}"
    if [[ -n "${ROMAV2_CUDA_SHIM_DIR:-}" ]]; then
        romav2_ld_library_path="${ROMAV2_CUDA_SHIM_DIR}:${romav2_ld_library_path}"
        romav2_library_path="${ROMAV2_CUDA_SHIM_DIR}:${romav2_library_path}"
    fi

    log "[runtime] running RoMaV2"
    PYTHONPATH="$romav2_pythonpath" \
        LD_LIBRARY_PATH="$romav2_ld_library_path" \
        LIBRARY_PATH="$romav2_library_path" \
        "$ROMAV2_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
            --run-method romav2_precise_mp2048 "${force_arg[@]}"

    log "[runtime] aggregating CSV, report, and figures"
    "$REPOSED_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
        --aggregate \
        --command "$(runtime_cmd_for_report)"

    log "[runtime] complete: $ROOT/output_v2/reports/final_runtime_comparison_report.md"
}

run_aggregate_only() {
    prepare_runtime_env
    setup_mkl_openmp_env
    "$REPOSED_PY" "$ROOT/scripts/final_runtime_benchmark.py" \
        --aggregate \
        --command "$(runtime_cmd_for_report)"
}

run_prepare_only() {
    prepare_runtime_env
    "$REPOSED_PY" "$ROOT/scripts/final_runtime_benchmark.py" --prepare-subset "${force_arg[@]}"
}

if [[ "$AGGREGATE_ONLY" -eq 1 ]]; then
    run_aggregate_only
    exit 0
fi

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    run_prepare_only
    exit 0
fi

if [[ "$RUN_WORKER" -eq 1 ]]; then
    run_all
    exit 0
fi

mkdir -p "$ROOT/output_v2/logs/final_runtime_benchmark"
screen_log="$ROOT/output_v2/logs/final_runtime_benchmark/final_runtime_benchmark.screen.log"
cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU_ID' GPU_ID='$GPU_ID' SCREEN_NAME='$SCREEN_NAME' '$ROOT/scripts/run_final_runtime_benchmark.sh' --run --gpu '$GPU_ID' $([[ "$FORCE" -eq 1 ]] && printf -- '--force') --command-for-report 'CUDA_VISIBLE_DEVICES=$GPU_ID scripts/run_final_runtime_benchmark.sh --launch --gpu $GPU_ID$([[ "$FORCE" -eq 1 ]] && printf ' --force')' > '$screen_log' 2>&1"

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
