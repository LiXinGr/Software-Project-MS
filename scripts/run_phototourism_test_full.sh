#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

ARGS=(--gpus "$GPU_SPEC")
[ "$DRY_RUN" -eq 1 ] && ARGS+=(--dry-run)

"$SCRIPT_DIR/run_phototourism_test_preprocess.sh" "${ARGS[@]}"
"$SCRIPT_DIR/run_phototourism_test_eval.sh" "${ARGS[@]}"

