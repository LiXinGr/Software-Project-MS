#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID="${1:-0}"
PAIR_LIMIT="${PAIR_LIMIT:-15000}"
SESSION_NAME="${SESSION_NAME:-stage2_151_eval}"

mkdir -p "$PROJECT_ROOT/logs"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/logs/stage2_151_evaluation_${LOG_TS}.log"

PAIR_LIMIT_EXPORT="PAIR_LIMIT=$PAIR_LIMIT "

screen -S "$SESSION_NAME" -dm bash -lc "cd '$PROJECT_ROOT' && echo 'Stage 2 151-scene evaluation started: \$(date)' | tee -a '$LOG_FILE' && ${PAIR_LIMIT_EXPORT}'$PROJECT_ROOT/scripts/run_stage2_151_evaluation.sh' '$GPU_ID' 2>&1 | tee -a '$LOG_FILE' && echo 'Stage 2 151-scene evaluation finished: \$(date)' | tee -a '$LOG_FILE'"

echo "Launched screen session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach with: screen -r $SESSION_NAME"
