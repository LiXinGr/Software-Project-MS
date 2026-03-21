#!/bin/bash
# Show Phase 2A sparse-extraction / training status.
# Usage: ./scripts/check_phase2a_status.sh

set -euo pipefail

cd /home.stud/gorbuden/datagrid/Software-Project-MS

SCENES=(0080 0042 0380 0000 0366 0001 0005 0237 0011 0148)
SPARSE_DIR="data/sparse_train"
OUTPUT_DIR="experiments/p2_projection_v1"
LATEST_LOG="$OUTPUT_DIR/phase2a_latest.log"

echo "=== Phase 2A Status ==="
echo "Time: $(date '+%F %T')"

completed=0
for scene in "${SCENES[@]}"; do
    if [ -f "$SPARSE_DIR/$scene.pt" ]; then
        size=$(du -sh "$SPARSE_DIR/$scene.pt" 2>/dev/null | cut -f1)
        echo "  sparse $scene: DONE ($size)"
        completed=$((completed + 1))
    else
        echo "  sparse $scene: PENDING"
    fi
done
echo "Sparse bundles: $completed/${#SCENES[@]}"

if [ -f "$SPARSE_DIR/summary.json" ]; then
    echo "Sparse summary: $SPARSE_DIR/summary.json"
fi

if [ -f "$OUTPUT_DIR/train_log.json" ]; then
    python - <<'PY'
import json
from pathlib import Path
path = Path("experiments/p2_projection_v1/train_log.json")
data = json.loads(path.read_text())
if not data:
    print("Training epochs logged: 0")
else:
    last = data[-1]
    epoch = int(last["epoch"]) + 1
    print(f"Training epochs logged: {epoch}")
    print(
        "Latest epoch: "
        f"train_loss={last['train_loss']:.4f} "
        f"train_acc={last['train_accuracy']:.4f} "
        f"val_loss={last['val_loss']:.4f} "
        f"val_acc={last['val_accuracy']:.4f}"
    )
PY
else
    echo "Training epochs logged: 0"
fi

if [ -L "$LATEST_LOG" ] || [ -f "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo "--- tail ---"
    tail -n 20 "$LATEST_LOG" || true
fi
