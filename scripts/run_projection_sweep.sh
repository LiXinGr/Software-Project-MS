#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/projection_sweep_${TIMESTAMP}.txt"
LATEST_LOG="$LOG_DIR/projection_sweep_latest.log"
SUMMARY_TSV="$LOG_DIR/projection_sweep_summary_${TIMESTAMP}.tsv"
LATEST_SUMMARY="$LOG_DIR/projection_sweep_summary_latest.tsv"

DEVICE="${DEVICE:-cuda:0}"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-4000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
GPU_POLL_SEC="${GPU_POLL_SEC:-300}"
SCENES=(0080 0042 0380 0000 0366 0001 0005 0237 0011 0148)
EVAL_SCENES=(sacre_coeur reichstag st_peters_square)
VARIANTS=(temp003 temp005 temp010 temp015 wide deep)
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ -z "${SPARSE_SCENE_CACHE_SIZE:-}" ]]; then
    if [[ "$HOST_SHORT" == "lie" ]]; then
        SPARSE_SCENE_CACHE_SIZE="${#SCENES[@]}"
    else
        SPARSE_SCENE_CACHE_SIZE="1"
    fi
fi
if [[ -z "${WAIT_FOR_GPU_BEFORE_EVAL:-}" ]]; then
    if [[ "$HOST_SHORT" == "lie" ]]; then
        WAIT_FOR_GPU_BEFORE_EVAL="0"
    else
        WAIT_FOR_GPU_BEFORE_EVAL="1"
    fi
fi

mkdir -p "$LOG_DIR"
ln -sfn "$LOG_FILE" "$LATEST_LOG"
ln -sfn "$SUMMARY_TSV" "$LATEST_SUMMARY"

load_anaconda_module() {
    local module_name
    for module_name in Anaconda3/2020.07 Anaconda3/2022.10 Anaconda3/2024.02-1; do
        if module load "$module_name" >/dev/null 2>&1; then
            echo "Using $module_name"
            return 0
        fi
    done
    echo "Failed to load any supported Anaconda3 module" >&2
    return 1
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_gpu() {
    while true; do
        read -r used util < <(
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
                | sed -n "$((GPU_INDEX + 1))p" \
                | tr -d ' %' \
                | tr ',' ' '
        )
        if [[ -z "${used:-}" || -z "${util:-}" ]]; then
            log "Unable to read GPU ${GPU_INDEX} state, retrying in ${GPU_POLL_SEC}s"
            sleep "$GPU_POLL_SEC"
            continue
        fi
        if (( used <= GPU_MAX_USED_MB && util <= GPU_MAX_UTIL )); then
            log "GPU ${GPU_INDEX} is available: used=${used}MiB util=${util}%"
            break
        fi
        log "GPU ${GPU_INDEX} busy: used=${used}MiB util=${util}% (thresholds ${GPU_MAX_USED_MB}MiB/${GPU_MAX_UTIL}%). Sleeping ${GPU_POLL_SEC}s."
        sleep "$GPU_POLL_SEC"
    done
}

log_gpu_state() {
    read -r used util < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
            | sed -n "$((GPU_INDEX + 1))p" \
            | tr -d ' %' \
            | tr ',' ' '
    )
    if [[ -n "${used:-}" && -n "${util:-}" ]]; then
        log "GPU ${GPU_INDEX} current state: used=${used}MiB util=${util}%"
    else
        log "GPU ${GPU_INDEX} current state unavailable"
    fi
}

append_summary() {
    local variant="$1"
    local out_dir="$2"
    local run_id="$3"
    local projection_tag="$4"
    python3 - "$variant" "$out_dir" "$run_id" "$projection_tag" "$SUMMARY_TSV" "${EVAL_SCENES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

variant, out_dir, run_id, projection_tag, summary_tsv, *scenes = sys.argv[1:]
config_path = Path(out_dir) / "config.json"

with config_path.open() as handle:
    config = json.load(handle)

scene_metrics = {}
for scene in scenes:
    results_dir = Path("output/results") / run_id / scene
    csv_candidates = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_candidates:
        raise SystemExit(f"No CSV results found in {results_dir}")

    with csv_candidates[-1].open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    target = None
    for row in rows:
        if row.get("Solver") == "3p_ours_shift_scale+12" and row.get("Exp.Type") == "calibrated":
            target = row
            break

    if target is None:
        raise SystemExit(f"Primary calibrated solver row not found in {csv_candidates[-1]}")

    scene_metrics[scene] = {
        "mAA@10": float(target["mAA@10"]),
        "Inliers": float(target["Inliers"]),
        "Num_Pairs": float(target["Num_Pairs"]),
        "results_csv": str(csv_candidates[-1]),
    }

avg_maa = sum(scene_metrics[scene]["mAA@10"] for scene in scenes) / len(scenes)
avg_inliers = sum(scene_metrics[scene]["Inliers"] for scene in scenes) / len(scenes)
avg_pairs = sum(scene_metrics[scene]["Num_Pairs"] for scene in scenes) / len(scenes)

summary_path = Path(summary_tsv)
write_header = not summary_path.exists()
with summary_path.open("a", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    if write_header:
        writer.writerow(
            [
                "variant",
                "projection_tag",
                "temperature",
                "hidden_dims",
                "mAA_sacre_coeur",
                "mAA_reichstag",
                "mAA_st_peters_square",
                "mAA_avg",
                "inliers_avg",
                "num_pairs_avg",
                "results_csv_sacre_coeur",
                "results_csv_reichstag",
                "results_csv_st_peters_square",
                "checkpoint",
            ]
        )
    writer.writerow(
        [
            variant,
            projection_tag,
            config["temperature"],
            ",".join(str(x) for x in config["hidden_dims"]),
            scene_metrics["sacre_coeur"]["mAA@10"],
            scene_metrics["reichstag"]["mAA@10"],
            scene_metrics["st_peters_square"]["mAA@10"],
            avg_maa,
            avg_inliers,
            avg_pairs,
            scene_metrics["sacre_coeur"]["results_csv"],
            scene_metrics["reichstag"]["results_csv"],
            scene_metrics["st_peters_square"]["results_csv"],
            str(Path(out_dir) / "best.pt"),
        ]
    )
PY
}

scene_eval_is_complete() {
    local run_id="$1"
    local scene="$2"
    python3 - "$run_id" "$scene" <<'PY'
import csv
import sys
from pathlib import Path

run_id, scene = sys.argv[1:]
results_dir = Path("output/results") / run_id / scene
csv_candidates = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
if not csv_candidates:
    raise SystemExit(1)

with csv_candidates[-1].open(newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    if row.get("Solver") == "3p_ours_shift_scale+12" and row.get("Exp.Type") == "calibrated":
        raise SystemExit(0)

raise SystemExit(1)
PY
}

training_is_complete() {
    local out_dir="$1"
    local expected_epochs="$2"
    python3 - "$out_dir" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])
train_log_path = out_dir / "train_log.json"
best_path = out_dir / "best.pt"

if not train_log_path.exists() or not best_path.exists():
    raise SystemExit(1)

with train_log_path.open() as handle:
    log = json.load(handle)

if not log:
    raise SystemExit(1)

last_epoch = int(round(float(log[-1]["epoch"])))
raise SystemExit(0 if last_epoch >= expected_epochs - 1 else 1)
PY
}

print_summary() {
    python3 - "$SUMMARY_TSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if not rows:
    raise SystemExit(0)

headers = [
    "variant",
    "temperature",
    "hidden_dims",
    "mAA_sacre_coeur",
    "mAA_reichstag",
    "mAA_st_peters_square",
    "mAA_avg",
]
widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
line = "  ".join(h.ljust(widths[h]) for h in headers)
print(line)
print("  ".join("-" * widths[h] for h in headers))
for row in rows:
    print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))
PY
}

variant_train_args() {
    local variant="$1"
    case "$variant" in
        temp003) echo "--temperature 0.03" ;;
        temp005) echo "--temperature 0.05" ;;
        temp010) echo "--temperature 0.10" ;;
        temp015) echo "--temperature 0.15" ;;
        wide) echo "--temperature 0.07 --hidden_dims 1024" ;;
        deep) echo "--temperature 0.07 --hidden_dims 512 512" ;;
        *)
            echo "Unknown variant: $variant" >&2
            return 1
            ;;
    esac
}

{
    cd "$PROJECT_ROOT"
    load_anaconda_module

    CONDA_BASE="$(conda info --base)"
    set +u
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate train
    set -u
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_projection_sweep}"
    mkdir -p "$MPLCONFIGDIR"

    printf "variant\tprojection_tag\ttemperature\thidden_dims\tmAA_sacre_coeur\tmAA_reichstag\tmAA_st_peters_square\tmAA_avg\tinliers_avg\tnum_pairs_avg\tresults_csv_sacre_coeur\tresults_csv_reichstag\tresults_csv_st_peters_square\tcheckpoint\n" > "$SUMMARY_TSV"

    log "Starting projection sweep"
    log "Project root: $PROJECT_ROOT"
    log "Host: $HOST_SHORT"
    log "Device: $DEVICE"
    log "GPU gate: index=$GPU_INDEX max_used=${GPU_MAX_USED_MB}MiB max_util=${GPU_MAX_UTIL}% poll=${GPU_POLL_SEC}s"
    log "Sparse scene cache size: $SPARSE_SCENE_CACHE_SIZE"
    log "Wait for GPU before eval: $WAIT_FOR_GPU_BEFORE_EVAL"
    log "Summary TSV: $SUMMARY_TSV"

    for variant in "${VARIANTS[@]}"; do
        out_dir="$PROJECT_ROOT/experiments/phase2_projection_${variant}"
        run_id="phase2_sweep_${variant}"
        projection_tag="projection_${variant}"
        checkpoint="$out_dir/best.pt"
        resume_checkpoint="$out_dir/latest.pt"

        log "=== Variant: $variant ==="
        log "Output dir: $out_dir"
        train_extra_args=$(variant_train_args "$variant")

        # shellcheck disable=SC2206
        TRAIN_EXTRA=($train_extra_args)
        RESUME_ARGS=()
        if [[ -f "$resume_checkpoint" ]]; then
            RESUME_ARGS=(--resume "$resume_checkpoint")
            log "Resuming $variant from $resume_checkpoint"
        fi
        if training_is_complete "$out_dir" 10; then
            log "Training already complete for $variant; skipping retraining"
        else
            log_gpu_state
            log "Starting training for $variant without GPU gate"
            python -u scripts/train_projection_head.py \
                --sparse_dir data/sparse_train \
                --scenes "${SCENES[@]}" \
                --epochs 10 \
                --pairs_per_epoch 50000 \
                --val_pairs_per_epoch 1000 \
                --sparse_scene_cache_size "$SPARSE_SCENE_CACHE_SIZE" \
                --lr 1e-3 \
                --num_correspondences 512 \
                --seed 42 \
                --device "$DEVICE" \
                --num_workers 0 \
                --log_interval 500 \
                --output_dir "$out_dir" \
                "${RESUME_ARGS[@]}" \
                "${TRAIN_EXTRA[@]}" \
                2>&1 | while IFS= read -r line; do
                    printf '[%s] [%s train] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$variant" "$line"
                done

            log "Training complete for $variant"
        fi
        for scene in "${EVAL_SCENES[@]}"; do
            if scene_eval_is_complete "$run_id" "$scene"; then
                log "Evaluation already complete for $variant on $scene; skipping"
                continue
            fi

            log "Evaluating $variant on $scene"

            if [[ "$WAIT_FOR_GPU_BEFORE_EVAL" == "1" ]]; then
                wait_for_gpu
            else
                log "Skipping GPU wait before eval"
                log_gpu_state
            fi
            ./run_thesis_benchmark.sh projection \
                --run_id "$run_id" \
                --scene "$scene" \
                --skip-depth \
                --device "$DEVICE" \
                --projection-checkpoint "$checkpoint" \
                --projection-tag "$projection_tag" \
                2>&1 | while IFS= read -r line; do
                    printf '[%s] [%s eval:%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$variant" "$scene" "$line"
                done
        done

        append_summary "$variant" "$out_dir" "$run_id" "$projection_tag"
        log "Current sweep summary:"
        print_summary
    done

    log "Sweep complete"
    log "Final summary:"
    print_summary
} 2>&1 | tee "$LOG_FILE"
