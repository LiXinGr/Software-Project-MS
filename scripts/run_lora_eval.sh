#!/usr/bin/env bash
# Quick eval of a LoRA checkpoint on one scene.
#
# Usage:
#   bash scripts/run_lora_eval.sh \
#       --checkpoint experiments/phase2_lora_smoke/best.pt \
#       --run_id phase2_lora_smoke_eval \
#       --scene sacre_coeur
#
# Steps:
#   1. Generate matches with lora_matches.py  (dinov3 env)
#   2. Pack benchmark HDF5 with pack_benchmark.py  (reposed env)
#   3. Run RePoseD eval.py  (reposed env)
#
# Log: logs/lora_eval_{run_id}_{scene}.log

set -euo pipefail

PYTHON_DINOV3=/home.stud/gorbuden/.conda/envs/dinov3/bin/python
PYTHON_REPOSED=/home.stud/gorbuden/.conda/envs/reposed/bin/python

# madpose (used by eval_shared_f.py) requires MKL — locate it dynamically
MKL_PKG=$(find "$HOME/.conda/envs/romav2/lib" "$HOME/.conda/pkgs" -maxdepth 3 \
    -name "libmkl_intel_lp64.so.2" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null)
OMP_PKG=$(find "$HOME/.conda/envs/romav2/lib" "$HOME/.conda/pkgs" -maxdepth 3 \
    -name "libiomp5.so" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null)
EXTRA_LIBS=""
[ -n "$MKL_PKG" ] && EXTRA_LIBS="$MKL_PKG"
[ -n "$OMP_PKG" ] && EXTRA_LIBS="${EXTRA_LIBS:+$EXTRA_LIBS:}$OMP_PKG"
[ -n "$EXTRA_LIBS" ] && export LD_LIBRARY_PATH="${EXTRA_LIBS}:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPOSED_DIR="$PROJECT_ROOT/external/RePoseD"

# ---- Defaults ----
CHECKPOINT=""
RUN_ID=""
SCENE="sacre_coeur"
LIMIT=15000

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --run_id)     RUN_ID="$2";     shift 2 ;;
        --scene)      SCENE="$2";      shift 2 ;;
        --limit)      LIMIT="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$RUN_ID" ]]; then
    echo "Usage: $0 --checkpoint <path> --run_id <id> [--scene <scene>] [--limit <N>]"
    exit 1
fi

# ---- Paths ----
cd "$PROJECT_ROOT"

CONFIG_KEY="lora_r4_proj_wide_sp_mnn_mp2000"
IMAGES_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/images_preprocessed"
SPARSE_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/sparse"
DEPTH_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/depth_unidepth"
PREPROCESS_INFO="$PROJECT_ROOT/datasets/phototourism/$SCENE/images_preprocessed/preprocess_info.json"
PAIRS_FILE="$PROJECT_ROOT/output/pairs_${SCENE}.txt"
MATCHES_DIR="$PROJECT_ROOT/output/matches/${CONFIG_KEY}/${SCENE}"
BENCHMARK_FILE="$PROJECT_ROOT/output/benchmarks/${CONFIG_KEY}_${SCENE}.h5"
RESULTS_DIR="$PROJECT_ROOT/output/results/${RUN_ID}/${SCENE}"

mkdir -p logs "$MATCHES_DIR" "$RESULTS_DIR"

LOG="$PROJECT_ROOT/logs/lora_eval_${RUN_ID}_${SCENE}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== LoRA eval: run_id=$RUN_ID scene=$SCENE ==="
log "Checkpoint: $CHECKPOINT"
log "Config key: $CONFIG_KEY"

# ============================================================
# Step 1: Generate matches
# ============================================================
EXISTING=$(ls "$MATCHES_DIR"/*.npz 2>/dev/null | wc -l || echo 0)
if [[ "$EXISTING" -ge "$LIMIT" ]]; then
    log "Step 1: Skipping matches (found $EXISTING >= $LIMIT .npz files)"
else
    log "Step 1: Generating matches ($EXISTING found, need $LIMIT)..."
    $PYTHON_DINOV3 -u scripts/lora_matches.py \
        --lora_checkpoint "$CHECKPOINT" \
        --pairs_file "$PAIRS_FILE" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$MATCHES_DIR" \
        --max_points 2000 \
        --img_size 768 768 \
        --t 0 \
        --up_ft_index 2 \
        --ensemble_size 8 \
        --alpha 0.5 \
        --limit "$LIMIT" \
        --device cuda:0 \
        2>&1 | tee -a "$LOG"
    log "Step 1: Done"
fi

# ============================================================
# Step 2: Pack benchmark HDF5
# ============================================================
if [[ -f "$BENCHMARK_FILE" ]]; then
    log "Step 2: Skipping pack (benchmark exists: $BENCHMARK_FILE)"
else
    log "Step 2: Packing benchmark..."
    $PYTHON_REPOSED -u scripts/pack_benchmark.py \
        --matches_dir "$MATCHES_DIR" \
        --depth_dir "$DEPTH_DIR" \
        --sparse_dir "$SPARSE_DIR" \
        --pairs_file "$PAIRS_FILE" \
        --output "$BENCHMARK_FILE" \
        --limit "$LIMIT" \
        2>&1 | tee -a "$LOG"
    log "Step 2: Done"
fi

# ============================================================
# Step 3: RePoseD eval (calibrated + shared_f + varying_f)
# ============================================================
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_JSON="$RESULTS_DIR/results_lora_${SCENE}_${TIMESTAMP}.json"
RESULTS_CSV="$RESULTS_DIR/results_lora_${SCENE}_${TIMESTAMP}.csv"

log "Step 3: Running RePoseD calibrated eval..."
cd "$REPOSED_DIR"
$PYTHON_REPOSED -u eval.py "$BENCHMARK_FILE" \
    -nw 8 --thesis \
    --output_dir "$RESULTS_DIR" \
    --preprocess_info "$PREPROCESS_INFO" \
    2>&1 | tee -a "$LOG"
log "Step 3: Calibrated eval done"

log "Step 3: Running shared_f eval..."
$PYTHON_REPOSED -u eval_shared_f.py "$BENCHMARK_FILE" \
    -nw 8 --thesis \
    --output_dir "$RESULTS_DIR" \
    --preprocess_info "$PREPROCESS_INFO" \
    2>&1 | tee -a "$LOG" || log "Warning: shared_f eval failed (non-fatal)"

log "Step 3: Running varying_f eval..."
$PYTHON_REPOSED -u eval_varying_f.py "$BENCHMARK_FILE" \
    -nw 8 --thesis \
    --output_dir "$RESULTS_DIR" \
    --preprocess_info "$PREPROCESS_INFO" \
    2>&1 | tee -a "$LOG" || log "Warning: varying_f eval failed (non-fatal)"

cd "$PROJECT_ROOT"

# ============================================================
# Step 4: Combine JSONs + generate CSV
# ============================================================
log "Step 4: Generating CSV..."
$PYTHON_REPOSED - "$RESULTS_DIR" "lora_r4" "$SCENE" "$RESULTS_JSON" "$CONFIG_KEY" << 'PYTHON_COMBINE'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
matcher = sys.argv[2]
scene = sys.argv[3]
output_path = sys.argv[4]
config_key = sys.argv[5] if len(sys.argv) > 5 else f"lora_{scene}"

basename = f"{config_key}_{scene}"
all_results = []
for path, exp_type in [
    (results_dir / f"calibrated-{basename}.json",    "calibrated"),
    (results_dir / f"shared_focal-{basename}.json",  "shared_f"),
    (results_dir / f"varying_focal-{basename}.json", "varying_f"),
]:
    if path.exists():
        with open(path) as f:
            results = json.load(f)
        for r in results:
            if isinstance(r, dict):
                r['exp_type'] = exp_type
        all_results.extend(results)
        print(f"Loaded {exp_type}: {len(results)} entries")
    else:
        print(f"No {exp_type} results at {path}")

with open(output_path, 'w') as f:
    json.dump(all_results, f)
print(f"Combined JSON saved to: {output_path}")
PYTHON_COMBINE

$PYTHON_REPOSED - "$RESULTS_JSON" "$RESULTS_CSV" "lora_r4" "UniDepth" "2000" "768" "-8" "2" "0" "" << 'PYTHON_CSV'
import json, csv, sys
import numpy as np

json_path, csv_path = sys.argv[1], sys.argv[2]
matcher_name = sys.argv[3] if len(sys.argv) > 3 else "lora_r4"
depth_method  = sys.argv[4] if len(sys.argv) > 4 else "UniDepth"
max_points    = sys.argv[5] if len(sys.argv) > 5 else "2000"
img_size      = sys.argv[6] if len(sys.argv) > 6 else "768"
feat_level    = sys.argv[7] if len(sys.argv) > 7 else "-8"
up_ft_index   = sys.argv[8] if len(sys.argv) > 8 else "2"
dift_t        = sys.argv[9] if len(sys.argv) > 9 else "0"
ratio_thresh  = sys.argv[10] if len(sys.argv) > 10 else ""

with open(json_path) as f:
    results = json.load(f)

experiments = {}
for r in results:
    if not isinstance(r, dict):
        continue
    key = f"{r.get('experiment','unknown')}|{r.get('exp_type','calibrated')}"
    if key not in experiments:
        experiments[key] = {'R_err': [], 't_err': [], 'runtime': [], 'inlier_ratio': [],
                            'f_err': [], 'exp': r.get('experiment','?'), 'exp_type': r.get('exp_type','calibrated')}
    experiments[key]['R_err'].append(r.get('R_err', float('nan')))
    experiments[key]['t_err'].append(r.get('t_err', float('nan')))
    if 'f_err' in r:
        experiments[key]['f_err'].append(r.get('f_err', float('nan')))
    info = r.get('info', {})
    experiments[key]['runtime'].append(info.get('runtime', float('nan')))
    experiments[key]['inlier_ratio'].append(info.get('inlier_ratio', float('nan')))

has_focal = any(len(d['f_err']) > 0 and not all(np.isnan(d['f_err'])) for d in experiments.values())

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Matches','Depth','Solver','Exp.Type','Opt.','εr(°)','εt(°)','mAA@10',
              'τ(ms)','Inliers','Num_Pairs','max_points','img_size','feat_level',
              'up_ft_index','dift_t','ratio_thresh']
    if has_focal:
        header.insert(8, 'mAA_f@10')
    writer.writerow(header)

    for key, data in sorted(experiments.items()):
        r_err = np.array(data['R_err'])
        t_err = np.array(data['t_err'])
        pose_err = np.maximum(r_err, t_err)
        pose_err[np.isnan(pose_err)] = 180.0
        mAA_10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
        mAA_f_10 = None
        if has_focal and data['f_err']:
            f_err = np.array(data['f_err'])
            f_err[np.isnan(f_err)] = 1.0
            mAA_f_10 = np.mean([np.sum(f_err < t/100) / len(f_err) for t in range(1, 11)]) * 100
        opt_type = 'H' if 'hybrid' in data['exp'].lower() else 'S'
        row = [matcher_name, depth_method, data['exp'], data['exp_type'], opt_type,
               f"{np.nanmedian(r_err):.2f}", f"{np.nanmedian(t_err):.2f}", f"{mAA_10:.1f}",
               f"{np.nanmean(data['runtime']):.1f}", f"{np.nanmean(data['inlier_ratio'])*100:.1f}",
               len(r_err), max_points, img_size, feat_level, up_ft_index, dift_t, ratio_thresh]
        if has_focal:
            row.insert(8, f"{mAA_f_10:.1f}" if mAA_f_10 is not None else "N/A")
        writer.writerow(row)

print(f"CSV saved to: {csv_path}")
PYTHON_CSV

log "Step 4: CSV written to $RESULTS_CSV"

# ============================================================
# Step 5: Print mAA@10 summary
# ============================================================
log "=== Results (best solver: 3p_ours_shift_scale+12) ==="
RESULT_JSON_CALIB=$(ls "$RESULTS_DIR"/calibrated-*.json 2>/dev/null | head -1)
if [[ -n "$RESULT_JSON_CALIB" ]]; then
    $PYTHON_REPOSED -c "
import json, numpy as np
SOLVER = '3p_ours_shift_scale+12'
with open('$RESULT_JSON_CALIB') as f:
    d = json.load(f)
entries = [e for e in d if e.get('experiment') == SOLVER] or d
pose_err = np.array([max(e['R_err'], e['t_err']) for e in entries])
pose_err[np.isnan(pose_err)] = 180.0
maa10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
print(f'mAA@10 (calibrated, {SOLVER}): {maa10:.1f}')
print(f'N pairs: {len(entries)}')
" 2>&1 | tee -a "$LOG"
else
    log "No calibrated result JSON found"
fi

log "=== Eval complete: $(date) ==="
log "Full log: $LOG"
