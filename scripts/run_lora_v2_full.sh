#!/usr/bin/env bash
# =============================================================================
# run_lora_v2_full.sh — Phase 2 LoRA v2 full overnight pipeline
#
# 5 stages, 4 eval configurations:
#   Stage 1: Train LoRA on DINOv3-only          (~3-4 h)
#   Stage 2: Eval A  — raw LoRA-DINOv3          (~30 min × 3 scenes)
#   Stage 3: Eval B  — LoRA-DINOv3 + DIFT       (~30 min × 3 scenes)
#   Stage 4: Train proj head (frozen, dino-only) (~2-3 h)
#   Stage 4b: Eval C — proj-dino-only           (~30 min × 3 scenes)
#   Stage 5: Train proj head (frozen, fusion)    (~3-4 h)
#   Stage 5b: Eval D — proj-fusion (KEY RESULT) (~30 min × 3 scenes)
#
# Expected wall-time: ~10-13 h
#
# Usage:
#   screen -S lora_v2
#   cd /mnt/datagrid/personal/gorbuden/Software-Project-MS
#   bash scripts/run_lora_v2_full.sh [--smoke]
#
# --smoke runs 1 scene × 200 pairs × 2 epochs for all stages (quick sanity check).
# =============================================================================
set -euo pipefail

PYTHON_DINOV3=/home.stud/gorbuden/.conda/envs/dinov3/bin/python
PYTHON_REPOSED=/home.stud/gorbuden/.conda/envs/reposed/bin/python

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPOSED_DIR="$PROJECT_ROOT/external/RePoseD"
EVAL_SCENES=(sacre_coeur reichstag st_peters_square)

cd "$PROJECT_ROOT"
mkdir -p logs

# --------------------------------------------------------------------------
# MKL shared-library path (required by madpose inside eval_shared_f.py)
# --------------------------------------------------------------------------
MKL_PKG=$(find "$HOME/.conda/envs/romav2/lib" "$HOME/.conda/pkgs" -maxdepth 3 \
    -name "libmkl_intel_lp64.so.2" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)
OMP_PKG=$(find "$HOME/.conda/envs/romav2/lib" "$HOME/.conda/pkgs" -maxdepth 3 \
    -name "libiomp5.so" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || true)
EXTRA_LIBS=""
[[ -n "$MKL_PKG" ]] && EXTRA_LIBS="$MKL_PKG"
[[ -n "$OMP_PKG" ]] && EXTRA_LIBS="${EXTRA_LIBS:+$EXTRA_LIBS:}$OMP_PKG"
[[ -n "$EXTRA_LIBS" ]] && export LD_LIBRARY_PATH="${EXTRA_LIBS}:${LD_LIBRARY_PATH:-}"

# --------------------------------------------------------------------------
# Smoke / full mode
# --------------------------------------------------------------------------
SMOKE=0
SUMMARY_ONLY=0
EVAL_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--smoke" ]] && SMOKE=1
    [[ "$arg" == "--summary-only" ]] && SUMMARY_ONLY=1
    [[ "$arg" == "--eval-only" ]] && EVAL_ONLY=1
done

if [[ "$SMOKE" == "1" ]]; then
    SCENES_ARG="0148"
    EPOCHS_LORA=2; PAIRS_LORA=200; EPOCHS_PROJ=2; PAIRS_PROJ=200
    LIMIT=200
    echo "[SMOKE MODE] 1 scene, 2 epochs, 200 pairs, limit=200"
else
    SCENES_ARG="0080 0042 0380 0000 0366 0001 0005 0237 0011 0148"
    EPOCHS_LORA=10; PAIRS_LORA=10000; EPOCHS_PROJ=10; PAIRS_PROJ=10000
    LIMIT=15000
    echo "[FULL MODE] 10 scenes, epochs=${EPOCHS_LORA}/${EPOCHS_PROJ}, pairs=${PAIRS_LORA}/${PAIRS_PROJ}"
fi

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
STAGE1_DIR="experiments/phase2_lora_dinov3only"
STAGE4_DIR="experiments/phase2_lora_proj_dinov3only"
STAGE5_DIR="experiments/phase2_lora_proj_fusion"
SPARSE_LORA_DIR="data/sparse_train_lora"

CK_A="phase2_lora_r4_raw_dinov3_sp_mnn_mp2000"
CK_B="phase2_lora_r4_fusion_noproj_sp_mnn_mp2000"
CK_C="phase2_lora_r4_proj_dinov3only_sp_mnn_mp2000"
CK_D="phase2_lora_r4_proj_fusion_sp_mnn_mp2000"

RUN_A="phase2_lora_eval_raw_dinov3"
RUN_B="phase2_lora_eval_fusion_noproj"
RUN_C="phase2_lora_proj_dinov3only"
RUN_D="phase2_lora_proj_fusion"

MAIN_LOG="$PROJECT_ROOT/logs/lora_v2_full_$(date '+%Y%m%d_%H%M%S').log"

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
banner() {
    log ""; log "================================================================"
    log "  $*"
    log "================================================================"; log ""
}

# --------------------------------------------------------------------------
# Generic eval function: runs matches + pack + eval × 3 + CSV for one scene
# --------------------------------------------------------------------------
run_eval_scene() {
    local SCENE="$1"
    local CONFIG_KEY="$2"
    local RUN_ID="$3"
    local CHECKPOINT="$4"
    local MATCHER_SCRIPT="$5"
    local MATCHER_TAG="$6"      # short label used in CSV "Matches" column
    local EXTRA_MATCH_ARGS="${7:-}"

    local IMAGES_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/images_preprocessed"
    local SPARSE_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/dense/sparse"
    local DEPTH_DIR="$PROJECT_ROOT/datasets/phototourism/$SCENE/depth_unidepth"
    local PREPROCESS_INFO="$PROJECT_ROOT/datasets/phototourism/$SCENE/images_preprocessed/preprocess_info.json"
    local PAIRS_FILE="$PROJECT_ROOT/output/pairs_${SCENE}.txt"
    local MATCHES_DIR="$PROJECT_ROOT/output/matches/${CONFIG_KEY}/${SCENE}"
    local BENCHMARK_FILE="$PROJECT_ROOT/output/benchmarks/${CONFIG_KEY}_${SCENE}.h5"
    local RESULTS_DIR="$PROJECT_ROOT/output/results/${RUN_ID}/${SCENE}"
    local SCENE_LOG="$PROJECT_ROOT/logs/lora_v2_eval_${RUN_ID}_${SCENE}.log"

    mkdir -p "$MATCHES_DIR" "$RESULTS_DIR"

    log "--- Eval scene=$SCENE  config=$CONFIG_KEY  run=$RUN_ID ---"

    # Step 1: Generate matches (skip if already complete)
    local EXISTING
    EXISTING=$(find "$MATCHES_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)
    if [[ "$EXISTING" -ge "$LIMIT" ]]; then
        log "  Step 1: skipping matches ($EXISTING >= $LIMIT)"
    else
        log "  Step 1: generating matches ($EXISTING found, need $LIMIT)..."
        $PYTHON_DINOV3 -u scripts/"$MATCHER_SCRIPT" \
            --lora_checkpoint "$CHECKPOINT" \
            --pairs_file "$PAIRS_FILE" \
            --images_dir "$IMAGES_DIR" \
            --output_dir "$MATCHES_DIR" \
            --max_points 2000 \
            --limit "$LIMIT" \
            --device cuda:0 \
            $EXTRA_MATCH_ARGS \
            2>&1 | tee -a "$SCENE_LOG" | tee -a "$MAIN_LOG"
        log "  Step 1: done"
    fi

    # Step 2: Pack benchmark HDF5 (repack if existing has fewer pairs than LIMIT)
    local EXISTING_PAIRS=0
    if [[ -f "$BENCHMARK_FILE" ]]; then
        EXISTING_PAIRS=$($PYTHON_REPOSED -c "
import h5py, sys
try:
    with h5py.File('$BENCHMARK_FILE', 'r') as f:
        print(len(f.keys()))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    fi
    if [[ "$EXISTING_PAIRS" -ge "$LIMIT" ]]; then
        log "  Step 2: skipping pack (benchmark has $EXISTING_PAIRS >= $LIMIT pairs)"
    else
        log "  Step 2: packing benchmark..."
        $PYTHON_REPOSED -u scripts/pack_benchmark.py \
            --matches_dir "$MATCHES_DIR" \
            --depth_dir "$DEPTH_DIR" \
            --sparse_dir "$SPARSE_DIR" \
            --pairs_file "$PAIRS_FILE" \
            --output "$BENCHMARK_FILE" \
            --limit "$LIMIT" \
            2>&1 | tee -a "$SCENE_LOG" | tee -a "$MAIN_LOG"
        log "  Step 2: done"
    fi

    # Step 3: RePoseD eval (3 experiment types)
    local TIMESTAMP; TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    local BASENAME="${CONFIG_KEY}_${SCENE}"

    cd "$REPOSED_DIR"
    log "  Step 3a: calibrated eval..."
    $PYTHON_REPOSED -u eval.py "$BENCHMARK_FILE" \
        -nw 8 --thesis \
        --output_dir "$RESULTS_DIR" \
        --preprocess_info "$PREPROCESS_INFO" \
        2>&1 | tee -a "$SCENE_LOG" | tee -a "$MAIN_LOG"

    log "  Step 3b: shared_f eval..."
    $PYTHON_REPOSED -u eval_shared_f.py "$BENCHMARK_FILE" \
        -nw 8 --thesis \
        --output_dir "$RESULTS_DIR" \
        --preprocess_info "$PREPROCESS_INFO" \
        2>&1 | tee -a "$SCENE_LOG" | tee -a "$MAIN_LOG" \
        || log "  Warning: shared_f eval failed (non-fatal)"

    log "  Step 3c: varying_f eval..."
    $PYTHON_REPOSED -u eval_varying_f.py "$BENCHMARK_FILE" \
        -nw 8 --thesis \
        --output_dir "$RESULTS_DIR" \
        --preprocess_info "$PREPROCESS_INFO" \
        2>&1 | tee -a "$SCENE_LOG" | tee -a "$MAIN_LOG" \
        || log "  Warning: varying_f eval failed (non-fatal)"
    cd "$PROJECT_ROOT"

    # Step 4: Combine per-type JSONs into one results JSON (same structure as Phase 1)
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    local RESULTS_JSON="$RESULTS_DIR/results_${MATCHER_TAG}_${SCENE}_${TIMESTAMP}.json"
    local RESULTS_CSV="$RESULTS_DIR/results_${MATCHER_TAG}_${SCENE}_${TIMESTAMP}.csv"

    log "  Step 4: combining JSONs and generating CSV..."
    $PYTHON_REPOSED - "$RESULTS_DIR" "$BASENAME" "$RESULTS_JSON" << 'PYTHON_COMBINE'
import json, sys
from pathlib import Path
results_dir = Path(sys.argv[1])
basename    = sys.argv[2]
out_path    = sys.argv[3]
all_results = []
for fname, exp_type in [
    (f"calibrated-{basename}.json",    "calibrated"),
    (f"shared_focal-{basename}.json",  "shared_f"),
    (f"varying_focal-{basename}.json", "varying_f"),
]:
    p = results_dir / fname
    if p.exists():
        with open(p) as f:
            records = json.load(f)
        for r in records:
            if isinstance(r, dict):
                r["exp_type"] = exp_type
        all_results.extend(records)
        print(f"  loaded {exp_type}: {len(records)} records")
    else:
        print(f"  missing: {fname}")
with open(out_path, "w") as f:
    json.dump(all_results, f)
print(f"  combined JSON -> {out_path}")
PYTHON_COMBINE

    $PYTHON_REPOSED - "$RESULTS_JSON" "$RESULTS_CSV" "$MATCHER_TAG" "UniDepth" \
        "2000" "768" "-8" "2" "0" "" << 'PYTHON_CSV'
import json, csv, sys
import numpy as np
json_path    = sys.argv[1]
csv_path     = sys.argv[2]
matcher_name = sys.argv[3]
depth_method = sys.argv[4]
max_points   = sys.argv[5]
img_size     = sys.argv[6]
feat_level   = sys.argv[7]
up_ft_index  = sys.argv[8]
dift_t       = sys.argv[9]
ratio_thresh = sys.argv[10] if len(sys.argv) > 10 else ""

with open(json_path) as f:
    results = json.load(f)

experiments = {}
for r in results:
    if not isinstance(r, dict):
        continue
    key = f"{r.get('experiment','unknown')}|{r.get('exp_type','calibrated')}"
    if key not in experiments:
        experiments[key] = {
            "R_err": [], "t_err": [], "f_err": [],
            "runtime": [], "inlier_ratio": [],
            "exp": r.get("experiment","?"), "exp_type": r.get("exp_type","calibrated"),
        }
    experiments[key]["R_err"].append(r.get("R_err", float("nan")))
    experiments[key]["t_err"].append(r.get("t_err", float("nan")))
    if "f_err" in r:
        experiments[key]["f_err"].append(r.get("f_err", float("nan")))
    info = r.get("info", {})
    experiments[key]["runtime"].append(info.get("runtime", float("nan")))
    experiments[key]["inlier_ratio"].append(info.get("inlier_ratio", float("nan")))

has_focal = any(d["f_err"] for d in experiments.values())

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Matches","Depth","Solver","Exp.Type","Opt.",
              "εr(°)","εt(°)","mAA@10",
              "τ(ms)","Inliers","Num_Pairs",
              "max_points","img_size","feat_level","up_ft_index","dift_t","ratio_thresh"]
    if has_focal:
        header.insert(8, "mAA_f@10")
    writer.writerow(header)
    for key, data in sorted(experiments.items()):
        r_err = np.array(data["R_err"]); t_err = np.array(data["t_err"])
        pose_err = np.maximum(r_err, t_err)
        pose_err[np.isnan(pose_err)] = 180.0
        mAA_10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
        mAA_f = None
        if has_focal and data["f_err"]:
            f_err = np.array(data["f_err"]); f_err[np.isnan(f_err)] = 1.0
            mAA_f = np.mean([np.sum(f_err < t/100) / len(f_err) for t in range(1, 11)]) * 100
        opt_type = "H" if "hybrid" in data["exp"].lower() else "S"
        row = [matcher_name, depth_method, data["exp"], data["exp_type"], opt_type,
               f"{np.nanmedian(r_err):.2f}", f"{np.nanmedian(t_err):.2f}", f"{mAA_10:.1f}",
               f"{np.nanmean(data['runtime']):.1f}",
               f"{np.nanmean(data['inlier_ratio'])*100:.1f}",
               len(r_err), max_points, img_size, feat_level, up_ft_index, dift_t, ratio_thresh]
        if has_focal:
            row.insert(8, f"{mAA_f:.1f}" if mAA_f is not None else "N/A")
        writer.writerow(row)
print(f"  CSV -> {csv_path}")
PYTHON_CSV

    log "  Step 4: done  ->  $RESULTS_CSV"

    # Step 5: Print mAA@10 for primary solver
    log "  Results (3p_ours_shift_scale+12, calibrated):"
    local CALIB_JSON="$RESULTS_DIR/calibrated-${BASENAME}.json"
    if [[ -f "$CALIB_JSON" ]]; then
        $PYTHON_REPOSED -c "
import json, numpy as np
SOLVER = '3p_ours_shift_scale+12'
with open('$CALIB_JSON') as f:
    d = json.load(f)
entries = [e for e in d if e.get('experiment') == SOLVER]
if not entries: entries = d
pose_err = np.array([max(e['R_err'], e['t_err']) for e in entries])
pose_err[np.isnan(pose_err)] = 180.0
maa10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
print(f'  mAA@10={maa10:.1f}%  N={len(entries)}  scene=$SCENE  run=$RUN_ID')
" 2>&1 | tee -a "$MAIN_LOG"
    else
        log "  No calibrated JSON found at $CALIB_JSON"
    fi

    log "--- Done: scene=$SCENE  run=$RUN_ID ---"
}

# --------------------------------------------------------------------------
# Run eval for all 3 scenes with one configuration
# --------------------------------------------------------------------------
run_eval_all() {
    local CONFIG_KEY="$1"
    local RUN_ID="$2"
    local CHECKPOINT="$3"
    local MATCHER_SCRIPT="$4"
    local MATCHER_TAG="$5"
    local EXTRA_MATCH_ARGS="${6:-}"

    banner "EVAL  run=$RUN_ID  config=$CONFIG_KEY"
    for SCENE in "${EVAL_SCENES[@]}"; do
        run_eval_scene "$SCENE" "$CONFIG_KEY" "$RUN_ID" "$CHECKPOINT" \
                       "$MATCHER_SCRIPT" "$MATCHER_TAG" "$EXTRA_MATCH_ARGS"
    done
}

# --------------------------------------------------------------------------
# mAA@10 extractor (reads calibrated JSON, returns scalar)
# --------------------------------------------------------------------------
extract_maa10() {
    local RESULTS_DIR="$1"
    local CONFIG_KEY="$2"
    local SCENE="$3"
    local CALIB_JSON="$RESULTS_DIR/$SCENE/calibrated-${CONFIG_KEY}_${SCENE}.json"
    if [[ ! -f "$CALIB_JSON" ]]; then
        echo "N/A"
        return
    fi
    $PYTHON_REPOSED -c "
import json, numpy as np
SOLVER = '3p_ours_shift_scale+12'
with open('$CALIB_JSON') as f:
    d = json.load(f)
entries = [e for e in d if e.get('experiment') == SOLVER]
if not entries: entries = d
pose_err = np.array([max(e['R_err'], e['t_err']) for e in entries])
pose_err[np.isnan(pose_err)] = 180.0
maa10 = np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100
print(f'{maa10:.1f}')
" 2>/dev/null || echo "N/A"
}

# ==========================================================================
# EARLY EXIT for --summary-only
# ==========================================================================
if [[ "$SUMMARY_ONLY" == "1" ]]; then
    log "[--summary-only] Skipping all training and eval stages."
    # jump straight to summary (defined below)
elif [[ "$EVAL_ONLY" == "1" ]]; then
    log "[--eval-only] Skipping training stages, running evals on existing checkpoints."
    banner "EVAL A — raw LoRA-DINOv3"
    run_eval_all "$CK_A" "$RUN_A" "$STAGE1_DIR/best.pt" "lora_raw_matches.py" "phase2_lora_r4_raw"
    banner "EVAL B — LoRA-DINOv3 + DIFT fusion (no proj)"
    run_eval_all "$CK_B" "$RUN_B" "$STAGE1_DIR/best.pt" "lora_fusion_noproj_matches.py" "phase2_lora_r4_fusion_noproj" \
        "--img_size 768 768 --t 0 --up_ft_index 2 --ensemble_size 8 --alpha 0.5"
    banner "EVAL C — proj head, DINOv3-only"
    run_eval_all "$CK_C" "$RUN_C" "$STAGE4_DIR/best.pt" "lora_matches.py" "phase2_lora_r4_proj_dinov3only"
    banner "EVAL D — proj head, DIFT+DINOv3 fusion (KEY RESULT)"
    run_eval_all "$CK_D" "$RUN_D" "$STAGE5_DIR/best.pt" "lora_matches.py" "phase2_lora_r4_proj_fusion" \
        "--img_size 768 768 --t 0 --up_ft_index 2 --ensemble_size 8 --alpha 0.5"
else

# ==========================================================================
# STAGE 1 — Train LoRA on DINOv3-only
# ==========================================================================
banner "STAGE 1 — LoRA training (DINOv3-only, $EPOCHS_LORA epochs × $PAIRS_LORA pairs/ep)"

log "Output dir: $STAGE1_DIR"
$PYTHON_DINOV3 -u scripts/train_lora.py \
    --megadepth_root /mnt/datasets/MegaDepth/MegaDepth_v1_SfM \
    --sparse_dir data/sparse_train \
    --scenes $SCENES_ARG \
    --dinov3_only \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_dropout 0.0 \
    --epochs "$EPOCHS_LORA" \
    --pairs_per_epoch "$PAIRS_LORA" \
    --num_correspondences 512 \
    --min_correspondences 50 \
    --temperature 0.07 \
    --lr_lora 5e-5 \
    --lr_proj 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --output_dir "$STAGE1_DIR" \
    --device cuda:0 \
    --log_interval 200 \
    --seed 42 \
    2>&1 | tee -a "$MAIN_LOG"

log "Stage 1 complete. Checkpoint: $STAGE1_DIR/best.pt"

# ==========================================================================
# STAGE 1.5 — Precompute LoRA-DINOv3 sparse features
# Replaces DINOv3 dims [640:1664] in sparse bundles with LoRA-merged DINOv3.
# Output: data/sparse_train_lora/{scene}.pt
# Enables ~75 pairs/sec fast path for Stages 4 and 5.
# ==========================================================================
banner "STAGE 1.5 — Precompute LoRA-DINOv3 sparse features"

log "Output dir: $SPARSE_LORA_DIR"
$PYTHON_DINOV3 -u scripts/update_sparse_lora.py \
    --lora_checkpoint "$STAGE1_DIR/best.pt" \
    --megadepth_root /mnt/datasets/MegaDepth/MegaDepth_v1_SfM \
    --input_dir data/sparse_train \
    --output_dir "$SPARSE_LORA_DIR" \
    --scenes $SCENES_ARG \
    --device cuda:0 \
    2>&1 | tee -a "$MAIN_LOG"

log "Stage 1.5 complete. Sparse LoRA bundles in: $SPARSE_LORA_DIR"

# ==========================================================================
# STAGE 2 — Eval A: raw LoRA-DINOv3 (1024-dim, no proj, no DIFT)
# Baseline: raw DINOv3 block 16 = 60.3% avg
# ==========================================================================
# lora_raw_matches.py doesn't use --img_size/--t/--up_ft_index; safe to pass them
# (they're parsed but unused). Config key: lora_r4_raw_dinov3_sp_mnn_mp2000
run_eval_all "$CK_A" "$RUN_A" "$STAGE1_DIR/best.pt" "lora_raw_matches.py" "phase2_lora_r4_raw"

# ==========================================================================
# STAGE 3 — Eval B: LoRA-DINOv3 + DIFT fusion (1664-dim, no proj)
# Baseline: raw fusion DINOv3+DIFT = 70.4% avg
# ==========================================================================
run_eval_all "$CK_B" "$RUN_B" "$STAGE1_DIR/best.pt" "lora_fusion_noproj_matches.py" "phase2_lora_r4_fusion_noproj" \
    "--img_size 768 768 --t 0 --up_ft_index 2 --ensemble_size 8 --alpha 0.5"

# ==========================================================================
# STAGE 4 — Train projection head on frozen LoRA-DINOv3 (1024→512→256)
# ==========================================================================
banner "STAGE 4 — Projection head (frozen backbone, DINOv3-only, $EPOCHS_PROJ epochs × $PAIRS_PROJ pairs/ep)"

log "Output dir: $STAGE4_DIR"
$PYTHON_DINOV3 -u scripts/train_lora.py \
    --sparse_dir "$SPARSE_LORA_DIR" \
    --scenes $SCENES_ARG \
    --freeze_backbone \
    --fast_sparse \
    --dinov3_only \
    --lora_checkpoint "$STAGE1_DIR/best.pt" \
    --epochs "$EPOCHS_PROJ" \
    --pairs_per_epoch "$PAIRS_PROJ" \
    --num_correspondences 1024 \
    --min_correspondences 50 \
    --temperature 0.07 \
    --lr_proj 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --output_dir "$STAGE4_DIR" \
    --device cuda:0 \
    --log_interval 500 \
    --seed 42 \
    2>&1 | tee -a "$MAIN_LOG"

log "Stage 4 complete. Checkpoint: $STAGE4_DIR/best.pt"

# ==========================================================================
# STAGE 4b — Eval C: projection head on LoRA-DINOv3 only
# Config key read from checkpoint config (dinov3_only=True → proj_dinov3only)
# ==========================================================================
run_eval_all "$CK_C" "$RUN_C" "$STAGE4_DIR/best.pt" "lora_matches.py" "phase2_lora_r4_proj_dinov3only"

# ==========================================================================
# STAGE 5 — Train projection head on frozen LoRA-DINOv3 + DIFT (1664→1024→256)
# ==========================================================================
banner "STAGE 5 — Projection head (frozen backbone, DIFT+DINOv3, $EPOCHS_PROJ epochs × $PAIRS_PROJ pairs/ep)"

log "Output dir: $STAGE5_DIR"
$PYTHON_DINOV3 -u scripts/train_lora.py \
    --sparse_dir "$SPARSE_LORA_DIR" \
    --scenes $SCENES_ARG \
    --freeze_backbone \
    --fast_sparse \
    --lora_checkpoint "$STAGE1_DIR/best.pt" \
    --epochs "$EPOCHS_PROJ" \
    --pairs_per_epoch "$PAIRS_PROJ" \
    --num_correspondences 1024 \
    --min_correspondences 50 \
    --temperature 0.07 \
    --lr_proj 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --output_dir "$STAGE5_DIR" \
    --device cuda:0 \
    --log_interval 500 \
    --seed 42 \
    2>&1 | tee -a "$MAIN_LOG"

log "Stage 5 complete. Checkpoint: $STAGE5_DIR/best.pt"

# ==========================================================================
# STAGE 5b — Eval D: projection head on DIFT + LoRA-DINOv3 (KEY RESULT)
# Must beat Phase 2a baseline: 76.7% avg
# ==========================================================================
run_eval_all "$CK_D" "$RUN_D" "$STAGE5_DIR/best.pt" "lora_matches.py" "phase2_lora_r4_proj_fusion" \
    "--img_size 768 768 --t 0 --up_ft_index 2 --ensemble_size 8 --alpha 0.5"

fi  # end of [[ "$SUMMARY_ONLY" == "0" ]] block

# ==========================================================================
# SUMMARY TABLE
# ==========================================================================
banner "FINAL SUMMARY"

log "Extracting mAA@10 per scene / configuration..."

SC="sacre_coeur"; RE="reichstag"; ST="st_peters_square"
RDIR="$PROJECT_ROOT/output/results"

# Gather all results
maa_A_sc=$(extract_maa10 "$RDIR/$RUN_A" "$CK_A" "$SC")
maa_A_re=$(extract_maa10 "$RDIR/$RUN_A" "$CK_A" "$RE")
maa_A_st=$(extract_maa10 "$RDIR/$RUN_A" "$CK_A" "$ST")

maa_B_sc=$(extract_maa10 "$RDIR/$RUN_B" "$CK_B" "$SC")
maa_B_re=$(extract_maa10 "$RDIR/$RUN_B" "$CK_B" "$RE")
maa_B_st=$(extract_maa10 "$RDIR/$RUN_B" "$CK_B" "$ST")

maa_C_sc=$(extract_maa10 "$RDIR/$RUN_C" "$CK_C" "$SC")
maa_C_re=$(extract_maa10 "$RDIR/$RUN_C" "$CK_C" "$RE")
maa_C_st=$(extract_maa10 "$RDIR/$RUN_C" "$CK_C" "$ST")

maa_D_sc=$(extract_maa10 "$RDIR/$RUN_D" "$CK_D" "$SC")
maa_D_re=$(extract_maa10 "$RDIR/$RUN_D" "$CK_D" "$RE")
maa_D_st=$(extract_maa10 "$RDIR/$RUN_D" "$CK_D" "$ST")

# Compute averages (skip if any N/A)
avg_maa() {
    local a="$1" b="$2" c="$3"
    if [[ "$a" == "N/A" || "$b" == "N/A" || "$c" == "N/A" ]]; then
        echo "N/A"
    else
        $PYTHON_REPOSED -c "print(f'{($a+$b+$c)/3:.1f}')" 2>/dev/null || echo "N/A"
    fi
}

avg_A=$(avg_maa "$maa_A_sc" "$maa_A_re" "$maa_A_st")
avg_B=$(avg_maa "$maa_B_sc" "$maa_B_re" "$maa_B_st")
avg_C=$(avg_maa "$maa_C_sc" "$maa_C_re" "$maa_C_st")
avg_D=$(avg_maa "$maa_D_sc" "$maa_D_re" "$maa_D_st")

SUMMARY_CSV="$PROJECT_ROOT/output/results/lora_v2_summary_$(date '+%Y%m%d_%H%M%S').csv"

{
echo "experiment,sacre_mAA10,reichstag_mAA10,st_peters_mAA10,avg_mAA10"
echo "$RUN_A,$maa_A_sc,$maa_A_re,$maa_A_st,$avg_A"
echo "$RUN_B,$maa_B_sc,$maa_B_re,$maa_B_st,$avg_B"
echo "$RUN_C,$maa_C_sc,$maa_C_re,$maa_C_st,$avg_C"
echo "$RUN_D,$maa_D_sc,$maa_D_re,$maa_D_st,$avg_D"
echo "---baselines---,,,,"
echo "baseline_dinov3_raw,67.4,50.8,62.6,60.3"
echo "baseline_fusion_raw,79.8,59.3,72.0,70.4"
echo "baseline_proj_wide,82.5,74.1,73.3,76.7"
} | tee "$SUMMARY_CSV" | tee -a "$MAIN_LOG"

log ""
log "Summary CSV saved to: $SUMMARY_CSV"
log ""
log "Eval D (LoRA + DIFT + proj head) avg mAA@10: $avg_D"
log "  Baseline Phase 2a (plain DINOv3 + DIFT + proj head): 76.7%"
if [[ "$avg_D" != "N/A" ]]; then
    $PYTHON_REPOSED -c "
d=$avg_D; b=76.7
delta = d - b
print(f'  Delta vs baseline: {delta:+.1f}% ({\"IMPROVEMENT\" if delta > 0 else \"REGRESSION\"})')
" 2>/dev/null | tee -a "$MAIN_LOG" || true
fi

banner "ALL DONE — $(date)"
log "Full log: $MAIN_LOG"
