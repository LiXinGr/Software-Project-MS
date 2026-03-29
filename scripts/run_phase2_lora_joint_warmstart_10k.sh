#!/usr/bin/env bash
set -euo pipefail

ROOT=/home.stud/gorbuden/datagrid/Software-Project-MS
TRAIN_PY=/home.stud/gorbuden/.conda/envs/train/bin/python
REPOSED_PY=/home.stud/gorbuden/.conda/envs/reposed/bin/python

cd "$ROOT"

STAGE1_CKPT=$ROOT/experiments/phase2_lora_dinov3only/best.pt
PROJ_CKPT=$ROOT/experiments/phase2_projection_wide/best.pt

# Joint LoRA training uses the raw sparse bundles:
# only the DIFT slice is read from them, while DINOv3 is recomputed online.
SPARSE_DIR=$ROOT/data/sparse_train
SCENES_TRAIN=(0080 0042 0380 0000 0366 0001 0005 0237 0011 0148)
SCENES_EVAL=(sacre_coeur reichstag st_peters_square)

RUN_TAG=jointwarm10k_$(date +%Y%m%d_%H%M%S)
RUN_ID=phase2_lora_joint_fusion_projwarm_${RUN_TAG}
EXP_DIR=$ROOT/experiments/$RUN_ID
MATCH_KEY=lora_r4_joint_fusion_projwarm_${RUN_TAG}_sp_mnn_mp2000
MATCH_ROOT=$ROOT/output/matches/$MATCH_KEY
RESULT_ROOT=$ROOT/output/results/$RUN_ID
BENCH_ROOT=$ROOT/output/benchmarks
LORA_EVAL_CACHE=$ROOT/cache/features/$MATCH_KEY
MAIN_LOG=$ROOT/logs/${RUN_ID}.log
MPLCONFIGDIR=/tmp/mpl_${RUN_TAG}

for path in "$EXP_DIR" "$MATCH_ROOT" "$RESULT_ROOT" "$LORA_EVAL_CACHE" "$MAIN_LOG" "$MPLCONFIGDIR"; do
  if [ -e "$path" ]; then
    echo "Refusing to reuse existing path: $path"
    exit 1
  fi
done

mkdir -p "$EXP_DIR" "$MATCH_ROOT" "$RESULT_ROOT" "$BENCH_ROOT" "$LORA_EVAL_CACHE" "$MPLCONFIGDIR"
export MPLCONFIGDIR

echo "[$(date '+%F %T')] RUN_ID=$RUN_ID" | tee "$MAIN_LOG"
echo "[$(date '+%F %T')] EXP_DIR=$EXP_DIR" | tee -a "$MAIN_LOG"
echo "[$(date '+%F %T')] MATCH_KEY=$MATCH_KEY" | tee -a "$MAIN_LOG"
echo "[$(date '+%F %T')] LORA_EVAL_CACHE=$LORA_EVAL_CACHE" | tee -a "$MAIN_LOG"
echo "[$(date '+%F %T')] No old LoRA match/benchmark cache will be reused." | tee -a "$MAIN_LOG"

echo "[$(date '+%F %T')] Joint LoRA + warm-started projection training starts" | tee -a "$MAIN_LOG"
"$TRAIN_PY" -u scripts/train_lora.py \
  --sparse_dir "$SPARSE_DIR" \
  --scenes "${SCENES_TRAIN[@]}" \
  --lora_checkpoint "$STAGE1_CKPT" \
  --projection_checkpoint "$PROJ_CKPT" \
  --epochs 10 \
  --pairs_per_epoch 10000 \
  --num_correspondences 512 \
  --min_correspondences 50 \
  --temperature 0.07 \
  --fusion_alpha 0.5 \
  --lr_lora 5e-5 \
  --lr_proj 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --output_dir "$EXP_DIR" \
  --device cuda:0 \
  --log_interval 100 \
  --seed 42 \
  2>&1 | tee -a "$MAIN_LOG"
echo "[$(date '+%F %T')] Training done" | tee -a "$MAIN_LOG"

"$TRAIN_PY" - "$STAGE1_CKPT" "$EXP_DIR/best.pt" <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import hashlib
import json
import sys
import torch

stage1_ckpt, run_ckpt = sys.argv[1:3]

def sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

with open(run_ckpt.replace("/best.pt", "/config.json")) as f:
    cfg = json.load(f)
s1 = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
s2 = torch.load(run_ckpt, map_location="cpu", weights_only=False)

print("TRAIN_CONFIG projection_checkpoint=", cfg.get("projection_checkpoint"))
print("TRAIN_CONFIG lora_checkpoint=", cfg.get("lora_checkpoint"))
print("TRAIN_CONFIG pairs_per_epoch=", cfg.get("pairs_per_epoch"))
print("TRAIN_CONFIG num_correspondences=", cfg.get("num_correspondences"))
print("TRAIN_CONFIG fusion_alpha=", cfg.get("fusion_alpha"))
print("TRAIN_CONFIG lr_lora=", cfg.get("lr_lora"))
print("TRAIN_CONFIG lr_proj=", cfg.get("lr_proj"))
print("lora_equal_to_stage1", sha_state(s1["lora_state_dict"]) == sha_state(s2["lora_state_dict"]))
PY

run_scene () {
  local scene="$1"
  local pairs_file="$ROOT/output/pairs_${scene}.txt"
  local images_dir="$ROOT/datasets/phototourism/${scene}/images_preprocessed"
  local depth_dir="$ROOT/datasets/phototourism/${scene}/depth_unidepth"
  local sparse_dir="$ROOT/datasets/phototourism/${scene}/dense/sparse"
  local preprocess_info="$images_dir/preprocess_info.json"
  local match_dir="$MATCH_ROOT/$scene"
  local bench_file="$BENCH_ROOT/${MATCH_KEY}_${scene}.h5"
  local scene_result_dir="$RESULT_ROOT/$scene"

  if [ -e "$bench_file" ]; then
    echo "Refusing to overwrite benchmark file: $bench_file" | tee -a "$MAIN_LOG"
    exit 1
  fi

  mkdir -p "$match_dir" "$scene_result_dir"

  echo "[$(date '+%F %T')] Eval starts for $scene" | tee -a "$MAIN_LOG"

  "$TRAIN_PY" -u scripts/lora_matches.py \
    --lora_checkpoint "$EXP_DIR/best.pt" \
    --pairs_file "$pairs_file" \
    --images_dir "$images_dir" \
    --output_dir "$match_dir" \
    --max_points 2000 \
    --img_size 768 768 \
    --t 0 \
    --up_ft_index 2 \
    --ensemble_size 8 \
    --alpha 0.5 \
    --limit 15000 \
    --device cuda:0 \
    --lora_cache "$LORA_EVAL_CACHE" \
    2>&1 | tee -a "$MAIN_LOG"

  "$REPOSED_PY" -u scripts/pack_benchmark.py \
    --matches_dir "$match_dir" \
    --depth_dir "$depth_dir" \
    --sparse_dir "$sparse_dir" \
    --pairs_file "$pairs_file" \
    --output "$bench_file" \
    --limit 15000 \
    2>&1 | tee -a "$MAIN_LOG"

  "$REPOSED_PY" -u external/RePoseD/eval.py \
    "$bench_file" \
    -nw 8 --thesis \
    --output_dir "$scene_result_dir" \
    --preprocess_info "$preprocess_info" \
    2>&1 | tee -a "$MAIN_LOG"

  "$TRAIN_PY" - "$scene_result_dir" "$MATCH_KEY" "$scene" <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import json
import os
import sys

result_dir, match_key, scene = sys.argv[1:4]
path = os.path.join(result_dir, f"calibrated-{match_key}_{scene}.json")
with open(path) as f:
    data = json.load(f)
vals = []
for row in data:
    if row.get("experiment") != "3p_ours_shift_scale+12":
        continue
    r = float(row["R_err"])
    t = float(row["t_err"])
    vals.append(180.0 if (r != r or t != t) else max(r, t))
n = len(vals)
maa = sum(sum(1 for v in vals if v < th) / float(n) for th in range(1, 11)) / 10.0 * 100.0
print(f"SCENE {scene} mAA@10={maa:.10f} N={n}")
PY

  echo "[$(date '+%F %T')] Eval done for $scene" | tee -a "$MAIN_LOG"
}

for scene in "${SCENES_EVAL[@]}"; do
  run_scene "$scene"
done

"$TRAIN_PY" - "$RESULT_ROOT" "$MATCH_KEY" <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import json
import os
import sys

result_root, match_key = sys.argv[1:3]
scenes = ["sacre_coeur", "reichstag", "st_peters_square"]
scores = {}
for scene in scenes:
    path = os.path.join(result_root, scene, f"calibrated-{match_key}_{scene}.json")
    with open(path) as f:
        data = json.load(f)
    vals = []
    for row in data:
        if row.get("experiment") != "3p_ours_shift_scale+12":
            continue
        r = float(row["R_err"])
        t = float(row["t_err"])
        vals.append(180.0 if (r != r or t != t) else max(r, t))
    n = len(vals)
    maa = sum(sum(1 for v in vals if v < th) / float(n) for th in range(1, 11)) / 10.0 * 100.0
    scores[scene] = maa
avg = sum(scores.values()) / len(scores)
print("FINAL_SUMMARY")
for scene in scenes:
    print(f"{scene}: {scores[scene]:.10f}")
print(f"avg: {avg:.10f}")
PY

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$MAIN_LOG"
echo "MAIN_LOG=$MAIN_LOG"
echo "RESULT_ROOT=$RESULT_ROOT"
