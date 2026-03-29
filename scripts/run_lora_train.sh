#!/usr/bin/env bash
# Full LoRA training launch script (Phase 2b).
# Run inside screen:  screen -S lora_train  then  bash scripts/run_lora_train.sh
#
# Steps:
#   1. Train LoRA + projection head on 10 MegaDepth scenes
#   2. Eval best checkpoint on all 3 PhotoTourism scenes (sacre_coeur, reichstag, st_peters_square)
#
# Expected duration:
#   Training: 10000 pairs/epoch × ~0.5 s/pair (AMP) × 10 epochs ≈ 14 h
#   Eval:     ~1 h per scene × 3 scenes ≈ 3 h
# GPU memory: ~10-15 GB (AMP fp16)
# Output: experiments/phase2_lora_r4_v1/{best.pt, latest.pt, config.json, train_log.json}

set -euo pipefail

PYTHON=/home.stud/gorbuden/.conda/envs/dinov3/bin/python

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

RUN_ID="phase2_lora_r4_v1"
OUTPUT_DIR="experiments/${RUN_ID}"
EVAL_SCENES=(sacre_coeur reichstag st_peters_square)

echo "=== Phase 2b LoRA training ===" | tee logs/lora_train_${RUN_ID}.log
echo "Output dir: ${OUTPUT_DIR}" | tee -a logs/lora_train_${RUN_ID}.log
echo "Start: $(date)" | tee -a logs/lora_train_${RUN_ID}.log

$PYTHON -u scripts/train_lora.py \
    --megadepth_root /mnt/datasets/MegaDepth/MegaDepth_v1_SfM \
    --sparse_dir data/sparse_train \
    --scenes 0080 0042 0380 0000 0366 0001 0005 0237 0011 0148 \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_dropout 0.0 \
    --projection_checkpoint experiments/phase2_projection_wide/best.pt \
    --epochs 10 \
    --pairs_per_epoch 10000 \
    --num_correspondences 256 \
    --min_correspondences 50 \
    --temperature 0.07 \
    --lr_lora 5e-5 \
    --lr_proj 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --log_interval 200 \
    --seed 42 \
    2>&1 | tee -a logs/lora_train_${RUN_ID}.log

echo "=== Training complete: $(date) ===" | tee -a logs/lora_train_${RUN_ID}.log
echo "Checkpoint: ${OUTPUT_DIR}/best.pt" | tee -a logs/lora_train_${RUN_ID}.log

# ============================================================
# Eval on all three PhotoTourism scenes
# ============================================================
echo "" | tee -a logs/lora_train_${RUN_ID}.log
echo "=== Starting eval on ${#EVAL_SCENES[@]} scenes ===" | tee -a logs/lora_train_${RUN_ID}.log

for SCENE in "${EVAL_SCENES[@]}"; do
    echo "" | tee -a logs/lora_train_${RUN_ID}.log
    echo "--- Eval: ${SCENE} ---" | tee -a logs/lora_train_${RUN_ID}.log
    bash scripts/run_lora_eval.sh \
        --checkpoint "${OUTPUT_DIR}/best.pt" \
        --run_id "${RUN_ID}_eval" \
        --scene "${SCENE}" \
        2>&1 | tee -a logs/lora_train_${RUN_ID}.log
done

echo "" | tee -a logs/lora_train_${RUN_ID}.log
echo "=== All done: $(date) ===" | tee -a logs/lora_train_${RUN_ID}.log
