#!/usr/bin/env bash
# Smoke test: 200 pairs, 2 epochs, scene 0148 (~5-10 min)
set -euo pipefail

PYTHON=/home.stud/gorbuden/.conda/envs/dinov3/bin/python

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs experiments/phase2_lora_smoke

echo "=== LoRA smoke test: $(date) ===" | tee logs/lora_smoke.log
echo "Python: $($PYTHON --version 2>&1)" | tee -a logs/lora_smoke.log

$PYTHON -u scripts/train_lora.py \
    --megadepth_root /mnt/datasets/MegaDepth/MegaDepth_v1_SfM \
    --sparse_dir data/sparse_train \
    --scenes 0148 \
    --epochs 2 \
    --pairs_per_epoch 200 \
    --projection_checkpoint experiments/phase2_projection_wide/best.pt \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_dropout 0.0 \
    --lr_lora 5e-5 \
    --lr_proj 1e-3 \
    --temperature 0.07 \
    --num_correspondences 256 \
    --min_correspondences 50 \
    --output_dir experiments/phase2_lora_smoke \
    --device cuda:0 \
    --log_interval 50 \
    2>&1 | tee -a logs/lora_smoke.log

echo "=== Done: $(date) ===" | tee -a logs/lora_smoke.log
