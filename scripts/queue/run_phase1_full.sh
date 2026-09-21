#!/bin/bash
# Phase 1: Martell-Hybrid with AMP, batch 16, patch 132 (200 epochs) + BD-rate and perceptual eval
set -e
cd /home/filip/vvc-cnn-inter-enhancement

# 1. Train 200 epok AMP @ batch 16, patch 132
echo "=================================================="
echo "  PHASE 1 TRAINING (Martell-Hybrid AMP, 200 epok)"
echo "=================================================="
uv run python train_martell_hybrid.py \
    --epochs 200 --batch-size 16 --patch-size 132 \
    --ckpt-prefix martell_hybrid_amp \
    --save-every 50

# 2. BD-rate eval (patches 132)
echo "=================================================="
echo "  PHASE 1 BD-RATE EVAL"
echo "=================================================="
uv run python evaluate_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_amp_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_hybrid_amp_patch.json

# 3. Perceptual eval (patches 132)
echo "=================================================="
echo "  PHASE 1 PERCEPTUAL EVAL"
echo "=================================================="
uv run python evaluate_perceptual_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_amp_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_hybrid_amp_perceptual_patch.json

echo "=================================================="
echo "  PHASE 1 DONE"
echo "=================================================="
