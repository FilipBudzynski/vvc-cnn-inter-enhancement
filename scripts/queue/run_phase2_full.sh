#!/bin/bash
# Phase 2: Martell-Hybrid with AMP, patch 256 (200 epochs) + eval on 256 and 132 patches
set -e
cd /home/filip/vvc-cnn-inter-enhancement

# 1. Train 200 epok @ patch 256, batch 8 (256^2 ~4x VRAM patcha 132)
echo "=================================================="
echo "  PHASE 2 TRAINING (Martell-Hybrid 256x256, 200 epok)"
echo "=================================================="
uv run python train_martell_hybrid.py \
    --epochs 200 --batch-size 8 --patch-size 256 \
    --ckpt-prefix martell_hybrid_256 \
    --save-every 50

# 2. BD-rate eval @ 256
echo "=================================================="
echo "  PHASE 2 BD-RATE EVAL @ 256x256"
echo "=================================================="
uv run python evaluate_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_256_best.pt \
    --patch-size 256 \
    --out bdrate_results/martell_hybrid_256_patch256.json

# 3. BD-rate eval @ 132 (for comparison with prior Faza 1)
echo "=================================================="
echo "  PHASE 2 BD-RATE EVAL @ 132x132"
echo "=================================================="
uv run python evaluate_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_256_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_hybrid_256_patch132.json

# 4. Perceptual @ 256
echo "=================================================="
echo "  PHASE 2 PERCEPTUAL EVAL @ 256x256"
echo "=================================================="
uv run python evaluate_perceptual_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_256_best.pt \
    --patch-size 256 \
    --out bdrate_results/martell_hybrid_256_perceptual_patch256.json

# 5. Perceptual @ 132 (porownanie)
echo "=================================================="
echo "  PHASE 2 PERCEPTUAL EVAL @ 132x132"
echo "=================================================="
uv run python evaluate_perceptual_patch.py \
    --model martell \
    --checkpoint checkpoints/martell_hybrid_256_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_hybrid_256_perceptual_patch132.json

echo "=================================================="
echo "  PHASE 2 DONE"
echo "=================================================="
