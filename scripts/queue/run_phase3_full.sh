#!/bin/bash
# Phase 3: Martell-UNet, patch 256 (200 epochs) + eval on 256 and 132 patches
set -e
cd "$(dirname "$0")/../.."

# 1. Train 200 epok
echo "=================================================="
echo "  PHASE 3 TRAINING (Martell-UNet 256x256, 200 epok)"
echo "=================================================="
uv run python train_martell_unet.py \
    --epochs 200 --batch-size 8 --patch-size 256 \
    --ckpt-prefix martell_unet \
    --save-every 50

# 2. BD-rate eval @ 256
echo "=================================================="
echo "  PHASE 3 BD-RATE EVAL @ 256x256"
echo "=================================================="
uv run python evaluate_patch.py \
    --model martell_unet \
    --checkpoint checkpoints/martell_unet_best.pt \
    --patch-size 256 \
    --out bdrate_results/martell_unet_patch256.json

# 3. BD-rate eval @ 132
echo "=================================================="
echo "  PHASE 3 BD-RATE EVAL @ 132x132"
echo "=================================================="
uv run python evaluate_patch.py \
    --model martell_unet \
    --checkpoint checkpoints/martell_unet_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_unet_patch132.json

# 4. Perceptual @ 256
echo "=================================================="
echo "  PHASE 3 PERCEPTUAL EVAL @ 256x256"
echo "=================================================="
uv run python evaluate_perceptual_patch.py \
    --model martell_unet \
    --checkpoint checkpoints/martell_unet_best.pt \
    --patch-size 256 \
    --out bdrate_results/martell_unet_perceptual_patch256.json

# 5. Perceptual @ 132
echo "=================================================="
echo "  PHASE 3 PERCEPTUAL EVAL @ 132x132"
echo "=================================================="
uv run python evaluate_perceptual_patch.py \
    --model martell_unet \
    --checkpoint checkpoints/martell_unet_best.pt \
    --patch-size 132 \
    --out bdrate_results/martell_unet_perceptual_patch132.json

echo "=================================================="
echo "  PHASE 3 DONE"
echo "=================================================="
