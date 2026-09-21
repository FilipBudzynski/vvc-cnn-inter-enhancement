#!/bin/bash
# Full evaluation of the no-metadata ablation (BD-rate full frame, patch, perceptual)
set -e
mkdir -p bdrate_results

CKPT="checkpoints/martell_hybrid_nometa_best.pt"
VIDEOS="Johnny_1280x720_60,vidyo1_720p_60fps,vidyo3_720p_60fps"

echo "=== Full-frame BD-rate (720p subset) ==="
uv run python evaluate_bd.py \
    --model martell_nometa \
    --checkpoint "$CKPT" \
    --videos "$VIDEOS" \
    --out bdrate_results/martell_hybrid_nometa_720p.json

echo "=== Patch BD-rate ==="
uv run python evaluate_patch.py \
    --model martell_nometa \
    --checkpoint "$CKPT" \
    --videos "$VIDEOS" \
    --out bdrate_results/martell_hybrid_nometa_patch.json

echo "=== Perceptual (SSIM/MS-SSIM/LPIPS) ==="
uv run python evaluate_perceptual.py \
    --model martell_nometa \
    --checkpoint "$CKPT" \
    --videos "$VIDEOS" \
    --out bdrate_results/martell_hybrid_nometa_perceptual.json

echo "=== Rebuilding tables ==="
uv run python build_bdrate_table.py
uv run python build_perceptual_table.py

echo "=== DONE ==="
