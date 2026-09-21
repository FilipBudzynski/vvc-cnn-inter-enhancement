#!/bin/bash
# Perceptual eval (SSIM, MS-SSIM, LPIPS) for the trained models, sequentially
set -e
mkdir -p bdrate_results

VIDEOS="Johnny_1280x720_60,vidyo1_720p_60fps,vidyo3_720p_60fps"

for spec in \
    "martell:checkpoints/martell_hybrid_best.pt:martell_hybrid_perceptual.json" \
    "bi_conv_lstm:checkpoints/bi_conv_lstm_best.pt:bi_conv_lstm_perceptual.json" \
    "qg_conv_lstm:checkpoints/qg_conv_lstm_best.pt:qg_conv_lstm_perceptual.json" \
    "stenet:checkpoints/stenet_2024_best.pt:stenet_perceptual.json" \
    "vvc_ppff:checkpoints/vvc_ppff_epoch_190.pt:vvc_ppff_perceptual.json"
do
    IFS=":" read -r MODEL CKPT OUT <<< "$spec"
    echo "=== ${MODEL} ==="
    uv run python evaluate_perceptual.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --videos "$VIDEOS" \
        --out "bdrate_results/$OUT"
done
echo "All done."
