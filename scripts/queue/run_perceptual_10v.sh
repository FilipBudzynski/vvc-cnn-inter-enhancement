#!/bin/bash
# Perceptual eval on the 10-video test set
set -e
cd "$(dirname "$0")/../.."
mkdir -p bdrate_results

VIDEOS="Johnny_1280x720_60,controlled_burn_1080p,pedestrian_area_1080p25,red_kayak_1080p,rush_hour_1080p25,sunflower_1080p25,touchdown_pass_1080p,tractor_1080p25,vidyo1_720p_60fps,vidyo3_720p_60fps"

for spec in \
    "martell:checkpoints/martell_hybrid_best.pt:martell_hybrid_perceptual_10v.json" \
    "bi_conv_lstm:checkpoints/bi_conv_lstm_best.pt:bi_conv_lstm_perceptual_10v.json" \
    "stenet:checkpoints/stenet_2024_best.pt:stenet_perceptual_10v.json" \
    "vvc_ppff:checkpoints/vvc_ppff_epoch_190.pt:vvc_ppff_perceptual_10v.json" \
    "martell_nometa:checkpoints/martell_hybrid_nometa_best.pt:martell_hybrid_nometa_perceptual_10v.json"
do
    IFS=":" read -r MODEL CKPT OUT <<< "$spec"
    echo "=== ${MODEL} ==="
    uv run python evaluate_perceptual.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --videos "$VIDEOS" \
        --out "bdrate_results/$OUT"
done
echo "All perceptual 10v done."
