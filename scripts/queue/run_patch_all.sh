#!/bin/bash
# Patch-based (132x132) BD-rate and perceptual evals on the 10-video test set
set -e
cd "$(dirname "$0")/../.."
mkdir -p bdrate_results

# 1. Patch BD-rate for missing models
echo "================================================="
echo "  PATCH BD-RATE (missing models)"
echo "================================================="

for spec in \
    "snow_wide:checkpoints/snow_wide_9ch_best.pt:snow_wide_patch.json" \
    "martell_nometa:checkpoints/martell_hybrid_nometa_best.pt:martell_hybrid_nometa_patch_10v.json"
do
    IFS=":" read -r MODEL CKPT OUT <<< "$spec"
    if [ -f "bdrate_results/$OUT" ] && [ "$(uv run python -c "import json; d=json.load(open('bdrate_results/$OUT')); print(len(d.get('videos', [])))" 2>/dev/null)" = "10" ]; then
        echo "=== ${MODEL} patch BD already done (10v), skipping ==="
        continue
    fi
    echo "=== ${MODEL} patch BD-rate ==="
    uv run python evaluate_patch.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --out "bdrate_results/$OUT"
done

# 2. Patch perceptual for all 6 models
echo "================================================="
echo "  PATCH PERCEPTUAL (all models)"
echo "================================================="

for spec in \
    "martell:checkpoints/martell_hybrid_best.pt:martell_hybrid_perceptual_patch.json" \
    "martell_nometa:checkpoints/martell_hybrid_nometa_best.pt:martell_hybrid_nometa_perceptual_patch.json" \
    "snow_wide:checkpoints/snow_wide_9ch_best.pt:snow_wide_perceptual_patch.json" \
    "bi_conv_lstm:checkpoints/bi_conv_lstm_best.pt:bi_conv_lstm_perceptual_patch.json" \
    "stenet:checkpoints/stenet_2024_best.pt:stenet_perceptual_patch.json" \
    "vvc_ppff:checkpoints/vvc_ppff_epoch_190.pt:vvc_ppff_perceptual_patch.json"
do
    IFS=":" read -r MODEL CKPT OUT <<< "$spec"
    if [ -f "bdrate_results/$OUT" ]; then
        echo "=== ${MODEL} patch perceptual already done, skipping ==="
        continue
    fi
    echo "=== ${MODEL} patch perceptual ==="
    uv run python evaluate_perceptual_patch.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --out "bdrate_results/$OUT"
done

echo "================================================="
echo "  ALL PATCH EVAL DONE"
echo "================================================="
