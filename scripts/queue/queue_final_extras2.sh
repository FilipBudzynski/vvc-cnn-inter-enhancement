#!/bin/bash
# VVC-PPFF repeat with late LR milestones (tag mqp2) + evals
set -x
cd /home/filip/vvc-cnn-inter-enhancement
{
  echo "=== $(date) [extras2] waiting for MQP ALL DONE ==="
  until grep -q "MQP ALL DONE" output_vtm/run_mqp.log; do sleep 300; done
  for V in off on; do
    echo "=== $(date) [extras2] train vvc_ppff mqp2 $V ==="
    uv run python train_mqp.py --model vvc_ppff --variant $V --num-workers 12 \
      --milestones 30,40,46,50 --tag mqp2 > vvc_ppff_mqp2_${V}_run.log 2>&1
    echo "=== $(date) [extras2] eval vvc_ppff mqp2 $V ==="
    uv run python ctc_vtm_evaluate.py --variant $V --classes A,B,C,D,E,F \
      --models vvc_ppff_mqp2_${V} > output_vtm/eval_mqp2_vvc_ppff_${V}.log 2>&1
  done
  echo "=== $(date) [extras2] A,F top-up for ON-round models ==="
  until [ -f af_on_ready.marker ]; do sleep 300; done
  for M in martell_hybrid vvc_ppff stenet_2024 bi_conv_lstm qg_conv_lstm martell_hybrid_nometa; do
    uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F \
      --models ${M}_mqp_on > output_vtm/eval_mqp_${M}_on_af.log 2>&1
  done
  echo "=== $(date) [extras2] deferred QG full-frame BD eval ==="
  uv run python evaluate_bd.py --model qg_conv_lstm \
    --checkpoint checkpoints/qg_conv_lstm_fixed2_best.pt \
    --out bdrate_results/qg_conv_lstm_fixed2.json > qg_fixed2_fullframe.log 2>&1
  echo "=== $(date) [extras2] ALL EXTRAS DONE ==="
} >> output_vtm/run_extras.log 2>&1
