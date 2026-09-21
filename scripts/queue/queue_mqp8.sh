#!/bin/bash
# Multi-QP training and CTC eval of all compared architectures, filters-off round
set -x
cd /home/filip/vvc-cnn-inter-enhancement
ON_PGID=$1; ON_PID=$2; AF_PGID=$3
OFF_MODELS="qg_conv_lstm martell_hybrid_nometa"
ON_MODELS="martell_hybrid vvc_ppff stenet_2024 bi_conv_lstm qg_conv_lstm martell_hybrid_nometa"
pause_enc()  { :; }  # encoders run 24/7: crop-first dataset freed the CPU
resume_enc() { kill -CONT -$ON_PGID 2>/dev/null; kill -CONT -$AF_PGID 2>/dev/null; }
{
  (
    while kill -0 $ON_PID 2>/dev/null; do sleep 60; done
    uv run python mqp_prepare.py --variant on --stage all --enc-workers 5 \
      >> output_vtm/prepare_on_finish.log 2>&1
    touch mqp_on_ready.marker
  ) &
  for M in $OFF_MODELS; do
    echo "=== $(date) [mqp8] train $M off (encoders paused) ==="
    pause_enc
    uv run python train_mqp.py --model $M --variant off --num-workers 12 > ${M}_mqp_off_run.log 2>&1
    resume_enc
    echo "=== $(date) [mqp8] eval $M off (A-F) ==="
    uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models ${M}_mqp_off > output_vtm/eval_mqp_${M}_off.log 2>&1
  done
  echo "=== $(date) [mqp8] A,F top-up evals (martell_hybrid, vvc_ppff, off) ==="
  uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models martell_hybrid_mqp_off > output_vtm/eval_mqp_martell_hybrid_off_af.log 2>&1
  uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models vvc_ppff_mqp_off > output_vtm/eval_mqp_vvc_ppff_off_af.log 2>&1
  echo "=== $(date) [mqp8] OFF round done, waiting for ON data ==="
  until [ -f mqp_on_ready.marker ]; do sleep 120; done
  for M in $ON_MODELS; do
    echo "=== $(date) [mqp8] train $M on (encoders paused) ==="
    pause_enc
    uv run python train_mqp.py --model $M --variant on --num-workers 12 > ${M}_mqp_on_run.log 2>&1
    resume_enc
    CL="B,C,D,E"; [ -f af_on_ready.marker ] && CL="A,B,C,D,E,F"
    echo "=== $(date) [mqp8] eval $M on (classes $CL) ==="
    uv run python ctc_vtm_evaluate.py --variant on --classes $CL --models ${M}_mqp_on > output_vtm/eval_mqp_${M}_on.log 2>&1
  done
  resume_enc
  echo "=== $(date) [mqp8] MQP ALL DONE ==="
} >> output_vtm/run_mqp.log 2>&1
