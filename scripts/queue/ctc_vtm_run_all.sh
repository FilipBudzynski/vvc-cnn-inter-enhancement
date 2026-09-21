#!/bin/bash
# VTM CTC study, classes B-E: prepare both in-loop filter variants, then evaluate all models
set -x
cd /home/filip/vvc-cnn-inter-enhancement
mkdir -p output_vtm
{
  echo "=== $(date) prepare OFF ==="
  uv run python ctc_vtm_prepare.py --variant off --classes D,C,E,B --stage all --enc-workers 10
  echo "=== $(date) eval OFF (bg) + prepare ON ==="
  uv run python ctc_vtm_evaluate.py --variant off > output_vtm/eval_off.log 2>&1 &
  EVAL_OFF=$!
  uv run python ctc_vtm_prepare.py --variant on --classes D,C,E,B --stage all --enc-workers 10
  echo "=== $(date) waiting for eval OFF (pid $EVAL_OFF) ==="
  wait $EVAL_OFF
  echo "=== $(date) eval ON ==="
  uv run python ctc_vtm_evaluate.py --variant on > output_vtm/eval_on.log 2>&1
  echo "=== $(date) ALL DONE ==="
} >> output_vtm/run_all.log 2>&1
