#!/bin/bash
# Multi-QP training of the proposed model on VTM data (both filter variants) + CTC eval
set -x
cd "$(dirname "$0")/../.."
until grep -q "AF ALL DONE" output_vtm/run_af.log 2>/dev/null && [ -f perc_done.marker ]; do sleep 300; done
{
  echo "=== $(date) mqp prepare OFF ==="
  uv run python mqp_prepare.py --variant off --stage all
  echo "=== $(date) mqp prepare ON ==="
  uv run python mqp_prepare.py --variant on --stage all
  echo "=== $(date) mqp train OFF ==="
  uv run python train_martell_mqp.py --variant off > martell_hybrid_mqp_off_run.log 2>&1
  echo "=== $(date) mqp train ON ==="
  uv run python train_martell_mqp.py --variant on > martell_hybrid_mqp_on_run.log 2>&1
  echo "=== $(date) mqp eval OFF ==="
  uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_mqp_off > output_vtm/eval_mqp_off.log 2>&1
  echo "=== $(date) mqp eval ON ==="
  uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models martell_hybrid_mqp_on > output_vtm/eval_mqp_on.log 2>&1
  echo "=== $(date) MQP ALL DONE ==="
} >> output_vtm/run_mqp.log 2>&1
