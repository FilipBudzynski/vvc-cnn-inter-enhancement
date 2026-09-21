#!/bin/bash
# Classes A/F evaluation, remaining models (smaller tiles on class A)
set -x
cd "$(dirname "$0")/../.."
{
  echo "=== $(date) [af-topup2] vvc_ppff: A,F (kafle 128) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=128 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models vvc_ppff_mqp_off >> output_vtm/eval_topup_vvc.log 2>&1
  echo "=== $(date) [af-topup2] qg: klasa A (kafle 192) ==="
  timeout 14400 env VVC_TILE_MAX_ROWS=192 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models qg_conv_lstm_mqp_off >> output_vtm/eval_topup_qg_a.log 2>&1
  echo "=== $(date) [af-topup2] DONE ==="
} >> output_vtm/run_mqp.log 2>&1
