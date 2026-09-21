#!/bin/bash
# Classes A/F evaluation: metadata-inactive variant, VVC-PPFF, QG-ConvLSTM
set -x
cd "$(dirname "$0")/../.."
{
  echo "=== $(date) [af-topup] martell bez metadanych: A,F ==="
  timeout 7200 env VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models martell_hybrid_mqp_off > output_vtm/eval_topup_martell_dead.log 2>&1
  echo "=== $(date) [af-topup] vvc_ppff: A,F (kafle 256) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models vvc_ppff_mqp_off > output_vtm/eval_topup_vvc.log 2>&1
  echo "=== $(date) [af-topup] qg: klasa A (kafle 192) ==="
  timeout 14400 env VVC_TILE_MAX_ROWS=192 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models qg_conv_lstm_mqp_off > output_vtm/eval_topup_qg_a.log 2>&1
  echo "=== $(date) [af-topup] DONE ==="
} >> output_vtm/run_mqp.log 2>&1
