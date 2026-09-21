#!/bin/bash
# Filters-on round: multi-QP trainings and CTC evals A-F
set -x
cd "$(dirname "$0")/../.."
{
  echo "=== $(date) [on-round] train martell_hybrid on (wd=0, mqpnwd) ==="
  uv run python train_mqp.py --model martell_hybrid --variant on --num-workers 12 --wd 0 --tag mqpnwd > martell_hybrid_mqpnwd_on_run.log 2>&1
  echo "=== $(date) [on-round] eval martell on A-F (kafle 256) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F --models martell_hybrid_mqpnwd_on > output_vtm/eval_mqpnwd_martell_on.log 2>&1
  echo "=== $(date) [on-round] train stenet on ==="
  uv run python train_mqp.py --model stenet_2024 --variant on --num-workers 12 > stenet_2024_mqp_on_run.log 2>&1
  echo "=== $(date) [on-round] eval stenet on A-F ==="
  timeout 10800 uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F --models stenet_2024_mqp_on > output_vtm/eval_mqp_stenet_2024_on.log 2>&1
  echo "=== $(date) [on-round] train bi on ==="
  uv run python train_mqp.py --model bi_conv_lstm --variant on --num-workers 12 > bi_conv_lstm_mqp_on_run.log 2>&1
  echo "=== $(date) [on-round] eval bi on A-F ==="
  timeout 10800 uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F --models bi_conv_lstm_mqp_on > output_vtm/eval_mqp_bi_conv_lstm_on.log 2>&1
  echo "=== $(date) [on-round] train qg on ==="
  uv run python train_mqp.py --model qg_conv_lstm --variant on --num-workers 12 > qg_conv_lstm_mqp_on_run.log 2>&1
  echo "=== $(date) [on-round] eval qg on B-F (kafle 192) ==="
  timeout 14400 env VVC_TILE_MAX_ROWS=192 uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E,F --models qg_conv_lstm_mqp_on > output_vtm/eval_mqp_qg_conv_lstm_on.log 2>&1
  echo "=== $(date) [on-round] eval qg on klasa A (kafle 96/64) ==="
  timeout 18000 env VVC_TILE_MAX_ROWS=96 VVC_TILE_OVERLAP=64 uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F --models qg_conv_lstm_mqp_on >> output_vtm/eval_mqp_qg_conv_lstm_on.log 2>&1
  echo "=== $(date) [on-round] ON ROUND ALL DONE ==="
} >> output_vtm/run_mqp.log 2>&1
