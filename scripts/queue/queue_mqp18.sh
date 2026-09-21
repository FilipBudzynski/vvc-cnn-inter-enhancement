#!/bin/bash
# VVC-PPFF (mqp3) evaluation A-F, remaining filters-on trainings and evals
set -x
cd "$(dirname "$0")/../.."
{
  echo "=== $(date) [mqp18] eval vvc_ppff mqp3 off A-F (mini-kafle, timeout 3h) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models vvc_ppff_mqp3_off > output_vtm/eval_mqp3_vvc_ppff_off.log 2>&1
  echo "=== $(date) [mqp18] qg off dokonczenie (male kafle, timeout 3h) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=384 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E,F --models qg_conv_lstm_mqp_off > output_vtm/eval_mqp_qg_conv_lstm_off3.log 2>&1
  echo "=== $(date) [mqp18] wznowienie: train martell on (wd=0) ==="
  uv run python train_mqp.py --model martell_hybrid --variant on --num-workers 12 --wd 0 --tag mqpnwd > martell_hybrid_mqpnwd_on_run.log 2>&1
  timeout 7200 uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models martell_hybrid_mqpnwd_on > output_vtm/eval_mqpnwd_martell_on.log 2>&1
  for M in stenet_2024 bi_conv_lstm; do
    echo "=== $(date) [mqp18] train $M on ==="
    uv run python train_mqp.py --model $M --variant on --num-workers 12 > ${M}_mqp_on_run.log 2>&1
    timeout 7200 uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models ${M}_mqp_on > output_vtm/eval_mqp_${M}_on.log 2>&1
  done
  echo "=== $(date) [mqp18] dolot A,F martell proc. pierwotna ==="
  timeout 10800 uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models martell_hybrid_mqp_off > output_vtm/eval_mqp_martell_hybrid_off_af.log 2>&1
  echo "=== $(date) [mqp18] train qg on ==="
  uv run python train_mqp.py --model qg_conv_lstm --variant on --num-workers 12 > qg_conv_lstm_mqp_on_run.log 2>&1
  timeout 14400 env VVC_TILE_MAX_ROWS=384 uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models qg_conv_lstm_mqp_on > output_vtm/eval_mqp_qg_conv_lstm_on.log 2>&1
  echo "=== $(date) [mqp18] MQP ALL DONE ==="
} >> output_vtm/run_mqp.log 2>&1
