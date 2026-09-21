#!/bin/bash
# VVC-PPFF trained with its original procedure (tag mqp3), remaining filters-on trainings
set -x
cd /home/filip/vvc-cnn-inter-enhancement
{
  echo "=== $(date) [mqp16] train vvc_ppff mqp3 off (wierna procedura: clamp, wd=1e-4) ==="
  uv run python train_mqp.py --model vvc_ppff --variant off --num-workers 12 --wd 1e-4 --tag mqp3 > vvc_ppff_mqp3_off_run.log 2>&1
  echo "=== $(date) [mqp16] eval vvc_ppff mqp3 off (A-F) ==="
  uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models vvc_ppff_mqp3_off > output_vtm/eval_mqp3_vvc_ppff_off.log 2>&1
  echo "=== $(date) [mqp16] qg off dokonczenie (male kafle, B-F) ==="
  VVC_TILE_MAX_ROWS=384 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E,F --models qg_conv_lstm_mqp_off > output_vtm/eval_mqp_qg_conv_lstm_off3.log 2>&1
  echo "=== $(date) [mqp16] dolot A,F: martell proc. pierwotna ==="
  uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F --models martell_hybrid_mqp_off > output_vtm/eval_mqp_martell_hybrid_off_af.log 2>&1
  echo "=== $(date) [mqp16] train martell_hybrid on (wd=0) ==="
  uv run python train_mqp.py --model martell_hybrid --variant on --num-workers 12 --wd 0 --tag mqpnwd > martell_hybrid_mqpnwd_on_run.log 2>&1
  uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models martell_hybrid_mqpnwd_on > output_vtm/eval_mqpnwd_martell_on.log 2>&1
  for M in stenet_2024 bi_conv_lstm; do
    echo "=== $(date) [mqp16] train $M on ==="
    uv run python train_mqp.py --model $M --variant on --num-workers 12 > ${M}_mqp_on_run.log 2>&1
    uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models ${M}_mqp_on > output_vtm/eval_mqp_${M}_on.log 2>&1
  done
  echo "=== $(date) [mqp16] train qg on ==="
  uv run python train_mqp.py --model qg_conv_lstm --variant on --num-workers 12 > qg_conv_lstm_mqp_on_run.log 2>&1
  VVC_TILE_MAX_ROWS=384 uv run python ctc_vtm_evaluate.py --variant on --classes B,C,D,E --models qg_conv_lstm_mqp_on > output_vtm/eval_mqp_qg_conv_lstm_on.log 2>&1
  echo "=== $(date) [mqp16] MQP ALL DONE ==="
} >> output_vtm/run_mqp.log 2>&1
