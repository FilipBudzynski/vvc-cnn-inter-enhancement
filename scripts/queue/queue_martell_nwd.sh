#!/bin/bash
# Proposed model, multi-QP training without weight decay (tag mqpnwd) + CTC eval A-F
cd "$(dirname "$0")/../.."
{
  echo "=== $(date) [nwd] waiting for OFF round done ==="
  until grep -q "OFF round done" output_vtm/run_mqp.log; do sleep 120; done
  echo "=== $(date) [nwd] train martell_hybrid off wd=0 ==="
  uv run python train_mqp.py --model martell_hybrid --variant off \
    --num-workers 12 --wd 0 --tag mqpnwd > martell_hybrid_mqpnwd_off_run.log 2>&1
  echo "=== $(date) [nwd] eval martell_hybrid mqpnwd off (A-F) ==="
  uv run python ctc_vtm_evaluate.py --variant off --classes A,B,C,D,E,F \
    --models martell_hybrid_mqpnwd_off > output_vtm/eval_mqpnwd_martell_off.log 2>&1
  echo "=== $(date) [nwd] done ==="
} >> output_vtm/run_mqp.log 2>&1
