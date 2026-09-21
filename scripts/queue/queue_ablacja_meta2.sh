#!/bin/bash
# Leave-one-out ablation, continued: all four MV channels jointly, then remaining channels
set -x
cd "$(dirname "$0")/../.."
NAMES=(QP PredMode Depth Boundary MVL0_X MVL0_Y MVL1_X MVL1_Y FrameType)
{
  echo "=== $(date) [abl2] train bez WSZYSTKICH wektorow ruchu (ch 4,5,6,7) ==="
  uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag abl_allMV --zero-meta 4,5,6,7 > martell_hybrid_abl_allMV_off_run.log 2>&1
  echo "=== $(date) [abl2] eval bez WSZYSTKICH wektorow ruchu B-E ==="
  timeout 7200 env VVC_ZERO_META=4,5,6,7 VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl_allMV_off > output_vtm/eval_abl_allMV_off.log 2>&1
  grep "clean-avg" output_vtm/eval_abl_allMV_off.log

  for CH in 7 2 3 1 8; do
    N=${NAMES[$CH]}
    echo "=== $(date) [abl2] train bez $N (ch $CH) ==="
    uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag abl$CH --zero-meta $CH > martell_hybrid_abl${CH}_off_run.log 2>&1
    echo "=== $(date) [abl2] eval bez $N B-E ==="
    timeout 7200 env VVC_ZERO_META=$CH VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl${CH}_off > output_vtm/eval_abl${CH}_off.log 2>&1
    grep "clean-avg" output_vtm/eval_abl${CH}_off.log
  done
  echo "=== $(date) [abl2] ABLACJA ALL DONE ==="
} >> output_vtm/run_ablacja.log 2>&1
