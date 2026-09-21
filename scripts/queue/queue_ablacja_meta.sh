#!/bin/bash
# Leave-one-out metadata channel ablation: 9 trainings, channel zeroed in training and eval
set -x
cd "$(dirname "$0")/../.."
NAMES=(QP PredMode Depth Boundary MVL0_X MVL0_Y MVL1_X MVL1_Y FrameType)
{
  for CH in 0 4 5 6 7 2 3 1 8; do
    N=${NAMES[$CH]}
    echo "=== $(date) [abl] train bez $N (ch $CH) ==="
    uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag abl$CH --zero-meta $CH > martell_hybrid_abl${CH}_off_run.log 2>&1
    echo "=== $(date) [abl] eval bez $N B-E ==="
    timeout 7200 env VVC_ZERO_META=$CH VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl${CH}_off > output_vtm/eval_abl${CH}_off.log 2>&1
    grep "clean-avg" output_vtm/eval_abl${CH}_off.log
  done
  echo "=== $(date) [abl] ABLACJA ALL DONE ==="
} >> output_vtm/run_ablacja.log 2>&1
# group variants
{
  for SPEC in "4,5,6,7:allMV" "1,2,3:struct"; do
    CH=${SPEC%%:*}; N=${SPEC##*:}; TAG=abl_${N}
    echo "=== $(date) [abl] train bez grupy $N (ch $CH) ==="
    uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag $TAG --zero-meta $CH > martell_hybrid_${TAG}_off_run.log 2>&1
    echo "=== $(date) [abl] eval bez grupy $N B-E ==="
    timeout 7200 env VVC_ZERO_META=$CH VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_${TAG}_off > output_vtm/eval_${TAG}_off.log 2>&1
    grep "clean-avg" output_vtm/eval_${TAG}_off.log
  done
  echo "=== $(date) [abl] GRUPY DONE ==="
} >> output_vtm/run_ablacja.log 2>&1
