#!/bin/bash
# Leave-one-out ablation, final part: Depth eval, PredMode resume, FrameType
set -x
cd /home/filip/vvc-cnn-inter-enhancement
{
  echo "=== $(date) [resume] eval bez Depth B-E (checkpoint already trained) ==="
  timeout 7200 env VVC_ZERO_META=2 VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl2_off > output_vtm/eval_abl2_off.log 2>&1
  grep "clean-avg" output_vtm/eval_abl2_off.log

  echo "=== $(date) [resume] resume train bez PredMode (ch 1) from epoch 45 ==="
  uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag abl1 --zero-meta 1 >> martell_hybrid_abl1_off_run.log 2>&1
  echo "=== $(date) [resume] eval bez PredMode B-E ==="
  timeout 7200 env VVC_ZERO_META=1 VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl1_off > output_vtm/eval_abl1_off.log 2>&1
  grep "clean-avg" output_vtm/eval_abl1_off.log

  echo "=== $(date) [resume] train bez FrameType (ch 8) ==="
  uv run python train_mqp.py --model martell_hybrid --variant off --num-workers 12 --wd 0 --tag abl8 --zero-meta 8 > martell_hybrid_abl8_off_run.log 2>&1
  echo "=== $(date) [resume] eval bez FrameType B-E ==="
  timeout 7200 env VVC_ZERO_META=8 VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant off --classes B,C,D,E --models martell_hybrid_abl8_off > output_vtm/eval_abl8_off.log 2>&1
  grep "clean-avg" output_vtm/eval_abl8_off.log

  echo "=== $(date) [resume] ABLACJA ALL DONE ==="
} >> output_vtm/run_ablacja.log 2>&1
