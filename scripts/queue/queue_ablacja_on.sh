#!/bin/bash
# Metadata-inactive variant in the filters-on setting (training + CTC eval A-F)
set -x
cd /home/filip/vvc-cnn-inter-enhancement
{
  echo "=== $(date) [ablacja-on] train martell_hybrid on (metadane nieaktywne) ==="
  uv run python train_mqp.py --model martell_hybrid --variant on --num-workers 12 > martell_hybrid_mqp_on_run.log 2>&1
  echo "=== $(date) [ablacja-on] eval A-F (kafle 256) ==="
  timeout 10800 env VVC_TILE_MAX_ROWS=256 uv run python ctc_vtm_evaluate.py --variant on --classes A,B,C,D,E,F --models martell_hybrid_mqp_on > output_vtm/eval_mqp_martell_hybrid_on.log 2>&1
  echo "=== $(date) [ablacja-on] DONE ==="
} >> output_vtm/run_mqp.log 2>&1
