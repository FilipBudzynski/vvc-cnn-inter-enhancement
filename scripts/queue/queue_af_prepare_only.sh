#!/bin/bash
# VTM CTC classes A and F: data preparation only
cd /home/filip/vvc-cnn-inter-enhancement
{
  echo "=== $(date) [af] waiting for OFF A,F prepare (pid 44530) ==="
  while kill -0 44530 2>/dev/null; do sleep 120; done
  echo "=== $(date) [af] OFF verify pass ==="
  nice -n 19 uv run python ctc_vtm_prepare.py --variant off --classes A,F --stage all
  echo "=== $(date) [af] prepare ON A,F ==="
  nice -n 19 uv run python ctc_vtm_prepare.py --variant on --classes A,F --stage all
  touch af_data_ready.marker
  echo "=== $(date) [af] AF DATA READY (no evals scheduled) ==="
} >> output_vtm/run_af.log 2>&1
