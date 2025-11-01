#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_80m.sh [RUN_NAME]

RUN_NAME=${1:-comp_run_80m}

export PYGAME_HIDE_SUPPORT_PROMPT=1

python scripts/auto_train.py \
  --run-name "$RUN_NAME" \
  --save-path checkpoints \
  --total-steps 80000000 \
  --segment-steps 2000000 \
  --save-freq 50000 \
  --max-saved 50 \
  --resume \
  --log none \
  --vec-envs 8 \
  --vec-type subproc
