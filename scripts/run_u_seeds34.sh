#!/usr/bin/env bash
# Train untied (U / A1) seeds 3 and 4 only. Does not overwrite existing seed 0–2.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
NW="${NW:-1}"
mkdir -p checkpoints results
LOG=results/u_seeds34_run.log
exec >>"$LOG" 2>&1

echo "===== $(date) U seeds 3–4 start ====="
for SEED in 3 4; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) seed=$SEED A1 untied from scratch ====="
  rm -f model.pth
  python train.py "exp_A1_seed${SEED}" --version 1 \
    --num-workers "$NW" --device cuda --seed "$SEED"
  cp -a model.pth "checkpoints/untied_A1_seed${SEED}.pth"
  echo "===== $(date) seed=$SEED done ====="
done
echo "===== $(date) U seeds 3–4 complete ====="
