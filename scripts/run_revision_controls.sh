#!/usr/bin/env bash
# Priority-1 architecture / position controls for the journal revision.
# Runs AFTER U seeds 3–4 when GPU is free, or can be launched independently.
#
# Variants:
#   U-L1   : parameter-matched untied (depth=1, same unique params as T)
#   T-MHA  : tied MHA (gqa_groups=8)
#   POS-*  : seed-0 indicative absolute / relative / both ablations
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
NW="${NW:-1}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LOG=results/revision_controls_run.log
mkdir -p checkpoints results
exec >>"$LOG" 2>&1

echo "===== $(date) revision controls start ====="

# --- Parameter-matched untied (depth=1) ---
for SEED in $SEEDS; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) U-L1 seed=$SEED ====="
  rm -f model.pth
  python train.py "exp_UL1_seed${SEED}" --version 1 --layers 1 \
    --num-workers "$NW" --device cuda --seed "$SEED"
  cp -a model.pth "checkpoints/untied_L1_seed${SEED}.pth"
done

# --- Tied-MHA (G=H=8) ---
for SEED in $SEEDS; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) T-MHA seed=$SEED ====="
  rm -f model.pth
  python train.py "exp_TMHA_seed${SEED}" --version 1 --share-layers --gqa-groups 8 \
    --num-workers "$NW" --device cuda --seed "$SEED"
  cp -a model.pth "checkpoints/tied_mha_seed${SEED}.pth"
done

# --- Position ablations (seed 0 only; indicative) ---
SEED=0
export PYTHONHASHSEED=$SEED
echo "===== $(date) POS abs-only seed=$SEED ====="
rm -f model.pth
python train.py "exp_POS_abs_seed${SEED}" --version 1 --share-layers --no-relative-pos \
  --num-workers "$NW" --device cuda --seed "$SEED"
cp -a model.pth "checkpoints/pos_abs_seed${SEED}.pth"

echo "===== $(date) POS rel-only seed=$SEED ====="
rm -f model.pth
python train.py "exp_POS_rel_seed${SEED}" --version 1 --share-layers --no-absolute-pos \
  --num-workers "$NW" --device cuda --seed "$SEED"
cp -a model.pth "checkpoints/pos_rel_seed${SEED}.pth"

# "both" is the existing T seed0 checkpoint; copy for clarity
cp -a checkpoints/dense_A3_seed0.pth checkpoints/pos_both_seed0.pth

echo "===== $(date) revision controls complete ====="
