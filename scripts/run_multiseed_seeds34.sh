#!/usr/bin/env bash
# Extend Phase 3: seeds 3 and 4 for A3 (tied) + C2 (α=0.20 structured+INT8) only.
# Does not retrain A1. Restores canonical dense_A3.pth -> model.pth on EXIT.
set -euo pipefail
cd /home/alizare84/LightTransformerKWS2
export PYTHONUNBUFFERED=1
NW=1
ROOT=results/multiseed
mkdir -p "$ROOT" checkpoints
LOG="$ROOT/run_seeds34.log"

CANONICAL=checkpoints/dense_A3.pth
BACKUP=checkpoints/dense_A3_canonical_backup.pth
KNOWN_MD5=14c41d51d5eb5ac7ce96ee43ab177179

if [[ ! -f "$BACKUP" ]]; then
  if [[ -f "$CANONICAL" ]]; then
    cp -a "$CANONICAL" "$BACKUP"
  else
    echo "ERROR: missing $CANONICAL and $BACKUP" >&2
    exit 1
  fi
fi
BACKUP_MD5=$(md5sum "$BACKUP" | awk '{print $1}')
if [[ "$BACKUP_MD5" != "$KNOWN_MD5" ]]; then
  echo "ERROR: $BACKUP md5=$BACKUP_MD5 expected $KNOWN_MD5" >&2
  exit 1
fi

restore() {
  echo "===== $(date) restoring canonical dense_A3.pth -> model.pth ====="
  cp -a "$BACKUP" "$CANONICAL"
  cp -a "$BACKUP" model.pth
}
trap restore EXIT

exec >>"$LOG" 2>&1

echo "===== $(date) seeds 3–4 (A3+C2) start ====="

for SEED in 3 4; do
  export PYTHONHASHSEED=$SEED
  STAGE="$ROOT/stage_seeds34.txt"

  echo "===== $(date) seed=$SEED A3 tied from scratch =====" | tee "$STAGE"
  rm -f model.pth
  python train.py "exp_A3_seed${SEED}" --share-layers --version 1 \
    --num-workers "$NW" --device cuda --seed "$SEED"
  cp -a model.pth "checkpoints/dense_A3_seed${SEED}.pth"

  echo "===== $(date) seed=$SEED C2 structured α=0.20 =====" | tee "$STAGE"
  python train.py "exp_C2_seed${SEED}" --share-layers --version 1 \
    --num-workers "$NW" --device cuda --seed "$SEED" \
    --skip-train --enable-pruning --prune-mode structured --prune-amounts 0.20 \
    --finetune-steps 2000 --max-acc-drop 0.005
  if [[ -f model_quantized.pt ]]; then
    cp -a model_quantized.pt "checkpoints/struct_seed${SEED}_quant.pt"
  fi
  cp -a "checkpoints/dense_A3_seed${SEED}.pth" model.pth
done

python scripts/parse_multiseed.py
python scripts/ttest_a3_c2.py || true
echo "===== $(date) seeds 3–4 done =====" | tee "$ROOT/stage_seeds34.txt"
