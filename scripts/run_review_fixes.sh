#!/usr/bin/env bash
# Reviewer fixes, ordered cheap -> expensive:
#   Phase 1: U2 (unstructured 0.15 + INT8) on the five seeded A3 parents
#   Phase 2: structured sweep {0.10,0.15,0.20,0.30} per seed (Table II -> 5 seeds)
#   Phase 3: NOISEX-92 eval per seed with the alpha=0.20 artifacts (Table VI redo)
set -euo pipefail
cd /home/alizare84/LightTransformerKWS2
export PYTHONUNBUFFERED=1
NW=1
ROOT=results/multiseed
mkdir -p "$ROOT" checkpoints
LOG="$ROOT/run_review_fixes.log"
STAGE="$ROOT/stage_review_fixes.txt"

restore() {
  echo "===== $(date) restoring canonical dense_A3.pth -> model.pth ====="
  if [[ -f checkpoints/dense_A3_canonical_backup.pth ]]; then
    cp -a checkpoints/dense_A3_canonical_backup.pth checkpoints/dense_A3.pth
    cp -a checkpoints/dense_A3_canonical_backup.pth model.pth
  fi
}
trap restore EXIT

exec >>"$LOG" 2>&1

echo "===== $(date) review-fixes run start ====="

# ---------- Phase 1: U2 on seeded parents ----------
for SEED in 0 1 2 3 4; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) seed=$SEED U2 unstructured 0.15 + INT8 =====" | tee "$STAGE"
  cp -a "checkpoints/dense_A3_seed${SEED}.pth" model.pth
  python train.py "exp_U2_seed${SEED}" --share-layers --version 1 \
    --num-workers "$NW" --device cuda --seed "$SEED" \
    --skip-train --enable-pruning --prune-mode unstructured --prune-amounts 0.15 \
    --finetune-steps 2000 --max-acc-drop 0.005
done

# ---------- Phase 2: structured sweep per seed ----------
for SEED in 0 1 2 3 4; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) seed=$SEED structured sweep 0.10/0.15/0.20/0.30 =====" | tee "$STAGE"
  cp -a "checkpoints/dense_A3_seed${SEED}.pth" model.pth
  python train.py "exp_sweep_seed${SEED}" --share-layers --version 1 \
    --num-workers "$NW" --device cuda --seed "$SEED" \
    --skip-train --enable-pruning --prune-mode structured \
    --prune-amounts 0.10,0.15,0.20,0.30 \
    --finetune-steps 2000 --max-acc-drop 0.005
done

# ---------- Phase 3: NOISEX-92 with alpha=0.20 artifacts ----------
for SEED in 0 1 2 3 4; do
  echo "===== $(date) seed=$SEED NOISEX-92 (alpha=0.20) =====" | tee "$STAGE"
  python eval_noisex92.py \
    --dense-ckpt "checkpoints/dense_A3_seed${SEED}.pth" \
    --quant-ckpt "checkpoints/struct_seed${SEED}_quant.pt" \
    --share-layers --seed "$SEED" \
    --outdir "results/noisex92_a020_seed${SEED}"
done

echo "===== $(date) ALL DONE =====" | tee "$STAGE"
