#!/usr/bin/env bash
# C1 (5-seed structured α sweep) then C2 (S-Q from-scratch, 5 seeds).
set -euo pipefail
cd /home/alizare84/LightTransformerKWS2
export PYTHONUNBUFFERED=1
export PATH="/home/alizare84/anaconda3/bin:$PATH"
LOG=results/c1_c2_run.log
mkdir -p results
exec >>"$LOG" 2>&1

echo "===== $(date) C1/C2 launcher start ====="

# Wait until no other train.py is using the GPU.
while pgrep -f 'python train.py' >/dev/null 2>&1; do
  echo "$(date) waiting for an existing train.py to finish..."
  sleep 30
done

echo "===== $(date) C1 alpha sweep ====="
python scripts/run_c1_alpha_sweep.py

echo "===== $(date) C2 S-Q from scratch ====="
for SEED in 0 1 2 3 4; do
  export PYTHONHASHSEED=$SEED
  echo "===== $(date) S-Q seed=$SEED ====="
  python train.py "exp_SQ_seed${SEED}" \
    --share-layers --version 1 --init-pruned-alpha 0.20 \
    --num-workers 1 --device cuda --seed "$SEED"
done

echo "===== $(date) C1/C2 ALL DONE ====="
echo done > results/c1_c2_done.txt
