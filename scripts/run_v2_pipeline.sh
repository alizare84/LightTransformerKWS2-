#!/usr/bin/env bash
# Fair GSC v2 rerun of the v1 recipe. Restores v1 model.pth on exit.
set -euo pipefail
cd /home/alizare84/LightTransformerKWS2
export PYTHONUNBUFFERED=1
NW=1
LOG=/tmp/gsc_v2_pipeline.log

exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) GSC v2 pipeline start ====="
mkdir -p checkpoints results

cp -a checkpoints/dense_A3.pth checkpoints/dense_A3_v1.pth
cp -a model.pth checkpoints/dense_A3.pth

restore_v1() {
  echo "===== $(date) restoring v1 model.pth ====="
  if [[ -f checkpoints/dense_A3.pth ]]; then
    cp -a checkpoints/dense_A3.pth model.pth
  fi
}
trap restore_v1 EXIT

run() {
  echo
  echo "===== $(date) $* ====="
  python train.py "$@"
}

# --- A3: tied dense, from scratch (do not load v1 weights) ---
rm -f model.pth
run exp_A3_v2 --share-layers --version 2 --num-workers "$NW" --device cuda
cp -a model.pth checkpoints/dense_A3_v2.pth
echo "Saved checkpoints/dense_A3_v2.pth"

# --- C0: INT8 only ---
run exp_C0_v2 --share-layers --version 2 --num-workers "$NW" --device cuda --skip-train
cp -a checkpoints/dense_A3_v2.pth model.pth

# --- U2: unstructured 0.15 + INT8 ---
run exp_U2_v2 --share-layers --version 2 --num-workers "$NW" --device cuda --skip-train \
  --enable-pruning --prune-mode unstructured --prune-amounts 0.15 \
  --finetune-steps 2000 --max-acc-drop 0.005
cp -a checkpoints/dense_A3_v2.pth model.pth

# --- C2: structured sweep + INT8 ---
run exp_struct_sweep_v2 --share-layers --version 2 --num-workers "$NW" --device cuda --skip-train \
  --enable-pruning --prune-mode structured --prune-amounts 0.1,0.15,0.2,0.3 \
  --finetune-steps 2000 --max-acc-drop 0.005
if [[ -f model_quantized.pt ]]; then
  cp -a model_quantized.pt checkpoints/struct_v2_quant.pt
fi
cp -a checkpoints/dense_A3_v2.pth model.pth

# --- A1: untied from scratch (strict=False would leak tied weights if model.pth exists) ---
rm -f model.pth
run exp_A1_v2 --version 2 --num-workers "$NW" --device cuda
cp -a model.pth checkpoints/untied_A1_v2.pth
echo "Saved checkpoints/untied_A1_v2.pth"

echo "===== $(date) GSC v2 pipeline done ====="
