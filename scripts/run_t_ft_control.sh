#!/bin/bash
# T + 2K FT control: Run 2000-step fine-tuning on dense T checkpoints
# without pruning to isolate the effect of continued training.

set -e

SEEDS=(0 1 2 3 4)
STEPS=2000
LR=1e-4
DEVICE=cuda
VERSION=1
NW=1

echo "=========================================="
echo "T + 2K FT Control Experiment"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo "Steps: $STEPS"
echo "LR: $LR"
echo "Device: $DEVICE"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Seed $SEED <<<"
    
    CKPT="checkpoints/dense_A3_seed${SEED}.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "⚠️  Checkpoint not found: $CKPT"
        continue
    fi
    
    OUTPUT_DIR="results/t_ft_control_seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    
    # Copy checkpoint to model.pth for train.py
    cp "$CKPT" model.pth
    
    # Run training for 2K steps (no pruning, just continue training)
    # We'll use a simple Python script instead of train.py
    python - <<EOF
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from models.kwt import KWT
from datagen import Datagen
from utils.label_smoothing import LabelSmoothingLoss

device = torch.device("$DEVICE")
seed = $SEED
steps = $STEPS
lr = $LR

# Set seed
torch.manual_seed(seed)

# Load model
model = KWT(
    img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
    dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
    channels=1, dropout=0.1, emb_dropout=0.05,
    use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
    share_layers=True,
)
state = torch.load("model.pth", map_location=device, weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.to(device)

# Data
train_gen = Datagen(part="train", version=$VERSION, num_workers=$NW, batch_size=512)
val_gen = Datagen(part="val", version=$VERSION, num_workers=$NW, batch_size=512)
test_gen = Datagen(part="test", version=$VERSION, num_workers=$NW, batch_size=512)

# Evaluate before
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            pred = output.squeeze().argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

print("Before fine-tuning:")
val_before = evaluate(model, val_gen.generator())
test_before = evaluate(model, test_gen.generator())
print(f"  Val:  {val_before:.2f}%")
print(f"  Test: {test_before:.2f}%")

# Fine-tune
model.train()
loss_fn = LabelSmoothingLoss(num_classes=12, smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

print(f"Fine-tuning for {steps} steps...")
step = 0
for data, target in tqdm(train_gen.generator(), total=steps, desc="FT"):
    if step >= steps:
        break
    data, target = data.float().to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output.squeeze(), target)
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    step += 1

# Evaluate after
print("After fine-tuning:")
val_after = evaluate(model, val_gen.generator())
test_after = evaluate(model, test_gen.generator())
print(f"  Val:  {val_after:.2f}%")
print(f"  Test: {test_after:.2f}%")
print(f"  Δ Val:  {val_after - val_before:+.2f}%")
print(f"  Δ Test: {test_after - test_before:+.2f}%")

# Save
torch.save(model.state_dict(), "${OUTPUT_DIR}/model.pth")
with open("${OUTPUT_DIR}/results.txt", "w") as f:
    f.write(f"seed,val_before,test_before,val_after,test_after,val_delta,test_delta\n")
    f.write(f"{seed},{val_before:.2f},{test_before:.2f},{val_after:.2f},{test_after:.2f},{val_after-val_before:+.2f},{test_after-test_before:+.2f}\n")

print(f"✅ Saved to: ${OUTPUT_DIR}/")
EOF

done

echo ""
echo "=========================================="
echo "✅ T + 2K FT control complete"
echo "=========================================="
echo "Results in: results/t_ft_control_seed*/"
