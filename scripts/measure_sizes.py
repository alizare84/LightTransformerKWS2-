#!/usr/bin/env python3
"""Re-measure dedup file sizes with per-channel quant metadata included.

Measures: dense A3 (float), C0 (dense + dynamic INT8), C2 (struct 0.20 + INT8).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.quantization
from torch.quantization import per_channel_dynamic_qconfig, quantize_dynamic

from train import get_model_size_kb_dedup
from models.kwt import KWT


def load_dense(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
        dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
        channels=1, dropout=0.1, emb_dropout=0.05,
        use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
        share_layers=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    dense = load_dense("checkpoints/dense_A3_seed0.pth")
    print(f"A3 dense (seed0):       {get_model_size_kb_dedup(dense):8.2f} KB")

    c0 = quantize_dynamic(dense, {torch.nn.Linear: per_channel_dynamic_qconfig},
                          dtype=torch.qint8)
    print(f"C0 dense+INT8 (seed0):  {get_model_size_kb_dedup(c0):8.2f} KB")

    for seed in range(5):
        c2 = torch.load(f"checkpoints/struct_seed{seed}_quant.pt",
                        map_location="cpu", weights_only=False)
        print(f"C2 struct+INT8 seed{seed}:  {get_model_size_kb_dedup(c2):8.2f} KB")


if __name__ == "__main__":
    main()
