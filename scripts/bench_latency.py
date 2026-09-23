#!/usr/bin/env python3
"""Proper host-CPU latency benchmark: warm-up, median/IQR, fixed threads.

Compares dense A3 (float, seed0) against C2 (struct 0.20 + dynamic INT8, seed0).
Protocol: torch.set_num_threads(1), 50 warm-up forwards, 1000 timed forwards,
batch size 1, report median and IQR.
"""
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

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
    return model


def bench(model, warmup=50, runs=1000):
    model.eval()
    x = torch.randn(1, 98, 40)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times.sort()
    n = len(times)
    med = statistics.median(times)
    q1 = times[n // 4]
    q3 = times[(3 * n) // 4]
    return med, q1, q3


def main():
    torch.set_num_threads(1)
    print(f"CPU: {platform.processor() or platform.machine()}, "
          f"PyTorch {torch.__version__}, threads=1, warmup=50, runs=1000, batch=1")

    dense = load_dense("checkpoints/dense_A3_seed0.pth")
    m_d, q1_d, q3_d = bench(dense)
    print(f"A3 dense float : median {m_d:7.3f} ms  IQR [{q1_d:.3f}, {q3_d:.3f}]")

    c2 = torch.load("checkpoints/struct_seed0_quant.pt", map_location="cpu",
                    weights_only=False)
    m_q, q1_q, q3_q = bench(c2)
    print(f"C2 struct+INT8 : median {m_q:7.3f} ms  IQR [{q1_q:.3f}, {q3_q:.3f}]")

    print(f"\nSpeedup (dense/quant, median): {m_d/m_q:.3f}x")


if __name__ == "__main__":
    main()
