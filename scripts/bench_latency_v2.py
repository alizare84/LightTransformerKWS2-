#!/usr/bin/env python3
"""CPU latency benchmark for T, T-Q, T-P-Q variants.

Benchmarks inference latency with proper warm-up and statistics.
Writes results to CSV for manuscript inclusion.
"""
import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.kwt import KWT


def load_dense(path, share_layers=True):
    """Load dense FP32 checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
        dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
        channels=1, dropout=0.1, emb_dropout=0.05,
        use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
        share_layers=share_layers,
    )
    model.load_state_dict(state)
    return model


def bench(model, warmup=100, runs=1000):
    """Run benchmark: return median, Q1, Q3, P95 in ms."""
    model.eval()
    x = torch.randn(1, 98, 40)
    
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    
    times.sort()
    n = len(times)
    return (
        statistics.median(times),
        times[n // 4],
        times[(3 * n) // 4],
        times[int(0.95 * n)]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/latency.csv")
    args = parser.parse_args()
    
    torch.set_num_threads(args.threads)
    
    print("=" * 70)
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"PyTorch: {torch.__version__}, threads={args.threads}")
    print(f"Warmup: {args.warmup}, runs: {args.runs}, batch: 1, seed: {args.seed}")
    print("=" * 70)
    
    results = []
    
    # T: Tied dense FP32
    path_t = f"checkpoints/dense_A3_seed{args.seed}.pth"
    if os.path.exists(path_t):
        print(f"\nT (Tied dense FP32): {path_t}")
        m = load_dense(path_t, share_layers=True)
        med, q1, q3, p95 = bench(m, args.warmup, args.runs)
        print(f"  Median: {med:.3f} ms, IQR: [{q1:.3f}, {q3:.3f}], P95: {p95:.3f}")
        results.append({"variant": "T", "description": "Tied dense FP32",
                       "median_ms": med, "q1_ms": q1, "q3_ms": q3, "p95_ms": p95})
    
    # T-Q: Tied dense + dynamic INT8
    if os.path.exists(path_t):
        print(f"\nT-Q (Tied + INT8): dynamic quantization")
        m_fp32 = load_dense(path_t, share_layers=True)
        m_q = torch.quantization.quantize_dynamic(m_fp32, {torch.nn.Linear}, dtype=torch.qint8)
        med, q1, q3, p95 = bench(m_q, args.warmup, args.runs)
        print(f"  Median: {med:.3f} ms, IQR: [{q1:.3f}, {q3:.3f}], P95: {p95:.3f}")
        results.append({"variant": "T-Q", "description": "Tied + INT8",
                       "median_ms": med, "q1_ms": q1, "q3_ms": q3, "p95_ms": p95})
    
    # T-P-Q: Structured pruned + INT8
    path_tpq = f"checkpoints/struct_seed{args.seed}_quant.pt"
    if os.path.exists(path_tpq):
        print(f"\nT-P-Q (Pruned + INT8): {path_tpq}")
        m_tpq = torch.load(path_tpq, map_location="cpu", weights_only=False)
        med, q1, q3, p95 = bench(m_tpq, args.warmup, args.runs)
        print(f"  Median: {med:.3f} ms, IQR: [{q1:.3f}, {q3:.3f}], P95: {p95:.3f}")
        results.append({"variant": "T-P-Q", "description": "Pruned + INT8",
                       "median_ms": med, "q1_ms": q1, "q3_ms": q3, "p95_ms": p95})
    
    # Write CSV
    if results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "description", "median_ms", "q1_ms", "q3_ms", "p95_ms"])
            w.writeheader()
            w.writerows(results)
        print("\n" + "=" * 70)
        print(f"✅ {args.output}")
        
        # Speedup
        t_med = next((r["median_ms"] for r in results if r["variant"] == "T"), None)
        tpq_med = next((r["median_ms"] for r in results if r["variant"] == "T-P-Q"), None)
        if t_med and tpq_med:
            print(f"Speedup (T / T-P-Q): {t_med / tpq_med:.3f}x")
        print("=" * 70)


if __name__ == "__main__":
    main()
