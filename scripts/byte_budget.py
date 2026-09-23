#!/usr/bin/env python3
"""Byte budget of a saved (possibly quantized) KWT checkpoint.

Decomposes the deduplicated on-disk payload (the number reported as
"file size" in the paper) into: INT8 Linear weights, their per-channel
scales/zero-points, fp32 biases, conv front-end, positional embedding,
relative bias, LayerNorm/BatchNorm, and classifier.

Usage:
  python scripts/byte_budget.py [checkpoint.pt]
Defaults to checkpoints/struct_seed0_quant.pt (a C2 artifact).
"""
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.quantized.dynamic as nnqd


def classify(name: str) -> str:
    if name.startswith("front"):
        return "Conv front-end"
    if name == "pos" or name.startswith("pos"):
        return "Abs. pos. embedding"
    if "rel_bias" in name or "rel_pos" in name:
        return "Relative bias"
    if name == "norm" or name.endswith(".norm1") or name.endswith(".norm2") or ".norm" in name:
        return "LayerNorm"
    if name.startswith("fc"):
        return "Classifier"
    if name.startswith("blocks"):
        return "Encoder block (tied, x1)"
    return f"Other ({name})"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/struct_seed0_quant.pt"
    model = torch.load(path, map_location="cpu", weights_only=False)

    groups: "OrderedDict[tuple, list]" = OrderedDict()

    def add(cat: str, sub: str, n: int, b: int) -> None:
        e = groups.setdefault((cat, sub), [0, 0])
        e[0] += n
        e[1] += b

    seen_params = set()
    for mod_name, module in model.named_modules():  # named_modules dedups tied modules
        if isinstance(module, nnqd.Linear):
            cat = classify(mod_name)
            w = module.weight()
            add(cat, "INT8 weight", w.numel(), w.numel() * w.element_size())
            try:
                s = w.q_per_channel_scales()
                z = w.q_per_channel_zero_points()
                add(cat, "scales+zero-points", 0,
                    s.numel() * s.element_size() + z.numel() * z.element_size())
            except RuntimeError:
                add(cat, "scale+zero-point", 0, 16)
            b = module.bias()
            if b is not None:
                add(cat, "fp32 bias", b.numel(), b.numel() * b.element_size())
        else:
            for pn, p in module.named_parameters(recurse=False):
                if id(p) in seen_params:
                    continue
                seen_params.add(id(p))
                full = f"{mod_name}.{pn}" if mod_name else pn
                add(classify(full), "fp32 param", p.numel(), p.numel() * p.element_size())
            for bn, buf in module.named_buffers(recurse=False):
                if id(buf) in seen_params:
                    continue
                seen_params.add(id(buf))
                full = f"{mod_name}.{bn}" if mod_name else bn
                add(classify(full), "buffer", buf.numel(), buf.numel() * buf.element_size())

    print(f"Checkpoint: {path}\n")
    print(f"{'Component':30s} {'Kind':22s} {'Count':>9s} {'Bytes':>9s} {'KB':>8s}")
    print("-" * 84)
    total_n, total_b = 0, 0
    cat_totals: "OrderedDict[str, list]" = OrderedDict()
    for (cat, sub), (n, b) in groups.items():
        print(f"{cat:30s} {sub:22s} {n:9d} {b:9d} {b/1024:8.2f}")
        total_n += n
        total_b += b
        e = cat_totals.setdefault(cat, [0, 0])
        e[0] += n
        e[1] += b
    print("-" * 84)
    print(f"{'TOTAL (dedup payload)':53s} {total_n:9d} {total_b:9d} {total_b/1024:8.2f}")

    print("\nPer-component totals:")
    fp32_b = 0
    int8_b = 0
    for cat, (n, b) in sorted(cat_totals.items(), key=lambda kv: -kv[1][1]):
        print(f"  {cat:30s} {n:9d} params {b/1024:8.2f} KB  ({100*b/total_b:5.1f}%)")
    for (cat, sub), (n, b) in groups.items():
        if "INT8" in sub:
            int8_b += b
        else:
            fp32_b += b
    print(f"\n  INT8 payload : {int8_b/1024:8.2f} KB ({100*int8_b/total_b:5.1f}%)")
    print(f"  float payload: {fp32_b/1024:8.2f} KB ({100*fp32_b/total_b:5.1f}%)")

    import os
    raw = os.path.getsize(path)
    print(f"\n  Raw file on disk: {raw/1024:8.2f} KB "
          f"(pickle + tied-module duplication overhead: {(raw-total_b)/1024:.2f} KB)")


if __name__ == "__main__":
    main()
