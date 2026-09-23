#!/usr/bin/env python3
"""Decompose the naive tied state_dict dump that reports ~389 KB."""
import io
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/struct_seed0_quant.pt"
    model = torch.load(path, map_location="cpu", weights_only=False)
    sd = model.state_dict()

    tmp = "/tmp/_c2_sd.pth"
    torch.save(sd, tmp)
    raw = os.path.getsize(tmp)
    print(f"torch.save(state_dict) = {raw} B = {raw/1024:.2f} KB  ({len(sd)} keys)")

    per_key = {}
    for k, v in sd.items():
        buf = io.BytesIO()
        torch.save(v, buf)
        per_key[k] = buf.tell()

    by_prefix = defaultdict(lambda: [0, 0])
    for k, b in per_key.items():
        prefix = k.split(".")[0] if "." in k else k
        if k.startswith("blocks."):
            prefix = k.split(".")[0] + "." + k.split(".")[1]  # blocks.0 ..
        by_prefix[prefix][0] += 1
        by_prefix[prefix][1] += b

    print("\nPer-prefix (sum of per-tensor torch.save):")
    tot = 0
    for p, (n, b) in sorted(by_prefix.items(), key=lambda x: -x[1][1]):
        print(f"  {p:28s}  {n:3d} tensors  {b/1024:8.2f} KB")
        tot += b
    print(f"  {'SUM of per-tensor saves':28s}           {tot/1024:8.2f} KB")
    print(f"  pickle container overhead vs one dump: {(raw - tot)/1024:+.2f} KB  "
          f"(negative = sharing / zip packing)")

    # data_ptr grouping
    print("\nShared storage (same data_ptr):")
    seen = defaultdict(list)
    for k, v in sd.items():
        if torch.is_tensor(v):
            seen[v.data_ptr()].append(k)
        elif hasattr(v, "data_ptr"):
            seen[v.data_ptr()].append(k)
    n_dup_groups = 0
    for ptr, keys in seen.items():
        if len(keys) > 1:
            n_dup_groups += 1
            print(f"  {len(keys):2d} copies of {keys[0]}  ({per_key[keys[0]]} B each)")
    if not n_dup_groups:
        print("  (no shared data_ptr — quantized packed params were re-materialized)")

    # Count block copies
    block_keys = [k for k in sd if k.startswith("blocks.")]
    idxs = sorted({k.split(".")[1] for k in block_keys})
    print(f"\nstate_dict visits {len(idxs)} block copies: {idxs}")
    print(f"keys starting with blocks: {len(block_keys)}")


if __name__ == "__main__":
    main()
