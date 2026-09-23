#!/usr/bin/env python3
"""Accurate parameter counting with deduplication for tied models.

Counts:
  - Unique learned parameters (trainable)
  - Non-learned buffers (BN running stats, etc.)
  - For quantized models: packed INT8 + scales/zero-points + fp32 biases

Outputs:
  - results/parameter_inventory.csv
"""
import sys
import os
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.quantized.dynamic as nnqd

from models.kwt import KWT


def count_unique_params(model):
    """Count unique parameter tensors (handles weight tying via id())."""
    seen = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    return trainable, total


def count_buffers(model):
    """Count non-parameter buffers (e.g., BN running_mean/var)."""
    seen = set()
    total = 0
    for buf in model.buffers():
        if id(buf) not in seen:
            seen.add(id(buf))
            total += buf.numel()
    return total


def count_quantized(model):
    """Count parameters in a quantized model (INT8 Linear + metadata)."""
    seen_params = set()
    fp32_params = 0
    int8_weights = 0
    int8_biases = 0
    metadata_elements = 0
    
    # Non-quantized parameters
    for p in model.parameters():
        if id(p) not in seen_params:
            seen_params.add(id(p))
            fp32_params += p.numel()
    
    # Quantized Linear layers
    for module in model.modules():
        if isinstance(module, nnqd.Linear):
            w = module.weight()
            int8_weights += w.numel()
            
            # Per-channel scales and zero-points
            try:
                s = w.q_per_channel_scales()
                z = w.q_per_channel_zero_points()
                metadata_elements += s.numel() + z.numel()
            except RuntimeError:
                # Per-tensor quantization
                metadata_elements += 2
            
            b = module.bias()
            if b is not None:
                int8_biases += b.numel()
    
    # Buffers
    buffers = count_buffers(model)
    
    return {
        "fp32_params": fp32_params,
        "int8_weights": int8_weights,
        "int8_biases": int8_biases,
        "metadata_elements": metadata_elements,
        "buffers": buffers,
        "total_elements": fp32_params + int8_weights + int8_biases + metadata_elements + buffers,
    }


def load_model(path, is_quantized=False, is_tied=True):
    """Load a KWT checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    if not is_quantized:
        # Float model
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        
        model = KWT(
            img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
            dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
            channels=1, dropout=0.1, emb_dropout=0.05,
            use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
            share_layers=is_tied,
        )
        model.load_state_dict(state, strict=False)  # strict=False because tied models have duplicate keys
        return model.eval()
    else:
        # Already quantized
        return ckpt.eval()


def check_weight_sharing(model, label):
    """Verify that tied models actually share weight tensors."""
    blocks = list(model.modules())
    encoder_blocks = [m for m in blocks if m.__class__.__name__ == "EncoderBlock"]
    
    if len(encoder_blocks) == 0:
        return f"{label}: No EncoderBlock found"
    
    if len(encoder_blocks) == 1:
        # Single block in list means either depth=1 or tied
        block_refs = [b for b in model.blocks]
        if len(block_refs) == 12 and all(b is block_refs[0] for b in block_refs):
            return f"{label}: ✓ 12 slots → 1 shared EncoderBlock (id={id(block_refs[0])})"
        elif len(block_refs) == 1:
            return f"{label}: 1 block only (depth=1?)"
        else:
            return f"{label}: Unexpected: 1 unique block but {len(block_refs)} slots"
    else:
        # Multiple unique blocks
        unique_ids = set(id(b) for b in encoder_blocks)
        return f"{label}: {len(unique_ids)} unique EncoderBlocks (untied)"


def main():
    root = Path(__file__).parent.parent
    ckpt_dir = root / "checkpoints"
    results_dir = root / "results"
    
    # Models to inventory
    models_to_check = [
        ("T", ckpt_dir / "dense_A3_seed0.pth", False, True),
        ("U", ckpt_dir / "untied_A1_seed0.pth", False, False),
        ("T-P-Q", ckpt_dir / "struct_seed0_quant.pt", True, True),
    ]
    
    rows = []
    
    print("="*70)
    print("PARAMETER INVENTORY")
    print("="*70)
    
    for model_id, path, is_quant, is_tied in models_to_check:
        if not path.exists():
            print(f"\n{model_id}: {path.name} NOT FOUND")
            continue
        
        print(f"\n{model_id}: {path.name}")
        
        # Also report state_dict size (for tied models, torch.save duplicates all blocks)
        ckpt_raw = torch.load(path, map_location="cpu", weights_only=False)
        if not is_quant:
            state_raw = ckpt_raw.get("model_state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw
            statedict_params = sum(v.numel() for v in state_raw.values())
            print(f"  State_dict size:   {statedict_params:>8,} (incl. duplicate blocks if tied)")
        
        model = load_model(path, is_quant, is_tied)
        
        if not is_quant:
            trainable, total = count_unique_params(model)
            buffers = count_buffers(model)
            print(f"  Trainable params:  {trainable:>8,}")
            print(f"  Non-trainable:     {total - trainable:>8,}")
            print(f"  Buffers:           {buffers:>8,}")
            print(f"  Total elements:    {total + buffers:>8,}")
            print(f"  → UNIQUE runtime:  {trainable:>8,}")
            
            rows.append({
                "model": model_id,
                "checkpoint": path.name,
                "quantized": "no",
                "trainable_params": trainable,
                "non_trainable_params": total - trainable,
                "fp32_params": total,
                "int8_weights": 0,
                "int8_biases": 0,
                "metadata_elements": 0,
                "buffers": buffers,
                "total_elements": total + buffers,
                "statedict_size": statedict_params,
                "is_tied": is_tied,
            })
        else:
            counts = count_quantized(model)
            print(f"  FP32 params:       {counts['fp32_params']:>8,}")
            print(f"  INT8 weights:      {counts['int8_weights']:>8,}")
            print(f"  INT8 biases:       {counts['int8_biases']:>8,}")
            print(f"  Metadata elements: {counts['metadata_elements']:>8,}")
            print(f"  Buffers:           {counts['buffers']:>8,}")
            print(f"  Total elements:    {counts['total_elements']:>8,}")
            
            rows.append({
                "model": model_id,
                "checkpoint": path.name,
                "quantized": "yes",
                "trainable_params": "—",
                "non_trainable_params": "—",
                "fp32_params": counts["fp32_params"],
                "int8_weights": counts["int8_weights"],
                "int8_biases": counts["int8_biases"],
                "metadata_elements": counts["metadata_elements"],
                "buffers": counts["buffers"],
                "total_elements": counts["total_elements"],
                "statedict_size": "—",
                "is_tied": is_tied,
            })
        
        # Check sharing
        sharing_status = check_weight_sharing(model, model_id)
        print(f"  {sharing_status}")
    
    # Write CSV
    out_path = results_dir / "parameter_inventory.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["model", "checkpoint", "quantized", "trainable_params", "non_trainable_params",
                     "fp32_params", "int8_weights", "int8_biases", "metadata_elements",
                     "buffers", "total_elements", "statedict_size", "is_tied"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Wrote {out_path}")


if __name__ == "__main__":
    main()