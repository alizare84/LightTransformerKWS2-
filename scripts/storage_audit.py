#!/usr/bin/env python3
"""Comprehensive storage audit for all KWT variants.

Computes:
  1. Unique tensor payload bytes (deduplicated)
  2. Quantization metadata bytes (scales, zero-points)
  3. Actual serialized file size on disk
  4. Per-dtype breakdown
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.quantized.dynamic as nnqd
from models.kwt import KWT


def analyze_model_memory(model):
    """Analyze a model (Module or state_dict)."""
    if isinstance(model, dict):
        return analyze_state_dict(model)
    else:
        return analyze_module(model)


def analyze_state_dict(state_dict):
    """Analyze state_dict payload."""
    seen = set()
    total_bytes = 0
    dtype_bytes = {}
    
    for k, v in state_dict.items():
        if id(v) in seen:
            continue
        seen.add(id(v))
        
        if isinstance(v, torch.Tensor):
            b = v.numel() * v.element_size()
            total_bytes += b
            dtype = str(v.dtype)
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
    
    return {
        "unique_params": len(seen),
        "total_bytes": total_bytes,
        "dtype_breakdown": dtype_bytes,
        "quantized": False,
    }


def analyze_module(model):
    """Analyze nn.Module with potential quantization."""
    seen_params = set()
    total_bytes = 0
    dtype_bytes = {}
    int8_weights = 0
    int8_bytes = 0
    metadata_bytes = 0
    is_quantized = False
    
    # Check for quantized layers
    for module in model.modules():
        if isinstance(module, nnqd.Linear):
            is_quantized = True
            w = module.weight()
            b = w.numel() * w.element_size()  # INT8
            int8_weights += w.numel()
            int8_bytes += b
            total_bytes += b
            
            # Metadata
            try:
                s = w.q_per_channel_scales()
                z = w.q_per_channel_zero_points()
                meta = s.numel() * s.element_size() + z.numel() * z.element_size()
            except RuntimeError:
                meta = 16  # Scalar scale + zero-point
            
            metadata_bytes += meta
            total_bytes += meta
            
            # Bias
            bias = module.bias()
            if bias is not None:
                b_bytes = bias.numel() * bias.element_size()
                total_bytes += b_bytes
                dtype_bytes[str(bias.dtype)] = dtype_bytes.get(str(bias.dtype), 0) + b_bytes
    
    # Regular parameters and buffers
    for p in model.parameters():
        if id(p) in seen_params:
            continue
        seen_params.add(id(p))
        
        b = p.numel() * p.element_size()
        total_bytes += b
        dtype = str(p.dtype)
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
    
    for buf in model.buffers():
        if id(buf) in seen_params:
            continue
        seen_params.add(id(buf))
        
        b = buf.numel() * buf.element_size()
        total_bytes += b
        dtype = str(buf.dtype)
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
    
    if is_quantized:
        dtype_bytes["torch.qint8"] = int8_bytes
        dtype_bytes["quantization_metadata"] = metadata_bytes
    
    return {
        "unique_params": len(seen_params) + (int8_weights if is_quantized else 0),
        "total_bytes": total_bytes,
        "dtype_breakdown": dtype_bytes,
        "quantized": is_quantized,
        "int8_weights": int8_weights,
        "metadata_bytes": metadata_bytes,
    }


def format_bytes(b):
    """Format bytes as KB/KiB."""
    kb = b / 1000
    kib = b / 1024
    return f"{b:,} B ({kb:.2f} KB / {kib:.2f} KiB)"


def main():
    checkpoints = [
        ("T (Tied dense)", "checkpoints/dense_A3_seed0.pth", "state_dict"),
        ("T-Q (Tied + INT8)", "checkpoints/struct_seed0_quant.pt", "model"),
        ("U (Untied)", "checkpoints/untied_A1_seed0.pth", "state_dict"),
    ]
    
    results = []
    
    for name, path, load_type in checkpoints:
        full_path = Path("/home/alizare84/LightTransformerKWS2") / path
        if not full_path.exists():
            print(f"⚠️  {name}: {path} NOT FOUND")
            continue
        
        if load_type == "state_dict":
            obj = torch.load(full_path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "model_state_dict" in obj:
                obj = obj["model_state_dict"]
        else:
            obj = torch.load(full_path, map_location="cpu", weights_only=False)
        
        analysis = analyze_model_memory(obj)
        file_size = os.path.getsize(full_path)
        
        results.append({
            "name": name,
            "path": path,
            "analysis": analysis,
            "file_size": file_size,
        })
    
    print("="*80)
    print("STORAGE AUDIT — KWT Variants")
    print("="*80)
    
    for r in results:
        a = r["analysis"]
        print(f"\n### {r['name']}")
        print(f"Path: {r['path']}")
        print(f"  File size on disk:     {format_bytes(r['file_size'])}")
        print(f"  Unique payload bytes:  {format_bytes(a['total_bytes'])}")
        print(f"  Overhead (pickle etc): {format_bytes(r['file_size'] - a['total_bytes'])}")
        print(f"  Quantized: {a['quantized']}")
        
        if a['quantized']:
            print(f"  INT8 weights: {a['int8_weights']:,} elements = {format_bytes(a['int8_weights'])}")
            print(f"  Metadata (scales/zero-points): {format_bytes(a['metadata_bytes'])}")
        
        print(f"  Dtype breakdown:")
        for dtype, b in sorted(a['dtype_breakdown'].items(), key=lambda x: -x[1]):
            pct = 100 * b / a['total_bytes'] if a['total_bytes'] > 0 else 0
            print(f"    {dtype:30s} {format_bytes(b):>30s} ({pct:5.1f}%)")
    
    # Generate CSV
    csv_path = Path("/home/alizare84/LightTransformerKWS2/results/storage_breakdown.csv")
    with open(csv_path, "w") as f:
        f.write("model,file_bytes,payload_bytes,overhead_bytes,quantized,int8_elements,metadata_bytes\n")
        for r in results:
            a = r["analysis"]
            f.write(f"{r['name']},{r['file_size']},{a['total_bytes']},{r['file_size']-a['total_bytes']},"
                    f"{a['quantized']},{a.get('int8_weights',0)},{a.get('metadata_bytes',0)}\n")
    
    print(f"\n✓ CSV saved to {csv_path}")


if __name__ == "__main__":
    main()