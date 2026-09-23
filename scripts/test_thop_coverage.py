#!/usr/bin/env python3
"""Test what thop actually counts in KWT-GQA.

Compares thop profile against analytical computation.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from thop import profile, clever_format
from models.kwt import KWT


def main():
    # Dense tied model
    model = KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
        dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
        channels=1, dropout=0.1, emb_dropout=0.05,
        use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
        share_layers=True,
    )
    model.eval()
    
    x = torch.randn(1, 98, 40)
    
    print("="*70)
    print("THOP PROFILING TEST")
    print("="*70)
    
    # Profile with thop
    macs, params = profile(model, inputs=(x,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    print(f"\nthop results:")
    print(f"  MACs:   {macs_str} ({macs:,})")
    print(f"  Params: {params_str} ({params:,})")
    
    # Compare with known values
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"\nActual runtime params: {actual_params:,}")
    print(f"thop params count:     {params:,}")
    
    if params != actual_params:
        print(f"⚠️  thop param count is WRONG (off by {abs(params - actual_params):,})")
    else:
        print(f"✓ thop param count matches")
    
    # Analytical MACs
    from scripts.compute_macs_analytical import compute_kwt_macs
    
    config = {
        "seq_len": 98,
        "mel_bins": 40,
        "dim": 48,
        "depth": 12,
        "heads": 8,
        "gqa_groups": 4,
        "mlp_dim": 96,
        "num_classes": 12,
        "conv_mid": 48,
    }
    
    analytical = compute_kwt_macs(config)
    
    print(f"\nAnalytical MACs:       {analytical['total_macs']:,} ({analytical['total_macs']/1e6:.3f}M)")
    print(f"thop MACs:             {macs:,} ({macs/1e6:.3f}M)")
    print(f"Ratio (analytical/thop): {analytical['total_macs']/macs:.2f}×")
    
    # Check attention-only
    attn_only = 2 * 12 * 98**2 * 8 * 6
    print(f"\nAttention matmuls only: {attn_only:,} ({attn_only/1e6:.3f}M)")
    
    if macs < attn_only:
        print(f"🔴 CRITICAL: thop MACs ({macs:,}) < attention-only ({attn_only:,})")
        print(f"   This proves thop is NOT counting attention matmuls correctly.")
    
    # Test with verbose to see what thop hooks
    print("\n" + "="*70)
    print("Verbose thop profile (first 50 lines):")
    print("="*70)
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        profile(model, inputs=(x,), verbose=True)
    
    lines = f.getvalue().split('\n')[:50]
    for line in lines:
        print(line)
    
    all_lines = f.getvalue().split('\n')
    if len(all_lines) > 50:
        print(f"... (truncated, total {len(all_lines)} lines)")


if __name__ == "__main__":
    main()