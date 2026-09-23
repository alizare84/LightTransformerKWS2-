#!/usr/bin/env python3
"""Analytical MAC/FLOPs computation for KWT-GQA.

Computes operations for each component:
  - Conv front-end
  - Absolute position embedding (add, no MACs)
  - Relative position bias (indexing, no MACs)
  - Encoder blocks (attention + FFN)
  - Final classifier

Does NOT use thop; pure shape arithmetic.
"""
import sys
from pathlib import Path


def conv1d_macs(in_ch, out_ch, kernel, seq_len):
    """MACs for 1D convolution: out_ch × in_ch × kernel × seq_len."""
    return out_ch * in_ch * kernel * seq_len


def linear_macs(in_feat, out_feat, batch_seq=1):
    """MACs for linear layer: out_feat × in_feat × batch_seq."""
    return out_feat * in_feat * batch_seq


def gqa_attention_macs(T, d, H, G, dh):
    """MACs for one GQA attention layer.

    Projections use residual width d:
      Q: d → H·dh,  K/V: d → G·dh,  out: H·dh → d

    QK^T / AV: H × T² × dh each (group broadcast does not change the
    per-query-head matmul count under the implementation used here).
    Softmax is excluded.
    """
    q_proj = d * (H * dh) * T
    k_proj = d * (G * dh) * T
    v_proj = d * (G * dh) * T
    qk = H * T * T * dh
    av = H * T * T * dh
    out_proj = (H * dh) * d * T
    total_per_layer = q_proj + k_proj + v_proj + qk + av + out_proj
    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "qk_matmul": qk,
        "av_matmul": av,
        "out_proj": out_proj,
        "total": total_per_layer,
    }


def ffn_macs(d, d_ff, T):
    """MACs for feed-forward network.
    
    fc1: d → d_ff
    GELU: not counted
    fc2: d_ff → d
    """
    fc1 = d * d_ff * T
    fc2 = d_ff * d * T
    return {"fc1": fc1, "fc2": fc2, "total": fc1 + fc2}


def layernorm_ops(d, T):
    """LayerNorm operations (not MACs, but FLOPs).
    
    Mean: T additions
    Variance: T multiplications + T additions
    Normalize: T subtractions + T divisions + T multiplications
    Affine: T multiplications + T additions
    
    Total ≈ 6T operations per LayerNorm (not MAC, excluded from main count).
    """
    return 6 * d * T


def compute_kwt_macs(config):
    """Compute MACs for KWT-GQA model."""
    T = config["seq_len"]
    C = config["mel_bins"]
    d = config["dim"]
    L = config["depth"]
    H = config["heads"]
    G = config["gqa_groups"]
    # Head dimension is fixed by the architecture (dense default d/H_dense = 6).
    # After structured head/group pruning, H decreases but dh does not grow.
    dh = config.get("head_dim", 6)
    d_ff = config["mlp_dim"]
    num_classes = config["num_classes"]
    conv_mid = config.get("conv_mid", d)  # Middle channels in conv front-end
    
    breakdown = {}
    
    # Front-end: Conv1d(C, conv_mid, k=3) → BN → GELU → Conv1d(conv_mid, d, k=3)
    conv1_macs = conv1d_macs(C, conv_mid, kernel=3, seq_len=T)
    conv2_macs = conv1d_macs(conv_mid, d, kernel=3, seq_len=T)
    breakdown["conv1"] = conv1_macs
    breakdown["conv2"] = conv2_macs
    
    # Positional embedding: just addition, 0 MACs
    breakdown["pos_embed"] = 0
    
    # Relative position bias: indexing, 0 MACs
    breakdown["rel_pos_bias"] = 0
    
    # Encoder blocks (repeated L times)
    attn = gqa_attention_macs(T, d, H, G, dh)
    ffn = ffn_macs(d, d_ff, T)
    
    breakdown["attn_q_proj_per_layer"] = attn["q_proj"]
    breakdown["attn_k_proj_per_layer"] = attn["k_proj"]
    breakdown["attn_v_proj_per_layer"] = attn["v_proj"]
    breakdown["attn_qk_matmul_per_layer"] = attn["qk_matmul"]
    breakdown["attn_av_matmul_per_layer"] = attn["av_matmul"]
    breakdown["attn_out_proj_per_layer"] = attn["out_proj"]
    breakdown["attn_total_per_layer"] = attn["total"]
    
    breakdown["ffn_fc1_per_layer"] = ffn["fc1"]
    breakdown["ffn_fc2_per_layer"] = ffn["fc2"]
    breakdown["ffn_total_per_layer"] = ffn["total"]
    
    breakdown["encoder_per_layer"] = attn["total"] + ffn["total"]
    breakdown["encoder_all_layers"] = L * breakdown["encoder_per_layer"]
    
    # Classifier: Linear(d, num_classes)
    # After pooling, seq_len=1
    classifier_macs = linear_macs(d, num_classes, batch_seq=1)
    breakdown["classifier"] = classifier_macs
    
    # Total
    total_macs = (
        breakdown["conv1"] + breakdown["conv2"]
        + breakdown["encoder_all_layers"]
        + breakdown["classifier"]
    )
    breakdown["total_macs"] = total_macs
    breakdown["total_flops"] = 2 * total_macs
    
    return breakdown


def format_number(n):
    """Format large numbers as K/M."""
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    elif n >= 1e3:
        return f"{n/1e3:.3f}K"
    else:
        return str(n)


def main():
    # Dense T configuration
    config_dense = {
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
    
    # Pruned T-P configuration
    config_pruned = {
        "seq_len": 98,
        "mel_bins": 40,
        "dim": 48,
        "depth": 12,
        "heads": 6,
        "gqa_groups": 3,
        "mlp_dim": 77,
        "num_classes": 12,
        "conv_mid": 38,
    }
    
    print("="*80)
    print("ANALYTICAL MAC/FLOP COMPUTATION — KWT-GQA")
    print("="*80)
    
    for name, cfg in [("Dense (T)", config_dense), ("Pruned (T-P)", config_pruned)]:
        print(f"\n### {name}")
        dh = cfg.get("head_dim", 6)
        print(f"T={cfg['seq_len']}, d={cfg['dim']}, L={cfg['depth']}, "
              f"H={cfg['heads']}, G={cfg['gqa_groups']}, dh={dh}, "
              f"d_ff={cfg['mlp_dim']}, conv_mid={cfg['conv_mid']}")
        print()
        
        bd = compute_kwt_macs(cfg)
        
        print(f"{'Component':<35s} {'MACs':>12s} {'% of total':>10s}")
        print("-"*60)
        
        components = [
            ("Conv1 (C→conv_mid, k=3)", "conv1"),
            ("Conv2 (conv_mid→d, k=3)", "conv2"),
            ("  Attention Q proj (×{})".format(cfg['depth']), "attn_q_proj_per_layer", cfg['depth']),
            ("  Attention K proj (×{})".format(cfg['depth']), "attn_k_proj_per_layer", cfg['depth']),
            ("  Attention V proj (×{})".format(cfg['depth']), "attn_v_proj_per_layer", cfg['depth']),
            ("  Attention QK^T matmul (×{})".format(cfg['depth']), "attn_qk_matmul_per_layer", cfg['depth']),
            ("  Attention AV matmul (×{})".format(cfg['depth']), "attn_av_matmul_per_layer", cfg['depth']),
            ("  Attention out proj (×{})".format(cfg['depth']), "attn_out_proj_per_layer", cfg['depth']),
            ("  FFN fc1 (×{})".format(cfg['depth']), "ffn_fc1_per_layer", cfg['depth']),
            ("  FFN fc2 (×{})".format(cfg['depth']), "ffn_fc2_per_layer", cfg['depth']),
            ("Classifier (d→num_classes)", "classifier"),
        ]
        
        for item in components:
            if len(item) == 2:
                label, key = item
                mult = 1
            else:
                label, key, mult = item
            
            val = bd[key] * mult
            pct = 100 * val / bd["total_macs"] if bd["total_macs"] > 0 else 0
            print(f"{label:<35s} {format_number(val):>12s} {pct:>9.2f}%")
        
        print("-"*60)
        print(f"{'TOTAL MACs':<35s} {format_number(bd['total_macs']):>12s} {100.0:>9.2f}%")
        print(f"{'TOTAL FLOPs (=2×MACs)':<35s} {format_number(bd['total_flops']):>12s}")
        print()
        
        # Attention-only check
        attn_only = bd["attn_qk_matmul_per_layer"] + bd["attn_av_matmul_per_layer"]
        attn_only_total = attn_only * cfg["depth"]
        dh = cfg.get("head_dim", 6)
        print(f"Attention matmuls only (QK^T + AV) × {cfg['depth']}: {format_number(attn_only_total)}")
        print(f"  Formula check: 2×L×T²×H×dh = 2×{cfg['depth']}×{cfg['seq_len']}²×{cfg['heads']}×{dh} = {format_number(2*cfg['depth']*cfg['seq_len']**2*cfg['heads']*dh)}")
    
    print("\n" + "="*80)
    print("SUMMARY (analytical; thop undercounts matmul):")
    print("="*80)
    dense_bd = compute_kwt_macs(config_dense)
    pruned_bd = compute_kwt_macs(config_pruned)
    print(f"Dense MACs:  {format_number(dense_bd['total_macs'])}")
    print(f"Pruned MACs: {format_number(pruned_bd['total_macs'])}")


if __name__ == "__main__":
    main()