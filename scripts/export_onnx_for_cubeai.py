#!/usr/bin/env python3
"""Export KWT variants to ONNX for offline MCU tooling (STM32Cube.AI Analyze).

Produces ONNX graphs only. Does NOT run STM32Cube.AI — that tool is Windows/
ST-installer specific and is not available in this Linux environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# onnx / older torch may expect removed NumPy aliases
for _name, _alias in (("object", object), ("bool", bool), ("str", str), ("long", int)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

import torch

from models.kwt import KWT


def build_tied(share=True, gqa=4, depth=12):
    return KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
        dim=48, depth=depth, heads=8, mlp_dim=96, pool="mean",
        channels=1, dropout=0.0, emb_dropout=0.0,
        use_conv_encoder=True, gqa_groups=gqa, use_relative_pos=True,
        use_absolute_pos=True, share_layers=share,
    )


def load_state(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def export(model, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 98, 40)
    torch.onnx.export(
        model, x, str(out_path),
        input_names=["mfcc"], output_names=["logits"],
        opset_version=13,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/onnx")
    args = p.parse_args()
    out = Path(args.out_dir)

    specs = [
        ("T", "checkpoints/dense_A3_seed0.pth", dict(share=True, gqa=4, depth=12)),
        ("U", "checkpoints/untied_A1_seed0.pth", dict(share=False, gqa=4, depth=12)),
    ]
    # Prefer pruned float if present; else skip T-P
    for cand in [
        "checkpoints/struct_seed0_quant.pt",
        "results/exp_C2_seed0/best.pth",
    ]:
        if Path(cand).exists() and not cand.endswith("_quant.pt"):
            specs.append(("T-P", cand, dict(share=True, gqa=4, depth=12)))
            break

    for name, path, kw in specs:
        if not Path(path).exists():
            print(f"SKIP {name}: missing {path}")
            continue
        if path.endswith("_quant.pt"):
            print(f"SKIP {name}: quantized module not exported to ONNX in this pass")
            continue
        model = build_tied(**kw)
        try:
            load_state(model, path)
        except Exception as e:
            print(f"SKIP {name}: load failed ({e})")
            continue
        export(model, out / f"{name}.onnx")

    readme = out / "README.md"
    readme.write_text(
        "# ONNX exports for STM32Cube.AI Analyze\n\n"
        "These graphs are for **desktop Analyze mode** (Flash/RAM/MAC estimates).\n"
        "STM32Cube.AI / ST Edge AI Core was **not** installed in the revision\n"
        "environment; run Analyze locally and paste numbers into the manuscript.\n"
        "\n"
        "Suggested targets: Cortex-M4 and Cortex-M7.\n"
        "Required manuscript caveat:\n"
        "> These figures are static architectural/compiler-level estimates from\n"
        "> STM32Cube.AI Analyze mode targeting the stated Cortex-M cores, not\n"
        "> measured on-silicon execution.\n"
    )
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
