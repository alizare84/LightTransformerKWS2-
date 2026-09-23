#!/usr/bin/env python3
"""C1: 5-seed structured-α sweep on dense T (A3) checkpoints.

For each seed in {0..4} and α in {0.10, 0.15, 0.20, 0.30}:
  clone dense T → structured prune (conv / FFN / GQA groups) →
  fine-tune 2000 AdamW steps at 1e-4 → record val before/after, test acc,
  params, FLOPs.

Does not apply the 0.5 pp selection rule during the run; selection is
done afterwards from the CSV (most compressive α with mean val drop ≤ 0.5 pp).
"""
from __future__ import annotations

import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from models.kwt import KWT  # noqa: E402
from models.structured_prune import apply_structured_pruning  # noqa: E402
from train import (  # noqa: E402
    Dataset,
    calculate_complexity,
    finetune_pruned_model,
    train_worker_init,
)
from utils.label_smoothing import LabelSmoothingLoss  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.10, 0.15, 0.20, 0.30)
FINETUNE_STEPS = 2000
FINETUNE_LR = 1e-4
CKPT_TMPL = os.path.join(ROOT, "checkpoints", "dense_A3_seed{seed}.pth")
OUT_DIR = os.path.join(ROOT, "results", "c1_alpha_sweep")
CSV_PATH = os.path.join(OUT_DIR, "sweep.csv")


def build_dense():
    return KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40,
        num_classes=12, dim=48, depth=12, heads=8, mlp_dim=96,
        pool="mean", channels=1, dropout=0.1, emb_dropout=0.05,
        gqa_groups=4, use_relative_pos=True, use_conv_encoder=True,
        share_layers=True,
    )


def val_accuracy(model, loader, device):
    model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.float().to(device), target.to(device)
            pred = model(data).argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            total += target.shape[-1]
    return correct / total


def gpu_test_accuracy(model, loader, device):
    model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.float().to(device), target.to(device)
            pred = model(data).argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            total += int(target.numel())
    return correct / total


def already_done(rows, seed, alpha):
    for r in rows:
        if int(r["seed"]) == seed and abs(float(r["alpha"]) - alpha) < 1e-9:
            return True
    return False


def load_rows():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def append_row(row, fieldnames):
    os.makedirs(OUT_DIR, exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    fieldnames = [
        "seed", "alpha", "val_before", "val_after", "val_drop_pp",
        "test_acc", "params", "flops", "conv_mid", "ffn", "heads_groups",
        "seconds",
    ]
    rows = load_rows()

    train_set = Dataset(part="train", steps=23000, batch_size=512, version=1, preprocess="mfcc")
    val_set = Dataset(part="val", batch_size=512, version=1, preprocess="mfcc")
    test_set = Dataset(part="test", batch_size=64, version=1, preprocess="mfcc")
    train_loader = torch.utils.data.DataLoader(
        train_set, num_workers=1, batch_size=None, persistent_workers=True,
        worker_init_fn=train_worker_init,
    )
    val_loader = torch.utils.data.DataLoader(val_set, num_workers=0, batch_size=None)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=0, batch_size=None)
    next(iter(train_loader))

    loss_fn = LabelSmoothingLoss(classes=12, smoothing=0.1).to(device)

    for seed in SEEDS:
        ckpt = CKPT_TMPL.format(seed=seed)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dense = build_dense()
        state = torch.load(ckpt, map_location="cpu")
        dense.load_state_dict(state, strict=True)
        dense.to(device)
        val_before = val_accuracy(dense, val_loader, device)
        print(f"\n===== seed={seed} dense-T val={val_before*100:.2f}% =====")

        for alpha in ALPHAS:
            if already_done(rows, seed, alpha):
                print(f"  α={alpha:.2f} already in {CSV_PATH}, skip")
                continue
            t0 = time.time()
            pruned, info = apply_structured_pruning(dense, alpha, prune_heads=True)
            params = int(info["params_after"])
            pruned = finetune_pruned_model(
                pruned, train_loader, loss_fn, device,
                lr=FINETUNE_LR, steps=FINETUNE_STEPS, preserve_zeros=False,
            )
            val_after = val_accuracy(pruned, val_loader, device)
            test_acc = gpu_test_accuracy(pruned, test_loader, device)
            _ff, _mf, _pf, flops_n, _macs_n, _pn = calculate_complexity(pruned)
            drop_pp = (val_before - val_after) * 100.0
            heads_groups = info["attn_shape"]
            row = {
                "seed": seed,
                "alpha": f"{alpha:.2f}",
                "val_before": f"{val_before:.6f}",
                "val_after": f"{val_after:.6f}",
                "val_drop_pp": f"{drop_pp:.4f}",
                "test_acc": f"{test_acc:.6f}",
                "params": params,
                "flops": int(flops_n),
                "conv_mid": info["conv_channels"],
                "ffn": info["ffn_hidden"],
                "heads_groups": "" if heads_groups is None else f"{heads_groups[0]}/{heads_groups[1]}",
                "seconds": f"{time.time() - t0:.1f}",
            }
            append_row(row, fieldnames)
            rows.append(row)
            ckpt_out = os.path.join(OUT_DIR, f"seed{seed}_a{alpha:.2f}.pth")
            torch.save(pruned.state_dict(), ckpt_out)
            print(
                f"  α={alpha:.2f} params={params} val {val_before*100:.2f}→{val_after*100:.2f} "
                f"(Δ {drop_pp:+.2f} pp) test={test_acc*100:.2f}%  {row['seconds']}s"
            )
            del pruned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del dense
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
