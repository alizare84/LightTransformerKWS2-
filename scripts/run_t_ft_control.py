#!/usr/bin/env python3
"""T + 2000-step fine-tuning control without pruning.

Continues each dense T checkpoint for 2000 AdamW steps at lr=1e-4
(matching the pruning fine-tune recipe) and records val/test accuracy.

Uses Datagen directly (no DataLoader workers) to avoid TF1 session
forking issues in multiprocessing.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Force line-buffered stdout so logs appear immediately when redirected
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kwt import KWT
from datagen import Datagen
from utils.label_smoothing import LabelSmoothingLoss


def load_tied(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = KWT(
        img_x=98, img_y=40, patch_x=1, patch_y=40, num_classes=12,
        dim=48, depth=12, heads=8, mlp_dim=96, pool="mean",
        channels=1, dropout=0.1, emb_dropout=0.05,
        use_conv_encoder=True, gqa_groups=4, use_relative_pos=True,
        use_absolute_pos=True, share_layers=True,
    )
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def evaluate_datagen(model, dg, part, device, eval_batch=256):
    """Evaluate accuracy using Datagen directly — no DataLoader subprocess.

    For val: uses dg.getData() which returns batch_size chunks.
    For test: fetches all samples in one call via audio_processor directly.
    """
    model.eval()
    correct = total = 0

    if part in ("train", "val"):
        n_batches = dg.dataLen(part)
        for i in range(n_batches):
            data, target = dg.getData(part, i)
            data_t = torch.from_numpy(data).float().to(device)
            tgt_t = torch.from_numpy(target).long().to(device).view(-1)
            pred = model(data_t).view(-1, 12).argmax(dim=1)
            correct += (pred == tgt_t).sum().item()
            total += tgt_t.numel()
    else:  # test — fetch in one big batch to avoid 3081 individual calls
        n_test = dg.dataLen("test")
        # Use audio_processor.get_data with full set size
        data_np, tgt_np = dg.audio_processor.get_data(
            n_test, 0, dg.flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, dg.sess)
        data_t = torch.from_numpy(data_np.astype("float32")).to(device)
        tgt_t = torch.from_numpy(tgt_np.astype("int64")).to(device).view(-1)
        # Evaluate in mini-batches to avoid OOM
        for s in range(0, len(tgt_t), eval_batch):
            pred = model(data_t[s:s+eval_batch]).view(-1, 12).argmax(dim=1)
            correct += (pred == tgt_t[s:s+eval_batch]).sum().item()
            total += tgt_t[s:s+eval_batch].numel()

    return 100.0 * correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Ignored; kept for CLI compat. Always 0 (avoid TF1 fork).")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    out_csv = Path("results/t_ft_control.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  seeds={seeds}  steps={args.steps}  lr={args.lr}")
    sys.stdout.flush()

    print("Initializing Datagen (TF1 AudioProcessor — may take a few minutes)...")
    sys.stdout.flush()
    dg = Datagen(batch_size=args.batch_size, version=1, preprocess="mfcc")
    n_train_batches = dg.dataLen("train")
    print(f"  train batches={n_train_batches}  val batches={dg.dataLen('val')}  "
          f"test samples={dg.dataLen('test')}")
    sys.stdout.flush()

    rows = []
    for seed in seeds:
        ckpt = Path(f"checkpoints/dense_A3_seed{seed}.pth")
        if not ckpt.exists():
            print(f"SKIP seed={seed}: missing {ckpt}")
            sys.stdout.flush()
            continue

        print(f"\n===== seed={seed} T+{args.steps} FT control =====")
        sys.stdout.flush()
        torch.manual_seed(seed)
        model = load_tied(ckpt, device)

        val_b = evaluate_datagen(model, dg, "val", device)
        test_b = evaluate_datagen(model, dg, "test", device)
        print(f"  before val={val_b:.2f}% test={test_b:.2f}%")
        sys.stdout.flush()

        model.train()
        loss_fn = LabelSmoothingLoss(classes=12, smoothing=0.1)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

        for step in range(args.steps):
            batch_idx = step % n_train_batches
            data, target = dg.getData("train", batch_idx)
            data_t = torch.from_numpy(data).float().to(device)
            tgt_t = torch.from_numpy(target).long().to(device).view(-1)
            opt.zero_grad()
            loss = loss_fn(model(data_t).view(-1, 12), tgt_t)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if (step + 1) % 200 == 0:
                print(f"  step {step+1}/{args.steps}  loss={loss.item():.4f}")
                sys.stdout.flush()

        val_a = evaluate_datagen(model, dg, "val", device)
        test_a = evaluate_datagen(model, dg, "test", device)
        print(f"  after  val={val_a:.2f}% test={test_a:.2f}%  "
              f"delta_test={test_a - test_b:+.2f} pp")
        sys.stdout.flush()

        out_dir = Path(f"results/t_ft_control_seed{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pth")
        row = {
            "seed": seed,
            "val_before": round(val_b, 4),
            "test_before": round(test_b, 4),
            "val_after": round(val_a, 4),
            "test_after": round(test_a, 4),
            "val_delta": round(val_a - val_b, 4),
            "test_delta": round(test_a - test_b, 4),
        }
        rows.append(row)
        with open(out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerow(row)
        print(f"  saved to {out_dir}/")
        sys.stdout.flush()

    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out_csv}")
        sys.stdout.flush()

    print("\nAll done.")


if __name__ == "__main__":
    main()
