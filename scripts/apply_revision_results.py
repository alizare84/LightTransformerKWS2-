#!/usr/bin/env python3
"""After the revision GPU queue finishes, rebuild metrics CSVs and print LaTeX snippets.

Safe to run repeatedly. Does not invent missing seeds.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_summary_acc(exp_dir: Path):
    for name in ("final_summary.csv",):
        p = exp_dir / name
        if not p.exists():
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            metric = (row.get("Metric") or "").lower()
            if "test accuracy" in metric or metric == "accuracy (confusion matrix)":
                # Prefer Original column for dense/untied; Pruned+Quant for C2
                for col in ("Original", "Pruned+Quant", "Quantized"):
                    if col in row and row[col] not in ("", "-", None):
                        try:
                            return float(row[col])
                        except ValueError:
                            pass
    return None


def mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def paired(a, b):
    pairs = [(x - y) for x, y in zip(a, b)]
    n = len(pairs)
    m = statistics.mean(pairs)
    s = statistics.stdev(pairs) if n > 1 else 0.0
    t = m / (s / math.sqrt(n)) if n > 1 and s > 0 else float("nan")
    return n, m, s, t


def main():
    # Base per-seed from existing file
    base = ROOT / "results" / "per_seed_metrics.csv"
    rows = {int(r["seed"]): r for r in csv.DictReader(open(base))}

    # Fill U from exp_A1_seed*
    for seed in range(5):
        acc = read_summary_acc(ROOT / "results" / f"exp_A1_seed{seed}")
        if acc is None:
            continue
        rows.setdefault(seed, {"seed": str(seed)})
        rows[seed]["U"] = f"{acc:.4f}"

    # U-L1 / T-MHA / POS from experiment dirs
    extras = {}
    for seed in range(5):
        for key, exp in [
            ("U-L1", f"exp_UL1_seed{seed}"),
            ("T-MHA", f"exp_TMHA_seed{seed}"),
        ]:
            acc = read_summary_acc(ROOT / "results" / exp)
            if acc is not None:
                extras.setdefault(seed, {})[key] = acc

    pos = {}
    for tag, exp in [
        ("abs", "exp_POS_abs_seed0"),
        ("rel", "exp_POS_rel_seed0"),
        ("both", "exp_A3_seed0"),
    ]:
        acc = read_summary_acc(ROOT / "results" / exp)
        if acc is not None:
            pos[tag] = acc

    # Write updated per_seed_metrics
    fields = ["seed", "T", "T-Q", "U", "U-Q", "T-P-Q", "S-Q"]
    out = ROOT / "results" / "per_seed_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for seed in sorted(rows):
            w.writerow({k: rows[seed].get(k, "") for k in fields})
    print(f"updated {out}")

    # Architecture control CSV
    if extras:
        ac = ROOT / "results" / "architecture_controls.csv"
        with open(ac, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "U-L1", "T-MHA"])
            for seed in sorted(extras):
                w.writerow([seed, extras[seed].get("U-L1", ""), extras[seed].get("T-MHA", "")])
        print(f"wrote {ac}")

    if pos:
        pc = ROOT / "results" / "position_ablation_seed0.csv"
        with open(pc, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "test_acc"])
            for k, v in pos.items():
                w.writerow([k, f"{v:.6f}"])
        print(f"wrote {pc}")

    # Print U summary if 5 seeds present
    u_vals = []
    for seed in range(5):
        v = rows.get(seed, {}).get("U", "")
        if v:
            u_vals.append(float(v) * 100)
    if u_vals:
        m, s = mean_std(u_vals)
        print(f"U: n={len(u_vals)} mean={m:.2f}±{s:.2f}%  values={[round(x,2) for x in u_vals]}")

    # T+2K
    ft = ROOT / "results" / "t_ft_control.csv"
    if ft.exists():
        print(f"T+2K control present: {ft}")
        print(ft.read_text())


if __name__ == "__main__":
    main()
