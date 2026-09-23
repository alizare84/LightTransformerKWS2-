#!/usr/bin/env python3
"""Aggregate Phase-3 multi-seed test accuracies from results/exp_*_seed*/final_summary.md."""
import csv
import os
import re
import statistics
from pathlib import Path

ROOT = Path("/home/alizare84/LightTransformerKWS2")
OUT = ROOT / "results" / "multiseed"
SEEDS = (0, 1, 2, 3, 4)
JOBS = (
    ("A3", "exp_A3_seed{seed}", "Original"),
    ("A3_int8", "exp_A3_seed{seed}", "Quantized"),
    ("C2", "exp_C2_seed{seed}", "Pruned+Quant"),
    ("A1", "exp_A1_seed{seed}", "Original"),
    ("A1_int8", "exp_A1_seed{seed}", "Quantized"),
)


def parse_test_accuracy(md_path, column_name):
    text = Path(md_path).read_text()
    for line in text.splitlines():
        if not line.startswith("| Test accuracy"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        header_line = None
        for hl in text.splitlines():
            if hl.startswith("| Metric"):
                header_line = [c.strip() for c in hl.strip("|").split("|")]
                break
        if header_line is None:
            return None
        try:
            idx = header_line.index(column_name)
        except ValueError:
            # C2 tables use "Pruned+Quant"; dense tables use "Quantized"
            if column_name == "Pruned+Quant" and "Quantized" in header_line:
                idx = header_line.index("Quantized")
            else:
                return None
        return float(cells[idx])
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    by_job = {name: [] for name, _, _ in JOBS}

    for seed in SEEDS:
        row = {"seed": seed}
        for name, tmpl, col in JOBS:
            md = ROOT / "results" / tmpl.format(seed=seed) / "final_summary.md"
            acc = parse_test_accuracy(md, col) if md.exists() else None
            row[name] = acc
            if acc is not None:
                by_job[name].append(acc)
        rows.append(row)

    csv_path = OUT / "multiseed_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed"] + [n for n, _, _ in JOBS])
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Multi-seed test accuracy (GSC v1)",
        "",
        "Values are test-set accuracy from each run's `final_summary.md`.",
        "Missing cells mean that seed has not finished.",
        "",
        "| Seed | A3 | A3 INT8 | C2 (α=0.20 + INT8) | A1 | A1 INT8 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(v):
            return f"{100*v:.2f}%" if v is not None else "—"
        lines.append(
            f"| {row['seed']} | {fmt(row['A3'])} | {fmt(row['A3_int8'])} | "
            f"{fmt(row['C2'])} | {fmt(row['A1'])} | {fmt(row['A1_int8'])} |"
        )

    lines += ["", "## Mean ± std (completed seeds only)", ""]
    for name, _, _ in JOBS:
        vals = by_job[name]
        if len(vals) >= 2:
            mu = 100 * statistics.mean(vals)
            sd = 100 * statistics.stdev(vals)
            lines.append(f"- **{name}**: {mu:.2f} ± {sd:.2f} pp  (n={len(vals)})")
        elif len(vals) == 1:
            lines.append(f"- **{name}**: {100*vals[0]:.2f}%  (n=1, std n/a)")
        else:
            lines.append(f"- **{name}**: not finished")

    md_path = OUT / "multiseed_results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
