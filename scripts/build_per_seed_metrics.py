#!/usr/bin/env python3
"""Extract per-seed metrics from all final_summary.csv files and compute paired statistics.

Outputs:
  - results/per_seed_metrics.csv
  - results/paired_statistics.csv
  - reports/statistical_audit.md
"""
import csv
import os
import statistics
from pathlib import Path
from collections import defaultdict
import math

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# Model definitions
MODELS = {
    "T": ("exp_A3_seed{seed}", "Original", "Tied dense"),
    "T-Q": ("exp_A3_seed{seed}", "Quantized", "Tied + INT8"),
    "U": ("exp_A1_seed{seed}", "Original", "Untied dense"),
    "U-Q": ("exp_A1_seed{seed}", "Quantized", "Untied + INT8"),
    "T-P-Q": ("exp_C2_seed{seed}", "Pruned+Quant", "Tied + Pruned α=0.20 + INT8"),
    "S-Q": ("exp_SQ_seed{seed}", "Quantized", "Scratch pruned arch + INT8"),
}

SEEDS = [0, 1, 2, 3, 4]


def parse_csv_value(csv_path, column_name):
    """Extract test accuracy from final_summary.csv."""
    if not csv_path.exists():
        return None
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Metric") == "Test accuracy":
                val = row.get(column_name)
                if val:
                    try:
                        return float(val)
                    except ValueError:
                        return None
    return None


def main():
    # Collect per-seed data
    data = defaultdict(dict)
    
    for model_id, (folder_tmpl, column, desc) in MODELS.items():
        for seed in SEEDS:
            folder = RESULTS / folder_tmpl.format(seed=seed)
            csv_file = folder / "final_summary.csv"
            acc = parse_csv_value(csv_file, column)
            if acc is not None:
                data[seed][model_id] = acc
    
    # Write per_seed_metrics.csv
    per_seed_path = RESULTS / "per_seed_metrics.csv"
    with open(per_seed_path, "w", newline="") as f:
        fieldnames = ["seed"] + list(MODELS.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in SEEDS:
            row = {"seed": seed}
            row.update(data[seed])
            writer.writerow(row)
    
    print(f"✓ Wrote {per_seed_path}")
    
    # Compute statistics
    stats = {}
    for model_id in MODELS.keys():
        vals = [data[s][model_id] for s in SEEDS if model_id in data[s]]
        if len(vals) >= 2:
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals)
            stats[model_id] = {
                "n": len(vals),
                "mean": mu,
                "std": sd,
                "seeds": [s for s in SEEDS if model_id in data[s]],
                "values": vals,
            }
        elif len(vals) == 1:
            stats[model_id] = {
                "n": 1,
                "mean": vals[0],
                "std": None,
                "seeds": [s for s in SEEDS if model_id in data[s]],
                "values": vals,
            }
        else:
            stats[model_id] = {"n": 0, "mean": None, "std": None, "seeds": [], "values": []}
    
    # Paired comparisons
    def paired_test(model_a, model_b):
        """Compute paired t-test between two models on common seeds."""
        common_seeds = sorted(set(stats[model_a]["seeds"]) & set(stats[model_b]["seeds"]))
        if len(common_seeds) < 2:
            return None
        
        vals_a = [data[s][model_a] for s in common_seeds]
        vals_b = [data[s][model_b] for s in common_seeds]
        diffs = [b - a for a, b in zip(vals_a, vals_b)]
        
        n = len(diffs)
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if n > 1 else 0.0
        se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0
        t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
        
        # 95% CI for mean difference
        # t-critical for df=n-1, two-tailed 0.05
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n, 2.0)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return {
            "n_pairs": n,
            "common_seeds": common_seeds,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "se_diff": se_diff,
            "t_stat": t_stat,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }
    
    comparisons = [
        ("T-Q", "T", "Effect of INT8 on tied dense"),
        ("T-P-Q", "T", "Effect of prune+INT8 on tied"),
        ("S-Q", "T-P-Q", "Scratch vs prune-then-finetune"),
        ("U", "T", "Untied vs tied (dense)"),
    ]
    
    paired_results = []
    for model_b, model_a, desc in comparisons:
        result = paired_test(model_a, model_b)
        if result:
            paired_results.append({
                "comparison": f"{model_b} - {model_a}",
                "description": desc,
                **result,
            })
    
    # Write paired_statistics.csv
    paired_path = RESULTS / "paired_statistics.csv"
    if paired_results:
        with open(paired_path, "w", newline="") as f:
            fieldnames = ["comparison", "description", "n_pairs", "common_seeds",
                         "mean_diff", "std_diff", "t_stat", "ci_95_lower", "ci_95_upper"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in paired_results:
                row = {k: r[k] for k in fieldnames}
                row["common_seeds"] = str(r["common_seeds"])
                writer.writerow(row)
        print(f"✓ Wrote {paired_path}")
    
    # Write statistical_audit.md
    report_path = ROOT / "reports" / "statistical_audit.md"
    lines = [
        "# Statistical Audit — KWT-GQA",
        "",
        f"**Generated:** {per_seed_path.name}, {paired_path.name}",
        "",
        "## Per-model summary",
        "",
        "| Model | n | Mean | Std | Seeds |",
        "|---|---:|---:|---:|---|",
    ]
    
    for model_id, (_, _, desc) in MODELS.items():
        s = stats[model_id]
        if s["n"] > 0:
            mean_str = f"{100*s['mean']:.2f}%" if s["mean"] is not None else "—"
            std_str = f"±{100*s['std']:.2f} pp" if s["std"] is not None else "—"
            seeds_str = ",".join(map(str, s["seeds"]))
            lines.append(f"| {model_id} ({desc}) | {s['n']} | {mean_str} | {std_str} | {seeds_str} |")
        else:
            lines.append(f"| {model_id} ({desc}) | 0 | — | — | none |")
    
    lines += ["", "## Paired comparisons", ""]
    
    if paired_results:
        for r in paired_results:
            lines.append(f"### {r['comparison']}")
            lines.append(f"**{r['description']}**")
            lines.append(f"- Common seeds: {r['common_seeds']}")
            lines.append(f"- n pairs: {r['n_pairs']}")
            lines.append(f"- Mean difference: {100*r['mean_diff']:.2f} pp")
            lines.append(f"- Std of differences: {100*r['std_diff']:.2f} pp")
            lines.append(f"- t-statistic: {r['t_stat']:.3f}")
            lines.append(f"- 95% CI: [{100*r['ci_95_lower']:.2f}, {100*r['ci_95_upper']:.2f}] pp")
            lines.append("")
    else:
        lines.append("No valid paired comparisons (insufficient overlapping seeds).")
    
    lines += [
        "## Notes",
        "",
        "- **U (Untied)** has only seeds 0,1,2 available (seeds 3,4 missing).",
        "- Paired t-tests assume normality; with n=3–5, interpret cautiously.",
        "- Non-rejection of H₀ is **not** evidence of equivalence.",
        "- For T-P-Q parent lineage, verify that T-P float checkpoint was used.",
        "",
    ]
    
    report_path.write_text("\n".join(lines))
    print(f"✓ Wrote {report_path}")
    
    # Print summary to terminal
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for model_id, (_, _, desc) in MODELS.items():
        s = stats[model_id]
        if s["n"] >= 2:
            print(f"{model_id:8s} ({desc:30s}): {100*s['mean']:5.2f} ± {100*s['std']:4.2f} pp  (n={s['n']})")
        elif s["n"] == 1:
            print(f"{model_id:8s} ({desc:30s}): {100*s['mean']:5.2f}%            (n=1)")
        else:
            print(f"{model_id:8s} ({desc:30s}): NO DATA")
    
    print("\nPaired comparisons:")
    for r in paired_results:
        ci_str = f"[{100*r['ci_95_lower']:.2f}, {100*r['ci_95_upper']:.2f}]"
        print(f"  {r['comparison']:20s} Δ={100*r['mean_diff']:+6.2f} pp  95%CI={ci_str:20s}  t={r['t_stat']:+6.3f}  (n={r['n_pairs']})")


if __name__ == "__main__":
    main()