#!/usr/bin/env python3
"""Paired t-test on per-seed A3 vs C2 test accuracies (same seed parent)."""
from __future__ import annotations

import statistics
from pathlib import Path

import sys

ROOT = Path("/home/alizare84/LightTransformerKWS2")
sys.path.insert(0, str(ROOT / "scripts"))
from parse_multiseed import SEEDS, parse_test_accuracy  # noqa: E402

OUT = ROOT / "results" / "multiseed" / "ttest_a3_c2.md"


def welch_t_p(a, b):
    """Two-sided Welch t-test p-value without scipy (n small)."""
    import math

    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1 = statistics.variance(a) if n1 > 1 else 0.0
    v2 = statistics.variance(b) if n2 > 1 else 0.0
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    t = (m1 - m2) / se
    # Welch–Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = 0.0
    if n1 > 1:
        den += (v1 / n1) ** 2 / (n1 - 1)
    if n2 > 1:
        den += (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float("nan")
    # two-sided p via regularized incomplete beta (Student-t CDF)
    try:
        from math import lgamma

        def betainc_reg(x, a, b):
            # continued-fraction approx of incomplete beta / B(a,b)
            # Use scipy if available
            raise NotImplementedError

        # Prefer scipy
        from scipy import stats  # type: ignore

        p = float(stats.t.sf(abs(t), df) * 2)
        return t, df, p
    except Exception:
        pass
    try:
        from scipy import stats  # type: ignore

        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        return t, df, p
    except Exception:
        return t, df, float("nan")


def paired_t(a, b):
    """Paired t-test on matched seed pairs."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return float("inf") if md != 0 else 0.0, n - 1, 0.0 if md != 0 else 1.0
    t = md / (sd / (n ** 0.5))
    df = n - 1
    try:
        from scipy import stats  # type: ignore

        p = float(stats.t.sf(abs(t), df) * 2)
        return t, df, p
    except Exception:
        # crude two-sided via erfc for large df; for n=5 leave p as nan if no scipy
        return t, df, float("nan")


def main():
    a3, c2, pairs = [], [], []
    for seed in SEEDS:
        md_a3 = ROOT / "results" / f"exp_A3_seed{seed}" / "final_summary.md"
        md_c2 = ROOT / "results" / f"exp_C2_seed{seed}" / "final_summary.md"
        if not md_a3.exists() or not md_c2.exists():
            continue
        va = parse_test_accuracy(md_a3, "Original")
        vc = parse_test_accuracy(md_c2, "Pruned+Quant")
        if va is None or vc is None:
            continue
        a3.append(va)
        c2.append(vc)
        pairs.append((seed, va, vc))

    lines = [
        "# A3 vs C2 seed-level comparison",
        "",
        "Per-seed test accuracy (fraction). C2 parent is the same-seed A3 checkpoint.",
        "",
        "| Seed | A3 | C2 | Δ (A3−C2) pp |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed, va, vc in pairs:
        lines.append(f"| {seed} | {100*va:.2f}% | {100*vc:.2f}% | {100*(va-vc):+.2f} |")

    n = len(pairs)
    lines += ["", f"Completed matched pairs: **n={n}**", ""]
    if n >= 2:
        mu_a = 100 * statistics.mean(a3)
        sd_a = 100 * statistics.stdev(a3)
        mu_c = 100 * statistics.mean(c2)
        sd_c = 100 * statistics.stdev(c2)
        lines.append(f"- A3: {mu_a:.2f} ± {sd_a:.2f}%")
        lines.append(f"- C2: {mu_c:.2f} ± {sd_c:.2f}%")
        lines.append(f"- Mean Δ: {mu_a - mu_c:+.2f} pp")
        t, df, p = paired_t(a3, c2)
        if p == p:  # not nan
            lines.append(
                f"- Paired two-sided t-test on seed accuracies: "
                f"t={t:.3f}, df={df}, p={p:.4f}"
            )
            if p >= 0.05:
                lines.append(
                    "- Interpretation: at α=0.05 we **do not** reject equal means; "
                    "report the gap as a descriptive mean shift, not a proven effect."
                )
            else:
                lines.append(
                    "- Interpretation: difference is statistically detectable at α=0.05 "
                    "under this paired seed test (still a small effect size)."
                )
        else:
            lines.append(
                f"- Paired t-statistic t={t:.3f}, df={df}; install scipy to get a p-value "
                f"(`pip install scipy`)."
            )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
