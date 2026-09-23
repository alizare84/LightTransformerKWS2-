#!/usr/bin/env python3
"""Paired tests on preds_*.npz.

Cross-run McNemar (A3 vs C2, A1 vs A3) is INVALID: the TF MFCC test loader does
not expose stable sample IDs, so label order differs across train.py invocations
even when accuracy is similar. Only within-run pairs (same process, same loader
pass) are paired: original vs quantized in the same experiment directory.
"""
from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np

ROOT = Path("/home/alizare84/LightTransformerKWS2/results")
OUT = ROOT / "multiseed" / "statistical_tests.md"


def load_preds(exp: str, which: str):
    data = np.load(ROOT / exp / f"preds_{which}.npz")
    return data["labels"], data["preds"]


def mcnemar(y, a, b):
    a_ok = a == y
    b_ok = b == y
    b01 = int(np.sum(~a_ok & b_ok))
    b10 = int(np.sum(a_ok & ~b_ok))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = max(b01, b10)
    # two-sided exact binomial under p=0.5
    p = 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return b01, b10, min(1.0, p)


def main():
    lines = [
        "# Statistical tests (paired predictions)",
        "",
        "## Important: cross-model pairing is not available",
        "",
        "Test clip order differs across `train.py` runs (no stable sample ID in the",
        "saved `preds_*.npz`). Therefore **A1 vs A3** and **A3 vs C2 across directories",
        "directories** cannot use McNemar. Report mean±std over seeds instead.",
        "",
        "## Within-run: float vs INT8 (same experiment, same loader pass)",
        "",
        "| Exp | Acc float | Acc INT8 | b01 | b10 | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in ("A3", "A1", "C2"):
        for s in range(3):
            exp = f"exp_{kind}_seed{s}"
            which_q = "quantized"
            y, a = load_preds(exp, "original")
            y2, b = load_preds(exp, which_q)
            assert np.array_equal(y, y2), f"label mismatch inside {exp}"
            b01, b10, p = mcnemar(y, a, b)
            lines.append(
                f"| {exp} | {(y==a).mean()*100:.2f}% | {(y==b).mean()*100:.2f}% | "
                f"{b01} | {b10} | {p:.4g} |"
            )

    lines += [
        "",
        "## Three-seed accuracy (primary reporting)",
        "",
        "See `multiseed_results.md`.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
