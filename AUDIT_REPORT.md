# Audit report — LightTransformerKWS2

Paper: *A 31K-Parameter Transformer for Keyword Spotting via Structured Pruning and INT8 Quantization*

This file is the living record of the Q1 audit. **No manuscript numbers were changed in Phases 0–2.** No results were invented.

Status key: `observed` = read from logs/checkpoints; `not run` = no seed/command exists; `blocked` = waiting on GPU/canonical A3.

---

# PHASE 0 — Feasibility and time budget

Estimates use wall-clock already observed in this repository, not guesses.

## Observed wall-clock (this machine, RTX 2060 6 GB, `--num-workers 1`)

| Job | Command / log | Wall-clock | What it did |
| --- | --- | ---: | --- |
| A1 v1 | `/tmp/ablation_A1.log`, terminal 311936 | **25.3 min** | Train 20 epochs + INT8 + 2× test (batch=1) |
| C0 v1 | `results/exp_C0/` timestamps 13:37–13:40 | **~3 min** | `--skip-train`, 2× test |
| U2 v1 | terminal 311935 | **62.3 min** | `--skip-train`, 2000-step FT, 2× test |
| Structured sweep v1 | `/tmp/struct_sweep.log` 10:53–13:02 | **~2.1 h** | **16 extra training epochs** (no `--skip-train`) + 4× (2000 FT + val) + 2× test |
| A3 v2 | `scripts/run_v2_pipeline.sh` 13:37–15:32 | **115 min** | Train 88 epochs (early stop) + INT8 + 2× test |
| C0 v2 | same log 15:32–15:39 | **7 min** | `--skip-train`, 2× test |
| U2 v2 | 15:39 31 Aug – 08:23 1 Sep | **~16.7 h** | Same recipe as U2 v1. This duration is **anomalous** (likely host sleep/stall). Do not use 16.7 h as the unit cost. |
| Structured sweep v2 | started 08:24 1 Sep, still running at audit time | **>2 h and counting** | `--skip-train`, α ∈ {0.10,0.15,0.20,0.30} |

Per-epoch training on v1 is ~1–2 min. Full test (`Dataset(part="test", batch_size=1)`) is the expensive tail: ~3–7 min per pass, two passes per `train.py` invocation (float + INT8) plus two FAR/FRR passes. Latency in `train.py` is 100 un-warmed CPU forwards of one clip — not a benchmark.

NOISEX-92 (`eval_noisex92.py`): 5 noises × 6 SNRs + Clean, full GSC test, two checkpoints. Prior run produced `results/noisex92/`. A full 5-crop repeat would be ~5× that job.

## Estimates for requested phases

| Phase | Naive full request | Estimated wall-clock | Decision for this session |
| --- | --- | ---: | --- |
| **3 multi-seed** | 5 seeds × (A1 + A3 + C2) | 5 × (~25 + ~40 + ~60) min ≈ **10–12 h**; 3 seeds ≈ **6.5–8 h** | **Reduced:** 3 seeds (0,1,2), **GSC v1 only**, C2 = **one** α after Phase-2 reselection (not a 4-point sweep per seed). Not started until canonical A3 is frozen. |
| **7 α boundary** | α ∈ {0.15…0.20} step 0.01, each a full `train.py` | 6 × ~45 min ≈ **4.5 h** if each run does full test | **Reduced:** val-only sweep on canonical A3, α ∈ **{0.15, 0.17, 0.20}** first (~30–45 min). Fill 0.16/0.18/0.19 only if 0.17 still passes the 0.5 pp rule. |
| **9 latency** | 4 models × 500–1000 iters | **15–25 min** | **Full scope** (feasible). |
| **10 NOISEX** | 5 crops × 5×7 conditions × 2 models | **tens of hours** | **Reduced:** keep existing single-crop table; add **3 crops** only if GPU time remains after Phases 2–3. Otherwise the paper must state that crop variance was not estimated. |
| **12 CNN baseline** | Reimplement DS-CNN-S / BC-ResNet-1 under this MFCC graph | **4–8 h** engineering + train; **no CNN trainer exists in this repo** | **Reduced:** do not fabricate a reimplementation this session. Document as not reproduced. Literature numbers stay **contextual**. |

Total if we ignored the cap: >24 h. **Capped plan after Phase 2:** Phase 2 re-sweep (mandatory) ~1–2 h; Phase 3 reduced ~7 h; Phase 7 reduced ~45 min; Phase 9 ~20 min. Phases 10 (extra crops) and 12 are explicitly partial/skipped.

A v2 pipeline (`scripts/run_v2_pipeline.sh`) was already running at audit time (structured sweep). It was **not killed**. It is **not** a substitute for the canonical v1 A3 protocol.

---

# PHASE 1 — Repository audit

## 1. How A1–A5 / C0–C2 / U1–U2 are produced

There is **no experiment config file and no `--seed` flag** in `train.py`. Rows are produced by CLI conventions:

| ID | Intended meaning | How it is actually produced |
| --- | --- | --- |
| **A1** | Untied GQA, train from scratch | `python train.py exp_A1 --num-workers 1` (**no** `--share-layers`). `model.pth` must be absent: `load_state_dict(..., strict=False)` would otherwise leak tied weights. |
| **A2** | Tied MHA (G=8) | **Not run.** Paper: `planned (P1)`. |
| **A3** | Tied GQA dense | Historically: train with `--share-layers`. Paper’s 95.55% is **not** a clean `--skip-train` eval of `checkpoints/dense_A3.pth` (see Phase 2). |
| **A4/A5** | GQA-2 / MQA | **Not run.** |
| **C0** | INT8 only, tensors unshrunk | `python train.py exp_C0 --share-layers --skip-train` |
| **C1** | Structured α=0.15, float, val | Float column of the structured sweep (`93.16% val` in Table II). Not a separate test-set float checkpoint in `results/`. |
| **C2** | Structured selected-α + INT8 | Same `train.py` process as the sweep: prune+FT then `quantize_dynamic`. Paper 94.42% from `results/exp_struct_sweep/`. |
| **U1** | Unstructured α=0.15, float val | Val column of the unstructured sweep inside `exp_U2` (95.38%). |
| **U2** | Unstructured α=0.15 + INT8 | `python train.py exp_U2 --share-layers --skip-train --enable-pruning --prune-mode unstructured --prune-amounts 0.15 --finetune-steps 2000 --max-acc-drop 0.005 --num-workers 1` |
| **H1** | `--keep-heads` | **Not run.** |

Every `train.py` run that is not `--skip-train` **overwrites `model.pth`**. Compression rows that need a frozen dense parent **must** copy `checkpoints/dense_A3.pth` → `model.pth` first.

## 2. Checkpoints

| File | Role | Notes |
| --- | --- | --- |
| `checkpoints/dense_A3.pth` | v1 tied dense state_dict (178 KB) | md5 `14c41d51d5eb5ac7ce96ee43ab177179`. Identical to `dense_A3_v1.pth`. |
| `checkpoints/dense_A3_best.pth` | 454 KB training blob | Contains optimizer; not the eval state_dict. |
| `checkpoints/struct_0.15_quant.pt` | C2 INT8 module | From the Table II sweep process. |
| `checkpoints/untied_A1.pth` | A1 state_dict | 916 KB. |
| `checkpoints/dense_A3_v2.pth` | v2 tied dense | md5 `1c168720…`. At audit time `model.pth` matched this (v2 pipeline). |
| `model.pth` | **ephemeral** | Last training run wins. Not a provenance record. |

## 3. Dataset split

`--version 1` → `data/speech_commands/google-speech-commands/data1` (GSC v0.01). `--version 2` → `data2` (v0.02). `--version 3` → `data3` 35-class pre-split.

12-class wanted-words come from `kws_streaming` defaults (not overridden except version 3). Train/val/test lists are the official `testing_list.txt` / `validation_list.txt` plus generated silence/unknown.

**Validation length actually used:** `dataLen('val') = set_size // 512`. Logs show **3072 = 6×512** examples on v1 and **4096 = 8×512** on v2. The remainder of the official val split is **dropped**.

**Val offset bug:** `Dataset.__getitem__` passes the DataLoader index as `offset` to `AudioProcessor.get_data`. Training correctly uses `offset * batch_size`. Validation uses `offset` ∈ {0,1,2,3,4,5} as a **sample** start index (`input_data.py:583,626`). The six val batches therefore start at clips 0,1,2,3,4,5 and **heavily overlap**. Unique val utterances seen ≈ 517, not 3072. Test uses `offset` = clip index and `how_many=1`, which is the intended sequential scan.

## 4. Preprocessing / MFCC

`datagen.py`: `preprocess="mfcc"`, 30 ms window, 10 ms hop, 80 mel bins, 40 DCT, `use_tf_fft=True`, TensorFlow graph in `kws_streaming`. Same graph is used by `eval_noisex92.py` **after** waveform mixing (`output_from_mixed_`). Table III (`get_data` test) vs Table VIII Clean differ because Clean in NOISEX still goes through the external-mix MFCC path (paper already notes +0.13 pp on C2).

## 5. Validation augmentation

`getData(..., 'val')` sets background frequency/volume, time shift, resample, volume jitter to 0. `get_data` applies background only when `mode == 'training'`. SpecAugment is configured on FLAGS but the val call does not pass training-time audio jitter. **Intended: val aug off. Residual risk:** a new TF session per worker can still change MFCC numerics slightly.

## 6. Random seed

`train.py` imports `random` and `numpy` but **never calls** `random.seed`, `np.random.seed`, `torch.manual_seed`, or `torch.cuda.manual_seed_all`. DataLoader workers are not seeded. `eval_noisex92.py` has `--seed` (default 0) and hashes it into the crop RNG. **All paper v1 numbers are one unseeded run.**

## 7. Train / fine-tune steps

- Train: AdamW `lr=1e-3`, wd=`5e-4`, batch 512, `num_steps` default 23000, `steps_per_epoch` **hardcoded 45**, warmup 10 epochs, cosine `T_max ≈ 511` (so LR barely decays before early stop). Early stop: patience 15 on val **loss**, `min_delta=5e-4`.
- A1 log: stopped epoch **20**.
- Paper text says A3 stopped epoch **16**. That “16” is the **continued** `exp_struct_sweep` run, not a documented first-from-scratch A3 log in `results/` under that name.
- Prune FT: 2000 steps, `lr=1e-4`, AdamW wd=0. Unstructured re-zeros the L1 mask after every step (`preserve_zeros=True`). Structured does not.

## 8. Evaluation

After train/prune/quant, `train.py` always:

1. `test_accuracy` on CPU, batch=1 TF MFCC test loader  
2. Repeats for the quantized model  
3. `evaluate_far_frr` (macro one-vs-rest on the 12×12 confusion matrix) for both  
4. `measure_inference_time_real` (100 runs, **no warmup**, one clip, CPU)  
5. Writes `results/<exp>/final_summary.{csv,md}`

Val for α selection is `quick_val_accuracy` on the **truncated overlapping** val loader, not the test set.

## 9–13. Parameter / FLOP / file-size counting

| Quantity | Function | Verdict |
| --- | --- | --- |
| Tied params | `count_parameters` = `sum(p.numel() for p in model.parameters())` | **Correct:** `nn.ModuleList([shared_block]*12)` still yields unique tensors once. Inventory 36,516 (tied) / 219,204 (untied) matches `tab:params`. |
| Quantized params | `count_parameters_quant_aware` walks `nnqd.Linear.weight()` | **Correct intent.** Reports 36,516 after INT8 in C0/U2/C2 logs. |
| Unstructured “params” | `numel()` unchanged; nonzero via `count_nonzero_dense` | **Correctly distinguished** in U2 logs (29,772→25,365 Linear/Conv; whole-model nonzero 32,109). |
| FLOPs | `thop.profile` on float model, dummy `(1,98,40)` | **Architecture-level, not bit-accurate.** Forward still runs 12 tied blocks, so tying does not cut FLOPs (20.717M both). Structured C2 16.920M is from profiling the rebuilt graph. Quantized modules are **not** profiled (correct). |
| File size | `get_model_size_kb_dedup` by `id(module)` | **Correct for tying.** Paper 143.41 KB → 94.52 KB (C0) / 82.62 KB (C2) matches logs. Raw `state_dict` save is known to duplicate quantized tied blocks; the paper uses the dedup path. |

## 14. Latency

`measure_inference_time_real`: 100 iterations, no warmup, no thread pin, no p50/p95, one sample. Table III 11.7 ms → 12.7 ms is **one mean of 100**. **Not statistically meaningful** as a hardware claim. Phase 9 must replace it.

## Other implementation notes

- Default `--num-workers 10` OOMs on this host; successful ablations used `1`.
- `--skip-train` still constructs the train DataLoader (persistent workers) before skipping the loop.
- `strict=False` checkpoint load can silently skip mismatched keys (dangerous for A1 if `model.pth` exists).
- A2, A4, A5, H1, multi-seed, MCU latency: **not in the repository as runs**.

---

# PHASE 2 — A3 / U1 / Table II inconsistency

## The discrepancy (not a typesetting error)

| Source | Dense val | α=0.15 val | Implied drop |
| --- | ---: | ---: | ---: |
| Table II (`exp_struct_sweep`) | **93.36%** = 2868/3072 | **93.16%** | 0.20 pp |
| U1 (`/tmp/ablation_U2.log`) | **96.16%** = 2954/3072 | **95.38%** | 0.78 pp |

Same val **denominator** (3072), different **numerator**, **2.80 pp** on the dense baseline.

## Cause (traced)

1. **Table II did not use `--skip-train`.**  
   Command in `/tmp/struct_sweep.log`:
   `python train.py exp_struct_sweep --share-layers --enable-pruning --prune-mode structured --prune-amounts 0.1,0.15,0.2,0.3 --finetune-steps 2000 --max-acc-drop 0.005 --num-workers 2`  
   `results/exp_struct_sweep/data.txt` has **16 training epochs**. Epoch 1 val is already 2880/3072 (93.75%) — this is **continued training** from whatever `model.pth` existed that morning, not a fresh A3. Early stopping then fires; `sweep_pruning_amounts` prints `Baseline (unpruned) validation accuracy: 93.36%` and selects α=0.15 (drop 0.20 pp). Test “Original” 95.55% / C2 94.42% are this **continued-training parent**, then prune+INT8.

2. **U1/U2 used `--skip-train` on `checkpoints/dense_A3.pth`.**  
   `/tmp/ablation_U2.log`: `Baseline (unpruned) validation accuracy: 96.16%`, chosen α=0.15, val 95.38%, drop 0.78 pp. Test original 95.39% / U2 95.65% (`results/exp_U2/final_summary.md`).

3. **C0 is a third eval of a skip-train dense snapshot:** original 95.59% / INT8 95.62% (`results/exp_C0/`). Compatible with U2’s parent within ~0.2 pp test noise, **not** with Table II’s 93.36% val.

4. **Not explained by:** val augmentation (off), a different GSC version (both v1), or a different MFCC graph. **Partly amplified by:** unseeded TF/PyTorch, truncated overlapping val, and GPU vs CPU eval paths.

5. **α=0.15 is not known to be valid under the stated rule** once the parent is the skip-train dense checkpoint. U1’s own drop vs 96.16% is **0.78 pp > 0.5 pp** (U2 would not have been selected by the same ceiling; only one amount was tried). Table II’s 0.20 pp drop is measured against a **degraded** 93.36% baseline obtained after 16 extra epochs. **Preserving α=0.15 because the paper already prints it is not allowed.**

## Provenance table (real values)

Val split = kws_streaming `validation`, but only 6 overlapping batches (3072 scored examples, ≈517 unique clips). Test = official testing list, batch=1, aug off. Preprocess = TF MFCC. Seed = **unset**.

| Experiment | Source checkpoint | Seed | Train / fine-tune | Val acc (3072) | Test acc | Log / result |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A1 | none (from scratch; `model.pth` absent or shape-mismatch) | unset | 20 epochs, early stop | last epoch 94.73% (2909/3072 in `data.txt` epoch 20) | **94.87%** (INT8 same) | `results/exp_A1/`, `/tmp/ablation_A1.log` |
| A2 | — | — | **not run** | — | — | paper “planned” |
| A3 paper Table III | **continued `model.pth` inside `exp_struct_sweep`** (16 extra epochs) | unset | those 16 epochs | **93.36%** (2868/3072) at sweep time | **95.55%** | `results/exp_struct_sweep/final_summary.md` |
| A3 file `dense_A3.pth` | `checkpoints/dense_A3.pth` | unset | original tied train (epoch of first save not logged under `exp_A3/`) | **96.16%** when measured as U2 baseline | **95.39%** (U2 “Original”); **95.59%** (C0 “Original”) | U2 + C0 summaries |
| C0 | `model.pth` ← dense A3 snapshot, `--skip-train` | unset | none | not printed | 95.59% → INT8 **95.62%** | `results/exp_C0/` |
| C1 | Table II parent (continued A3) | unset | 2000 FT after structured 0.15 | **93.16%** | not separately exported | sweep table |
| C2 | same parent as Table II | unset | 2000 FT + INT8 | 93.16% val | **94.42%** | `results/exp_struct_sweep/` |
| U1 | `dense_A3.pth`, `--skip-train` | unset | 2000 FT unstructured 0.15 | **95.38%** | — (val only) | `/tmp/ablation_U2.log` |
| U2 | same as U1 | unset | 2000 FT + INT8 | 95.38% | **95.65%** | `results/exp_U2/` |

v2 (not used for Table II; recorded so it is not mixed in): A3_v2 test 94.09%/94.13% INT8; C0_v2 93.82%/93.93%; U2_v2 original 93.97% / pruned+INT8 94.99%; unstructured baseline val 94.14%. Structured v2 sweep was still running at audit time (baseline print 93.36% on that run’s val loader).

## Canonical A3 decision

**Canonical v1 dense A3 is `checkpoints/dense_A3.pth` (md5 `14c41d51d5eb5ac7ce96ee43ab177179`), evaluated with `--skip-train` and no extra training epochs.**

Reasons:

- It is the parent actually used for C0 and U1/U2.
- Table II’s parent is a **different weight vector** (16-epoch continuation). Using it as “the” A3 makes INT8-only, unstructured, and structured comparisons incommensurable.
- The selection rule must be applied to `quick_val` of **this** file, not to 93.36%.

**Until a `--skip-train` structured sweep from this file is completed, α=0.15 is unproven and C2 94.42% must not be described as the compression of canonical A3.**

Paper numbers that currently attach to the wrong parent (do not silently replace):

- Table II dense 93.36% / α=0.15 val 93.16% / C2 test 94.42% / A3 test 95.55%
- Abstract/conclusion 95.55% → 94.42% (−1.13 pp)

C0 95.62% and U2 95.65% stay associated with `dense_A3.pth`.

## Canonical structured sweep — completed 2026-09-03

Command (log: `results/exp_struct_sweep_canonical_A3/run.log`):

```bash
cp -a checkpoints/dense_A3.pth model.pth
python train.py exp_struct_sweep_canonical_A3 \
  --share-layers --skip-train --enable-pruning --prune-mode structured \
  --prune-amounts 0.1,0.15,0.2,0.3 --finetune-steps 2000 --max-acc-drop 0.005 \
  --num-workers 1 --version 1
```

Baseline val this run: **95.12%** (same overlapping 3072-example loader; not equal to U2’s 96.16% or Table II’s 93.36%).

| α | Stored params | Val acc | Drop vs 95.12% | ≤ 0.5 pp? |
| ---: | ---: | ---: | ---: | --- |
| 0.10 | 34,211 | 94.47% | 0.65 | no |
| 0.15 | 31,147 | 94.47% | 0.65 | no |
| **0.20** | **29,861** | **95.25%** | **−0.13** | **yes** |
| 0.30 | 27,823 | 94.53% | 0.59 | no |

**Selected: α=0.20** (most compressive amount that stayed under the ceiling). Paper Table II’s α=0.15 **fails** this rule on the canonical parent.

Test (`results/exp_struct_sweep_canonical_A3/final_summary.md`): dense re-eval **95.78%** → pruned+INT8 **94.51%**; 29,861 params; 78.98 KB; 16.277M FLOPs. INT8 checkpoint: `checkpoints/struct_canonical_0.20_quant.pt`.

Keep `results/exp_struct_sweep/` as the **legacy continued-training** run (α=0.15, test 94.42%). Do not overwrite it.

Val nondeterminism remains: 0.10 drop was 0.39 pp on the killed 94.50% baseline and 0.65 pp here. Selection is therefore **conditional on this loader pass**, not a stable optimum. Phase 7 (finer α) is still useful; Phase 3 should train new seeds rather than reuse this single pass.

---

# REFERENCE [10]

`paper/refs.bib` entry `zareprev` is `@unpublished` with note *“Fill in venue/title before submission.”* Repository metadata does not contain a DOI, conference name, or camera-ready title beyond that placeholder.

**REFERENCE_10_NEEDS_MANUAL_VERIFICATION**

Missing: venue, year as published, final title, pages/DOI. Do not invent them.

---

# Phase status

| Phase | Status |
| --- | --- |
| 0 Time budget | **done** (reductions declared above) |
| 1 Repo audit | **done** |
| 2 Provenance + canonical A3 | **documented**; canonical **re-sweep not yet run** (GPU busy / required) |
| 3–24 | **not started** (3 and 7 blocked on Phase 2 re-sweep) |
