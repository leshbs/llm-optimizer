# H24: evaluation fix, missing inputs, structural models

**Outcome:**
* **One real, reproducible improvement:** C0 = config G + ICD code +
  diagnosis text. It narrowly fails the strict pre-declared confirmation gate.
* **Promoted by user approval** as a documented judgement call (see the
  update at the end). The candidate is written; **nothing is submitted.**
* **The core problem (collapsed titles) is still unsolved.** Step D (an LLM
  feature) is prepared for Colab, because this machine has no GPU.

---

## Step A: evaluation harness (`src/stage2_h24_harness.py`)

The frozen single split could not detect gains under ~0.02 M2: config G
scores 0.694 ± 0.019 across 10 different splits, and one frozen fold sits at
0.544 against ~0.74 for the other four.

This protocol was fixed before any candidate was scored:

* 5 × 5 repeated stratified K-fold (seeds 0–4), with identical splits for
  every candidate, so every comparison is paired.
* Paired fold-level deltas, with the variance inflated by the Nadeau–Bengio
  correction for overlapping training sets.
* **Gate:** mean delta > 0, AND corrected one-sided p < 0.10, AND pooled M2
  improves in ≥ 4 of 5 repeats.
* Threshold fixed at 0.5 everywhere.

**Negative control:** G plus a pure-noise column scored **−0.012**, with 0/5
repeats improved, and was rejected. This also shows how much M2 at a fixed
threshold moves from tiny probability shifts.

## Step B: missing inputs (`src/stage2_h24_step_b.py`, `src/stage2_h24_features.py`)

Each candidate is G with one change.

| Candidate | M2 (5×5) | Δ vs G | Repeats + | p | Frozen split | Collapsed-title FN |
|---|---|---|---|---|---|---|
| G | 0.6993 | — | — | — | 0.7024 | 35.8 / 36 |
| B0 control: + noise | 0.6872 | −0.0121 | 0/5 | 0.905 | 0.6911 | 35.8 |
| **B1 + ICD code** | 0.7183 | **+0.0191** | **5/5** | 0.110 | 0.7037 | 31.6 |
| B2 + ICD category | 0.7145 | +0.0152 | 5/5 | 0.114 | 0.7108 | 35.4 |
| B3 + title × ICD | 0.7167 | +0.0175 | 3/5 | 0.234 | 0.6883 | 34.0 |
| B4 + ICD + title × ICD | 0.7161 | +0.0169 | 3/5 | 0.245 | 0.6912 | 30.6 |
| **B5 + diagnosis text** | 0.7147 | **+0.0154** | **5/5** | 0.170 | 0.6980 | 33.0 |
| B6 narrative → clean full text | 0.7069 | +0.0076 | 4/5 | 0.256 | 0.7065 | 35.6 |
| B7 + clean full text | 0.6955 | −0.0037 | 2/5 | 0.592 | 0.6910 | 34.6 |

In the "clean full text" variants, the cleaning removes only the lab-value
`Показатель:` lines. That takes the median protocol from 3,953 to 1,609
characters, and the maximum from 32,765 to 6,794.

## Step C: combined and structural models (`src/stage2_h24_step_c.py`)

| Candidate | M2 (5×5) | Δ vs G | Repeats + | p | FN | FP | Collapsed AUC | Collapsed FN |
|---|---|---|---|---|---|---|---|---|
| **C0 G + ICD + diagnosis text** | **0.7215** | **+0.0222** | **5/5** | **0.079 PASS** | 76.0 | 40.8 | 0.484 | 30.6 |
| C1 C0 + per-title copies | 0.7123 | +0.0131 | 4/5 | 0.337 | 74.4 | 44.0 | 0.478 | 28.0 |
| C2 CatBoost | 0.6827 | −0.0165 | 2/5 | 0.690 | 68.6 | 54.6 | 0.517 | 26.6 |
| C3 CatBoost + protocol id | 0.6962 | −0.0031 | 2/5 | 0.541 | 65.6 | 53.4 | 0.524 | 26.4 |
| C4 mean(C0, C3) | 0.7128 | +0.0135 | 3/5 | 0.317 | 69.0 | 47.4 | 0.486 | 29.0 |
| C5 C0, C = 0.3 | 0.6780 | −0.0212 | 0/5 | 0.792 | 74.4 | 52.0 | 0.483 | 32.2 |
| C5 C0, C = 3.0 | 0.7161 | +0.0168 | 3/5 | 0.241 | 77.0 | 41.4 | 0.461 | 30.0 |

* **Trees and per-title models** recover the most collapsed-title positives,
  but they pay for them in false positives. None beats C0.
* **C3 is the leak-free redo of H21-F.** CatBoost's ordered target statistics
  encode the protocol id without leakage. It no longer collapses (−0.003
  against H21-F's −0.147), which confirms that H21-F's result was an
  implementation artifact. It still doesn't help.
* **The default regularisation C = 1.0 is about right.**

## Confirmation on fresh seeds (`src/stage2_h24_confirm_build.py`)

C0 was assembled after seeing step B on seeds 0–4, so it was re-tested on
unused seeds 5–9:

| | M2 | Δ | Repeats + | p |
|---|---|---|---|---|
| G | 0.6888 | — | — | — |
| C0 | 0.7107 | +0.0219 | 5/5 | **0.136 (gate fails)** |

**The effect replicates:** same size (+0.022), same direction in all 10
repeats, and better on 39 of 50 folds (10 worse, 1 tie). It misses the
p < 0.10 bar because the Nadeau–Bengio correction is conservative: its
`n_test/n_train` term does not shrink with more repeats. Under the
pre-declared rule, no candidate was written.

**What C0 would change on test (computed in memory, not written):**
6 of 173 Stage 2 predictions, all 0 → 1. The positive rate would go from
0.543 to 0.578. Every test ICD code appears in train (115 of 173 test rows
have one). A refit of G reproduces the production Stage 2 labels exactly, so
the comparison is like for like.

## What remains unsolved

The collapsed titles (36 positives) are still the core problem:

* **Within-title AUC:** 0.40 → 0.48 for C0. That is still chance level.
* **False negatives:** 35.8 → 30.6.

These positives depend on free-text facts (pregnancy, upper-GI involvement,
cancer) that bag-of-words models can't carry from one title to another. The
next planned step is a validated open-source LLM feature (step D). It needs a
GPU, and this machine has none (torch CPU-only, no CUDA).

## Integrity

* The production submission `h17/submission_stage1_clean.csv` (sha
  `5cd35e4d…`) and checkpoint `653c2b36…` were asserted unchanged.
* No earlier artifacts were modified.
* The only change to shared code is the new `seeds=` parameter on
  `repeated_splits`, whose default is unchanged.

---

## Update: C0 promoted as a documented judgement call

The user approved promoting C0 despite the failed fresh-seed gate. The grounds
are listed in `src/stage2_h24_promote_c0.py` and `h24/candidate_manifest.json`.

* **Candidate:** `h24/submission_candidate_h24_c0.csv`
  (sha `367e14c98b9cd728…`). Stage 1 rows are identical to the H17 production
  file; 6 Stage 2 rows flip, all 0 → 1. **Not submitted.**
* **Notebook:** section 133 (`src/append_h24_section.py`, cells `h24_*`)
  rebuilds the candidate end to end and checks it row for row.
* **Production** `h17/submission_stage1_clean.csv` is unchanged.

## Step D: prepared, waiting for a GPU run

* **`colab/stage2_llm_feature.ipynb`** (generated by `src/build_llm_colab.py`):
  * computes a zero-shot Qwen2.5-7B-Instruct probability of "1 = applicable",
    read from next-token logits (no sampling);
  * 4-bit on a T4; per-row cache on Drive; model sha and prompt sha recorded;
  * text helpers copied verbatim and verified identical on all 863 protocols.
* **`src/stage2_h24_step_d_eval.py`** tests C0 + LLM against C0 on both seed
  sets and writes `h24/submission_candidate_h24_d1.csv` only if both pass.
  A dry run with a pure-noise "LLM" ran cleanly and was rejected
  (−0.004 and −0.010 against C0).
* **Prompt design note:** the plausibility rule and the postmenopause example
  in the prompt come from reading **training** labels during the H23 audit.
  No test information is used.

## Final outcome (2026-09-17)

* **C0 on the leaderboard:** the user submitted the C0 candidate and it scored
  **54.799**, below production's 55.168 (implied M2 ≈ 0.723 against 0.731).
  **Rejected.**
* **Step D (LLM feature):** zero-shot Qwen2.5-7B-Instruct.
  * Alone: AUC 0.735, collapsed-title AUC 0.515.
  * D1 = C0 + LLM against C0: Δ −0.011 (seeds 0–4) and −0.008 (seeds 5–9),
    2/5 repeats improved on each. **Rejected**; no D1 candidate was written.
  * Results: `h24/step_d_llm_results.*`.
* **Final model:** H17 Stage 1 + config G Stage 2 (production, 55.168), frozen.
  `solution.ipynb` (generated by `src/build_solution_notebook.py`) rebuilds it
  from the four CSVs alone. Its output is byte-identical to the production
  file (sha `5cd35e4d…`).
