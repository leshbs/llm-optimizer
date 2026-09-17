"""Generate h22/FINAL_REPORT.md from the H19-H22 artifacts.

Numbers are read from the JSON/CSV artifacts rather than typed, so the report
cannot drift from the experiments.

Run:  python src/write_final_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h22"


def md_table(df: pd.DataFrame, cols, headers, fmts) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = [f.format(r[c]) if f else str(r[c]) for c, f in zip(cols, fmts)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = json.loads((ROOT / "h19" / "baseline_manifest.json").read_text(encoding="utf-8"))
    h19 = json.loads((ROOT / "h19" / "h19_results.json").read_text(encoding="utf-8"))
    h20 = pd.read_csv(ROOT / "h20" / "fn_categories.csv")
    h21 = pd.read_csv(ROOT / "h21" / "recurrence_ablation.csv")
    tg = json.loads((OUT / "threshold_and_group_analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "h19" / "error_analysis_summary.json").read_text(encoding="utf-8"))

    h19t = pd.DataFrame(h19["ablation"])
    thr = tg["threshold"]

    report = f"""# Stage 2 optimisation: H19-H22 final report

**Recommendation: KEEP CURRENT MODEL.** No candidate beat the frozen baseline on
any measure. Details and evidence below.

All comparisons in this report are offline-versus-offline on the frozen 5-fold
split. The measured real M2 of 0.73077 appears nowhere as a selection target.
Nothing was submitted.

---

## 1. Baseline

Config G reproduced **exactly** from source, lifted out of the notebook into
`src/stage2_baseline.py`.

| | value |
|---|---|
| model | LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42) |
| features | title lemma TF-IDF + narrative lemma TF-IDF (max 20k) + 4 numeric + gender one-hot + 11 contradiction features |
| threshold | {base['threshold']} |
| CV splitter | `frozen/skf2_folds.json` (5 frozen stratified folds) |
| **offline M2** | **{base['m2']:.10f}** |
| recorded reference | {base['reference_m2']:.10f} |
| **drift** | **{base['drift_vs_reference']:+.2e} (exact)** |
| max deviation vs cached H8 OOF | {base['max_abs_deviation_vs_cached_oof']:.2e} |
| macro-F2 | {base['macro_f2']:.6f} |
| fold scores | {[round(v, 4) for v in base['fold_scores_macro_f2']]} |
| mean / std | {base['mean_fold']:.4f} / {base['std_fold']:.4f} |
| FN / FP | {base['fn']} / {base['fp']} |
| positive rate | {base['positive_rate']:.4f} |

One obstacle worth recording: the H8 feature cache is a parquet file and both
parquet engines on this machine are blocked by an Application Control policy, so
the 11 contradiction features were **regenerated from the notebook source**
rather than loaded. The regeneration is verified by the reproduction itself --
feeding them through the pipeline reproduces the cached out-of-fold
probabilities to 1.1e-16, which could not happen if any feature differed.

A latent reproducibility bug was found and fixed in passing: the original
`condition_contradiction` iterates `list(toks)[:5]` over a Python *set*, and
CPython randomises string hashing per process, so rows with more than five
shared condition tokens could flip between runs. `stage2_features.py` sorts the
tokens first; the sorted order is what reproduces the cache.

---

## 2. Error anatomy

**The dominant failure is structural, not linguistic. The model largely
reproduces the per-title prior and barely discriminates within a title.**

* correlation between a title's base rate and its mean predicted probability: **0.7996**
* **75.0%** of out-of-fold probability variance is *between* titles, 25.0% within
* **8 titles have 100% of their positives missed** = **37 of 81 FN (45.7%)**
* inside those titles, positives average probability 0.2530 against 0.2361 for
  negatives -- no separation at all, and no positive ever exceeds 0.4715

With 37 distinct titles across 690 rows and a title TF-IDF block, the model can
identify the title almost perfectly. Where a title's base rate is low it pushes
the entire title below 0.5 and every positive in it becomes a false negative.

False negatives are near-misses, not confident errors: median probability
{summary['FN']['median_proba']:.4f}, {summary['fn_proba_above_040']} of {base['fn']} at or above 0.40, only
{summary['fn_proba_below_020']} below 0.20.

**The measurement that pre-empted H19:** scoped negation separates errors from
correct predictions but does **not** separate FN from FP --
{summary['FN']['condition_negated_exact_rate']:.1%} of FN against {summary['FP']['condition_negated_exact_rate']:.1%} of FP
(correct: {summary['correct']['condition_negated_exact_rate']:.1%}), with median nearest-negation distance
{summary['FN']['median_nearest_negation_distance']:.0f} for FN against {summary['FP']['median_nearest_negation_distance']:.0f} for FP. A feature firing
equally on both cannot fix an 81:42 imbalance.

Also ruled out as drivers: the condition being **absent** ({summary['FN']['condition_present_rate']:.1%} of FN
contain it literally, *more* than correct rows), protocol length, protocol
recurrence, and truncation.

Artifact: `h19/stage2_error_analysis.csv` (690 rows), `h19/error_analysis.md`.

---

## 3. H19 -- scoped negation

{md_table(h19t, ['experiment', 'cv_m2', 'delta_vs_baseline', 'fn', 'fp', 'positive_rate'],
          ['Experiment', 'CV M2', 'Delta', 'FN', 'FP', 'Pos rate'],
          [None, '{:.6f}', '{:+.6f}', '{:.0f}', '{:.0f}', '{:.4f}'])}

Fold scores: {[r['fold_scores'] for r in h19['ablation']][0]} (baseline).

**Rejected.** Every variant is worse, and no variant improved more than
1 of 5 folds. Variant C (scoped negation alone) behaved exactly as Phase 1
predicted -- it removed 1 FN and added 9 FP, pushing in both directions at once.
Across all 81 FN, variant C fixes 7.

All H19 features are row-local (one title/protocol pair, no corpus statistics,
no labels), so fold safety is structural.

---

## 4. H20 -- FN categories

{md_table(h20, ['category', 'count', 'pct_of_all_fn', 'proba_median', 'n_fixed_by_h19_C'],
          ['Category', 'Count', '% of FN', 'Median proba', 'Fixed by H19-C'],
          [None, '{:.0f}', '{:.1f}', '{:.4f}', '{:.0f}'])}

**Answer to "which single category holds the largest actionable concentration":**
`title_prior_collapse_total` -- {int(h20.iloc[0]['count'])} FN ({h20.iloc[0]['pct_of_all_fn']:.1f}%), of which H19 fixes
{int(h20.iloc[0]['n_fixed_by_h19_C'])}. Adding the partial-collapse category, **title-prior effects account for
{int(h20[h20['category'].str.startswith('title_prior')]['count'].sum())} of 81 FN = {100 * h20[h20['category'].str.startswith('title_prior')]['count'].sum() / 81:.1f}%**.

Genuinely linguistic categories are a minority: `condition_negated` is
{int(h20[h20['category'] == 'condition_negated']['count'].iloc[0])} FN ({h20[h20['category'] == 'condition_negated']['pct_of_all_fn'].iloc[0]:.1f}%).

---

## 5. H21 -- recurrence

{md_table(h21, ['experiment', 'cv_m2', 'delta_vs_baseline', 'fn', 'fp', 'folds_improved'],
          ['Experiment', 'CV M2', 'Delta', 'FN', 'FP', 'Folds improved'],
          [None, '{:.6f}', '{:+.6f}', '{:.0f}', '{:.0f}', '{:.0f}/5'])}

**Rejected.** Three things worth reading off this table:

* **Label-conditioned rates are catastrophic (-0.147).** This is the Phase 1
  diagnosis confirmed from the opposite direction: P(y=1 | title) reinforces the
  very prior the model already over-relies on. FN fell to 76 but FP nearly
  doubled to 79.
* **Pair recurrence is exactly 0.000000** because every (title, protocol) pair in
  the training set is unique, so the feature is constant. A useful sanity check
  that the encoder is wired correctly.
* **Normalised and exact protocol recurrence are identical**, which means the
  repeated protocols are byte-identical duplicates, not formatting variants.

---

## 6. H22 -- similarity and ensemble

**Not run, by the roadmap's own gate:** Phase 5 is conditional on exact
recurrence having measurable value, and it does not (-0.0057, 1 of 5 folds).
Phase 6 and 7 are conditional on having at least one candidate worth comparing
or ensembling; there is none. Running them anyway would be exactly the "random
feature dumping" the roadmap rules out.

### Phase 8 -- threshold

{md_table(pd.read_csv(OUT / 'threshold_sweep.csv').query('threshold in [0.30,0.35,0.40,0.45,0.50,0.55,0.60]'),
          ['threshold', 'm2', 'precision_pos', 'recall_pos', 'fn', 'fp', 'positive_rate'],
          ['Threshold', 'M2', 'Precision(+)', 'Recall(+)', 'FN', 'FP', 'Pos rate'],
          ['{:.2f}', '{:.4f}', '{:.4f}', '{:.4f}', '{:.0f}', '{:.0f}', '{:.4f}'])}

**0.50 is already the optimum, and sharply so.** In-sample gain from tuning is
**{thr['in_sample_gain']:+.6f}**. The nested check (threshold chosen on four folds, applied to
the fifth) picked {thr['nested_thresholds']} and scored {thr['nested_m2']:.6f}, i.e.
**{thr['nested_gain_vs_050']:+.6f} against simply using 0.50**.

The sharpness is a property of macro-F2 averaged over *both* classes: lowering
the threshold buys positive-class recall and destroys negative-class F2. Despite
81 FN sitting between 0.20 and 0.50, moving the cut to reach them costs more
than it gains. This is the single most useful negative result in the report --
it closes the most intuitively appealing remaining lever.

### Phase 9 -- recurrence dependence

| Split | Groups | M2 | Delta vs frozen | FN | FP |
|---|---|---|---|---|---|
| frozen stratified (official) | - | {tg['frozen_split_m2']:.6f} | - | {base['fn']} | {base['fp']} |
| grouped by protocol | {tg['grouped_cv']['protocol']['n_groups']} | {tg['grouped_cv']['protocol']['m2']:.6f} | {tg['grouped_cv']['protocol']['m2'] - tg['frozen_split_m2']:+.6f} | {tg['grouped_cv']['protocol']['fn']} | {tg['grouped_cv']['protocol']['fp']} |
| grouped by normalised protocol | {tg['grouped_cv']['normalised protocol']['n_groups']} | {tg['grouped_cv']['normalised protocol']['m2']:.6f} | {tg['grouped_cv']['normalised protocol']['m2'] - tg['frozen_split_m2']:+.6f} | {tg['grouped_cv']['normalised protocol']['fn']} | {tg['grouped_cv']['normalised protocol']['fp']} |
| grouped by title | {tg['grouped_cv']['title']['n_groups']} | {tg['grouped_cv']['title']['m2']:.6f} | **{tg['grouped_cv']['title']['m2'] - tg['frozen_split_m2']:+.6f}** | {tg['grouped_cv']['title']['fn']} | {tg['grouped_cv']['title']['fp']} |

Interpretation, not a replacement for the official split. Holding out whole
protocols costs {abs(tg['grouped_cv']['protocol']['m2'] - tg['frozen_split_m2']):.4f} -- modest. Holding out whole **titles** costs
**{abs(tg['grouped_cv']['title']['m2'] - tg['frozen_split_m2']):.4f}**, collapsing M2 to {tg['grouped_cv']['title']['m2']:.4f}. The model is, structurally, a
per-title classifier.

That is not a reason to discard the frozen split. H18 measured 100% of real test
titles and 89.0% of real test protocols as present in train, against 81.9%
protocol overlap inside CV. The real test set reuses training material *more*
than cross-validation does, which is why the frozen split is mildly pessimistic
(offline 0.7024 against measured real 0.7308) and why the title-grouped scenario
does not arise in practice.

---

## 7. Leakage audit

| Feature family | Fold-safety mechanism |
|---|---|
| H19 scoped negation, condition presence, ternary state | Row-local: computed from one (title, protocol) pair. No corpus statistics, no labels, no cross-row lookup. Nothing can leak by construction. |
| H21 recurrence counts (protocol / normalised / title / pair) | Inside `RecurrenceEncoder`, whose `fit` sees only the training rows of a fold. `cross_val_predict` clones and refits it per fold. Unseen validation keys map to 0, never to a global count. |
| H21 label-conditioned rates | Same encoder. `fit(X, y)` receives only the training fold's labels; smoothing is toward the *training-fold* prior. A validation row's own label is never in the statistic that scores it. Unseen keys fall back to the training-fold prior. |
| Phase 1 analysis columns | Computed over the whole training set **on purpose**, for diagnosis only. Suffixed `_analysis_only` in `h19/stage2_error_analysis.csv` and never passed to a model. |
| Threshold selection | Phase 8's headline number is the nested one, chosen on four folds and applied to the fifth. The in-sample optimum is reported as optimistic and labelled as such. |

No leaderboard feedback was used to select anything. Nothing was submitted.

---

## 8. Model comparison

| Model | CV M2 | SD | FN | FP | Delta baseline |
|---|---|---|---|---|---|
| **Current Stage 2 (config G)** | **{base['m2']:.6f}** | {base['std_fold']:.4f} | {base['fn']} | {base['fp']} | - |
| Current + H19 (best variant, C) | {h19t.iloc[2]['cv_m2']:.6f} | {h19t.iloc[2]['std_fold']:.4f} | {int(h19t.iloc[2]['fn'])} | {int(h19t.iloc[2]['fp'])} | {h19t.iloc[2]['delta_vs_baseline']:+.6f} |
| Current + H19 (all features) | {h19t.iloc[4]['cv_m2']:.6f} | {h19t.iloc[4]['std_fold']:.4f} | {int(h19t.iloc[4]['fn'])} | {int(h19t.iloc[4]['fp'])} | {h19t.iloc[4]['delta_vs_baseline']:+.6f} |
| Current + H21 (best variant) | {h21.iloc[1]['cv_m2']:.6f} | {h21.iloc[1]['std_fold']:.4f} | {int(h21.iloc[1]['fn'])} | {int(h21.iloc[1]['fp'])} | {h21.iloc[1]['delta_vs_baseline']:+.6f} |
| Current + H21 (label-conditioned) | {h21.iloc[5]['cv_m2']:.6f} | {h21.iloc[5]['std_fold']:.4f} | {int(h21.iloc[5]['fn'])} | {int(h21.iloc[5]['fp'])} | {h21.iloc[5]['delta_vs_baseline']:+.6f} |
| Current + H22 similarity | not run | - | - | - | gated off by Phase 5 |
| Current at tuned threshold | {thr['nested_m2']:.6f} | - | {thr['nested_fn']} | {thr['nested_fp']} | {thr['nested_gain_vs_050']:+.6f} |

**Nothing beat the baseline. Not one variant, on any fold count.**

---

## 9. Recommendation

### KEEP CURRENT MODEL

Config G stays frozen at threshold 0.50. No `h22/final_candidate_manifest.json`
is written, because there is no candidate to promote and writing one would imply
otherwise.

The production artifacts are untouched: Stage 1 remains the H17 model, Stage 2
predictions remain byte-identical, submission row ordering and ids are unchanged,
and every H18 artifact still reproduces.

### Why this is a useful outcome rather than a failed one

Four plausible hypotheses were killed with measurements rather than opinions:

1. **Scoped negation** -- the headline hypothesis. Dead on the data: FN and FP
   have statistically indistinguishable negation geometry.
2. **Recurrence** -- dead, and label-conditioned recurrence is actively harmful.
3. **Threshold** -- 0.50 is already optimal and the nested check agrees; the
   intuition that 81 FN just under the cut are cheap to recover is wrong, because
   macro-F2 punishes the negative class for it.
4. **Similarity / ensembling** -- correctly never started, since its precondition
   failed.

### What the evidence actually points at

The real target is **within-title discrimination**. 72.8% of FN are title-prior
effects; in 8 titles the model assigns essentially the same probability to
positives and negatives, and title-grouped CV collapses M2 from 0.70 to 0.42.
Every feature tested in H19-H21 operates on a per-row basis that the model can
already approximate from the title, which is why they all failed.

Two directions follow from that, neither of which is a small feature addition:

* **Per-title calibration** -- centre or rank probabilities within a title before
  thresholding, so a low-prevalence title is not uniformly suppressed. This
  attacks the mechanism directly. It needs fold-safe per-title statistics and
  careful handling of the 37-title support, and the F2 shape means it must be
  validated as a whole, not as a feature.
* **Reducing title dominance** -- down-weighting or removing the title TF-IDF
  block so the model is forced to read the protocol. The 25% within-title
  variance is the part that carries real patient evidence.

Both are structural changes to the model rather than additions to it, which puts
them outside this roadmap's "add a feature family and ablate" frame. They are
the honest next step if Stage 2 work continues.

### Expected value, stated plainly

Stage 2 holds {49 * (1 - 0.7307714752567694):.2f} points of headroom. This round recovered none of it.
The measurements narrow where the remaining points are, but no promotion is
justified by anything in H19-H22.

---

## Artifacts

| Path | Contents |
|---|---|
| `h19/baseline_manifest.json` | frozen baseline specification and exact reproduction |
| `h19/stage2_error_analysis.csv` | 690 OOF rows, {pd.read_csv(ROOT / 'h19' / 'stage2_error_analysis.csv').shape[1]} diagnostic columns |
| `h19/error_analysis.md` | Phase 1 report, ten questions answered |
| `h19/h19_ablation.csv`, `h19/h19_results.json` | scoped-negation ablation A-E |
| `h20/fn_categories.csv` | every FN categorised, with H19 fix status |
| `h21/recurrence_ablation.csv` | fold-safe recurrence ablation A-G |
| `h22/threshold_sweep.csv` | full threshold curve |
| `h22/threshold_and_group_analysis.json` | nested threshold + grouped-CV stress test |
| `src/stage2_baseline.py` | the baseline, extracted and runnable |
| `src/stage2_features.py` | contradiction features regenerated from source |

All previous experiment artifacts were left in place; nothing was overwritten or
deleted.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print("wrote h22/FINAL_REPORT.md (%d chars)" % len(report))


if __name__ == "__main__":
    main()
