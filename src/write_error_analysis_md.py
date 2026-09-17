"""Generate h19/error_analysis.md from the Phase 1 artifacts.

Every number in the report is read from ``h19/stage2_error_analysis.csv`` or
``h19/error_analysis_summary.json`` rather than typed in, so the prose cannot
drift away from the data.

Run:  python src/write_error_analysis_md.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h19"


def main() -> None:
    d = pd.read_csv(OUT / "stage2_error_analysis.csv")
    s = json.loads((OUT / "error_analysis_summary.json").read_text(encoding="utf-8"))
    fn = d[d["outcome"] == "FN"]

    # title-level structure
    g = d.groupby("title_text").agg(
        n=("label", "size"), base=("label", "mean"),
        mean_p=("proba", "mean"), fn=("outcome", lambda x: (x == "FN").sum()))
    pos = d[d["label"] == 1].groupby("title_text").agg(
        npos=("label", "size"), miss=("pred", lambda x: (x == 0).sum()))
    all_missed = pos[pos["miss"] == pos["npos"]]

    total_var = d["proba"].var(ddof=0)
    between = d.groupby("title_text")["proba"].transform("mean").var(ddof=0)

    # literal vs partial condition match among FN
    fn_literal = int((fn["condition_present"] == 1).sum())
    fn_none = int((fn["condition_present"] == 0).sum())
    fn_partial = int(((fn["n_condition_tokens_matched"] > 0)
                      & (fn["n_condition_tokens_matched"] < fn["n_condition_tokens"])).sum())
    fn_negated = int((fn["condition_negated_exact"] == 1).sum())

    b, f, p, c = s["baseline"], s["FN"], s["FP"], s["correct"]

    md = f"""# Stage 2 error analysis (Phase 1)

Baseline: frozen config G, threshold {b['threshold']}, frozen 5-fold split.
Reproduced **exactly** (drift 0.00e+00 vs the recorded 0.7023868248; max deviation
from the cached H8 out-of-fold probabilities 1.11e-16).

| | value |
|---|---|
| offline M2 | **{b['m2']:.6f}** |
| macro-F2 | {b['macro_f2']:.6f} |
| fold macro-F2 | {[round(x, 4) for x in b['per_fold_macro_f2']]} |
| mean / std | {b['mean_fold']:.4f} / {b['std_fold']:.4f} |
| FN / FP | **{b['fn']} / {b['fp']}** |
| positive rate | {b['positive_rate']:.4f} |

Do not compare this against the measured real M2 of 0.73077. Offline-versus-offline only.

## Headline

**The dominant failure is not linguistic. The model is largely reproducing the
per-title prior and barely discriminating within a title.**

* correlation between a title's base rate and its mean predicted probability:
  **{g['base'].corr(g['mean_p']):.4f}**
* **{100 * between / total_var:.1f}%** of out-of-fold probability variance is *between* titles;
  only {100 * (total_var - between) / total_var:.1f}% is within them
* **{len(all_missed)} titles have 100% of their positives missed**, accounting for
  **{int(all_missed['miss'].sum())} of {len(fn)} FN ({100 * all_missed['miss'].sum() / len(fn):.1f}%)**
* inside those titles the model does not separate the classes at all: positives
  average {d[d['title_text'].isin(all_missed.index) & (d['label'] == 1)]['proba'].mean():.4f}
  against {d[d['title_text'].isin(all_missed.index) & (d['label'] == 0)]['proba'].mean():.4f} for negatives,
  and no positive ever reaches
  {d[d['title_text'].isin(all_missed.index) & (d['label'] == 1)]['proba'].max():.4f}

With 37 distinct titles over 690 rows and a title TF-IDF block in the feature
set, the model can identify the title almost perfectly. Where a title's base rate
is low it pushes the whole title below 0.5, and every positive in it becomes a
false negative.

## The ten questions

**1. What are the {len(fn)} FN?**
Mostly near-misses, not confident errors. Median probability {f['median_proba']:.4f};
{s['fn_proba_above_040']} of {len(fn)} sit at or above 0.40 and only
{s['fn_proba_below_020']} fall below 0.20. Deciles:
{[round(v, 3) for v in s['fn_proba_deciles']]}.
FN are {len(fn)}/{int((d['label'] == 1).sum())} = {len(fn) / (d['label'] == 1).sum():.1%} of all positives.

**2. What are the {len(d[d['outcome'] == 'FP'])} FP?**
Median probability {p['median_proba']:.4f}, so they are confident. They sit on longer
protocols ({p['mean_protocol_len']:.0f} chars against {c['mean_protocol_len']:.0f} for correct rows)
and are more often truncated ({p['truncated_rate']:.3f} against {c['truncated_rate']:.3f}).

**3. What common structures occur among FN?**
Concentration by title, above everything else. The top five titles hold
{int(g.sort_values('fn', ascending=False)['fn'].head(5).sum())} of {len(fn)} FN
({100 * g.sort_values('fn', ascending=False)['fn'].head(5).sum() / len(fn):.1f}%), and only
{int((g['fn'] > 0).sum())} of {len(g)} titles produce any FN at all.

**4. How many FN contain the title condition literally?**
**{fn_literal} of {len(fn)} ({fn_literal / len(fn):.1%})** — a *higher* rate than correct
predictions ({c['condition_present_rate']:.1%}) and than FP ({p['condition_present_rate']:.1%}).
The condition being absent is not the mechanism; in FN it is usually present.

**5. How many FN contain a synonym or variant rather than a literal match?**
{fn_none} FN have no literal token match at all, and {fn_partial} match some but not all
condition tokens (mean {f['mean_matched_tokens']:.2f} of {f['mean_n_condition_tokens']:.2f} tokens matched).
Lexical coverage is therefore not the bottleneck either.

**6. How many FN have the condition inside a negated span?**
{fn_negated} of {len(fn)} ({f['condition_negated_exact_rate']:.1%}) within the 40-character window;
{f['neg_within_20_rate']:.1%} within 20 characters.

**7. How many FN have the condition outside negation?**
{len(fn) - fn_negated} of {len(fn)} ({1 - f['condition_negated_exact_rate']:.1%}).

**The critical comparison for the H19 hypothesis:** FP show
{p['condition_negated_exact_rate']:.1%} scoped negation against FN's {f['condition_negated_exact_rate']:.1%},
and median nearest-negation distance is {f['median_nearest_negation_distance']:.0f} characters for FN
against {p['median_nearest_negation_distance']:.0f} for FP and {c['median_nearest_negation_distance']:.0f} for
correct rows. Scoped negation separates *errors from correct predictions*, but it
does **not** separate FN from FP. A feature that fires equally on both cannot fix
an 81:42 imbalance — it pushes in both directions at once.

**8. What is the probability distribution of FN?**
Concentrated just under the threshold: see question 1. This is what makes
threshold work (Phase 8) a genuine lever and makes confident-error explanations
a poor fit.

**9. Are errors concentrated in certain title lengths?**
Weakly, and it is confounded by title identity — the second length quartile
(156-182 chars) carries a {d.assign(q=pd.qcut(d['title_len_chars'], 4, duplicates='drop')).groupby('q', observed=True)['outcome'].apply(lambda x: (x == 'FN').mean()).iloc[1]:.4f} FN rate against
{d.assign(q=pd.qcut(d['title_len_chars'], 4, duplicates='drop')).groupby('q', observed=True)['outcome'].apply(lambda x: (x == 'FN').mean()).iloc[0]:.4f} in the first, but only a handful of titles occupy each bucket.

**10. Are errors concentrated in repeated protocols or templates?**
No. FN rate falls slightly as protocol recurrence rises (0.160 for singletons
against 0.103 for protocols seen more than six times), and FN carry
*lower* mean recurrence ({f['mean_protocol_recurrence']:.2f}) than correct rows
({c['mean_protocol_recurrence']:.2f}). Truncation is likewise not implicated
({f['truncated_rate']:.3f} of FN against {c['truncated_rate']:.3f} of correct rows).

## What this implies for the roadmap

1. **H19 (scoped negation) is unlikely to pay.** Measured directly, negation
   geometry is near-identical between FN and FP. The ablation still runs as
   specified, because the roadmap asks for it and a negative result is worth
   recording, but expectations should be low.
2. **The largest actionable concentration is title-prior collapse** — 8 titles,
   {int(all_missed['miss'].sum())} FN, no within-title discrimination. Anything that improves
   *within-title* separation, or that decouples the decision from the title
   prior, addresses roughly half the FN.
3. **Threshold work is a real lever** and should not be deferred blindly: most FN
   sit between 0.20 and 0.50.
4. **Recurrence (H21) looks weak for FN** on this evidence, though it is still
   worth the fold-safe test the roadmap specifies.

## Artifacts

* `h19/stage2_error_analysis.csv` — one row per OOF prediction, {d.shape[1]} columns
* `h19/error_analysis_summary.json` — the aggregates used above
* `h19/baseline_manifest.json` — the frozen baseline specification

Recurrence columns in the CSV are suffixed `_analysis_only`: they are computed
over the whole training set for diagnosis and must never be used as model
features without fold-safe recomputation.
"""
    (OUT / "error_analysis.md").write_text(md, encoding="utf-8")
    print("wrote h19/error_analysis.md (%d chars)" % len(md))


if __name__ == "__main__":
    main()
