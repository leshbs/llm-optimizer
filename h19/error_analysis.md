# Stage 2 error analysis (Phase 1)

Baseline: frozen config G, threshold 0.5, frozen 5-fold split.
Reproduced **exactly** (drift 0.00e+00 vs the recorded 0.7023868248; max deviation
from the cached H8 out-of-fold probabilities 1.11e-16).

| | value |
|---|---|
| offline M2 | **0.702387** |
| macro-F2 | 0.816074 |
| fold macro-F2 | [0.7448, 0.8299, 0.8361, 0.8344, 0.8344] |
| mean / std | 0.8159 / 0.0398 |
| FN / FP | **81 / 42** |
| positive rate | 0.6000 |

Do not compare this against the measured real M2 of 0.73077. Offline-versus-offline only.

## Headline

**The dominant failure is not linguistic. The model is largely reproducing the
per-title prior and barely discriminating within a title.**

* correlation between a title's base rate and its mean predicted probability:
  **0.7996**
* **75.0%** of out-of-fold probability variance is *between* titles;
  only 25.0% is within them
* **8 titles have 100% of their positives missed**, accounting for
  **37 of 81 FN (45.7%)**
* inside those titles the model does not separate the classes at all: positives
  average 0.2530
  against 0.2361 for negatives,
  and no positive ever reaches
  0.4715

With 37 distinct titles over 690 rows and a title TF-IDF block in the feature
set, the model can identify the title almost perfectly. Where a title's base rate
is low it pushes the whole title below 0.5, and every positive in it becomes a
false negative.

## The ten questions

**1. What are the 81 FN?**
Mostly near-misses, not confident errors. Median probability 0.3358;
25 of 81 sit at or above 0.40 and only
9 fall below 0.20. Deciles:
[0.026, 0.195, 0.232, 0.268, 0.301, 0.336, 0.368, 0.403, 0.444, 0.48, 0.497].
FN are 81/453 = 17.9% of all positives.

**2. What are the 42 FP?**
Median probability 0.6660, so they are confident. They sit on longer
protocols (12833 chars against 11632 for correct rows)
and are more often truncated (0.310 against 0.212).

**3. What common structures occur among FN?**
Concentration by title, above everything else. The top five titles hold
40 of 81 FN
(49.4%), and only
20 of 37 titles produce any FN at all.

**4. How many FN contain the title condition literally?**
**76 of 81 (93.8%)** — a *higher* rate than correct
predictions (88.5%) and than FP (83.3%).
The condition being absent is not the mechanism; in FN it is usually present.

**5. How many FN contain a synonym or variant rather than a literal match?**
5 FN have no literal token match at all, and 76 match some but not all
condition tokens (mean 1.43 of 6.52 tokens matched).
Lexical coverage is therefore not the bottleneck either.

**6. How many FN have the condition inside a negated span?**
25 of 81 (30.9%) within the 40-character window;
14.8% within 20 characters.

**7. How many FN have the condition outside negation?**
56 of 81 (69.1%).

**The critical comparison for the H19 hypothesis:** FP show
33.3% scoped negation against FN's 30.9%,
and median nearest-negation distance is 64 characters for FN
against 65 for FP and 100 for
correct rows. Scoped negation separates *errors from correct predictions*, but it
does **not** separate FN from FP. A feature that fires equally on both cannot fix
an 81:42 imbalance — it pushes in both directions at once.

**8. What is the probability distribution of FN?**
Concentrated just under the threshold: see question 1. This is what makes
threshold work (Phase 8) a genuine lever and makes confident-error explanations
a poor fit.

**9. Are errors concentrated in certain title lengths?**
Weakly, and it is confounded by title identity — the second length quartile
(156-182 chars) carries a 0.2022 FN rate against
0.0511 in the first, but only a handful of titles occupy each bucket.

**10. Are errors concentrated in repeated protocols or templates?**
No. FN rate falls slightly as protocol recurrence rises (0.160 for singletons
against 0.103 for protocols seen more than six times), and FN carry
*lower* mean recurrence (5.47) than correct rows
(6.17). Truncation is likewise not implicated
(0.173 of FN against 0.212 of correct rows).

## What this implies for the roadmap

1. **H19 (scoped negation) is unlikely to pay.** Measured directly, negation
   geometry is near-identical between FN and FP. The ablation still runs as
   specified, because the roadmap asks for it and a negative result is worth
   recording, but expectations should be low.
2. **The largest actionable concentration is title-prior collapse** — 8 titles,
   37 FN, no within-title discrimination. Anything that improves
   *within-title* separation, or that decouples the decision from the title
   prior, addresses roughly half the FN.
3. **Threshold work is a real lever** and should not be deferred blindly: most FN
   sit between 0.20 and 0.50.
4. **Recurrence (H21) looks weak for FN** on this evidence, though it is still
   worth the fold-safe test the roadmap specifies.

## Artifacts

* `h19/stage2_error_analysis.csv` — one row per OOF prediction, 45 columns
* `h19/error_analysis_summary.json` — the aggregates used above
* `h19/baseline_manifest.json` — the frozen baseline specification

Recurrence columns in the CSV are suffixed `_analysis_only`: they are computed
over the whole training set for diagnosis and must never be used as model
features without fold-safe recomputation.
