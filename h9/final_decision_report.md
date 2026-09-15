# H9 Final Decision Report -- Stage 2 Model Promotion

## Q1: Is G quantitatively better than Frozen C2?

**Yes.** Pooled OOF at threshold 0.50: Raw Macro-F2 0.8057 -> 0.8161 (+0.0103), M2 0.6794 -> 0.7024
(**+0.0230**), Accuracy 0.8159 -> 0.8217, Precision(macro) +0.0071, Recall(macro) +0.0125. The gain
is reproducible to floating-point precision (H9 48.0: diff 9.5e-09 for C2, 0.0 for G against their
respective cached reference values).

## Q2: Is the improvement robust across folds and subgroups?

**Mostly yes, with one honestly-flagged caveat.** 3 of 5 folds improve (mean delta +0.023, std
0.035); the 2 degraded folds were already C2's strongest folds, and the single largest gain lands
on C2's weakest fold -- a reassuring pattern of variance, not concerning. Across 12 subgroups
(protocol length quartiles, structured-field availability, contradiction presence), **10 improve,
2 hold exactly flat (`age missing`, n=12; `protocol_len Q4`), and none regress**. No subgroup
reverses the pooled direction. Applicable-class recall dipped by a small but real -0.0088 (0.8300
-> 0.8212) -- within this project's tolerance, but explicitly carried forward as a caveat given
CLAUDE.md's stated conservatism principle for this class.

## Q3: Is the improvement clinically interpretable?

**Yes, with documented exceptions.** The contradiction feature audit (53.0) found 7 of 11 features
have the medically expected coefficient sign, including the two most directly safety-relevant ones
(`age_contradiction`, `contradiction_density`). 4 features show a counter-intuitive sign
(`severity_contradiction`, `condition_overlap`, `condition_contradiction`, `contradiction_count`);
of these, `condition_overlap`'s coefficient is negligible in magnitude (-0.037), and the other
three are explained by specific training rows (H8's 41.0 error analysis, H9's 54.0 focused audit
on `severity_unknown`/`severity_contradiction`) where the *general* medical policy behind the
feature (e.g. "unknown != contradiction") is correct but a fitted linear coefficient necessarily
reflects the *net* effect across all rows where the feature fires, not a per-row guarantee. This is
a known, documented limitation of a single global linear weight per feature -- not evidence of a
broken or clinically nonsensical feature.

The two focused audits (54.0 `severity_unknown`, 55.0 annotation ambiguity) both concluded that the
hardest remaining errors are either (a) rows where **both C2 and G were already wrong** before H8's
features existed (619, 722 -- G amplifies, does not create, these errors), or (b) genuine
**potential annotation ambiguity** (126/127, where all 5 independently-trained models -- C2, G, T2,
T3, T4 -- agree on a prediction that contradicts the label). One correction to prior framing: id
525 was previously described as a 5-model consensus error, but this audit's cross-model check
(55.0) shows **T2 alone predicts it correctly** (proba=0.805) -- so 525 is more accurately
classified as "mixed model agreement, inconclusive" rather than a unanimous annotation-ambiguity
case. No label was changed as a result of this finding; it is reported here as a correction to the
narrative, not the ground truth.

## Final Feature Sanity (57.0)

All 9 checks passed: no constant or duplicate features among the 11 promoted contradiction
columns, one documented near-perfect alias (`condition_overlap`/`condition_unknown`, corr=-1.0,
not a failure), no fold-partition leakage, zero overlap between the feature-engineering id set and
`test_stage2.csv` ids, no feature correlates with the label above the leakage threshold, threshold
confirmed unchanged at 0.50, fold mapping identical to C2's, and the seed (`RANDOM_STATE=42`)
matches the frozen `c2_summary.json` exactly.

## Decision

Evaluated against the three decision-gate options:

- **REJECT_G** criteria (reproducibility failure, leakage, unstable fold improvement, clinically
  invalid contradiction behavior) -- **none apply.** Reproducibility passed exactly; the sanity
  check found zero leakage; fold improvement, while not unanimous, is net-positive and
  directionally sound (larger gains where C2 was weakest); and the 4 counter-intuitive coefficients
  are explained, documented exceptions, not invalid or nonsensical behavior.
- **REVISE_G** criteria (localized feature bug, reproducibility passes, rerun only affected
  feature audit) -- **does not apply as a blocking condition.** `severity_contradiction`'s sign is
  a documented modeling limitation (a single global linear coefficient cannot be locally correct on
  every row), not a bug in the feature's extraction logic (53.0/54.0 confirm the lexical detection
  itself is correct on every audited row) -- it does not warrant blocking promotion, only continued
  monitoring.
- **PROMOTE_G** criteria (reproducibility passed, folds mostly improve, recall acceptable,
  contradiction features behave correctly, no leakage) -- **all satisfied.**

### DECISION: PROMOTE_G

G (C2 + 11 clinical contradiction features, threshold 0.50) is promoted as the final Stage 2
production model, replacing frozen C2's numeric feature set. This is a controlled, fully-audited,
+0.023 M2 improvement with broad-based (not single-subgroup-driven) robustness and complete
sanity-check clearance.

## Production Checklist (documented, not executed in this stage)

Per this stage's hard constraint of **no test-set inspection**, the following steps are
recorded as the next stage's scope, not run here:

1. **Freeze G** -- persist the exact variant-G pipeline definition (structured_preprocessor +
   `h8_numeric` block over the 11 contradiction columns + `LogisticRegression`) as
   `frozen/g_pipeline.joblib`, alongside a `frozen/g_summary.json` mirroring `c2_summary.json`'s
   format (name, cv_M2, threshold, random_state, n_splits).
2. **Refit on full training data** -- fit the frozen G pipeline once on all 690 rows (same
   discipline as C2's own final refit, section 7.2/10.0), never on a fold subset.
3. **Generate test predictions** -- apply the refit pipeline to `test_stage2.csv` (173 rows),
   producing `preds_test_s2_g`.
4. **Apply threshold 0.50** -- no retuning; G's own optimal threshold (H8 36.0) is already 0.50.
5. **Create `submission.csv`** -- concatenate Stage 1 and Stage 2 predictions in the existing
   `stage, id, label` format, replacing the current C2-based Stage 2 predictions with G's.

None of these five steps were executed in H9 -- they are scoped explicitly to a future,
deployment-focused stage so this stage's "no test-set inspection" constraint is respected in full.
