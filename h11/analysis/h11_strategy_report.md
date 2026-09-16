# H11 Strategy Report

## 1. Did probe submissions isolate Stage1/Stage2 successfully?

**Partially.** The probe *infrastructure* succeeded completely: Probe A (`probe_stage2_only.csv`),
Probe B (`probe_stage1_only.csv`), and Probe C (`probe_trivial.csv`) were built, validated (12/12
integrity checks each, H11 Part 1), and are ready to submit. However, **Probe A and Probe B have
not yet been submitted to the leaderboard** -- only the final combined submission's overall score
(49.2/70) is available. Stage-level isolation therefore has not
yet been *exercised*, only prepared. Submitting the two probes remains the single highest-value next
action to complete this diagnosis.

## 2. Which stage explains leaderboard degradation?

**Neither -- no degradation was detected.** Offline expected overall score: 44.18/70
(Stage1 M1=0.4648, Stage2 M2=0.7024).
Actual leaderboard: 49.2/70. Gap: **+5.02 points**,
i.e. the real submission *outperforms* the offline estimate rather than underperforming it. This is
the opposite of a degradation pattern, so none of the collapse/gap failure branches apply.

## 3. Is LLM worth pursuing as a Stage1 replacement?

**Not as a direct drop-in replacement, but worth further investigation.** The zero-shot LLM and `C1`
disagree on a large share of titles (LLM predicts 'Special' on 67.9%
of test rows vs. `C1`'s 5.9%), and the LLM's high-confidence
disagreements (92.0) show it correctly identifying population qualifiers (age category, subgroup
terms) directly in the title text, which `C1` -- trained on the cp1251-corrupted vocabulary --
systematically under-detects. This is a real, structural weakness in `C1`, not noise. But the LLM's
own bias (over-predicting 'Special') is also uncalibrated for this specific label distribution and
untested against ground truth. **Recommended**: hand-label a small validation slice and
calibrate/threshold the LLM's confidence before considering any replacement or ensemble role.

## 4. Is LLM worth pursuing as a Stage2 assistant?

**Possibly, as an assistant/auditor, not a replacement -- with one important caveat.** The LLM's
93.6% 'Applicable' rate vs. `G`'s
54.3% reflects the LLM applying the project's own "unknown !=
contradiction" principle very literally (CLAUDE.md section 8) -- it essentially never says "Not
applicable" unless a structured field directly contradicts the title. However, cross-referencing
disagreements against genuine contradiction signals (`contradiction_count > 0`, 91.0) found only
**5 of 68 disagreements (7.4%)** occur on a contradiction-flagged row, actually *below* that flag's
12.1% base rate across all 173 test rows -- so disagreements are **not** concentrated where the
engineered features detect a contradiction; if anything the two signals agree slightly more often
there than elsewhere. This means the LLM and G disagree for a *different* reason than the
contradiction features (most likely the LLM's much higher baseline optimism), not because the LLM is
independently confirming or catching contradiction-feature edge cases. The audit-layer idea is still
worth testing, but on evidence, not on the (now-falsified) hypothesis that the two disagree
specifically on contradiction-flagged rows -- since G's offline M2 is already validated and the
LLM's real-world precision is unmeasured, any assistant role needs its own calibration study first.

## 5. Recommended H12 direction

Per the decision matrix (**Branch C**: LLM pathway becomes primary research direction.):

1. **Submit Probe A and Probe B** to the leaderboard to complete the stage-level attribution this
   report could not finish (highest priority, lowest cost).
2. **Treat the LLM pathway as the primary research direction for H12**, per the Plateau/no-
   degradation classification -- specifically:
   - Build a small human-labeled disagreement-review set from `high_value_disagreements.csv` to
     measure real LLM precision/recall on the cases where it disagrees with production.
   - Prototype an LLM-as-auditor layer for Stage2: flag G's predictions for manual/LLM review only
     on the marginal probability band (H9 56.0's ~0.44-0.56 zone) rather than all rows, to control cost.
   - Do **not** yet replace `C1` or `G` with the LLM directly -- neither model has been degraded,
     and the LLM's own calibration against ground truth is untested.
