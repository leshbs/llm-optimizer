# H12 Final Strategy Report

## 1. Which Stage1 approach wins?

**Undetermined -- PENDING.** Neither Probe B (`C1`-only) nor the LLM-Stage1 probe has been
submitted to the leaderboard (H12 Part 3, 104.0), so `M1_classical` and `M1_llm` are both `None`
and 105.0's comparison matrix correctly reports `winner = PENDING` rather than guessing from
offline confidence or CV. Qualitatively (H11 Part 3, question 3): the LLM catches population
qualifiers in titles that `C1` misses due to its corrupted training vocabulary, but the LLM itself
over-predicts 'Special' (67.9%) and is uncalibrated against ground truth -- neither
weakness is resolved by this report alone.

## 2. Which Stage2 approach wins?

**Undetermined -- PENDING**, for the same reason: Probe A (`G`-only) and the LLM-Stage2 probe are
both unsubmitted, and isolating `M2_classical`/`M2_llm` additionally requires Probe C's trivial-
baseline score (104.0's `M1_trivial_reference`). `G`'s predictions are the only Stage2 contribution
with any real leaderboard confirmation at all -- they are part of the actual 49.2/70 overall score
(H11 Part 3) -- so they remain the safer default until the LLM's Stage2 probe is measured.

## 3. Does hybrid outperform pure classical/LLM?

**Cannot be determined from leaderboard data yet**, but two real, already-measured findings narrow
the question: (a) all three hybrid/ensemble candidates prepared in H12 Part 3 (107.0) are
ready to submit, and (b) the rule-engine ensemble candidate turned out **byte-identical** to the
plain LLM-Stage2 swap on this 173-row test set -- the age/gender rule fired on 6 rows but changed 0
LLM predictions, since the LLM had already reached the same conclusion independently. This means,
at least on this data, the rule engine currently adds no *measurable* value beyond what the LLM
already catches -- a real, non-obvious result worth re-checking once real Stage2 leaderboard scores
exist and a larger disagreement set can be reviewed.

## 4. Was corruption recovery worthwhile?

**Not applicable this cycle.** 109.0's Stage1 corruption recovery pipeline never ran because
108.0's gate reports `PENDING_NO_PROBE_SCORE`, not `COLLAPSE_DETECTED` -- no leaderboard
evidence yet indicates `C1` has actually collapsed relative to its offline estimate
(M1=0.4648). No engineering time was spent on
reconstruction, so the 4-hour time-box was never at risk of being exceeded. The pipeline itself
(exact skeleton-matching against `test_stage1.csv`'s intact vocabulary) is fully built and ready if
Probe B's real score ever does confirm a collapse.

## 5. Recommended submission candidate for Day 2

Per PENDING_INSUFFICIENT_DATA (H12 Part 3, 106.0's decision matrix): *"No branch can be selected yet -- per the hard constraints this decision must be based entirely on observed leaderboard probe scores. Still missing: Stage1 (Probe B / C1 and the LLM-Stage1 probe leaderboard scores); Stage2 (Probe A / G, the LLM-Stage2 probe, AND Probe C for the M1_trivial reference -- see 104.0)."*

The checkpoint built in this part (110.0-112.0) is therefore **Classical** -- `C1` (Stage1) + `G`
(Stage2), unchanged from production, checksum `653c2b360bdf450d...`.

**RECOMMENDATION: SUBMIT.** This checkpoint carries zero regression risk: its content is identical
to the pipeline that already produced the real, known 49.2/70 leaderboard score (H11 Part 3), so
submitting it cannot make the leaderboard position worse, and it keeps a current entry in place
while the real blocking work happens elsewhere. That blocking work is unchanged from H11 Part 3's
own conclusion and is the actual Day 2 priority, ranked by cost/value:

1. **Submit the 5 prepared diagnostic probes** (`probe_stage2_only.csv`, `probe_stage1_only.csv`,
   `probe_trivial.csv`, `probe_llm_stage1.csv`, `probe_llm_stage2.csv`) -- this is the only action
   that can turn any of the `PENDING` values in this report (104.0's score table, 105.0's
   comparison matrix, 106.0's decision tree, 108.0's recovery gate) into a real, evidence-based
   answer, and none of it requires new engineering.
2. Once real scores land, simply re-run 104.0-113.0 -- no code changes are needed for the router to
   resolve to a real branch, select a real candidate (possibly switching away from Classical), and
   regenerate this report with actual findings instead of `PENDING` placeholders.
