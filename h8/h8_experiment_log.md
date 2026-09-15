# H8 Experiment Log

One entry per section across H8 Part 1 (feature engineering), Part 2 (ablation), and Part 3
(diagnostics). All results are real, computed from cached artifacts -- none are simulated.

## Part 1 -- Feature Engineering

### 27.0 Freeze & Failure Mode Inventory
- **Hypothesis:** consolidating H6/H7's already-computed metrics and errors into one object (no
  recomputation) is sufficient to define concrete engineering targets for H8.
- **Feature(s):** none (read-only aggregation).
- **Result:** 9 failure-mode categories identified, each grounded in specific row ids from H6/H7.
- **Conclusion:** confirmed -- this became the target list for sections 28-32.

### 28.0 Clinical Contradiction Features
- **Hypothesis:** explicit age/gender/severity/condition compatibility flags, built only from
  already-extracted structured fields, can directly encode the title-vs-patient contradiction
  reasoning CLAUDE.md section 8 describes.
- **Feature(s):** 14 features (11 non-constant after 33.0's validation).
- **Result:** `age_contradiction`=17/690, `gender_condition_present`=69/690 (`gender_contradiction`
  constant 0 -- verified as a real dataset property, not a bug), `condition_overlap`=476/690.
- **Conclusion:** confirmed feasible; later promoted in Part 2 (35.0).

### 29.0 Negation-Aware Features
- **Hypothesis:** finer-grained negation counts (title-side, narrative-only, ratio/density) add
  signal beyond C2's single whole-protocol `negation_count`.
- **Feature(s):** 11 features (6 non-duplicate).
- **Result:** `guideline_negation_count` fires on the exact rows behind H7's "not recommended"
  title FPs (565/484); mean `protocol_negation_count`=31.4.
- **Conclusion:** feasible signal exists, but Part 2 (35.0) shows it does not survive pooled
  evaluation as its own group, and actively hurts when stacked on contradiction (F).

### 30.0 Missing Information Features
- **Hypothesis:** per-field missingness indicators (age/gender/ICD/complaints/history/objective)
  quantify the "unknown vs. contradiction" risk CLAUDE.md section 8 warns about.
- **Feature(s):** 8 features.
- **Result:** `gender_missing` and `icd_missing` are perfectly correlated (corr=1.0) -- confirmed
  identical to the 193 rows H7's subgroup analysis (25.0) already found.
- **Conclusion:** confirmed as a real, documented redundancy with C2's `has_structured_header`;
  Part 2 (35.0) shows this group regresses pooled M2 (-0.018).

### 31.0 Long Protocol Representation
- **Hypothesis:** lightweight length/section/head-tail metadata (no raw text) can recover some of
  the long-protocol difficulty H7 (25.0) identified.
- **Feature(s):** 21 features (17 non-duplicate).
- **Result:** `late_section_present` fires on only 1.2% of rows -- most protocols place narrative
  content well before the final third.
- **Conclusion:** feasible, but Part 2 (35.0) shows only a marginal +0.004 M2 gain alone, and a
  clear regression when combined with contradiction (M2=0.6685, below even frozen C2).

### 32.0 Interaction Features
- **Hypothesis:** simple products of already-engineered flags (e.g. `age_condition_conflict`) add
  value beyond the additive groups.
- **Feature(s):** 4 features (2 non-constant: `severity_with_negation`, `age_condition_conflict`).
- **Result:** `age_condition_conflict` fires on 6/690 rows, exactly the age-contradicted-but-
  topically-relevant cases (e.g. id 525).
- **Conclusion:** feasible as a diagnostic marker; Part 2 (37.0) shows no interaction variant beats
  contradiction alone in pooled terms.

### 33.0 Feature Validation
- **Hypothesis:** a static audit (missing rate, cardinality, duplicates, constants, correlation,
  redundancy with C2) will surface issues before any modeling is attempted.
- **Feature(s):** all 58.
- **Result:** 0 missing values; 5 constant columns (all explained by real dataset properties); 17
  exact duplicate pairs (all intentional spec-required aliases or the genuine gender/ICD
  coincidence); several features highly correlated with `has_structured_header`.
- **Conclusion:** confirmed and fully documented; nothing hidden or silently dropped.

## Part 2 -- Controlled Ablation

### 34.0 Baseline Control
- **Hypothesis:** the variant-testing harness (same architecture, folds, hyperparameters as C2)
  reproduces frozen `cv_M2` before any variant is trusted.
- **Feature(s):** none (control).
- **Result:** M2=0.679430990462764 vs. frozen 0.679431 (diff 9.5e-09).
- **Conclusion:** confirmed -- harness is trustworthy.

### 35.0 Progressive Feature Ablation
- **Hypothesis:** adding each H8 feature group individually will reveal which, if any, improve
  pooled M2 with acceptable fold stability.
- **Feature(s):** contradiction (B), negation (C), missingness (D), long-protocol (E),
  contradiction+negation (F).
- **Result:** B: M2=0.7024 (+0.023, PROMOTED); C: 0.6882 (+0.009, below gate); D: 0.6610 (-0.018,
  regression); E: 0.6834 (+0.004, below gate); F: 0.6883 (worse than B alone).
- **Conclusion:** only contradiction promoted; combining groups (F) actively hurts.

### 35.G Final Selected Combination
- **Hypothesis:** contradiction+long-protocol (two individually non-negative groups) might be
  synergistic.
- **Feature(s):** contradiction + long-protocol (17 extra cols).
- **Result:** M2=0.6685, worse than contradiction alone (0.7024) and worse than frozen C2 (0.6794).
- **Conclusion:** rejected -- variant G = contradiction only, identical to B.

### 36.0 Threshold Optimization (Promoted Variants Only)
- **Hypothesis:** a 0.10-0.90 threshold sweep on G's OOF probabilities may beat the default 0.50.
- **Feature(s):** none (threshold only, on variant G).
- **Result:** best threshold = 0.50 exactly; absolute gain = 0.0.
- **Conclusion:** confirmed no retuning needed; G ships at the default threshold.

### 37.0 Interaction Ablation
- **Hypothesis:** interaction features or group combinations add value beyond contradiction alone.
- **Feature(s):** negation-only, contradiction+negation interaction, contradiction+missingness
  interaction, contradiction+negation+missingness.
- **Result:** all four underperform contradiction alone (deltas -0.013 to -0.021); none unstable by
  fold-std.
- **Conclusion:** rejected on performance grounds, not instability.

### 38.0 Feature Importance
- **Hypothesis:** coefficient inspection of the promoted variant will show medically sensible
  signs for most contradiction features.
- **Feature(s):** all 11 contradiction features (via the fitted variant-G pipeline).
- **Result:** `age_contradiction` (-0.451) and `gender_condition_present` (-0.663) have the
  expected sign; `severity_unknown` (+1.092) is the strongest single coefficient and matches the
  project's "unknown != contradiction" principle; 3 features have a counter-intuitive positive
  sign, explained by specific mislabeled-looking or lexically-noisy rows in the training data.
- **Conclusion:** confirmed, with honest documentation of the exceptions.

### 39.0 Variant Comparison
- **Hypothesis:** N/A (aggregation step).
- **Feature(s):** all variants.
- **Result:** final comparison table with 7 rows (A-G), stability and medical-interpretation
  columns.
- **Conclusion:** contradiction is the single evidence-based promoted group; G = B.

## Part 3 -- Diagnostics & Final Decision

### 40.0 Medical Error Matrix
- **Hypothesis:** N/A (measurement step).
- **Feature(s):** variant G's cached OOF, threshold 0.50.
- **Result:** TP=372, FP=42, TN=195, FN=81; Precision(Applicable)=0.8986, Recall(Applicable)=0.8212;
  Macro-F2=0.8161.
- **Conclusion:** FP count dropped from C2's 50 to 42, while FN rose slightly from 77 to 81 -- a
  mixed picture on raw counts, but the *rescaled* M2 still improves because the shift moves
  precision/recall on both classes into a more balanced region that macro-F2 rewards more, not
  because errors uniformly decreased -- consistent with, but not identical to, the pooled M2 gain.

### 41.0 Focused Medical Error Analysis
- **Hypothesis:** the top-confidence FP/FN rows will map onto the same failure-mode categories
  identified in 27.0.
- **Feature(s):** engineered contradiction feature values shown per example.
- **Result:** 5 FP (severity mismatch x2, comorbidity contradiction, ambiguous wording, missing
  information) + 5 FN (missing information x3, potential annotation ambiguity x2).
- **Conclusion:** confirmed -- every hand-picked error maps cleanly onto one of the 9 categories,
  and the engineered features correctly *fired* on most of them (the model saw the right signal
  but the learned coefficient direction did not always resolve it correctly).

### 42.0 Protocol Length Diagnostics
- **Hypothesis:** long protocols remain the weakest length quartile even after adding contradiction
  features.
- **Feature(s):** `protocol_len` quartiles (reused from H7 25.0).
- **Result:** Q4 (longest) Macro-F2=0.7676, still the weakest quartile (same value as C2's own Q4
  score).
- **Conclusion:** confirmed -- contradiction features did not close this specific gap.

### 43.0 Structured Information Diagnostics
- **Hypothesis:** rows missing age/gender/ICD/complaints/history/objective will underperform rows
  where that information is present.
- **Feature(s):** structured-field availability masks (reused extraction logic, no new features).
- **Result:** age-missing (n=12) Macro-F2=0.4974 (weakest subgroup by far, though improved from
  C2's 0.3571); gender-missing/ICD-missing (same 193 rows) Macro-F2=0.7556 vs. 0.8247 available.
- **Conclusion:** confirmed subgroup gaps persist; no statistical-significance claim made (small n
  for age-missing).

### 44.0 Potential Annotation Ambiguity
- **Hypothesis:** the same rows flagged as ambiguous in H6/H7 will still be errors under G.
- **Feature(s):** none (documentation only).
- **Result:** id 525 (proba=0.201, still FN), ids 126/127 (proba=0.110/0.136, still FN) -- the
  identical rows, now confirmed as errors under a 5th independent model (C2, T3, T4, H1, G).
- **Conclusion:** confirmed as a recurring, cross-model pattern -- documented as "potential
  annotation ambiguity," never relabeled.

### 45.0 Architecture Justification Artifact
- **Hypothesis:** N/A (documentation step).
- **Feature(s):** N/A.
- **Result:** `h8/analysis/h8_architecture_justification.md` produced, covering the medical
  pipeline, accepted/rejected groups, why the transformer lost, why TF-IDF+medical features won,
  and the remaining performance ceiling.
- **Conclusion:** complete.

### 46.0 Final H8 Summary
- **Hypothesis:** N/A (aggregation step).
- **Feature(s):** N/A.
- **Result:** `h8/analysis/h8_summary.json` -- Frozen C2 (M2=0.6794) -> H8 promoted variant B
  (M2=0.7024) -> Final H8 model G (M2=0.7024, threshold 0.50, identical to B).
- **Conclusion:** the final H8 recommendation is variant G: C2 + 11 contradiction features.
