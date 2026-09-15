# Architecture Justification -- Stage 2 (Applicability Assessment)

## Medical Pipeline

The final Stage 2 pipeline is: raw `protocol_text` -> `ProtocolFieldExtractor` (regex-based
structured field extraction: age, gender, negation count, protocol length, narrative text) ->
`lemmatize_ru` (pymorphy3 morphological normalization) -> two TF-IDF vectorizers (title, narrative)
+ a numeric `ColumnTransformer` block -> `LogisticRegression`. On top of this frozen C2
architecture, H8 adds one additional numeric block: 11 **clinical contradiction features**
(age/gender/severity/condition compatibility flags), computed by comparing structured title-side
rules (`extract_age_bound`, `gender_requirement`, explicit severity keyword lists) against the
same structured/narrative fields the extractor already produces. No new NLP model, no changed
preprocessing, no changed labels or folds -- the contradiction features are a pure feature-space
extension of the exact frozen C2 pipeline.

## Accepted Feature Groups

**Contradiction features (11 columns)** were the only group promoted, delivering **+0.023 M2**
(0.6794 -> 0.7024) with *improved* fold-to-fold stability (std 0.0398 vs. 0.0498 for frozen C2).
They improve C2 because they make explicit, as direct numeric signals, exactly the kind of
title-vs-patient compatibility reasoning CLAUDE.md's own problem framing (section 8) asks for --
age bound vs. patient age, required gender vs. patient gender (with pregnancy treated as its own
condition, never inferred from gender alone), title severity vs. narrative severity, and lexical
condition-token overlap/contradiction. Feature importance confirms the model uses them sensibly:
`age_contradiction` and `gender_condition_present` have the medically expected sign, and
`severity_unknown`'s strong positive coefficient (+1.09, the single largest contradiction
coefficient) shows the model independently recovered the project's "unknown != contradiction"
principle without being told to.

## Rejected Feature Groups

- **Negation features** (finer-grained negation counts/ratios, title-side negation): +0.009 M2,
  below the +0.01 promotion bar. C2's own existing `negation_count` already captures most of this
  signal; the additional granularity did not add reliable pooled value, and combining it with
  contradiction (variant F) *reduced* M2 to 0.6883 -- clear evidence of interference, not
  complementarity.
- **Missingness features** (per-field missing indicators): -0.018 M2, a regression. Several of
  these features are near-perfectly collinear with C2's existing `has_structured_header`
  (`gender_missing`/`icd_missing` correlate at exactly -1.0 with it), so they mostly duplicate
  existing signal while adding noise to a linear model.
- **Long-protocol features** (length/section/head-tail metadata): +0.004 M2, below the promotion
  bar; combined with contradiction it actively hurt (M2=0.6685, worse than even frozen C2).
- **Interaction features** (`severity_with_negation`, `age_condition_conflict` combined with other
  groups): all four tested combinations underperformed contradiction alone; none were unstable by
  fold-std, so the rejection is a genuine performance regression, not noise.

## Why Transformer Lost

H6 (sections 12-18) fine-tuned `cointegrated/rubert-tiny2` on three input representations (T2:
title+narrative, T3: head+tail, T4: structured context) and one hybrid design (H1: transformer
embedding + C2's engineered features). **All four transformer variants scored below frozen C2**
(T2 M2=0.4961, T3=0.4774, T4=0.5397 -- the best of the four -- H1=0.4543), a gap of 0.14-0.23
absolute M2. The best transformer variant (T4) succeeded specifically by *rendering* structured
information as text for the model to read, confirming that the signal is there -- but a
29M-parameter model fine-tuned on 552 rows per fold could not extract or weight that signal as
reliably as a linear model given the same information directly as numeric features. This matches
CLAUDE.md's own stated modeling philosophy: "do not assume a single large transformer is
automatically best" on a dataset this small.

## Why TF-IDF + Medical Features Won

Three complementary mechanisms explain the win: **(1) Lexical matching** -- TF-IDF on lemmatized
title/narrative text directly captures the disease-name and treatment vocabulary that determines
topical relevance (`title_tfidf` terms dominate the top individual coefficients). **(2)
Contradiction detection** -- the 11 new features convert title-vs-patient compatibility checks
(age, gender, severity, condition) into explicit numeric signals a linear model can weight
directly, rather than requiring the model to infer this structure from raw text. **(3) Negation
and missingness awareness** -- C2's own `negation_count` and the structured extractor's
missing-value handling already give the linear model most of the benefit available from these
signals; H8's attempt to add more granularity on top showed diminishing, even negative, returns,
suggesting the ceiling for these particular signal types (given this architecture and dataset
size) was already close to reached by C2 itself.

## Remaining Performance Ceiling

Two structural limits remain, neither fixable by more feature engineering on this architecture:

1. **Potential annotation ambiguity.** Rows 525 (a pediatric-only surgical subsection applied to a
   73-year-old, structured `Возраст: 73`) and 126/127 (a pregnancy/breastfeeding-specific asthma
   subsection with no structured gender field at all) are labeled Applicable despite an
   unambiguous literal title contradiction (or, for 126/127, a genuinely unknown gender). These
   same rows were independently flagged as errors by five different models across three project
   stages (C2, T3, T4, H1, and now G) -- strong evidence they reflect labeling decisions this
   pipeline cannot and should not try to "fix" by relabeling.
2. **Missing information treated as unknown, not contradiction, by design -- but still costly.**
   Rows like 651/601 (an "Endometriosis and cancer" subsection with no oncology evidence in the
   protocol) and 13/619 (severity/subtype information absent or ambiguous) show that when the
   protocol simply does not contain the evidence needed to confirm or deny a title's condition,
   even a well-calibrated contradiction feature (`severity_unknown`, `condition_unknown`) can only
   express "no evidence of contradiction," which is sometimes the wrong call for this specific
   row even though it is the medically correct *general* policy. Closing this gap would require
   more information in the source protocols, not a better classifier.
