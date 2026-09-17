# Presentation materials: Optimizing LLM context for clinical-guideline evaluation

**AIIJC · Sber Health Industry Center case**

**Final score:** 55.168 / 70 on the metric. Stage 1: M1 = 0.922 on the
leaderboard. Stage 2: M2 = 0.731 on the leaderboard.

**What this file contains**
* **Slides 1–15:** the text for each slide (title, key message, content,
  suggested visual) plus speaker notes.
* **Part B:** prepared answers to likely jury questions.
* **Part C:** where every number comes from.

**Sources.** Every number comes from `solution.ipynb` (executed; it reproduces
the submitted file byte for byte) or from the experiment logs in `h*/`.
Validation numbers are always computed on train only.

**Suggested deck:** 15 main slides plus 4 backup slides, for a 10–12 minute
talk.

---

# Part A: Slides

---

## Slide 1: Title

**Filtering clinical-guideline sections before the LLM reads them**
*A two-stage, condition-matching approach for Russian medical text*

* Team / names / AIIJC 2026
* Result: **55.17 / 70** on the metric
* Deliverables: `solution.ipynb` (reproducible in Colab, about 1.5 min on a
  CPU, no GPU) and `outputs/submission.csv`

**Speaker notes.** "Our main claim: this task is condition matching, not
generic text classification, and we will show evidence for that. We will also
show where the model fails, because that matters for a medical use case."

---

## Slide 2: The problem, and why it matters

**Key message:** Every Treatment subsection we can safely drop saves LLM
tokens. Every subsection we drop by mistake can hide a relevant
recommendation.

* A clinical guideline has dozens of Treatment subsections. Sending all of
  them to an LLM for every patient is expensive and dilutes the context.
* The task has two filters.
  * **Stage 1 (title only).** Is the subsection **General** (applies to
    everyone) or **Special** (applies to a restricted group)?
  * **Stage 2 (title + patient protocol).** For a Special subsection, is it
    **Applicable** to this patient?
* The costs of the two errors are asymmetric.

| Error | Consequence | How the metric treats it |
|---|---|---|
| Stage 2 drops an applicable section | The LLM never sees a relevant recommendation (a clinical risk) | F2: recall weighted 4× |
| Stage 2 keeps an inapplicable section | Wasted tokens | tolerated |
| Stage 1 calls a General section Special | The section can then be dropped in Stage 2 | F0.5: precision weighted 4× |

* Score: `70 · (0.3·M1 + 0.7·M2)`. **+0.01 in M2 is worth 2.3× as much as
  +0.01 in M1**, so about 70% of our effort went into Stage 2.

**Visual:** a flow diagram: guideline → Stage 1 → (General: always keep) /
(Special → Stage 2 → keep / drop) → LLM.

**Speaker notes.** Stress the asymmetry. It explains both metrics, and later
it explains our threshold decisions.

---

## Slide 3: The data, and what it forced on us

**Key message:** Three data properties shaped every design decision.

| | Stage 1 | Stage 2 |
|---|---|---|
| Train / test rows | 1,767 / 442 | 690 / 173 |
| Class balance | 78% General / 22% Special | 66% Applicable / 34% Not applicable |
| Unique units | 491 guidelines in train | **37 titles**, 221 protocols in train |

1. **Small data.** 690 labelled pairs is too few to fine-tune a transformer
   reliably. On a single 5-fold split, M2 moves by **±0.02** from the choice
   of split alone.
2. **Stage 2 protocols are long mostly because of lab tables.**
   * Median protocol length is about 4,800 characters.
   * 187 of 863 protocol texts are cut at exactly 32,765 characters.
   * **78% of all protocol characters are lab-value lines**
     (`Показатель: … Результат: …`).
3. **The title alone explains a large part of the Stage 2 label.**
   * The per-title Applicable rate ranges from 0 to 1.
   * 13 of 37 titles are all-Applicable or all-Not-applicable.
   * The title rate alone reaches an in-sample AUC of **0.874**.
   * 102 protocols appear under more than one title, and **62 of them get
     different labels**, so the label depends on the (title, patient) pair.

**Encoding incident.** The first public copy of `train_stage1.csv` had every
Cyrillic letter replaced by `?`, so about 84% of the characters were lost. The
first three Stage 1 generations were built on that copy. A clean re-download
lifted Stage 1 from **M1 0.753 to 0.843** (nested CV).

**Visual:** (a) histogram of log protocol length; (b) a bar chart of the
Applicable rate per title, sorted (notebook cell 3.3).

---

## Slide 4: The solution at a glance

**Key message:** Two transparent linear models. Each one encodes the task
definition directly as features.

```
STAGE 1  title ─► parse hierarchy ─┬─ lemmatised last line (word 1–2-grams)
                                   ├─ lemmatised guideline + age + parent sections
                                   ├─ char 3–5-grams of the last line
                                   └─ layout + condition-marker counts
                                          ▼
                     LogReg (C=10, balanced) ─► threshold 0.69 ─► + Stage-2-title rule

STAGE 2  title ────────► required conditions (age bound, sex, severity, disease terms)
         protocol ─────► patient facts (Пол, Возраст, narrative, negations)
                                          ▼
                satisfied / CONTRADICTED / UNKNOWN  (11 features)
                  + lemmatised title & narrative TF-IDF + demographics
                                          ▼
                     LogReg (balanced) ─► threshold 0.5
```

| | Validation (train only) | Leaderboard |
|---|---|---|
| Stage 1, M1 | 0.843 (nested 5-fold) | **0.922** |
| Stage 2, M2 | 0.702 (fixed 5-fold); 0.699 ± 0.007 (5 × 5 repeated) | **0.731** |
| Points | 52.1 | **55.17** |

---

## Slide 5: Russian medical text processing

**Key message:** The medical-language handling is part of the model, not an
afterthought.

| Problem | What we do | Where |
|---|---|---|
| Rich morphology (`тяжёлой / тяжелая / тяжелый`) | pymorphy3 lemmatisation (cached); char n-grams catch what lemmas miss | both stages |
| Function words that change meaning (`при`, `у пациентов с`) | Stage 1 keeps every word; Stage 2 removes stop words | Stage 1 / Stage 2 |
| Patient-specificity vocabulary | Lemma sets for 7 condition groups: age, sex/pregnancy, severity, form, comorbidity, refractory disease, restriction | Stage 1 |
| Negation (`не …`, `нет …`, `отсутств…`, `отрицательн…`, `не выявлено`, `не определяется`) | Negation count per protocol, plus a 40-character negation window around each disease term. **Gap:** `отрицает`, `без` and a sentence-final `нет.` are not matched. | Stage 2 |
| Structured fields hidden in free text | Regex extraction of `Пол`, `Возраст` (with a free-text fallback), and the `Жалобы / Анамнез / Объективный статус` sections | Stage 2 |
| Sex inference | **Only** from the structured `Пол:` field. Guessing from "пациентка" created false contradictions. | Stage 2 |
| Severity terms | Mapped to levels (mild 1 / moderate 2 / severe 3), compared between title and narrative | Stage 2 |
| Long protocols | **Field extraction instead of truncation.** The narrative is a median 19% of the text; the lab tables are dropped. | Stage 2 |

**Example:** `"Жалоб нет. Отёков не выявлено, беременность отрицает."`
* Lemmatised: `жалоба отёк выявить беременность отрицать`.
* Negations matched: only `не выявлено`.
* **Missed:** `нет.` (at the end of a sentence) and `отрицает`. This is one
  reason the negation feature is weak (slide 7).

**Speaker notes.** Be honest here. Truncation strategies (head, head + tail)
were tested with transformers and lost (slide 9). Extraction was the only
long-text strategy that worked.

---

## Slide 6: Stage 1, what makes a title "Special"

**Key message:** "Special" means "the title names a restricted population".
We model that directly, and each feature block earns its place.

* **The hierarchy matters.** The Special rate is **15%** at heading depth 2
  and **52%** at depth 3: deeper subsections are more often population-specific.
* **Block ablation** (5-fold, M1 at threshold 0.5):

| Blocks | M1 |
|---|---|
| lemmatised last line | 0.634 |
| + guideline / age / parent-section context | 0.702 |
| + char 3–5-grams | 0.749 |
| + layout & condition markers (**final**) | **0.784** |

* **Nested CV** picks the hyperparameters. The inner folds choose the
  threshold, and the outer folds score it.

| C | 1 | 3 | **10** | 30 |
|---|---|---|---|---|
| nested M1 | 0.782 | 0.821 | **0.843** | 0.840 |

  The final threshold is **0.69**, the median of the inner-fold choices
  (0.73, 0.64, 0.69, 0.65, 0.79). A threshold above 0.5 fits F0.5, which
  rewards precision on Special.
* **Stage-2-title rule.** Stage 2 contains only Special subsections, so any
  title that also occurs in the Stage 2 data is Special.
  * On train: 32 matches, **100% precise**.
  * On test: 5 matches, **0 predictions changed**. The rule is a safety net,
    not a score source.

**Visual:** the ablation as a step chart; the C × nested-M1 table.

---

## Slide 7: Stage 2, the core idea: unknown ≠ contradiction

**Key message:** A section is "Not applicable" only when the patient
**contradicts** the title's condition. Missing evidence keeps it applicable,
so the features separate the two states.

| Title condition | Patient fact | Features |
|---|---|---|
| upper age bound `до N лет` | age | `age_contradiction` |
| female / male topic | structured `Пол` | `gender_condition_present`, `gender_unknown` |
| severity level | severity words in the narrative | `severity_match / _contradiction / _unknown` |
| informative disease terms | narrative lemmas + negation window | `condition_overlap / _contradiction / _unknown` |
| aggregate | | `contradiction_count`, `contradiction_density` |

**What the labels say** (training data, share Applicable):

| State | Rows | Applicable |
|---|---|---|
| **age contradicted** | 17 | **0.00** |
| sex-specific title, sex **known** | 51 | 0.04 |
| sex-specific title, sex **unknown** or pregnancy title | 18 | **0.33** |
| disease term never mentioned (unknown) | 214 | 0.55 |
| disease term present, not negated | 418 | 0.71 |
| disease term present **and negated** | 58 | 0.71 |
| all rows | 690 | 0.66 |

* **Age is a hard contradiction.** Every one of the 17 contradicted rows is
  Not applicable.
* **Sex shows the unknown-vs-contradicted split.** Unknown sex keeps 33% of
  rows applicable, against 4% when sex is known. A rule that treated unknown
  as contradicted would drop those rows.
* **Critical point: our disease-term negation flag has no signal** (0.71 vs
  0.71). A character window is too crude a negation scope. This is a known
  weakness (slides 11 and 13).

**Speaker notes.** This slide answers "how do you tell contradiction from
missing information?" Show the sex rows first, then admit the negation flag
doesn't work.

---

## Slide 8: Stage 2, what each component contributes (honest version)

**Key message:** Most of the lift comes from title + narrative. The
contradiction block's gain is within noise under repeated CV. We say so.

| Model | M2, fixed split | M2, 5 × 5 repeated (mean ± sd) | AUC |
|---|---|---|---|
| title TF-IDF only | 0.596 | 0.585 ± 0.010 | 0.842 |
| narrative TF-IDF only | 0.436 | 0.436 ± 0.021 | 0.753 |
| title + narrative | 0.682 | **0.706 ± 0.011** | 0.896 |
| + demographics & counts (C2) | 0.679 | 0.692 ± 0.007 | 0.892 |
| **+ contradiction / unknown features (G, final)** | **0.702** | 0.699 ± 0.007 | 0.888 |

* **On the original fixed split, G was +0.023 over C2.** That result, plus
  robustness checks (3/5 folds improved; 10/12 patient subgroups improved,
  none worse), is why it was promoted (experiment H8).
* **On 25 repeated splits, the last three rows are within about one
  standard deviation.** The fixed-split gain was mostly split luck.
* **Why G is still final:**
  1. The leaderboard verified it (55.17).
  2. No alternative beats it under repeated CV.
  3. Its explicit condition states make each decision inspectable, e.g. the
     age contradiction.
  4. Switching now, on a 0.006 difference, would be selection on noise.
* **What we learned about our own process:** a single 5-fold split can't
  detect gains below about 0.02 M2. We built a 5 × 5 repeated-CV harness with
  a corrected paired t-test and a noise-feature control, and used it for
  every later decision.

**Visual:** a bar chart with error bars (5 × 5) for the five models.

---

## Slide 9: Hypotheses we tested, and why they were rejected

**Key message:** 20+ controlled experiments. Most failed. The failures show
where the signal actually is.

| # | Hypothesis | Result | Verdict and reason |
|---|---|---|---|
| T2 | A transformer (rubert-tiny2) on title + narrative beats TF-IDF | M2 0.496 | ✗ 690 rows is too few to fine-tune on |
| T3 | Head + tail of the protocol recovers late negations | M2 0.477 | ✗ more raw text didn't help |
| T4 | A structured field template helps the transformer | M2 0.540 | ✗ best transformer, still −0.14; 72% of inputs truncated |
| H1 | Transformer embedding + engineered features | M2 0.454 | ✗ worst of the family |
| H7 | Raise the threshold to 0.54 (on C2) | M2 0.706 (+0.027 over C2) | ✗ buys M2 by giving up Applicable recall |
| **H8** | **Explicit contradiction / unknown features** | **M2 0.702 (+0.023, fixed split)** | ✓ promoted (see slide 8 caveat) |
| H19 | Finer negation scope, condition presence | Δ −0.036 | ✗ more rules added noise |
| H21 | How often a protocol / title recurs (fold-safe) | Δ −0.006 | ✗ an early −0.147 result turned out to be a leak and was re-run leak-free |
| H22 | Threshold tuned in nested CV | Δ −0.014 | ✗ 0.5 is already the right cut |
| H23 | Down-weight the title; per-title calibration | best Δ +0.001 at one seed; mean over seeds −0.008 | ✗ artifact of the random seed |
| H24-B/C | + ICD-10 code and diagnosis text (C0) | 5 × 5 Δ +0.022, 10/10 repeats; p = 0.08, then **0.14 on fresh seeds** | ✗ **leaderboard 54.80 < 55.17** |
| H24-C | CatBoost; per-title feature copies | Δ −0.017; +0.013 (not significant) | ✗ |
| H24-D | Zero-shot **Qwen2.5-7B-Instruct** probability as a feature | alone: AUC 0.735; with C0: Δ −0.011 / −0.008 | ✗ says "applicable" for 83% of rows; no within-title signal (AUC 0.52) |

**Three takeaways**
1. **Model capacity isn't the bottleneck**, as transformers, CatBoost and an
   LLM all showed. The bottleneck is **labelled evidence per title**.
2. **Rules only help when they are precise.** Age and sex rules work;
   free-text negation windows don't.
3. **The CV-to-leaderboard check matters.** C0 looked like the best CV result
   of the project and still scored lower on the leaderboard. We kept the
   verified model.

---

## Slide 10: Error analysis, Stage 1

**Key message:** The errors come from how a population is named, not from
missing vocabulary.

Nested out-of-fold results: precision on Special **0.84**, recall **0.75**.
Confusion: 95 FN, 55 FP out of 1,767.

| Error | How many | Pattern | Examples |
|---|---|---|---|
| **Missed Special** (FN) | 95, of which **57 have no condition marker** | The population is named *implicitly*, through a situation or disease variant | "Показания для **повторного** хирургического вмешательства", "**Лептоменингеальное** поражение", "Терапия пациентов с **хронической ИТП**" |
| **False Special** (FP) | 55, of which **33 do contain a marker** | A marker word appears in a heading the annotators considered general | "Лечение ортопедических **осложнений** гемофилии", "Терапия **рецидивов и резистентных форм** ОМЛ", "Лечение локальных **стадий** заболевания (I–II)" |

* **Marker words point towards Special but don't settle it.** That is why
  they are one feature block among four, not a hard rule.
* **Some labels are arguable.** "Therapy of relapsed/refractory AML" names a
  patient group, yet it is labelled General. The annotation convention seems
  to depend on the guideline's structure.
* **Risk:** an FP sends a General section into Stage 2, where it can be
  dropped. At F0.5 with a threshold of 0.69, only **4% of General titles
  (55 of 1,384)** are exposed.

---

## Slide 11: Error analysis, Stage 2: where and why the model is wrong

**Key message:** The false negatives are concentrated in a few titles, and
the model can't see the evidence that decides those rows.

Out-of-fold results: recall on Applicable **0.82**, precision 0.90.
Confusion: **81 FN**, 42 FP out of 690.

**1. The errors are concentrated.** Seven titles hold **50 of the 81 FN**.

| Title (last line) | Rows | Share Applicable | FN | Within-title AUC |
|---|---|---|---|---|
| ГБН with medication-overuse headache (ЛИГБ) | 21 | 0.48 | 10 | **0.22** |
| Surgery for Crohn's disease, terminal ileitis | 19 | 0.74 | 9 | 0.71 |
| Perianal Crohn's disease | 17 | 0.41 | 7 | **0.31** |
| Erosive gastritis / duodenitis | 20 | 0.80 | 7 | **0.34** |
| Crohn's disease, upper GI tract | 22 | 0.32 | 7 | 0.74 |

**2. FN categories** (experiment H20, all 81 FN):

| Category | FN | Share |
|---|---|---|
| the title's low prior pulls the score to the bottom | 37 | 46% |
| the title's prior partially dominates the score | 22 | 27% |
| the condition is stated in the protocol, but the model misses it | 12 | 15% |
| the condition term appears negated (window false alarm) | 10 | 12% |

**3. A concrete case: "Лечение БК с перианальными проявлениями".**

| | Protocol | True label | Model |
|---|---|---|---|
| Patient A | "Болезнь Крона, язвенный колит с 14–15 лет. Анемия." (K50.8). Perianal disease is **not mentioned**. | **Applicable** (no contradiction) | 0.14 → dropped ✗ |
| Patient B | "Болезнь Крона около 4 лет…" (K50.9), a normal exam | Not applicable | 0.73 → kept ✗ |

**Why the model gets it wrong**
* **The decisive fact is often an *absence*.** For patient A, perianal disease
  is not mentioned, which is not a contradiction, so the section applies.
* **A bag-of-words model scores what is present**, so it can't represent
  "nothing here contradicts the title".
* **Within such a title, the model has only 17–22 examples** to learn the
  distinction, so its ranking there is at or below chance.

**Speaker notes.** This is the most important slide for the critical analysis.
The main limitation is structural: it comes from the data, not from tuning.

---

## Slide 12: Critical assessment: what the score doesn't show

**Key message:** 55.17 is the score on titles the model has already seen.
Deployment would be harder.

1. **The test titles are all seen in training.**
   * All 32 test titles, and 89% of test protocols, also appear in the
     training data.
   * With cross-validation grouped by title (the model never sees the
     title), **M2 falls from 0.702 to 0.422** (experiment H22).
   * **So the model does not yet transfer to new guidelines.** It works
     mostly as a per-title prior refined by the patient's text.
2. **The competition metric doesn't match clinical safety.**
   * At threshold 0.5, **18% of applicable special sections are dropped**
     (recall 0.82).
   * At threshold 0.30, recall rises to **0.93**, but M2 falls to 0.54.
   * For a real deployment we would pick the high-recall operating point and
     accept lower token savings.

| Threshold | M2 | Applicable recall | Positive rate |
|---|---|---|---|
| 0.30 | 0.539 | **0.934** | 0.77 |
| 0.40 | 0.622 | 0.876 | 0.69 |
| **0.50 (submitted)** | **0.702** | 0.821 | 0.60 |

3. **The token saving is modest and unmeasured.**
   * Only Special sections (about 22%) can be dropped, and Stage 2 drops
     about 46% of them.
   * That removes **roughly 10% of subsections** per patient, by count. The
     saving in tokens depends on section lengths, which the data doesn't
     include.
4. **The validation noise is large.**
   * A single 5-fold split moves by ±0.02 M2.
   * The Stage 1 nested-CV folds range from M1 0.79 to 0.90.
   * Any claim of a gain below about 0.02 needs repeated CV, and we applied
     that bar to our own results (slide 8).

**Speaker notes.** Say this proactively. The jury values knowing the limits
more than a bigger number.

---

## Slide 13: What to develop next

**Key message:** The next gains need better evidence representation and more
labelled pairs, not bigger classifiers.

| Priority | Direction | Why | Expected effect |
|---|---|---|---|
| 1 | **Condition extraction as NER + assertion status** (e.g. a Russian biomedical BERT tagging present / negated / absent / historical) | Replaces the negation window, which has no signal (slide 7); represents "absence" explicitly (slide 11) | Targets the 22 FN where the condition is stated or wrongly read as negated in the text |
| 2 | **Title → structured criteria**, parsed once per guideline (age, sex, form, localisation, line of therapy) | Moves decisions from "title identity" to "criteria vs facts", so the model can transfer to unseen guidelines | Aims at closing the 0.70 → 0.42 gap under title-grouped CV |
| 3 | **An LLM as a verifier, not a feature**: called only on low-confidence rows, with few-shot examples of the annotation rules (e.g. "not mentioned = applicable") and calibrated per title | Zero-shot, the LLM was biased towards "applicable" (83%) and had no within-title signal | Improves recall on collapsed titles, at a small LLM cost |
| 4 | **More labels for low-prevalence titles** (active learning on uncertain rows) | 17–22 rows per title is too few to learn within-title distinctions | The most reliable lever |
| 5 | **Safety operating point + monitoring**: a recall target on Applicable, and logging of dropped sections | The clinical cost of an FN is asymmetric | Keeps deployment risk controlled |
| 6 | **Measure token savings** with real section lengths | The business goal is tokens, not subsection counts | Gives the true ROI |

---

## Slide 14: Reproducibility and engineering

**Key message:** One notebook reproduces the submitted file exactly, from the
four CSVs alone.

* **`solution.ipynb`: 47 cells.**
  * Contents: setup, data, metric, EDA, text processing, Stage 1, Stage 2,
    submission, results.
  * Runs in about 1.5 min on a CPU; no GPU.
* **Pinned:** scikit-learn 1.9.0, pymorphy3 2.0.6 and its dictionary. These
  decide the predictions and install on Colab without a runtime restart.
* **Seed 42.**
  * All CV splitters and models are seeded.
  * An old set-iteration nondeterminism in the negation feature was found and
    removed.
* **The notebook checks itself.**
  * It re-derives C and the threshold with nested CV and asserts that they
    match the recorded run.
  * It re-checks the Stage 2 CV score (0.7024).
  * It compares a SHA-256 fingerprint of each stage's predictions and
    **stops if the predictions differ**.
  * The output is byte-identical to the submitted file (sha `5cd35e4d…`).
* **Fold safety.**
  * All vectorisers and scalers are fitted inside the pipeline, per fold.
  * The contradiction features are row-local; no labels go in.
  * A leak found in our own recurrence experiment (H21) was caught and
    re-run.
* **Experiment log.** Each hypothesis has its own folder (`h7/` … `h24/`)
  with a script, results table and report. Every rejected idea is kept.

---

## Slide 15: Summary

* **Framing.** Stage 1 is *condition detection*; Stage 2 is *condition
  matching* with **unknown ≠ contradiction**.
* **Result:** 55.17 / 70 (M1 0.922, M2 0.731), with transparent linear models
  that beat transformers, CatBoost and a 7B LLM in controlled tests.
* **Honest limits.**
  * 18% of applicable special sections are still dropped at the submitted
    threshold.
  * The model relies on seeing a title in training (grouped-by-title M2 is
    0.42).
  * The negation-window feature carries no signal.
* **Next:** assertion-aware medical NER, structured title criteria, an LLM as
  a verifier on uncertain rows, and more labels for low-prevalence titles.

---

## Backup slides

**B1. Stage 1 model history**

| Model | Training text | Validation M1 | Leaderboard M1 |
|---|---|---|---|
| char TF-IDF + 6 meta features → LinearSVC | damaged `?` copy | 0.465 | 0.629 |
| rubert-tiny2 fine-tune (T1) | damaged copy | 0.458 | – |
| "word-shape" corruption-invariant features (H14) | damaged copy | 0.711 (nested) | – |
| invariant features + partial Cyrillic decoding (H15) | damaged copy | 0.753 (nested) | – |
| **lemmatised word + char + condition markers (H17)** | **clean** | **0.843 (nested)** | **0.922** |

**B2. Leaderboard submissions referenced in this deck**

| Submission | Score |
|---|---|
| Stage 1 v1 + G | 49.02 |
| **H17 + G (final)** | **55.17** |
| H17 + C0 (G + ICD code + diagnosis text) | 54.80 |

**B3. Score decomposition of the final submission**
* Stage 1: 19.36 of 21 possible points, so **1.64 points** of headroom.
* Stage 2: 35.81 of 49 possible points, so **13.19 points** of headroom.
* The remaining headroom is almost all in Stage 2.

**B4. Validation protocol (5 × 5 harness)**
* 5 × 5 repeated stratified K-fold, with paired fold-level differences.
* Nadeau–Bengio corrected t-test for overlapping training sets.
* **Gate:** mean Δ > 0, AND one-sided p < 0.10, AND at least 4 of 5 repeats
  improved.
* Negative control: a pure-noise feature scored Δ −0.012 with 0/5 repeats
  improved, and the gate rejected it.
* A candidate chosen on seeds 0–4 must pass again on fresh seeds 5–9.

---

# Part B: Prepared answers to jury questions

**Q1. Why didn't you use a transformer or an LLM as the main model?**
We tested both under the same splits.
* **Fine-tuned rubert-tiny2** scored M2 0.45–0.54 across four input
  strategies, against 0.70 for the linear model.
* **A zero-shot Qwen2.5-7B** said "applicable" for 83% of rows and had no
  within-title signal (AUC 0.52). As an extra feature, it lowered M2.

With 690 labelled pairs over 37 titles, a large model can't learn the
annotation conventions. A linear model with explicit condition features can.
Complexity was not a goal; the evidence decided.

**Q2. How does the model understand Special vs General?**
The brief defines Special as a section for a restricted population, so we
encode that definition:
* condition-marker lemma sets (age, sex/pregnancy, severity, form,
  comorbidity, refractory disease, restriction words);
* lemmatised n-grams of the subsection line, with the parent context kept
  separate;
* hierarchy depth, since 52% of depth-3 subsections are Special against 15%
  at depth 2.

Each block improves the ablation (0.634 → 0.784).

**Q3. What makes a title patient-specific?**
A title is patient-specific when it names a group smaller than the
guideline's population: an age bound, a sex or pregnancy, a disease form,
localisation or stage, a comorbidity, or a treatment line such as relapsed or
refractory disease. The hard cases name the group implicitly ("повторное
вмешательство") or use a marker word in a general heading. The first kind
drives our Stage 1 false negatives, the second our false positives.

**Q4. How do you handle negation?**
Two ways, and we are candid about the results.
* **Negation count per protocol**, a model feature (patterns: не, нет,
  отсутств…, отрицательн…, не выявлен…, не определя…).
* **A 40-character negation window** around each disease term shared by the
  title and the narrative.

The window flag turned out to carry no label signal (Applicable rate 0.71
whether or not the term is negated), and finer scoping (H19) lowered M2 by
0.036. The patterns also miss `отрицает`, `без` and a sentence-final `нет`.
Character windows are too crude for Russian clinical text. The fix is
assertion-aware NER (slide 13); adding the missing patterns is the cheap first
step.

**Q5. How do you tell a contradiction from missing information?**
Every condition gets separate *contradicted* and *unknown* features, so
missing information never looks like a contradiction to the model.

The data supports the split. For sex-specific titles:
* when the patient's sex is known, 4% of rows are Applicable;
* when it is unknown, 33% are.

An age contradiction gives 0% Applicable across 17 rows. Sex is read only
from the structured `Пол:` field, because guessing it from the text created
false contradictions.

**Q6. How do you handle long protocols?**
We extract fields instead of truncating: demographics plus the
complaints / anamnesis / objective-status narrative. That makes sense
because 78% of protocol characters are lab tables, and 187 of 863 texts are
cut at 32,765 characters anyway. Head and head + tail truncation were tested
with transformers and lost (M2 0.48).

**Q7. Why this final model?**
* It is the best-validated option: no alternative beat it under 5 × 5
  repeated CV, including the ICD / diagnosis model, CatBoost and the LLM
  feature.
* It is leaderboard-verified.
* It is reproducible byte for byte.
* It is interpretable: every prediction breaks down into TF-IDF weights plus
  named condition states.

**Q8. Which errors remain?**
* **Stage 2:** 81 false negatives out of 453 applicable rows; 50 of them are
  in seven titles. 73% happen because the title's low prior dominates the
  score. The decisive fact is often an absence ("perianal disease not
  mentioned"), which a bag-of-words model can't represent.
* **Stage 1:** implicitly named populations (missed) and generic uses of
  marker words (false Special).

**Q9. Why is the solution appropriate for medical text?**
* **Morphology:** lemmatisation plus char n-grams.
* **Terminology:** a condition lexicon and a severity scale.
* **Structure:** the protocol fields are parsed.
* **Negation:** handled explicitly.
* **The clinical asymmetry is respected:**
  * balanced class weights in both models;
  * a recall-oriented metric in Stage 2;
  * the unknown ≠ contradiction principle;
  * a documented high-recall operating point for deployment.

**Q10. How much does each component improve the metric?**
* **Stage 1** (M1 at 0.5): lemmatised subsection 0.634; + context +0.068;
  + char n-grams +0.047; + layout / markers +0.035. Tuning C and the
  threshold (nested) brings it to 0.843.
* **Stage 2** (5 × 5): title 0.585; + narrative +0.121. Demographics and
  contradiction features are within noise on average. The contradiction
  block's +0.023 on a single split did not replicate. We report that
  openly.

**Q11. Doesn't the Stage-2-title rule leak test information?**
It uses the test *titles* only, never labels, and it follows from the task
definition (Stage 2 contains only Special sections). It was enabled only
because it was 100% precise on train (32 of 32). On test it matched 5 titles
and **changed 0 predictions**, so it has no effect on the score.

**Q12. Did you tune on the leaderboard?**
No hyperparameter, threshold or feature was tuned on leaderboard feedback.
Everything was chosen by CV on train.

We made three submissions. The only leaderboard-informed decision was a
single choice between two pre-built models:
* C0 scored 54.80;
* the existing model scored 55.17;
* we kept the existing one.

That choice keeps the more conservative model; it doesn't fit anything new
to the test set.

**Q13. Why are your leaderboard scores higher than validation?**
Two reasons, and we didn't rely on the gap for any decision.
* **More training data.** The final models train on 100% of the training
  data, while each validation fold sees 80%.
* **Seen titles.** Every test title appears in training, and 89% of test
  protocols do too, so the test set is "in distribution".

We never compare a validation score with a leaderboard score to choose a
model.

**Q14. Why a threshold of 0.69 for Stage 1 but 0.5 for Stage 2?**
The metrics differ.
* **Stage 1 (F0.5) rewards precision**, so the threshold sits above 0.5. It
  was chosen on inner folds only, as the median of 0.64–0.79.
* **Stage 2 (F2) rewards recall.** The model already uses balanced class
  weights, and in the threshold sweep 0.5 gives the best M2. Nested tuning
  (H22) scored lower (0.688), and a higher threshold (H7) gave up recall.

**Q15. If false negatives are so costly, why is Applicable recall only
0.82?**
Because the competition metric is macro-F2 over *both* classes: pushing
recall up costs the Not-applicable class more than it gains. At a threshold
of 0.30, recall is 0.93 but M2 falls to 0.54.

The competition submission uses the metric-optimal point. In deployment we
would set a recall target (e.g. ≥ 0.93) and accept smaller token savings.
Slide 12 shows the table.

**Q16. The ICD-code model looked better in CV. Why did you reject it?**
* **It failed our pre-declared gate on fresh seeds** (p = 0.14). It had been
  chosen after looking at the first seeds, so its first p-value was
  optimistic.
* **The leaderboard agreed:** 54.80 against 55.17.

It changed only 6 of 173 test predictions, all towards "applicable". That
the change still cost 0.37 points shows how much small edits cost at this
data size.

**Q17. Does the model work for a new guideline?**
Not well yet, and we measured it: with titles held out of training, M2 is
0.42 instead of 0.70. The current model learns a per-title prior and refines
it with patient text. Parsing each title into structured criteria (slide 13,
direction 2) is how we would fix that.

**Q18. How reliable are your validation numbers?**
Our measurements show:
* one 5-fold split varies by about ±0.02 M2;
* the 5 × 5 standard deviation is about 0.007;
* the Stage 1 nested folds range from 0.79 to 0.90.

That is why we built the repeated-CV harness with a corrected paired test
and a noise-feature control (which the gate rejected). We also report which
of our own earlier gains don't survive it.

**Q19. How much does this reduce the LLM's token usage?**
We can estimate the reduction by subsection count, but not in tokens:
* about 22% of subsections are Special;
* Stage 2 drops about 46% of those;
* so roughly 10% of subsections are removed per patient.

The token saving depends on section lengths, which the data doesn't include.
Measuring it is part of our next steps. The framing caps the gain: General
sections are always sent.

**Q20. Is it reproducible?**
Yes.
* `solution.ipynb` runs from the four CSVs, with pinned scikit-learn and
  pymorphy3 and seed 42.
* It re-derives every selected hyperparameter and checks a fingerprint of
  the predictions.
* Its output is byte-identical to the submitted file.
* It ran in a clean folder containing only the notebook and the data.

**Q21. What about the damaged Stage 1 training file?**
The first public `train_stage1.csv` had its Cyrillic replaced by `?`.
* We first built corruption-invariant models: word shapes and partial
  decoding, reaching M1 0.711 and 0.753.
* Once a clean copy was available, we switched to reading real words: M1
  0.843 nested, 0.922 on the leaderboard.

The notebook refuses to run on the damaged file instead of degrading
silently.

**Q22. Why did the zero-shot LLM fail?**
The labels follow annotation conventions that the prompt didn't capture
well, for example "not mentioned means applicable" and the per-title
habits. The LLM was positive-biased (83% "applicable" at 0.5) and couldn't
rank rows within a title (AUC 0.52). Used as a verifier on uncertain rows
only, with few-shot examples and per-title calibration, it could still help.
That is a hypothesis for future work, not a result.

---

# Part C: Where the numbers come from

| Numbers | Source |
|---|---|
| Final score, M1/M2 on the leaderboard, points decomposition | `h18/metric_decomposition.json` |
| C0 leaderboard 54.80 | `h24/H24_REPORT.md` (final outcome) |
| EDA (lengths, lab share, title AUC, mixed labels) | `solution.ipynb` §3 |
| Stage 1 ablation, nested CV, errors | `solution.ipynb` §5.3–5.7 |
| Stage 2 state table, ablation, threshold sweep, per-title errors | `solution.ipynb` §6.2–6.7 |
| FN categories | `h20/fn_categories.csv` |
| Transformer runs (T1–T4, H1) | `H6_control_summary.json`, `baseline.ipynb` |
| H7, H8, H9 | `baseline.ipynb`, `h8/`, `h9/` |
| H19–H22 (incl. grouped-by-title 0.422) | `h22/FINAL_REPORT.md` |
| H23 | `h23/H23_REPORT.md` |
| H24 (harness, C0, CatBoost, LLM) | `h24/H24_REPORT.md`, `h24/step_*_results.csv` |
| Title / protocol overlap between train and test | computed from the four CSVs |
| Stage 1 history | `h14/`, `h15/`, `h17/` manifests |
