# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a data science competition project (AIIJC case): "Оптимизация LLM-контекста для оценки медицинских протоколов" (Optimizing LLM context for evaluating medical protocols). All text data is in Russian. The task has two subtasks:

- **Stage 1** — binary classification of clinical-guideline section titles into "Общий" (General, label 0) or "Специальный" (Special, label 1).
- **Stage 2** — binary classification of (title, protocol) pairs into "Применимо" (Applicable, label 1) / "Не применимо" (Not applicable, label 0): given a section title and a patient's clinical protocol text, decide whether that guideline subsection applies to the patient.

## Data layout

Data files sit in the repository root (`DATA_DIR = Path(".")` in the notebook, not `data/`):

- `train_stage1.csv` — 1,767 rows: `id, title_text, label`
- `test_stage1.csv` — 442 rows: `id, title_text`
- `train_stage2.csv` — 690 rows: `id, title_text, protocol_text, label`
- `test_stage2.csv` — 173 rows: `id, title_text, protocol_text`
- `submission.csv` — example/reference submission format: `stage, id, label`

Submission format: a single `submission.csv` with columns `stage` (1 or 2), `id`, `label`. Predictions for both subtasks are concatenated into one file, UTF-8, comma-separated, with row order matching the corresponding `test_stageN.csv`.

### `train_stage1.csv` is corrupted — read with `encoding="cp1251"`

`train_stage1.csv` fails to load as UTF-8. It is `cp1251`, but the Cyrillic content itself is **unrecoverably lost**: ~84% of the characters in `title_text` are the literal ASCII byte `?` (0x3F), not a decoding artifact — Cyrillic words were replaced with `?` runs of the same length at some earlier stage in the data pipeline (only ASCII survives: Latin script, digits, punctuation, and four cp1251 punctuation codes for quotes/dashes). `test_stage1.csv` is unaffected and fully readable text. This asymmetry matters for modeling: a lexical BoW/TF-IDF vectorizer fit on the corrupted train vocabulary barely transfers to the intact test vocabulary, so any Stage 1 model should lean on structural/meta features that survive the corruption (title length, `#`/`##` heading depth, presence of Latin script or digits) rather than relying on word content alone — see `baseline.ipynb` EDA section 2.1 for the quantified justification, and the ablation study (section 6.1) for why char n-gram + meta-feature configs beat plain word BoW.

## Pipeline (`baseline.ipynb`)

The notebook goes well beyond a naive baseline; current structure:

1. **Load** (`train_stage1.csv` via `cp1251`, everything else UTF-8).
2. **EDA** — encoding corruption (above), class balance, text-length stats, `protocol_text` field-structure coverage, negation-pattern frequency.
3. **Preprocessing v2** — `ProtocolFieldExtractor` (parses `Пол`/`Возраст`/`Код МКБ-10` etc. out of `protocol_text`, with regex fallbacks for the ~28% of protocols that lack the structured header, and separates out a `narrative_text` from `Жалобы`/`Анамнез`/`Объективный статус`), plus `lemmatize_ru()` (pymorphy3-based lemmatization with `lru_cache` per word).
4. **Rule-based age/gender contradiction detector** for Stage 2 (`rule_based_prediction`) — overrides the model to `0` when the title's explicit age bound (`"до N лет"`) or gender-specific topic contradicts the patient's structured `Возраст`/`Пол`. Gender inference intentionally uses **only** the structured `Пол:` field, never free-text guesses (`"пациент"`/`"пациентка"` heuristics were tried and produced false positives — see the comment in `_extract_gender`), to keep the rule's precision at 100% on train.
5. **Ablation study (5-fold `StratifiedKFold`, out-of-fold `cross_val_predict`)** comparing classical configs per stage (word/char TF-IDF, meta-features, field-extractor + lemmatized TF-IDF + numeric features, LogisticRegression/LinearSVC/SGDClassifier) scored with the *official* `stage1_score`/`stage2_score`, not raw sklearn metrics. The winning config per stage is picked programmatically from the CV results table, not hardcoded.
6. **Final refit** — winning pipeline refit on full train, holdout classification report, then test predictions with the rule-based override applied on top where it doesn't hurt CV score.
7. **Submission** — same `stage,id,label` format as the original baseline.

`RANDOM_STATE = 42` throughout; an 80/20 stratified holdout is used for the human-readable classification report alongside the 5-fold CV.

## Scoring formula

Reimplemented locally in the notebook to match the platform's official metric (`stage1_score`, `stage2_score`, `total_metric_points` in the last markdown/code section):

- **M1** (Stage 1): macro-F0.5 score, linearly rescaled from `[0.5, 0.95]` to `[0, 1]` (clipped).
- **M2** (Stage 2): macro-F2 score (F-beta with beta=2, averaged over both classes — biased toward recall since missing an applicable subsection is the worse error), rescaled from `[0.5, 0.95]` to `[0, 1]` (clipped). Averaging F2 across both classes means a trivial "always applicable" prediction only scores macro-F2 ≈ 0.45, which clips to M2 = 0.
- **Final points** (out of 70, the "accuracy" grading criterion): `70 * (0.3 * M1 + 0.7 * M2)` — Stage 2 is weighted more heavily.

When iterating on models, validate against these exact functions rather than raw sklearn metrics, since the competition score is the rescaled/weighted version, not raw F-scores.

## Environment

No dependency manifest exists yet. The notebook requires: `numpy`, `pandas`, `scikit-learn`, `scipy`, `pymorphy3` + `pymorphy3-dicts-ru` (the first code cell auto-installs pymorphy3 via `pip` if missing). There is no git repository, build system, lint config, or test suite currently set up.

To execute the notebook headlessly (e.g. to verify it still runs end-to-end after an edit), use `nbclient` with a registered kernel rather than the `jupyter` CLI (its `kernelspec` subcommand isn't available in this environment):

```python
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("baseline.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="<your kernel>").execute()
nbformat.write(nb, "baseline.ipynb")
```


## MAIN TASK

Sber Health Industry Center — Competition Context
1. Project Overview

This project is a machine learning competition organized by Sber's Health Industry Center.

The goal is to develop NLP models that reduce the amount of clinical guideline (CG) text that must be passed into an LLM when evaluating whether a medical protocol is applicable to a patient.

The core problem is relevance filtering of clinical guideline subsections.

Clinical guidelines are often very long. Passing the entire guideline into an LLM for every patient consumes excessive tokens, increases cost, and introduces unnecessary information that can reduce model performance.

Our solution should identify which parts of the Treatment section of a clinical guideline are relevant to a particular patient.

2. Competition Tasks

There are two independent NLP classification subtasks.

All dataset text is in Russian.

Subtask 1 — Clinical Guideline Title Classification
Objective

Determine whether a clinical guideline subsection is:

0 = General
1 = Special

The prediction is based only on the subsection title/path, without considering a patient.

Definition

A General subsection applies broadly to all patients.

A Special subsection describes treatment principles or methods intended for a specific patient group, for example patients:

with a particular comorbidity or condition;
of a specific gender;
in a particular age group;
with a particular form/severity/type of disease or condition.
Input

title_text

This contains the full hierarchical path to the subsection, including:

clinical guideline title;
section hierarchy;
final subsection title.
Dataset
train_stage1.csv
- 1,767 rows
- columns:
    id: int
    title_text: str
    label: int
        0 = General
        1 = Special

test_stage1.csv
- 442 rows
- columns:
    id: int
    title_text: str

Class distribution:

General: ~78%
Special: ~22%
Imbalance: ~3.6 : 1

The dataset is stratified by label.

Metric

Primary metric:

Macro-F0.5

For each class:

F0.5_k = 1.25 * P_k * R_k / (0.25 * P_k + R_k)

Then:

Macro-F0.5 = (F0.5_0 + F0.5_1) / 2

F0.5 weights precision more heavily than recall.

Therefore, avoid simply predicting General for most examples.

3. Subtask 2 — Applicability Assessment
Objective

Determine whether a Special clinical guideline subsection applies to a patient's examination report.

The basic diagnosis has already been matched using ICD-10.

We need to classify the relationship between:

the special clinical guideline title, and
the patient's examination protocol.
Classes
1 = Applicable
0 = Not applicable

Applicable means:

the patient's report does not contradict the conditions described by the title; OR
there is insufficient information to determine that the patient does not belong to the group.

Not applicable means:

there is a clear contradiction;
the patient clearly does not belong to the population described by the title.
Important principle

False negatives are particularly costly.

The system should be conservative when determining that a section is irrelevant.

The relevant section of the clinical guideline must not be skipped when it may apply.

This is reflected by the use of F2, which emphasizes recall.

Input

title_text

Special subsection title with hierarchy.

protocol_text

Raw text of the patient's examination protocol.

The diagnosis has already been matched according to ICD-10.

Dataset
train_stage2.csv
- 690 rows
- columns:
    id: int
    title_text: str
    protocol_text: str
    label: int
        1 = Applicable
        0 = Not applicable

test_stage2.csv
- 173 rows
- columns:
    id: int
    title_text: str
    protocol_text: str

Class distribution:

Applicable: ~66%
Not applicable: ~34%

The dataset is stratified by label.

Metric

Primary metric:

Macro-F2

For each class:

F2_k = 5 * P_k * R_k / (4 * P_k + R_k)

Then:

Macro-F2 = (F2_0 + F2_1) / 2

F2 strongly favors recall.

4. Overall Competition Score

The ML metric contributes up to 70/100 points.

The competition combines the two subtasks as:

Metric score = 70 * (0.3 * M1 + 0.7 * M2)

where:

M1 = max(0, (Macro-F0.5 - 0.5) / 0.45)
M1 <= 1.0

M2 = max(0, (Macro-F2 - 0.5) / 0.45)
M2 <= 1.0

Therefore:

Subtask 1 contributes 30% of the metric score.
Subtask 2 contributes 70% of the metric score.
Strategic implication

Subtask 2 is substantially more important.

When allocating experimentation and engineering effort:

Priority:
1. Subtask 2
2. Subtask 1

Do not optimize Subtask 1 at the expense of a significantly better Subtask 2.

The target should ideally be:

Subtask 1: strong Macro-F0.5
Subtask 2: very strong Macro-F2

with particular attention to recall in Subtask 2.

5. Expert Evaluation — Additional 30 Points

The remaining 30 points are awarded by experts.

The solution should therefore not be optimized purely for leaderboard performance.

The final solution must also demonstrate:

clean and reproducible code;
justified architecture selection;
domain-aware processing of Russian medical text;
strong presentation and error analysis.
5.1 Code Quality — 6 points

Aim for:

modular code;
meaningful variable/function names;
minimal duplication;
consistent style;
comments around important steps;
experiment logging;
reproducibility.

Use functions/classes where appropriate instead of one extremely long notebook.

5.2 Architecture Justification — 8 points

Aim for the highest evaluation tier.

The solution should ideally include:

a baseline;
multiple model families;
validation;
hyperparameter experiments;
quantitative comparison;
cross-validation where appropriate;
explanation of why the final architecture was selected.

Do not choose a model simply because it is more complex.

The final report should answer:

Why is this architecture better for this task and dataset?

with experimental evidence.

5.3 Medical Text Handling — 8 points

This is especially important.

The solution should account for characteristics of Russian medical language.

Potential areas:

Russian morphology

Consider:

lemmatization;
stemming where appropriate;
robust tokenization;
morphological normalization.
Medical terminology

Consider:

abbreviations;
medical terminology;
disease names;
clinical synonyms;
structured medical expressions.
Negation

Explicitly handle expressions such as:

не
отрицает
без
нет
не выявлено
анамнез не отягощен

Negation is particularly important in Subtask 2.

For example, the presence of a disease term does not necessarily mean the patient has that condition.

Conditional attributes

Special titles may encode conditions involving:

age;
gender;
disease form;
disease severity;
comorbidities;
clinical characteristics.

A strong solution should consider explicitly extracting these conditions and comparing them against the patient protocol.

Potential approaches include:

rule-based condition extraction;
structured feature extraction;
auxiliary classifiers;
specialized Russian-language embeddings;
medical-domain embeddings/models.
6. Recommended Modeling Philosophy

Do not assume that a single large transformer is automatically the best solution.

The datasets are relatively small:

Stage 1: 1,767 training examples
Stage 2: 690 training examples

Therefore, overfitting is a major concern.

We should compare several approaches.

Potential model families include:

Classical NLP

Examples:

TF-IDF + Logistic Regression
TF-IDF + Linear SVM
character n-grams
word n-grams
word + character feature combinations

These should serve as strong baselines.

Transformer-based models

Potential approaches:

Russian-language BERT models;
multilingual BERT models;
RuBERT-style models;
sentence-transformer embeddings;
other open-source Russian-language models.
Hybrid approaches

Consider combining:

lexical features
+
semantic embeddings
+
structured medical features

Especially for Subtask 2.

7. Important Insight for Subtask 1

Subtask 1 is not ordinary topic classification.

The task is fundamentally about detecting whether the title contains a patient-specific condition.

Therefore, investigate features such as:

age
gender
comorbidity
disease subtype
disease form
severity
specific population
clinical condition

The hierarchical title path may also contain useful contextual information.

Do not blindly discard the hierarchy.

For example, the parent section may change the interpretation of the final subsection.

Potential strategy:

title hierarchy
        ↓
normalize text
        ↓
extract conditional signals
        ↓
semantic representation
        ↓
classification
8. Important Insight for Subtask 2

Subtask 2 is best viewed as a conditional matching / contradiction detection problem, not merely generic binary text classification.

Conceptually:

Special title
      ↓
Extract required patient conditions
      ↓
Extract patient facts
      ↓
Determine:
    condition satisfied?
    condition contradicted?
    condition unknown?
      ↓
Applicable / Not applicable

The distinction between:

UNKNOWN

and

CONTRADICTED

is critical.

According to the competition definition:

Lack of evidence against applicability should generally not cause the section to be classified as Not applicable.

Therefore:

unknown ≠ contradiction

This should influence both feature engineering and model design.

9. Long Text Handling

protocol_text may be long.

Do not arbitrarily truncate the protocol without investigation.

Possible strategies:

Strategy A — Head truncation

Use only the beginning of the protocol.

Simple but potentially dangerous.

Strategy B — Tail truncation

Useful if conclusions or diagnoses tend to appear near the end.

Strategy C — Head + tail

Preserve both initial examination information and final conclusions.

Strategy D — Chunking

Split the protocol into chunks and encode each chunk separately.

Then aggregate:

max similarity
mean similarity
attention pooling
max evidence score
Strategy E — Information extraction

Extract medically relevant structured information first:

age
sex
diagnoses
symptoms
comorbidities
severity
clinical findings
negations

Then compare the extracted information with conditions encoded in the title.

The final choice must be based on validation experiments.

10. Validation

Because the datasets are small, validation must be handled carefully.

Use stratified splits.

Potential setup:

Stratified K-Fold

or repeated stratified validation where computationally feasible.

Track:

Macro-F0.5 for Stage 1;
Macro-F2 for Stage 2;
per-class precision;
per-class recall;
confusion matrix;
validation variance.

Do not rely on accuracy.

Accuracy is not the competition metric.

11. Class Imbalance

Stage 1:

General ~78%
Special ~22%

Stage 2:

Applicable ~66%
Not applicable ~34%

Potential techniques:

class weights;
threshold tuning;
focal loss where appropriate;
balanced sampling;
decision-threshold optimization.

However, threshold optimization must be performed only using training/validation data.

Never use the test set to tune thresholds.

12. Data Leakage Rules

Avoid any form of test-set leakage.

Strictly prohibited:

manually labeling test examples;
inspecting test labels;
using test labels indirectly;
tuning based on leaderboard feedback in a way that effectively overfits the test set;
external proprietary medical databases that are not publicly available.

All preprocessing and model-selection decisions should be reproducible from the training data.

13. Reproducibility Requirements

The final solution must run in Google Colab without manual code edits.

Requirements:

Python 3.10+;
all dependencies installed programmatically;
datasets loaded programmatically;
fixed random_seed;
training pipeline reproducible;
inference pipeline reproducible;
generated predictions must exactly match the submitted predictions, except for insignificant floating-point differences;
final model training code must be executable from start to finish.

Training is allowed to exceed the Colab time limit, but the code itself must be functional.

Recommended reproducibility setup:

SEED = 42

and explicitly seed:

Python;
NumPy;
PyTorch;
CUDA where applicable;
model/data-loader randomness.
14. Allowed Technologies

Python 3.10+.

Open-source libraries are allowed, including:

pandas
numpy
scikit-learn
PyTorch
Hugging Face Transformers
sentence-transformers
spaCy

Open pre-trained models are allowed.

LLMs may be used.

However, the solution must remain reproducible under the competition requirements.

15. Submission Requirements
Automatic Evaluation

submission.csv

Required columns:

stage
id
label

Where:

stage = 1

for Subtask 1 and:

stage = 2

for Subtask 2.

The file contains predictions for both subtasks.

Expert Evaluation
solution.ipynb

Must contain:

complete preprocessing;
training of both models;
inference;
comments;
reproducible execution;
experiment methodology.
presentation.pdf

Should explain:

problem understanding;
dataset;
preprocessing;
baseline;
candidate architectures;
experiments;
final architecture;
validation results;
error analysis;
medical-text-specific techniques;
limitations;
possible future improvements.
16. Expert-Facing Presentation Strategy

The presentation should not merely say:

"We used BERT and achieved X."

It should demonstrate reasoning.

Recommended narrative:

Problem
  ↓
Why full clinical guidelines are expensive
  ↓
Two-stage filtering formulation
  ↓
Data analysis
  ↓
Baseline
  ↓
Hypotheses
  ↓
Experiments
  ↓
Architecture comparison
  ↓
Medical-specific improvements
  ↓
Error analysis
  ↓
Final solution
  ↓
Limitations + future work

Important questions we should be able to answer:

Why does the model understand Special versus General?
What makes a title patient-specific?
How does the model handle negation?
How does the model distinguish contradiction from missing information?
How does it handle long protocols?
Why was the final model selected?
Which errors remain?
Why is the solution appropriate for medical text?
How much does each component improve the metric?
17. Competition Strategy

The competition score is strongly weighted toward Subtask 2.

Therefore, engineering effort should roughly prioritize:

Subtask 2: ~70%
Subtask 1: ~30%

This does not mean Subtask 1 can be ignored.

A strong final solution should establish a reliable baseline for both, then focus deeper experimentation on Subtask 2.

18. Development Workflow

Follow this order unless experiments justify changing it.

Phase 1 — Data Understanding

Inspect:

class balance;
title lengths;
protocol lengths;
duplicate examples;
near-duplicates;
common words/phrases;
common special-condition patterns;
label distribution;
possible leakage.

Perform qualitative inspection of examples from every class.

Phase 2 — Strong Baselines

Build:

Stage 1
TF-IDF word + character n-grams
→ Logistic Regression / Linear SVM
Stage 2

Start with:

TF-IDF title + protocol features
→ linear classifier

Then test a semantic embedding approach.

Record all validation metrics.

Phase 3 — Transformer Models

Evaluate appropriate Russian/multilingual pretrained models.

Compare against the classical baseline rather than assuming transformers will win.

Use stratified validation.

Phase 4 — Medical-Aware Features

Investigate:

negation;
age;
gender;
comorbidity;
disease subtype;
severity;
clinical conditions;
hierarchy;
medical terminology.

For Stage 2, investigate explicit title-vs-protocol condition matching.

Phase 5 — Long-Protocol Strategy

Experiment with:

truncation;
head + tail;
chunking;
embedding aggregation;
information extraction.

Select based on validation Macro-F2.

Phase 6 — Threshold Optimization

Optimize classification thresholds against the actual competition metric.

For Stage 1:

Macro-F0.5

For Stage 2:

Macro-F2

Do this using validation data only.

Phase 7 — Error Analysis

For every major model, inspect:

false positives;
false negatives;
per-class performance;
common linguistic failure modes;
negation errors;
age/gender errors;
contradiction vs unknown errors;
long-context errors.

Create representative examples for the presentation.

Phase 8 — Final Model

Only select the final architecture after quantitative comparison.

The final model should balance:

leaderboard performance
+
validation robustness
+
reproducibility
+
interpretability
+
expert-evaluation quality
19. Engineering Principles for This Project

When modifying the code:

Do not over-engineer prematurely.
Establish a measurable baseline first.
Change one major component at a time when running controlled experiments.
Record every experiment and its validation score.
Never optimize against the test set.
Prefer reproducible pipelines over manual notebook operations.
Keep preprocessing identical between training and inference.
Preserve the competition's exact label conventions.
Be especially careful with negation in medical Russian.
Treat unknown and contradiction as different semantic states in Stage 2.
Prefer evidence-based architecture decisions.
Keep the final notebook clean enough for expert review.
20. Target Outcome

The goal is not merely to produce a leaderboard submission.

The target is a competition-grade, expert-defensible NLP system that:

achieves strong Macro-F0.5 on Stage 1;
achieves especially strong Macro-F2 on Stage 2;
handles Russian medical text appropriately;
explicitly considers negation and patient-specific conditions;
is reproducible in Google Colab;
has a clean experimental history;
provides meaningful error analysis;
can be clearly defended during expert evaluation.

The strongest conceptual direction is:

Transform clinical guideline applicability from generic text classification into structured condition matching between a special guideline section and patient evidence, while preserving high recall.

This principle should guide model design, feature engineering, experiments, and presentation.