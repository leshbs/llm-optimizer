# Presentation Content — Sber Health Industry Center Case
## LLM-Context Optimization for Clinical Guideline Evaluation

**Note on sourcing.** Every quantitative claim below cites the exact repository artifact it came from. Numbers that could not be traced to a trustworthy source were excluded rather than estimated — see the "Data provenance note" at the end of this document for what was excluded and why.

---

## Slide 1 — Problem Framing

**Objective:** Establish why this is a hard, high-stakes filtering problem, not a generic classification exercise.

**Bullets:**
- Clinical guidelines are long; passing them whole into an LLM for every patient is expensive and dilutes relevance. The task: decide which Treatment subsections apply to a given patient *before* the LLM sees them.
- Two independent subtasks, weighted unevenly: `Score = 70 × (0.3·M1 + 0.7·M2)` — Stage 2 (applicability matching) carries **70%** of the score, Stage 1 (title triage) carries 30%.
- Small data: 1,767 / 442 rows (Stage 1 train/test), 690 / 173 rows (Stage 2 train/test) — overfitting risk dominates architecture choice.
- Class imbalance: Stage 1 is ~78% General / 22% Special (3.6:1); Stage 2 is ~66% Applicable / 34% Not Applicable.
- **`train_stage1.csv` is irreversibly corrupted**: ~84% of characters in `title_text` were replaced with literal `?` at an earlier pipeline stage (not a decoding bug — confirmed cp1251-readable but content-lost). `test_stage1.csv` is intact. This creates a train/test vocabulary mismatch that shapes every Stage 1 modeling decision.
- Naive lexical classification is insufficient on both counts: Stage 1 needs structural/meta signal because the words are gone; Stage 2 needs explicit contradiction reasoning because "no evidence against" must not be scored the same as "evidence against" (F2's recall bias exists precisely to penalize wrongly dropping an applicable subsection).

**Suggested evidence:**
- Table: dataset sizes, class balance, both stages.
- Before/after example of a corrupted vs intact title string (CLAUDE.md, EDA §2.1).
- Protocol length histogram + negation-position histogram (baseline.ipynb §10.1: negation is 65.9% concentrated in the final third of `protocol_text` — motivates head+tail / structured-extraction strategies later).

**Speaker notes:** Lead with the scoring formula — it is the single fact that should shape every subsequent engineering-priority decision in the talk.

---

## Slide 2 — Methodology Journey

**Objective:** Show a disciplined, evidence-gated progression from classical baselines through transformer experiments to clinical feature engineering — including the negative results.

**Naming note for presenter:** the repo's experiment labels ("T1–T4, H1, H6–H9") span **two tracks**, not one linear sequence: Track A (T1–T4, H1, synthesized in H6) is architecture comparison (classical vs. transformer); Track B (H7→H8→H9) is validation and clinical-feature engineering built *on top of* the Track-A winner. Present them as two connected phases, not 9 sequential steps.

### Phase 1 — Classical baselines (winner of a 6-config architecture ablation, `baseline.ipynb §6`)

| Stage | Winning config | Offline CV score | Runner-up | Worst config |
|---|---|---|---|---|
| 1 | Char TF-IDF (2–5gram) + 6 structural meta-features + LinearSVC | M1 = 0.4648 | — | Word CountVectorizer + LogReg: M1 = 0.3094 |
| 2 | Field-extractor + lemmatized TF-IDF + numeric features + LogReg ("C2") | M2 = 0.6794 | +LinearSVC: M2 = 0.6603 | Plain word TF-IDF: M2 = 0.3911 |

Confirms the CLAUDE.md hypothesis directly: char n-grams + structural meta-features survive the Stage-1 corruption far better than word-level lexical features.

### Phase 2 — Transformer experiments (Track A: T1–T4, H1)

| Experiment | Hypothesis | Input | Offline M2 (pooled OOF @0.5) | vs. classical | Verdict |
|---|---|---|---|---|---|
| T1 (Stage 1) | rubert-tiny2 beats char-TF-IDF+meta on raw titles | `title_text` | M1 = 0.4580 (best tuned: 0.5279 @ thr. 0.70) | C1 = 0.4648 | **Rejected** — best score required post-hoc threshold tuning on a small CV set, and the corrupted train vocabulary makes transformer generalization to the intact test vocabulary unverifiable; classical C1 retained |
| T2 (Stage 2) | Transformer on title+narrative beats C2 | title + `narrative_text` | M2 = 0.4961 | C2 = 0.6794 (Δ −0.1833) | **Rejected** |
| T3 (Stage 2) | Head+tail input recovers late-document negation (65.9% of negation sits in the final third) | title + head(protocol) + tail(protocol) | M2 = 0.4774 | Δ −0.2020 | **Rejected** — more raw text alone did not close the gap |
| T4 (Stage 2) | Structured templated context (Gender/Age/ICD/Complaints/etc. as labeled fields) helps the transformer exploit structure | templated fields, max_len 256 | M2 = 0.5397 (best transformer variant) | Δ −0.1397 | **Rejected**, though best of the transformer family; 71.6% of rows were truncated at max_length — flagged limitation |
| H1 (Stage 2) | Hybrid: transformer CLS embedding + C2's engineered features | CLS + numeric features | M2 = 0.4543 (worst of the family) | Δ −0.2251 | **Rejected** |

**Why transformers were abandoned (evidence, not opinion):** every transformer variant, across four different input strategies, scored below the classical C2 pipeline on identical 5-fold CV splits (`skf2_folds.json`), with no positive trend as input engineering improved (T2→T4 did climb, 0.496→0.540, but never crossed C2's 0.679). With 690 Stage-2 training rows, this is consistent with transformer overfitting/undertraining rather than a fixable input problem — H6's synthesis explicitly closes this line of investigation and returns to the classical pipeline.

### Phase 3 — Clinical feature engineering on the classical winner (Track B: H7→H8→H9)

- **H7 (validation-only, no new architecture):** reproduced C2's CV score to 9.5×10⁻⁹; found a threshold of 0.54 raises M2 to 0.7064 (+0.027) but **rejected it** — it trades Applicable-class recall for precision, contradicting the project's conservative-recall principle (Section 2's "false negatives are costly"). Feature-group ablation (structural → demographic → negation) showed text-only actually scored marginally higher in pooled OOF (0.6819) than the full feature set (0.6794); the full set was kept anyway for interpretability and subgroup robustness, not raw score — a **neutral, evidence-documented** trade-off. Feature importance ranking: protocol length > negation count > structural features > clinical terms > lexical > demographics (`feature_importance.csv`).
- **H8 (headline result):** 11 explicit contradiction features (age/gender/severity/condition contradiction flags, built from the already-extracted structured fields) → **M2 = 0.7024, +0.023 over C2** (`h8/analysis/h8_summary.json`). Alternatives tested and rejected: negation-refinement (+0.009, below promotion gate), missingness indicators (−0.018, regression), long-protocol metadata (+0.004, below gate), all interaction/combined variants (underperformed contradiction alone).
- **H9 (robustness gate):** confirmed the H8 gain is not a fold artifact — 3/5 folds improved individually (mean Δ +0.023, std 0.035), 10/12 subgroups improved and none regressed, 9/9 feature-sanity checks passed (no leakage, no duplicate/constant features). One documented trade-off: Applicable-class recall dipped slightly (0.830→0.821). Decision: **PROMOTE_G** — contradiction-feature model becomes the new Stage-2 production candidate.

**Why H8 worked when transformers didn't:** the contradiction features directly encode the *reasoning* the task requires (does the title's implied age/gender/severity/condition contradict the patient's structured fields?) rather than asking a small transformer to learn that reasoning from 690 examples of raw text. It is a case for structured condition-matching over generic text classification — exactly the framing CLAUDE.md's Section 8 argues for.

**Suggested evidence:** experiment-timeline diagram (Phase 1→2→3), the M2-by-experiment bar chart (C2 0.679, T2 0.496, T3 0.477, T4 0.540, H1 0.454, H8/G 0.702), H9's per-fold and per-subgroup robustness table.

---

## Slide 3 — Real-World Score Check

**Objective:** Report the one genuine external validation event honestly, and be explicit about what it did and did not establish.

**Bullets:**
- The frozen classical pipeline (C1 + C2, pre-H8) was submitted once and scored **49.2/70** overall (`h12/platform/submission_history.csv` — the only row ever recorded).
- Offline CV predicted an expected combined score of **44.18/70** (from M1=0.4648, M2=0.7024 fed through the scoring formula). The real score came in **above** the offline expectation, not below — i.e., no evidence of a generalization gap on this one data point.
- A stage-level attribution plan was designed to isolate exactly how much of that score came from Stage 1 vs. Stage 2: three isolation-probe submission files (Stage-2-only, Stage-1-only, and a trivial-baseline sanity check) were built and integrity-checked (`h11/probes/probe_manifest.json`).
- **These probes were never actually submitted.** The repository's own audit trail (`h11/analysis/h11_strategy_report.md`, `h12/platform/platform_diagnosis.json`) records this candidly: `n_scored_probes = 0`, stage-level M1/M2 attribution is `PENDING`/`None`. We report this as an incomplete diagnostic, not a resolved one.
- Takeaway for judges: the single available real-world data point is consistent with the offline validation (no red flag), but a rigorous per-stage confirmation was designed and not executed — an honest limitation, listed again in Slide 7.

**Suggested evidence:** offline-expected vs. real combined score bar (44.18 vs. 49.2/70); probe-design diagram showing what Stage-1-only / Stage-2-only isolation *would* have measured, annotated "prepared, not submitted."

**Speaker notes:** Do not claim a specific M1/M2 split for the real submission — the repo has exactly one number for this (49.2/70 combined), and any finer breakdown reported elsewhere in the project's internal logs could not be independently verified against the submission record and is excluded here on that basis.

---

## Slide 4 — Post-Baseline Exploration

**Objective:** Show that exploration continued past the frozen classical+contradiction model, evaluated fairly, and that unsuccessful or inconclusive leads were not quietly dropped.

### A. Zero-shot LLM (GPT-4o-mini) as a Stage 1/Stage 2 replacement

| | Stage 1 | Stage 2 |
|---|---|---|
| LLM positive rate | 67.9% predict 'Special' | 93.6% predict 'Applicable' |
| Production model positive rate | 5.9% ('Special') | 54.3% ('Applicable') |
| Qualitative finding | LLM correctly catches explicit population qualifiers (age category, named subgroups) in titles that classical C1 misses — attributable to C1's corrupted training vocabulary | LLM's high "Applicable" rate reflects a very literal reading of the "unknown ≠ contradiction" rule; disagreements with the production model do **not** concentrate on contradiction-flagged rows (7.4% vs. 12.1% base rate) — a specific "LLM catches contradiction edge cases" hypothesis was tested and **falsified** |
| Verdict | Not adopted as drop-in replacement; a real structural weakness in C1 was identified, but the LLM's own bias is uncalibrated against ground truth | Not adopted as replacement; possible future role as an audit layer on marginal-probability predictions only |

Rule-engine ensemble candidate (age/gender override + LLM) was found **byte-identical** to a plain LLM swap on this 173-row test set — the rule fired on 6 rows but changed 0 final labels, since the LLM already reached the same conclusions independently. **Neutral / no gain.**

### B. Patient-repeat / matrix-completion approach

**Not attempted.** A targeted search of the repository found no implementation, cache, or analysis file for this idea. Rather than reconstruct a narrative for an experiment that didn't happen, we report this honestly as an unexplored direction (see Slide 7, Future Work).

### C. Subsection base-rate blend

- Implementation: for each of the 173 test rows, blend G's prediction with the historical base rate of "Applicable" for that guideline subsection in train (all 173 rows matched a known subsection). Effect: 61 of 173 predictions were nudged by the base rate, but only **1 final label actually flipped** relative to plain G@0.50 (`h13/blend/base_rate_blend_manifest.json`).
- An improvement claim for this blend exists in the project's internal logs, but it is derived arithmetically from the same unconfirmed submission-based numbers flagged in Slide 3, and is excluded here on that basis.
- On the one trustworthy, descriptive measure available — a 1-label net change out of 173 — the blend's practical effect is negligible. **Verdict: Rejected as not worth the added complexity**, independent of the unverified score claim.

**Suggested evidence:** table above; a one-row example of the base-rate blend changing (or, more tellingly, not changing) a prediction.

---

## Slide 5 — Final Architecture

**Objective:** Describe exactly what is in the frozen production pipeline — not the exploratory branches.

**Bullets:**
- **Stage 1 (`C1`):** char-level TF-IDF (2–5gram, char_wb, min_df=2) + 6 structural meta-features (length, line count, Latin-script presence, digit presence, quote count, max word length), combined via `FeatureUnion`, scaled → LinearSVC. 1,016 total features. Trained on the full (corrupted) 1,767-row train set.
- **Stage 2 (`G`):** the frozen C2 pipeline (field-extractor + lemmatized title/narrative TF-IDF + 4 numeric features + gender one-hot, 2,673 base features) **plus the 11 H8/H9-promoted contradiction features** (age/gender/severity/condition contradiction flags) → LogisticRegression (`class_weight='balanced'`, `max_iter=2000`). Fixed threshold = 0.50 (re-confirmed optimal by H8's threshold re-sweep).
- **Important correction, disclosed for transparency:** the notebook's separately-coded rule-based age/gender hard-override function (`rule_based_prediction()`, CLAUDE.md §4) is **not invoked in the actual production inference path** — it exists in the codebase and is used only by an offline ensemble-candidate builder, not by the frozen submission pipeline. What *is* in production is the 11 contradiction signals as trained **features** inside G, not a hard post-hoc override. This was caught and corrected during a pre-freeze documentation audit (`h12/checkpoint/checkpoint_manifest.json`).
- Freeze status: checksummed, immutable production artifact (`FROZEN per FINAL FREEZE DIRECTIVE`), refit once on full train data, never retrained after.
- Submission integrity: automated 12-point structural checklist passed (row counts 442/173, ordering, schema match, checksum) — a correctness check, not a performance claim (`final_submission/final_review.json`).

**Suggested evidence:** end-to-end inference-pipeline diagram: raw title/protocol → `ProtocolFieldExtractor` + lemmatization → feature union (lexical + structural + contradiction) → C1 / G → thresholded labels → `submission.csv`.

---

## Slide 6 — Conclusions

**Objective:** State the core finding plainly and without overclaiming.

**Bullets:**
- Within the available time and data (690–1,767 labeled rows per stage), **classical lexical modeling combined with targeted clinical contradiction-feature engineering** produced the strongest *validated* result: C2 + 11 contradiction features, +0.023 M2 over the classical baseline, confirmed reproducible and robust across folds and subgroups (H8/H9).
- Four independent transformer strategies (raw title/narrative, head+tail, structured-template input, hybrid CLS+features) were tested under identical CV conditions and **none matched the classical baseline** — evidence that generic sequence modeling under-uses the structured signal (age/gender/severity/condition) explicit contradiction features exploit directly, at this data scale.
- This is presented as the strongest **validated** solution found within this project's time and data constraints — not a claim of global optimality.
- One real-world submission point (49.2/70) is consistent with, and did not contradict, the offline validation; a full per-stage confirmation of that consistency was designed but not executed.

**Suggested evidence:** single summary table — Frozen C2 (0.679) → +Contradiction features (0.702, validated) vs. best transformer (T4, 0.540, rejected) vs. LLM zero-shot (not comparably scored, not adopted).

---

## Slide 7 — Limitations & Future Work

**Objective:** Be explicit about what remains open, distinguishing implemented work from ideas not yet tried.

**Limitations (implemented, evidence-documented):**
- Stage 1 training text is irreversibly corrupted (~84% of characters lost); the model relies on structural/meta features as a workaround, not a fix.
- Small labeled sets (1,767 / 690 rows) constrain both transformer fine-tuning and confidence in subgroup-level metrics.
- Real-world stage-level validation is incomplete: isolation probes were designed and integrity-checked but never submitted, so Stage 1 vs. Stage 2 real-world contribution is not separately confirmed.
- 4 of 11 contradiction features have counter-intuitive coefficient signs; documented as net-effect artifacts of linear fitting, not verified as bug-free at the individual-row level.
- Known blind spot: age-missing subgroup (n=12) scores macro-F2 = 0.357, far below the 0.816 overall — a small but severe failure mode.
- Longest-protocol quartile is consistently the weakest subgroup across every experiment phase (H7 and H9 both flag it).
- LLM zero-shot predictions were never calibrated against ground truth; any audit-layer role remains unvalidated.

**Future work (not implemented — explicitly distinguished from the above):**
- Submit the prepared isolation probes to obtain a real per-stage (M1 vs. M2) score.
- Tree-structured parsing of guideline hierarchy (currently flattened into `title_text`).
- Explicit gating variables for clinically ambiguous subsections (e.g., EF/AF-based CHF criteria) rather than relying on general contradiction flags.
- A richer clinical ontology/synonym layer for condition matching.
- A calibrated LLM-as-auditor layer scoped to G's marginal-probability band (~0.44–0.56), per H11's recommendation.
- A hand-labeled disagreement-review set to measure real LLM precision/recall before any ensemble role.

---

## Data provenance note (for internal use, not a slide)

Excluded from this document, per explicit decision: a set of "real leaderboard" per-stage numbers (`M1_real ≈ 0.638`, `M2_real ≈ 0.731`) that appear in some of the project's internal experiment logs (`h12/checkpoint/checkpoint_manifest.json`, `h13/analysis/threshold_variant_manifest.json`). These are mathematically underdetermined from the one real combined score on record (one equation, two unknowns) and are contradicted by the same project's own audit files (`h11/analysis/h11_strategy_report.md`, `h12/platform/platform_diagnosis.json`, `h12/analysis/h12_final_strategy.md`), which state plainly that the isolation probes needed to measure them were never submitted (`n_scored_probes = 0`). Everywhere this document mentions a real-world score, it is the single audited value: **49.2/70 combined**, sourced from `h12/platform/submission_history.csv`.
