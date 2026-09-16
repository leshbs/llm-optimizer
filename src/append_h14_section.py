"""Append the H14 (Stage 1 rebuild) section to baseline.ipynb.

The notebook is edited programmatically rather than by hand so the section can
be regenerated deterministically. Running this twice is a no-op: any existing
H14 cells are removed first.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"
PREFIX = "h14"

CELLS = [
    (
        "markdown",
        "h14_intro",
        """## 129. H14 -- Stage 1 rebuild on a corruption-invariant feature space

Every preceding section treated Stage 1 as settled: config `C1`
(`char_wb` TF-IDF + 6 meta features + `LinearSVC`) was picked by the section 6
ablation and never revisited, on the reasonable grounds that Stage 2 carries 70%
of the metric weight. This section revisits it, because an audit of the
train/test feature transfer shows the production Stage 1 model is largely
inoperative at inference time.

**The defect.** `train_stage1.csv` had every Cyrillic character replaced by the
literal ASCII byte `?`. The substitution is *length preserving* and leaves all
non-Cyrillic characters intact. `test_stage1.csv` is clean. So a lexical
vectoriser fit on train learns weights on n-grams that cannot occur in test:

| | train | test |
|---|---|---|
| tf-idf mass on `?`-bearing n-grams | **0.9468** | **0.0000** |
| mean active features per row | 51.4 | 12.4 |

95% of the model's learned weight is dead at inference, and L2 normalisation
then rescales the surviving ASCII n-grams differently on the two splits. Section
2.1 anticipated the *direction* of this problem and section 6.1 responded by
adding meta-features, but the magnitude was never measured -- and the six meta
features are not enough to carry the model alone (CV M1 = 0.4171 for meta only).

**The fix.** Apply the same corruption to the test text and build features only
from what survives it. Masking *both* sides makes the feature space provably
identical across splits, which also makes cross-validation an honest estimate of
transfer rather than an optimistic one. Two invariant representations are used:

* a **numeric skeleton** -- 23 counts, lengths and punctuation statistics that
  take identical values on the corrupted and clean forms of a title;
* **shape n-grams** -- the sequence of word *lengths* in the subsection title
  (`w7 w5 w9`), which survives masking exactly and recovers a useful amount of
  phrase structure, plus the age-category line emitted as a categorical (it
  survives masking bijectively and its label rate ranges 0.079-0.292 across its
  four values, which the production meta-features never extracted).

The implementation lives in `src/stage1_invariant.py` and `src/stage1_rules.py`
rather than in this notebook, so that it is importable, testable and reusable
from the inference path.""",
    ),
    (
        "code",
        "h14_imports",
        """# ---- 129.0: import the Stage 1 rebuild modules ----
import sys, json, hashlib
from pathlib import Path

SRC_DIR_129 = Path("src").resolve()
if str(SRC_DIR_129) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_129))

from stage1_invariant import (
    SEED as SEED_129,
    SKELETON_FEATURE_NAMES,
    SkeletonFeatures,
    InvariantTokenizer,
    build_stage1_model,
    invariant_tokens,
    mask_cyrillic,
    parse_title,
)
from io_utils import read_competition_csv
from stage1_eval import stage1_score as stage1_score_129, tune_threshold
from stage1_rules import (
    stage2_title_index,
    audit_stage2_title_rule,
    apply_stage2_title_rule,
)
from stage1_tune import configure, nested_cv

H14_DIR = Path("h14")
H14_DIR.mkdir(exist_ok=True)

# The frozen checkpoint must never be touched by anything in this section.
CHECKPOINT_129 = Path("h12") / "checkpoint" / "checkpoint_submission.csv"
_checkpoint_hash_before_129 = hashlib.sha256(CHECKPOINT_129.read_bytes()).hexdigest()

train_s1_129 = read_competition_csv("train_stage1.csv")
test_s1_129 = pd.read_csv("test_stage1.csv")
X_129 = train_s1_129["title_text"].to_numpy(dtype=object)
y_129 = train_s1_129["label"].to_numpy()

print("train %d rows, positive rate %.4f" % (len(y_129), y_129.mean()))
print("test  %d rows" % len(test_s1_129))
print("invariant tokens for one title:")
print(" ", invariant_tokens(test_s1_129["title_text"].iloc[0]))""",
    ),
    (
        "markdown",
        "h14_audit_md",
        """### 129.1 Quantifying the transfer defect

Before replacing anything, the claim that the production feature space does not
transfer is measured directly, and the claim that the replacement *does* is
measured the same way. `src/stage1_transfer_check.py` produces this comparison.""",
    ),
    (
        "code",
        "h14_transfer",
        """# ---- 129.1: transfer audit, production vs rebuild feature spaces ----
import subprocess

subprocess.run([sys.executable, "src/stage1_transfer_check.py"], check=True)

with open(H14_DIR / "stage1_transfer_check.json", encoding="utf-8") as f:
    transfer_129 = json.load(f)

transfer_table_129 = pd.DataFrame(
    {
        "production_char_tfidf": transfer_129["production_char_tfidf"],
        "rebuild_shape_tfidf": transfer_129["rebuild_shape_tfidf"],
    }
)
display(transfer_table_129)

skeleton_drift_129 = pd.read_csv(H14_DIR / "stage1_skeleton_drift.csv")
print("skeleton features more than 0.5 SD apart between train and test: %d"
      % transfer_129["skeleton_drift"]["n_features_above_0.5_sd"])
display(skeleton_drift_129.head(8))""",
    ),
    (
        "markdown",
        "h14_transfer_res",
        """**Result.** The production char TF-IDF puts **94.68%** of train mass on n-grams
carrying **0.00%** of test mass, and sees 51.4 active features per train row
against 12.4 per test row -- a 4x mismatch that L2 normalisation converts into a
systematic rescaling of the surviving features.

The rebuild puts **41.13%** of train mass and **41.66%** of test mass on the same
`?`-bearing features (these are the `AGE=` categorical, which is invariant by
construction and so is *meant* to fire on both splits), with **35.6** active
features per train row against **34.8** per test row, and **no test row with zero
active features**. On the numeric skeleton, the largest standardised mean
difference between train and test across all 23 features is **0.14 SD**, and
**zero** features differ by more than 0.5 SD.

The two splits are in the same feature space. That is what makes the
cross-validation below a usable estimate of leaderboard transfer.""",
    ),
    (
        "markdown",
        "h14_cv_md",
        """### 129.2 Hyperparameter sweep and nested cross-validation

The section 6 ablation compared feature *configurations* but never swept the
estimator's hyperparameters. Here a 48-point grid over the shape n-gram order,
`min_df` and `C` is scored out of fold with the official metric.

Tuning a decision threshold on the same folds that report the score is
optimistic, so the grid is used only to **rank** configurations. The number the
shipping decision rests on comes from a **nested** cross-validation: the
threshold is chosen on inner folds and applied to a held-out outer fold, so no
fold ever scores a threshold that was fit on it. This also respects rule 11 --
threshold optimisation uses training data only.""",
    ),
    (
        "code",
        "h14_tune",
        """# ---- 129.2: grid sweep + nested CV (cached; set FORCE_RERUN_129 = True to recompute) ----
FORCE_RERUN_129 = False
_tuning_path_129 = H14_DIR / "stage1_tuning.json"

if FORCE_RERUN_129 or not _tuning_path_129.exists():
    subprocess.run([sys.executable, "src/stage1_tune.py"], check=True)

with open(_tuning_path_129, encoding="utf-8") as f:
    tuning_129 = json.load(f)

grid_129 = pd.read_csv(H14_DIR / "stage1_grid.csv")
print("grid points evaluated: %d" % len(grid_129))
display(grid_129.head(8))

nested_129 = tuning_129["nested_cv"]
print("\\nbest params: %s" % tuning_129["best_params"])
print("inner-fold thresholds: %s  (median %.2f)"
      % ([round(t, 2) for t in nested_129["thresholds"]], nested_129["threshold_median"]))
print("nested-CV pooled M1 = %.4f" % nested_129["pooled_m1"])
print("per-fold M1 = %.4f +/- %.4f" % (nested_129["mean_fold_m1"], nested_129["std_fold_m1"]))
print("pooled positive rate = %.4f  (train prior %.4f)"
      % (nested_129["pooled_posrate"], y_129.mean()))""",
    ),
    (
        "code",
        "h14_compare",
        """# ---- 129.3: head-to-head against the production configuration ----
with open(H14_DIR / "stage1_cv_comparison.json", encoding="utf-8") as f:
    comparison_129 = json.load(f)

rows_129 = [
    {"model": "production C1 (char TF-IDF + meta + LinearSVC)",
     "cv_m1": comparison_129["results"]["production_F_char_meta_linearsvc"]["cv_m1"],
     "posrate": comparison_129["results"]["production_F_char_meta_linearsvc"]["oof_positive_rate"],
     "estimate": "optimistic (features do not transfer)"},
    {"model": "rebuild, invariant features @ t=0.50",
     "cv_m1": comparison_129["results"]["invariant_logreg_t050"]["cv_m1"],
     "posrate": comparison_129["results"]["invariant_logreg_t050"]["oof_positive_rate"],
     "estimate": "honest"},
    {"model": "rebuild, tuned threshold (same folds)",
     "cv_m1": comparison_129["results"]["invariant_logreg_tuned"]["cv_m1"],
     "posrate": comparison_129["results"]["invariant_logreg_tuned"]["oof_positive_rate"],
     "estimate": "optimistic (threshold fit in-fold)"},
    {"model": "rebuild, nested CV (shipping estimate)",
     "cv_m1": nested_129["pooled_m1"],
     "posrate": nested_129["pooled_posrate"],
     "estimate": "honest"},
]
stage1_comparison_129 = pd.DataFrame(rows_129)
display(stage1_comparison_129)

REAL_M1_PRODUCTION_129 = 0.6382  # real leaderboard-derived value, see H13
print("production real M1 (leaderboard-derived): %.4f" % REAL_M1_PRODUCTION_129)
print("rebuild nested-CV M1:                     %.4f" % nested_129["pooled_m1"])
_delta_m1_129 = nested_129["pooled_m1"] - REAL_M1_PRODUCTION_129
print("expected metric points from Stage 1: %+.2f  (70 * 0.3 * %.4f)"
      % (70 * 0.3 * _delta_m1_129, _delta_m1_129))""",
    ),
    (
        "markdown",
        "h14_calib_md",
        """### 129.4 The second defect: Stage 1 under-predicts the positive class

Independently of which features are used, production `C1` predicts Special on
**5.88%** of test rows against a train prior of **21.68%**. Macro-F0.5 averages
over both classes, so starving the positive class collapses its recall and
therefore its F0.5, even though precision on it is high.

The cost is measurable: constraining a well-specified model to emit only a 5.88%
positive rate drops its out-of-fold M1 from **0.6954** to **0.5000**. Under-prediction
alone accounts for **0.1954** M1, about 4.1 metric points.""",
    ),
    (
        "code",
        "h14_calib",
        """# ---- 129.4: what the under-prediction costs, measured on the rebuild's OOF probabilities ----
import numpy as np

oof_proba_129 = np.load(H14_DIR / "stage1_invariant_oof_proba.npy")

def _score_at_rate(proba, y, rate):
    \"\"\"Score the model when forced to emit exactly `rate` positives.\"\"\"
    k = int(round(rate * len(proba)))
    cut = np.sort(proba)[::-1][k - 1] if k > 0 else 1.1
    return stage1_score_129(y, (proba >= cut).astype(int))

calibration_129 = pd.DataFrame(
    [
        {"positive_rate": r,
         "cv_m1": _score_at_rate(oof_proba_129, y_129, r),
         "note": note}
        for r, note in [
            (0.0588, "production C1 test rate"),
            (0.1000, ""),
            (0.1500, ""),
            (0.1629, "rebuild test rate"),
            (0.2168, "train prior"),
            (0.3000, ""),
        ]
    ]
)
display(calibration_129)
print("cost of predicting at C1's rate instead of the rebuild's: %.4f M1"
      % (calibration_129.loc[3, "cv_m1"] - calibration_129.loc[0, "cv_m1"]))""",
    ),
    (
        "markdown",
        "h14_rule_md",
        """### 129.5 A deterministic correction: Stage 2 titles are Special by construction

Stage 2 is defined over *Special* subsections only -- the brief states the task
is to decide whether a "special clinical guideline subsection" applies to a
patient. Any title appearing in the Stage 2 data is therefore Special, and this
is checkable against Stage 1's labels rather than merely assumed.

Matching is done on the **masked** form so corrupted Stage 1 titles and clean
Stage 2 titles are comparable. This uses test *inputs* only, never test labels,
which rule 12 permits. The rule ships only if it reaches the same 100% precision
bar already applied to the Stage 2 age/gender detector.""",
    ),
    (
        "code",
        "h14_rule",
        """# ---- 129.5: audit the stage-2-title rule against Stage 1 labels ----
train_s2_129 = pd.read_csv("train_stage2.csv")
test_s2_129 = pd.read_csv("test_stage2.csv")

s2_titles_129 = stage2_title_index(train_s2_129, test_s2_129)
rule_audit_129 = audit_stage2_title_rule(train_s1_129, s2_titles_129)

print("distinct masked stage-2 titles:            %d" % len(s2_titles_129))
print("train_stage1 rows matching one of them:    %d" % rule_audit_129["n_matched"])
print("  of which Special (label 1):              %d" % rule_audit_129["n_special"])
print("  of which General (label 0):              %d" % rule_audit_129["n_general"])
print("  precision:                               %.4f" % rule_audit_129["precision"])

RULE_ENABLED_129 = rule_audit_129["n_matched"] > 0 and rule_audit_129["precision"] == 1.0
print("\\nDECISION: rule %s"
      % ("ENABLED (meets the 100% precision bar)" if RULE_ENABLED_129
         else "REJECTED (below the 100% precision bar)"))""",
    ),
    (
        "markdown",
        "h14_final_md",
        """### 129.6 Final fit, prediction and submission assembly

Stage 2 is left untouched. Five structurally different Stage 2 feature sets
(title one-hot, title one-hot + protocol TF-IDF at three `C` values, 1-2 gram
variants) all land within 0.007 of the production config's CV M2 of 0.7024, and
H13 established that thresholds 0.45 and 0.48 are real-confirmed worse while
0.52 is byte-identical to 0.50. Stage 2 is at a plateau; the rebuild changes
Stage 1 only, so any leaderboard movement is attributable to Stage 1 alone.

The shipped threshold is the **median of the inner-fold optima**, not the single
best out-of-fold threshold: the threshold curve is a broad plateau between 0.70
and 0.80 and the median is the stable point inside it.""",
    ),
    (
        "code",
        "h14_build",
        """# ---- 129.6: fit on full train, predict test, assemble submission ----
subprocess.run([sys.executable, "src/stage1_build.py"], check=True)

with open(H14_DIR / "stage1_rebuild_manifest.json", encoding="utf-8") as f:
    rebuild_manifest_129 = json.load(f)

print(json.dumps(rebuild_manifest_129, indent=2))

submission_129 = pd.read_csv(H14_DIR / "submission_stage1_rebuild.csv")
assert len(submission_129) == 615, "submission must have 442 + 173 rows"
assert list(submission_129.columns) == ["stage", "id", "label"]
assert submission_129["label"].isin([0, 1]).all()
assert list(submission_129.loc[submission_129["stage"] == 1, "id"]) == list(test_s1_129["id"])
assert list(submission_129.loc[submission_129["stage"] == 2, "id"]) == list(test_s2_129["id"])
print("\\nsubmission validated: %d rows" % len(submission_129))

# Stage 2 must be byte-identical to the production submission.
_prod_sub_129 = pd.read_csv("final_submission/submission.csv")
_a = submission_129[submission_129["stage"] == 2].reset_index(drop=True)
_b = _prod_sub_129[_prod_sub_129["stage"] == 2].reset_index(drop=True)
assert _a.equals(_b), "Stage 2 predictions must be unchanged by the Stage 1 rebuild"
print("Stage 2 predictions confirmed unchanged vs production")

_n_diff_129 = int((submission_129[submission_129["stage"] == 1]["label"].to_numpy()
                   != _prod_sub_129[_prod_sub_129["stage"] == 1]["label"].to_numpy()).sum())
print("Stage 1 predictions changed on %d / %d test rows" % (_n_diff_129, len(test_s1_129)))

_checkpoint_hash_after_129 = hashlib.sha256(CHECKPOINT_129.read_bytes()).hexdigest()
assert _checkpoint_hash_before_129 == _checkpoint_hash_after_129, \\
    "129.6: checkpoint_submission.csv was modified -- THIS MUST NEVER HAPPEN"
print("checkpoint intact: %s" % _checkpoint_hash_after_129[:16])""",
    ),
    (
        "markdown",
        "h14_close",
        """### H14 Summary

**Two defects were found in Stage 1 and both are fixed.**

*Defect 1 -- the feature space did not transfer.* 94.68% of the production
model's TF-IDF mass sat on `?`-bearing n-grams carrying 0.00% of test mass.
Rebuilding on corruption-invariant features (numeric skeleton + word-shape
n-grams + the age categorical) moves cross-validated M1 from **0.4648** to
**0.7111** under nested cross-validation, and the transfer audit confirms the
two splits now occupy one feature space (35.6 vs 34.8 active features per row,
max skeleton drift 0.14 SD).

*Defect 2 -- the positive class was starved.* Production predicted Special on
5.88% of test rows against a 21.68% prior. Threshold tuning against the official
metric, with the threshold selected on inner folds only, brings the test positive
rate to **16.29%**.

*A deterministic correction was added.* 32 of 32 training titles that also occur
in the Stage 2 data are labelled Special -- 100% precision, so the rule clears
the same bar as the Stage 2 age/gender detector. It fires on 5 test rows and
flips 1 that the rebuilt model had still called General.

**Expected effect.** Production's real M1 is 0.6382 (H13, leaderboard-derived).
The rebuild's honest estimate is 0.7111, worth `70 * 0.3 * 0.0729 = +1.53`
metric points, i.e. roughly **49.21 -> 50.7**. That is an estimate from
cross-validation, not a measured leaderboard result, and the per-fold standard
deviation of 0.0615 is wide enough that the realised gain could plausibly fall
anywhere in the +0.4 to +2.7 range.

**What this does not fix.** Stage 1 still never sees a Cyrillic character at
training time. The corruption is partly *reversible* -- masked-pattern matching
against the clean corpus recovers the age line for 1767/1767 rows, line 0 for
922/1767, and a full subsection title for 339 -- which would allow a genuine
lexical model to be trained on decoded text. That is the largest remaining
Stage 1 opportunity, and section 130 carries it out.""",
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    before = len(nb["cells"])
    nb["cells"] = [c for c in nb["cells"] if not str(c.get("id", "")).startswith(PREFIX)]
    removed = before - len(nb["cells"])

    for cell_type, cid, source in CELLS:
        cell = {
            "cell_type": cell_type,
            "id": "%s_%s" % (PREFIX, cid),
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("removed %d stale H14 cells, appended %d, total now %d"
          % (removed, len(CELLS), len(nb["cells"])))


if __name__ == "__main__":
    main()
