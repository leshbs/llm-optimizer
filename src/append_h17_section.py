"""Append the H17 (clean-text Stage 1) section to baseline.ipynb.

Same contract as the H14 and H15 generators: deterministic, idempotent, and the
notebook is edited programmatically rather than by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"
PREFIX = "h17"

CELLS = [
    (
        "markdown",
        "intro",
        """## 131. H17 -- Stage 1 on the repaired training file

Sections 129 and 130 were built around a single constraint: the Cyrillic in
`train_stage1.csv` had been destroyed by a cp1252 export (section 2.1). Section
129 responded by throwing away lexical information entirely and modelling word
*shapes*; section 130 recovered about half the subsection lines by matching
masked patterns against the readable files. Both were the right response to the
data as it stood.

That constraint is gone. The file has been re-downloaded from the competition
platform and is clean UTF-8 -- same 1,767 ids in the same order, same labels,
same 0.2168 positive rate, verified in section 2.1. **Stage 1 is now an ordinary
Russian text-classification problem**, and this section treats it as one.

Nothing here masks, decodes or matches against a corpus. The model reads the
words. What carries over from the earlier work is the *evidence*, not the
machinery:

* section 130's ablation showed the subsection line and its surrounding
  hierarchy carry different signal and that both are needed, so they stay as
  separate feature blocks;
* the brief's section 7 defines Special as "names a restricted patient group",
  so the condition-marker block is kept -- and now fires on every row instead of
  the 19% that used to decode;
* lemmatisation matters for Russian morphology (brief section 5.3), so
  `тяжёлой` / `тяжелая` / `тяжелый` collapse to one feature;
* char n-grams are retained to absorb morphological variation and
  out-of-vocabulary medical terms that lemmatisation misses.

The implementation is in `src/stage1_clean.py`, evaluated by
`src/stage1_clean_eval.py` and `src/stage1_clean_select_C.py`, and built by
`src/stage1_clean_build.py`.""",
    ),
    (
        "code",
        "imports",
        """# ---- 131.0: the clean-text Stage 1 model ----
import sys, json, hashlib
from pathlib import Path

SRC_DIR_131 = Path("src").resolve()
if str(SRC_DIR_131) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_131))

from stage1_clean import (
    STRUCT_FEATURE_NAMES,
    Field,
    StructFeatures,
    build_clean_features,
    build_clean_model,
    title_fields,
)
from io_utils import is_corrupted

H17_DIR = Path("h17")
H17_DIR.mkdir(exist_ok=True)

CHECKPOINT_131 = Path("h12") / "checkpoint" / "checkpoint_submission.csv"
_checkpoint_hash_before_131 = hashlib.sha256(CHECKPOINT_131.read_bytes()).hexdigest()

# This whole section assumes the repaired file. Fail loudly rather than
# silently producing a worse model if the damaged one is ever restored.
assert not is_corrupted("train_stage1.csv"), (
    "train_stage1.csv is the damaged copy -- use section 129/130 instead"
)

_g, _age, _hier, _sub = title_fields(train_s1["title_text"].iloc[0])
print("a training title, now readable:")
print("  guideline  :", _g)
print("  age        :", _age)
print("  hierarchy  :", _hier)
print("  subsection :", _sub)""",
    ),
    (
        "markdown",
        "transfer_md",
        """### 131.1 The defect that started all of this

Section 129 opened by measuring how badly the production Stage 1 features failed
to transfer: 0.9468 of the training tf-idf mass sat on `?`-bearing n-grams that
could never fire at inference, against 0.0000 of the test mass, and the model
saw 51.4 active features per training row but only 12.4 per test row.

With real text on both sides that failure mode simply does not arise. The check
is repeated here rather than assumed.""",
    ),
    (
        "code",
        "transfer",
        """# ---- 131.1: does the lexical vocabulary transfer now? ----
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_eval_131 = json.loads((H17_DIR / "stage1_clean_eval.json").read_text(encoding="utf-8"))
print(pd.Series(_eval_131["transfer"]).to_string())
print()
print("compare with the corrupted-era pipeline (section 129.1):")
print("  train tf-idf mass on dead n-grams : 0.9468   ->  not applicable, no dead n-grams")
print("  active features per row train/test: 51.4 / 12.4  ->  %.1f / %.1f"
      % (_eval_131["transfer"]["mean_active_features_train"],
         _eval_131["transfer"]["mean_active_features_test"]))""",
    ),
    (
        "markdown",
        "ablation_md",
        """### 131.2 Which blocks earn their place

Re-run from scratch on the clean text -- the H15 ablation cannot be carried over,
because every block now sees different input.""",
    ),
    (
        "code",
        "ablation",
        """# ---- 131.2: ablation and hyperparameter selection ----
# Regenerate with:  python src/stage1_clean_eval.py   (a few minutes)
print("ablation, 5-fold, threshold tuned in-fold (optimistic -- ranking only):")
print(pd.DataFrame(_eval_131["ablation"]).round(4).to_string(index=False))

_sel_131 = json.loads((H17_DIR / "stage1_clean_C_selection.json").read_text(encoding="utf-8"))
print()
print("C chosen on the NESTED estimate, not the in-fold one:")
_rows = {c: {"pooled_m1": v["pooled_m1"], "per_fold_sd": v["std_fold_m1"],
             "threshold_median": v["threshold_median"]}
         for c, v in _sel_131["nested_by_C"].items()}
print(pd.DataFrame(_rows).T.round(4).to_string())
print()
print("selected C =", _sel_131["best_C"], " threshold =", _sel_131["threshold_median"])
print("top two are within %.4f M1 of each other" % _sel_131["spread_top_two"])""",
    ),
    (
        "markdown",
        "ablation_res",
        """The blocks stack cleanly: subsection alone 0.7347, adding the hierarchy context
0.7431, adding char n-grams 0.7955, adding structural and condition features
**0.8125** (all in-fold tuned, so optimistic and comparable only to each other).

Selecting `C` deserves a note. The in-fold sweep prefers `C=30`, but that column
tunes its threshold on the folds it scores and overstates every configuration.
Choosing on the *nested* estimate instead puts `C=10` ahead (0.8434 against
0.8396) with by far the tightest spread across folds (SD 0.0373 against 0.0526).
The top two land within 0.0038 of each other, so the honest read is "about 0.84
anywhere in the 10-30 range", not a precise optimum. Selecting on the nested
estimate does make the winner mildly optimistic -- that caveat is recorded in
`h17/stage1_clean_C_selection.json` rather than buried.""",
    ),
    (
        "markdown",
        "nested_md",
        """### 131.3 Comparison against the corruption-era models

Same folds, same seed, same nested protocol, same official metric as sections
129 and 130. The repaired file has identical ids in identical order with
identical labels, so `StratifiedKFold(seed=42)` produces literally the same five
folds -- these numbers are directly comparable, not approximately so.""",
    ),
    (
        "code",
        "nested",
        """# ---- 131.3: H14 vs H15 vs H17 ----
_n14 = json.loads((Path("h14") / "stage1_tuning.json").read_text(encoding="utf-8"))["nested_cv"]
_n15 = json.loads((Path("h15") / "stage1_hybrid_eval.json").read_text(encoding="utf-8"))["nested_cv"]
_n17 = _sel_131["nested_by_C"][str(_sel_131["best_C"])]

print(pd.DataFrame({
    "H14 shape features": {"pooled M1": _n14["pooled_m1"], "per-fold SD": _n14["std_fold_m1"],
                           "positive rate": _n14["pooled_posrate"]},
    "H15 decoded hybrid": {"pooled M1": _n15["pooled_m1"], "per-fold SD": _n15["std_fold_m1"],
                           "positive rate": _n15["pooled_posrate"]},
    "H17 clean text": {"pooled M1": _n17["pooled_m1"], "per-fold SD": _n17["std_fold_m1"],
                       "positive rate": _n17["pooled_posrate"]},
}).T.round(4).to_string())

print("\\nper-outer-fold M1")
for _name, _s in (("H14", _n14["fold_scores"]), ("H15", _n15["fold_scores"]),
                  ("H17", _n17["fold_scores"])):
    print("  %s: %s" % (_name, ["%.4f" % v for v in _s]))
_wins = sum(c > b for b, c in zip(_n15["fold_scores"], _n17["fold_scores"]))
print("  H17 ahead of H15 on %d of %d folds" % (_wins, len(_n17["fold_scores"])))

for _label, _ref in (("H15", _n15["pooled_m1"]), ("H14", _n14["pooled_m1"])):
    _d = _n17["pooled_m1"] - _ref
    print("\\nvs %s: %+.4f M1 = %+.2f metric points" % (_label, _d, 70 * 0.3 * _d))""",
    ),
    (
        "markdown",
        "final_md",
        """### 131.4 Final build

Refit on all 1,767 rows at the median inner-fold threshold. The Stage-2-title
rule from section 129.6 carries over unchanged -- it never depended on the
corruption, only on the fact that a title appearing in the Stage 2 data is
Special by construction -- and is now an exact string match rather than a masked
one. Stage 2 is copied byte for byte from the production predictions and the
frozen checkpoint is hashed before and after.""",
    ),
    (
        "code",
        "build",
        """# ---- 131.4: load the built submission and re-verify ----
# Regenerate with:  python src/stage1_clean_build.py
_man_131 = json.loads((H17_DIR / "stage1_clean_manifest.json").read_text(encoding="utf-8"))
_sub_131 = pd.read_csv(H17_DIR / "submission_stage1_clean.csv")
_s2_frozen = pd.read_csv(Path("final") / "predictions" / "stage2_predictions.csv")

_s1 = _sub_131[_sub_131["stage"] == 1]
_s2 = _sub_131[_sub_131["stage"] == 2]
assert len(_sub_131) == 615, "submission must have 615 rows"
assert _sub_131["label"].isin([0, 1]).all(), "labels must be binary"
assert list(_s1["id"]) == list(test_s1["id"]), "stage 1 row order"
assert list(_s2["label"]) == list(_s2_frozen["label"]), "stage 2 must be untouched"

print("threshold             %.2f" % _man_131["threshold"])
print("stage 1 positive rate %.4f   (train prior %.4f)"
      % (_man_131["test_positive_rate"], _man_131["train_positive_rate"]))
print("stage-2 title rule    %d matched, precision %.4f, %d flipped"
      % (_man_131["stage2_title_rule"]["n_matched"],
         _man_131["stage2_title_rule"]["precision"],
         _man_131["stage2_title_rule"]["n_flipped"]))
print()
for _tag, _d in _man_131["deltas"].items():
    print("  vs %-11s agreement %.4f  (%d rows differ)" % (_tag, _d["agreement"], _d["n_differ"]))

_after_131 = hashlib.sha256(CHECKPOINT_131.read_bytes()).hexdigest()
assert _after_131 == _checkpoint_hash_before_131, "checkpoint modified -- MUST NEVER HAPPEN"
print("\\ncheckpoint intact:", _after_131[:16])
print("submission sha256:", _man_131["submission_sha256"][:16])""",
    ),
    (
        "markdown",
        "close",
        """### H17 summary

**Result.** Nested-CV M1 **0.7527 -> 0.8434**, ahead of the decoder hybrid on
**5 of 5** outer folds, with the per-fold spread tightening from 0.0593 to
0.0373. That is **+0.0907 M1 = +1.90 metric points** over H15 and +0.1323 =
+2.78 over H14.

**The calibration problem solved itself.** Stage 1's predicted positive rate
goes to **0.2195** against a training prior of 0.2168. The production model sat
at 0.0588 and section 129 measured that under-prediction as costing 0.1954 M1 on
its own. No threshold engineering was needed here -- a model that can read the
words is simply calibrated.

**The rule is now redundant but harmless.** The Stage-2-title rule still audits
at 32/32 = 100% precision on train, but on test it flips **zero** predictions:
the clean-text model already classifies all five matching titles as Special
unaided. It is kept because it costs nothing and remains a correct constraint.

**What this says about sections 129 and 130.** They were not wasted, but they
were working around a data defect rather than solving the task. The honest
lesson is that the largest single gain in this notebook's Stage 1 history came
from diagnosing a broken input file and re-downloading it -- not from any
modelling choice. Section 2.1 records the diagnosis; it is worth more in the
write-up than the feature engineering it displaced.

**Where the remaining headroom is.** Stage 1 now contributes
`21 * 0.8434 = 17.7` of its 21 available points. Stage 2 is still frozen at CV
M2 0.7024 and carries 49 points, of which it earns about 34.4 -- so roughly
**14.6 points** remain there against **3.3** here. Every further hour belongs to
Stage 2, and the h16 audit narrowed the target: false negatives outnumber false
positives 81 to 42 under a recall-weighted metric, protocol length is not the
cause, and document-level negation is useless because it fires on 97.5% of
protocols. Scoped negation is the open question.""",
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
    print("removed %d stale H17 cells, appended %d, total now %d"
          % (removed, len(CELLS), len(nb["cells"])))


if __name__ == "__main__":
    main()
