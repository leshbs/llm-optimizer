"""Append the H18 (Stage 2 data integrity + metric decomposition) section.

Idempotent and deterministic, like the other section generators.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"
PREFIX = "h18"

CELLS = [
    (
        "markdown",
        "intro",
        """## 132. H18 -- Stage 2 data integrity, and what the leaderboard actually measures

Section 2.1 found that `train_stage1.csv` arrived damaged from a cp1252 export.
That raises an obvious question about the rest of the download: the Stage 2
files came out of the same batch on the same day. This section audits them, and
then uses the two leaderboard readings now available to separate the score into
its Stage 1 and Stage 2 halves exactly.""",
    ),
    (
        "code",
        "audit",
        """# ---- 132.0: Stage 2 file and frame integrity ----
# Regenerate with:  python src/audit_stage2_data.py
import json
from pathlib import Path
import pandas as pd

H18_DIR = Path("h18")
_a = json.loads((H18_DIR / "stage2_data_audit.json").read_text(encoding="utf-8"))

print("byte-level integrity")
print(pd.DataFrame(_a["byte_profile"]).T.to_string())
print()
print("frame-level integrity")
_keep = ["shape", "nulls", "duplicate_ids", "duplicate_pairs", "empty_protocol",
         "mojibake_rows", "n_distinct_protocols", "n_distinct_titles"]
print(pd.DataFrame({k: {c: _a[k][c] for c in _keep} for k in ("train", "test")}).to_string())""",
    ),
    (
        "markdown",
        "audit_res",
        """**The Stage 2 files are not encoding-damaged.** Both are valid UTF-8, 83% of
their bytes are high bytes (i.e. real Cyrillic), the literal-`?` share is
0.0015%, there are no mojibake rows, no nulls, no duplicate ids and no duplicate
(title, protocol) pairs. Whatever went wrong with `train_stage1.csv` did not
happen here.

Two structural facts are worth noting before the next cell, because they shape
how Stage 2 should be validated:

* 690 training rows contain only **221 distinct protocols** and **37 distinct
  titles** -- the median title appears in 20 rows. Stage 2 is a
  many-to-many join, not 690 independent observations.
* the test set reuses training protocols heavily, so this is not a case where
  grouped cross-validation would be more honest. The next cell measures it.""",
    ),
    (
        "code",
        "truncation",
        """# ---- 132.1: the one real defect -- export truncation ----
from io_utils import read_competition_csv

_tr2 = read_competition_csv("train_stage2.csv")
_te2 = read_competition_csv("test_stage2.csv")
_all = pd.concat([_tr2["protocol_text"], _te2["protocol_text"]]).astype(str).drop_duplicates()
_len = _all.str.len()

print("distinct protocols: %d" % len(_all))
print("  at exactly 32,765 chars : %d" % (_len == 32765).sum())
print("  longest below that cap  : %d" % _len[_len < 32765].max())
print("  -> a %d-character gap with nothing in it" % (32765 - _len[_len < 32765].max()))
print()
for _name, _df in (("train_stage2", _tr2), ("test_stage2", _te2)):
    _n = (_df["protocol_text"].astype(str).str.len() == 32765).sum()
    print("%s: %d / %d rows affected (%.1f%%)" % (_name, _n, len(_df), 100 * _n / len(_df)))
print()
_ex = _tr2.loc[_tr2["protocol_text"].str.len() == 32765, "protocol_text"].iloc[0]
print("how a truncated protocol ends:")
print("  ..." + _ex[-90:].replace("\\n", " "))""",
    ),
    (
        "markdown",
        "truncation_res",
        """**17 distinct protocols are cut off at exactly 32,765 characters**, and the
longest untruncated protocol is 26,986 -- a 5,779-character gap with nothing in
between. That is a hard ceiling, not a length distribution. 32,767 is the
per-cell character limit of Excel and several CSV exporters, so these protocols
went through a spreadsheet on the way out, which fits the cp1252 damage in
section 2.1 exactly.

Because long protocols are reused across rows, this touches **147 of 690
training rows (21.3%)** and **40 of 173 test rows (23.1%)**.

Nothing in this repository caused it: no cell and no module writes either Stage
2 file, and the copies in the first commit are identically truncated. It arrived
with the download, like the Stage 1 damage.""",
    ),
    (
        "code",
        "impact",
        """# ---- 132.2: does the truncation actually cost anything? ----
import re

_p = _tr2["protocol_text"].astype(str)
_trunc = _p.str.len() == 32765

print("what sits in the last 400 chars of UNtruncated protocols:")
_tails = _p[~_trunc].str[-400:]
for _pat, _label in ((r"Показатель", "lab-result row"), (r"Диагноз", "diagnosis"),
                     (r"Рекомендации", "recommendations"), (r"Заключение", "conclusion")):
    print("  %-18s %.1f%%" % (_label, 100 * _tails.str.contains(_pat).mean()))

print()
print("are the decision-relevant fields inside the surviving prefix?")
for _f in ("Код МКБ-10", "Пол:", "Возраст:", "Жалобы", "Анамнез"):
    _first = _p.str[:2000].str.contains(_f, regex=False).mean()
    _any = _p.str.contains(_f, regex=False).mean()
    print("  %-12s first 2k chars %.3f   anywhere %.3f" % (_f, _first, _any))

print()
print("error rate on truncated vs intact rows (cached config-G out-of-fold):")
_oof = pd.read_csv(Path("h8") / "variants" / "oof_g_final.csv")
_oof["truncated"] = _oof["protocol_text"].astype(str).str.len() == 32765
_oof["pred"] = (_oof["oof_proba_g"] >= 0.50).astype(int)
_oof["err"] = _oof["pred"] != _oof["label"]
_oof["fn"] = (_oof["label"] == 1) & (_oof["pred"] == 0)
print(_oof.groupby("truncated").agg(n=("err", "size"), error_rate=("err", "mean"),
                                    false_negative_rate=("fn", "mean")).round(4).to_string())""",
    ),
    (
        "markdown",
        "impact_res",
        """**The truncation is real but harmless for this task**, and that conclusion is
measured rather than assumed:

* what gets cut is the **laboratory table**. 58.7% of untruncated protocols end
  in a `Показатель:` row, while a diagnosis appears in the final 400 characters
  of only 3.5% and a conclusion of 0.4%. The competition brief's "Strategy B --
  tail truncation, useful if conclusions appear near the end" does not apply to
  this dataset: the conclusions are not at the end.
* every decision-relevant structured field sits in the **first 2,000
  characters** -- `Код МКБ-10`, `Пол`, `Возраст` and `Анамнез` each appear in
  the opening 2k at exactly the rate they appear anywhere (0.720 vs 0.720), so
  the 32k cut never removes one.
* truncated rows are **not harder**: out-of-fold error rate 0.1837 against
  0.1768 on intact rows, and their false-negative rate is *lower*, 0.0952
  against 0.1234.

So re-downloading the Stage 2 files is worth doing for correctness, but the
expected score effect is approximately zero, and no modelling decision in this
notebook rests on the missing text.""",
    ),
    (
        "markdown",
        "metric_md",
        """### 132.3 Separating M1 from M2 in the leaderboard score

The platform reports a single number, `70 * (0.3 * M1 + 0.7 * M2)`, so one score
cannot tell the stages apart. Two facts make the split exact.

An all-zero Stage 1 prediction forces macro-F0.5 below 0.5, which the M1 clip
maps to exactly **0** -- section 13 verified this empirically to a residual of
8e-6. A probe submitted with that trivial Stage 1 therefore reads out `49 * M2`
directly. That probe scored **35.8078**, which makes

$$M2 = 35.8078 / 49 = 0.73077$$

a **measured** quantity, not an estimate. Every later submission that leaves
Stage 2 byte-identical can then be inverted for its real M1.""",
    ),
    (
        "code",
        "metric",
        """# ---- 132.3: decompose both leaderboard readings ----
# Regenerate with:  python src/metric_decomposition.py
_m = json.loads((H18_DIR / "metric_decomposition.json").read_text(encoding="utf-8"))

print(pd.DataFrame(_m["submissions"]).T.round(4).to_string())
print()
print("M2 real (measured)   : %.5f" % _m["M2_real_measured"])
print("M2 offline (5-fold)  : %.5f" % _m["M2_offline_cv"])
print("  CV runs low by     : %.5f" % _m["M2_cv_underestimates_by"])
print()
print("M1 H17 nested CV     : %.4f" % _m["offline_reference"]["M1_h17_nested_cv"])
print("M1 H17 real          : %.4f" % _m["M1_h17_real"])
print("  CV runs low by     : %.4f" % _m["M1_h17_cv_underestimates_by"])
print()
print("remaining headroom   : Stage 1 %.2f pts, Stage 2 %.2f pts"
      % (21 * (1 - _m["M1_h17_real"]), 49 * (1 - _m["M2_real_measured"])))""",
    ),
    (
        "markdown",
        "metric_res",
        """| | offline CV | real | CV error |
|---|---|---|---|
| M1 (H17 Stage 1) | 0.8434 | **0.9219** | −0.0785 |
| M2 (config G) | 0.7024 | **0.7308** | −0.0284 |

**Cross-validation understates both stages, consistently.** Two causes, and
neither is a bug:

1. every CV model trains on 80% of the data while the shipped model trains on
   100%, which matters a lot at n=1,767 and n=690;
2. the folds are *harder* than the real test set. 81.9% of held-out Stage 2 rows
   share a protocol with their training part, against 89.0% for the real test
   set; the Stage 1 equivalent is 94.1% against 94.6% (section h16). The test
   set reuses training material slightly more than cross-validation does, so CV
   is mildly pessimistic by construction.

**Which M2 to quote, and when.** They are not competing estimates of the same
thing, and both belong in the project:

* **M2_real = 0.7308 for score accounting** -- reporting the result, computing
  how many points remain, and gating a candidate that has actually been scored
  on the leaderboard. It is measured, so it needs no defending.
* **M2_offline = 0.7024 for model selection** -- comparing an unscored Stage 2
  variant against the incumbent. It is the only figure obtainable for a
  candidate that has not been submitted, and competition rule 12 forbids
  selecting on leaderboard feedback anyway.
* **Never compare the two.** An offline reading runs about 0.028 low, so
  measuring a new offline M2 against the real 0.7308 would reject a genuinely
  better model for free. Offline goes against offline; real goes against real.
  Sections 35-47 already follow this (every promotion gate there is
  offline-versus-offline) and section 13's probe gate is real-versus-real.""",
    ),
    (
        "markdown",
        "close",
        """### H18 summary

**Stage 2 loading and data: no bugs to fix.** Both files are valid UTF-8 with no
corruption, no nulls, no duplicate ids, no duplicated pairs and no mojibake. The
inference path is aligned end to end -- `stage2_predictions.csv` carries the
same 173 ids in the same order as `test_stage2.csv`, the shipped labels are
reproduced exactly by thresholding the saved probabilities at 0.50 (agreement
1.0000), and the Stage 2 block of `submission.csv` matches the predictions row
for row.

**One upstream data defect, quantified and dismissed.** 21.3% of training rows
and 23.1% of test rows carry a protocol truncated at the 32,765-character export
limit. It removes laboratory tables, never the structured fields or the
conclusions, and truncated rows are no harder to classify. Re-downloading the
Stage 2 files is worthwhile hygiene, not a score opportunity. The loader in
section 1 now reports the truncation count so it stays visible.

**A bookkeeping inconsistency worth correcting.**
`h12/checkpoint/checkpoint_manifest.json` records the pre-H17 leaderboard total
as 49.21 and derives `M1_real(C1) = 0.6382` from it, but the total reported for
that submission was 49.021, which implies 0.6292. Only the size of the
historical baseline delta depends on which is right; H17's real M1 of 0.9219 is
derived from its own total against the measured M2 anchor and stands either way.

**Where the points are now.** Stage 1 earns 19.36 of 21 and has **1.64 points**
left. Stage 2 earns 35.81 of 49 and has **13.19 points** left -- eight times as
much. Stage 1 is finished; everything remaining is Stage 2, and the h16 error
audit already narrowed the target there: false negatives outnumber false
positives 81 to 42 under a recall-weighted metric, protocol length is not the
driver, and document-level negation is useless because it fires on 97.5% of
protocols. Scoped negation -- negation within the span of the title's condition
term -- is the open question.""",
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
    print("removed %d stale H18 cells, appended %d, total now %d"
          % (removed, len(CELLS), len(nb["cells"])))


if __name__ == "__main__":
    main()
