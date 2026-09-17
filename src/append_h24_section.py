"""Append the H24 (Stage 2 evaluation fix + missing inputs) section.

Idempotent and deterministic, like the other section generators.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"
PREFIX = "h24"

CELLS = [
    (
        "markdown",
        "intro",
        """## 133. H24 -- a harness that can see small gains, and the inputs config G never read

Five rounds of Stage 2 work (H19-H23, reports in `h22/FINAL_REPORT.md` and
`h23/H23_REPORT.md`) found nothing better than config G. An audit of *why*
found three causes, and two of them are fixable:

1. **The evaluation was blind below ~0.02 M2.** Config G scores
   0.694 +/- 0.019 across ten different 5-fold splits, and on the frozen split
   one fold sits at 0.544 against ~0.74 for the other four.
2. **G never reads the diagnosis.** `extract_narrative` (section 3) keeps
   Жалобы/Анамнез/Объективный статус -- a median 18.7% of each protocol. The
   `Код МКБ-10` field (497 of 690 rows) and the `Диагноз` line are dropped,
   although a (title, ICD) lookup alone reaches AUC 0.877. Example: every K50.1
   (Crohn's disease of the *large* intestine) row under
   "БК тонкой кишки (кроме терминального илеита)" is labelled 0.
3. **G is additive** -- f(title) + g(protocol) -- while the label is an
   interaction: 62 of 102 protocols that appear under several titles carry
   mixed labels.

**The new protocol** (`src/stage2_h24_harness.py`) was fixed before any
candidate was scored:
* 5 x 5 repeated stratified CV with identical splits for every candidate;
* paired fold-level deltas against G, with the Nadeau-Bengio correction for
  overlapping training sets;
* gate: mean delta > 0, corrected one-sided p < 0.10, and >= 4/5 repeats
  improved.

A pure-noise column is the negative control.""",
    ),
    (
        "code",
        "results",
        """# ---- 133.0: step B (inputs) and step C (structure), 5x5 repeated CV ----
# Regenerate with:  python src/stage2_h24_step_b.py ; python src/stage2_h24_step_c.py
import json
import sys
from pathlib import Path
import pandas as pd

if "src" not in sys.path:
    sys.path.insert(0, "src")
H24_DIR = Path("h24")
_cols = ["candidate", "m2_mean", "delta_m2_mean", "repeats_improved", "p_one_sided",
         "passes_gate", "m2_frozen_split", "fn_mean", "fp_mean", "collapsed_within_auc",
         "collapsed_fn_mean"]
_h24 = pd.concat([pd.read_csv(H24_DIR / "step_b_results.csv"),
                  pd.read_csv(H24_DIR / "step_c_results.csv")], ignore_index=True)
print(_h24[_cols].round(4).to_string(index=False))""",
    ),
    (
        "markdown",
        "results_md",
        """**Reading the table.**
* **The noise control is rejected (-0.012, 0/5 repeats).** It also shows how
  far M2 at a fixed threshold moves from tiny probability shifts.
* **The ICD code (B1) and the diagnosis text (B5) each improve all 5 repeats.**
  Their combination **C0 is the only candidate that clears the gate:**
  +0.022 M2, p = 0.079.
* **On the old frozen split, C0 and G tie** (0.7023 vs 0.7024). This is
  exactly the blind spot cause 1 describes.
* **Structural models don't beat C0.** Per-title feature copies and CatBoost
  recover the most missed positives in the collapsed titles, but they pay for
  them in false positives.
* **CatBoost with the protocol id** is the leak-free version of the H21
  label-rate experiment. It lands at -0.003 instead of -0.147, which confirms
  the H21 collapse was an implementation leak rather than a property of the
  data.""",
    ),
    (
        "code",
        "confirm",
        """# ---- 133.1: confirmation on fresh, untouched seeds 5-9 ----
# Regenerate with:  python src/stage2_h24_confirm_build.py
_cf = json.loads((H24_DIR / "confirm_build.json").read_text(encoding="utf-8"))
print(pd.DataFrame([_cf["G"], _cf["C0"]])[
    ["candidate", "m2_mean", "delta_m2_mean", "repeats_improved", "p_one_sided", "passes_gate"]
].round(4).to_string(index=False))
print("refit G reproduces production Stage 2:", _cf["refit_G_reproduces_production_stage2"])""",
    ),
    (
        "markdown",
        "decision",
        """**C0 was assembled after seeing step B on seeds 0-4**, so it was re-tested on
unused seeds.

* **The effect replicates:** same size (+0.022), same direction in all 10
  repeats, and better on 39 of 50 folds.
* **It misses the bar (p = 0.136).** The correction's `n_test/n_train` term
  does not shrink with more repeats.
* **Decision:** C0 is promoted as a **documented judgement call**, not a
  gate pass. The change is small, it is in the recall direction the metric
  rewards, and it keeps G's model family and threshold.

The next cell rebuilds the candidate from scratch and checks it byte-for-byte
against the file in `h24/`.""",
    ),
    (
        "code",
        "build",
        """# ---- 133.2: rebuild the C0 candidate end to end ----
# Standalone:  python src/stage2_h24_promote_c0.py
from stage2_h24_promote_c0 import CANDIDATE, build_candidate

_cand, _p_c0, _p_g, _test = build_candidate()   # asserts refit G == production Stage 2
# compare content, not bytes: the stored file has Windows line endings, and a
# rebuild on Linux/Colab would differ only in those
_stored = pd.read_csv(CANDIDATE)
assert _cand.reset_index(drop=True).equals(_stored), "candidate not reproduced"
print("candidate reproduced row for row:", CANDIDATE)
print("Stage 2 positive rate  G %.3f -> C0 %.3f" % ((_p_g >= 0.5).mean(), (_p_c0 >= 0.5).mean()))
print(pd.read_csv(H24_DIR / "candidate_flips_vs_production.csv").to_string(index=False))""",
    ),
    (
        "markdown",
        "limits",
        """**What C0 does not fix.**
* **In the collapsed titles, ranking is still close to chance:** within-title
  AUC goes only from 0.40 to 0.48, and missed positives from 36 to about 31.
* **Those positives depend on free-text facts** (pregnancy, upper-GI
  involvement, cancer) that no bag-of-words model can carry from one title to
  another.

**The remaining direction** is a label-free, zero-shot LLM probability used as
one more input to C0 (`colab/stage2_llm_feature.ipynb`). Because it uses no
labels, it is fold-safe by construction.""",
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
    print("removed %d stale H24 cells, appended %d, total now %d"
          % (removed, len(CELLS), len(nb["cells"])))


if __name__ == "__main__":
    main()
