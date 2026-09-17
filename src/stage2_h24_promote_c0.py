"""H24: build the C0 candidate submission -- a documented judgement call.

C0 (config G + ICD code + diagnosis text) passed the pre-declared gate on
seeds 0-4 (p = 0.079) and **failed** it on the fresh confirmation seeds 5-9
(p = 0.136), with the same effect size both times (+0.022 M2), improving
10 of 10 repeats and 39 of 50 folds. ``stage2_h24_confirm_build.py`` therefore
wrote nothing, as its rule requires.

The user approved promoting C0 anyway, on these grounds (recorded in the
manifest):

* the effect is consistent in size and direction on 10 independent repeat seeds;
* the miss comes from the conservative Nadeau-Bengio correction, whose
  ``n_test/n_train`` term does not shrink with more repeats;
* the change is small and recall-directed: 6 of 173 test predictions, all 0 -> 1;
* the model family, threshold and all other inputs are unchanged from G.

This writes a **candidate** only. The production file is asserted unchanged,
and nothing is submitted.

Run:  python src/stage2_h24_promote_c0.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_h24_confirm_build import (  # noqa: E402
    CHECKPOINT, CHECKPOINT_SHA_PREFIX, PROD_SUBMISSION, PROD_SUBMISSION_SHA, build_test_frame, sha256,
)
from stage2_h24_features import build_h24_frame  # noqa: E402
from stage2_h24_harness import OUT, THRESHOLD  # noqa: E402
from stage2_h24_step_b import build_lr  # noqa: E402
from stage2_h24_step_c import C0_KW  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE = OUT / "submission_candidate_h24_c0.csv"


def build_candidate() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (candidate submission, C0 test proba, G test proba, test frame)."""
    X = build_h24_frame()
    y = X["label"].to_numpy()
    test = build_test_frame()
    prod = pd.read_csv(PROD_SUBMISSION)
    prod_s2 = prod[prod["stage"] == 2].reset_index(drop=True)
    assert (prod_s2["id"].to_numpy() == test["id"].to_numpy()).all(), "test order drifted"

    p_g = build_lr().fit(X, y).predict_proba(test)[:, 1]
    assert ((p_g >= THRESHOLD).astype(int) == prod_s2["label"].to_numpy()).all(), \
        "refit G no longer reproduces production Stage 2"
    p_c0 = build_lr(**C0_KW).fit(X, y).predict_proba(test)[:, 1]

    cand = prod.copy()
    cand.loc[cand["stage"] == 2, "label"] = (p_c0 >= THRESHOLD).astype(int)
    return cand, p_c0, p_g, test


def main() -> None:
    assert sha256(PROD_SUBMISSION) == PROD_SUBMISSION_SHA, "production submission changed"
    assert sha256(CHECKPOINT).startswith(CHECKPOINT_SHA_PREFIX), "checkpoint changed"

    cand, p_c0, p_g, test = build_candidate()
    prod = pd.read_csv(PROD_SUBMISSION)
    assert list(cand.columns) == ["stage", "id", "label"]
    assert (cand[["stage", "id"]].to_numpy() == prod[["stage", "id"]].to_numpy()).all()
    assert (cand.loc[cand["stage"] == 1, "label"].to_numpy()
            == prod.loc[prod["stage"] == 1, "label"].to_numpy()).all()
    cand.to_csv(CANDIDATE, index=False)

    g_lab = (p_g >= THRESHOLD).astype(int)
    c_lab = (p_c0 >= THRESHOLD).astype(int)
    flips = pd.DataFrame({
        "id": test["id"], "title": test["title_text"].str.split("\n").str[-1],
        "icd": test["icd"], "G_proba": p_g.round(4), "C0_proba": p_c0.round(4),
        "G": g_lab, "C0": c_lab,
    })[g_lab != c_lab]
    flips.to_csv(OUT / "candidate_flips_vs_production.csv", index=False)
    pd.DataFrame({"id": test["id"], "c0_proba": p_c0, "g_proba": p_g}).to_csv(
        OUT / "c0_test_proba.csv", index=False)

    confirm = json.loads((OUT / "confirm_build.json").read_text(encoding="utf-8"))
    manifest = {
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": sha256(CANDIDATE),
        "model": "C0 = config G + OneHot(icd) + TF-IDF(diagnosis_lemma); LogisticRegression "
                 "(C=1.0, class_weight='balanced', max_iter=2000, random_state=42); threshold 0.5",
        "stage1_source": str(PROD_SUBMISSION.relative_to(ROOT)) + " (rows copied unchanged)",
        "evidence": {
            "seeds_0_4": {"delta_m2": 0.0222, "repeats_improved": "5/5", "p_one_sided": 0.079,
                          "gate": "PASS"},
            "seeds_5_9": {"delta_m2": confirm["C0"]["delta_m2_mean"],
                          "repeats_improved": "%d/5" % confirm["C0"]["repeats_improved"],
                          "p_one_sided": confirm["C0"]["p_one_sided"], "gate": "FAIL"},
            "folds_improved_all_10_repeats": "39/50 (10 worse, 1 tie)",
            "frozen_split_m2": {"G": 0.702387, "C0": 0.702292},
        },
        "decision": "promoted as a documented judgement call, approved by the user, despite the "
                    "failed fresh-seed gate (see module docstring for the grounds)",
        "stage2_positive_rate": {"production_G": float(g_lab.mean()), "candidate_C0": float(c_lab.mean())},
        "stage2_flips": {"total": int(len(flips)), "0_to_1": int((flips["C0"] == 1).sum()),
                         "1_to_0": int((flips["C0"] == 0).sum())},
        "production_untouched": sha256(PROD_SUBMISSION) == PROD_SUBMISSION_SHA,
        "submitted": False,
    }
    (OUT / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
