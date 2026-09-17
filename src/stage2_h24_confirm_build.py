"""H24 confirmation and candidate build.

C0 (config G + ICD code + diagnosis text) passed the gate on seeds 0-4, but it
was assembled *after* the step-B results on those same splits were visible, so
its p-value is optimistic. This script:

1. re-tests C0 against G on five **fresh** repeat seeds (5-9) that no H24
   decision has touched, under the same pre-declared gate;
2. only if that holds, refits C0 on all 690 rows and writes a **candidate**
   submission to ``h24/`` -- Stage 1 rows copied byte-for-byte from the
   production H17 file, Stage 2 rows from C0.

Nothing in production is overwritten and nothing is submitted.

Run:  python src/stage2_h24_confirm_build.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import CONTRADICTION_COLS, build_base_frame, load_h8_features, merge_extra_cols  # noqa: E402
from stage2_h24_features import add_h24_columns, build_h24_frame  # noqa: E402
from stage2_h24_harness import OUT, THRESHOLD, oof_matrix, print_row, repeated_splits, summarise  # noqa: E402
from stage2_h24_step_b import build_lr  # noqa: E402
from stage2_h24_step_c import C0_KW  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRESH_SEEDS = (5, 6, 7, 8, 9)
PROD_SUBMISSION = ROOT / "h17" / "submission_stage1_clean.csv"
PROD_SUBMISSION_SHA = "5cd35e4dcba7f5a8bcef21e21b9689440a413e2712c6a65258136a957e8945a6"
CHECKPOINT = ROOT / "h12" / "checkpoint" / "checkpoint_submission.csv"
CHECKPOINT_SHA_PREFIX = "653c2b360bdf450d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_test_frame() -> pd.DataFrame:
    base = build_base_frame(ROOT / "test_stage2.csv")
    frame = merge_extra_cols(base, load_h8_features(base), CONTRADICTION_COLS)
    return add_h24_columns(frame)


def main() -> None:
    assert sha256(PROD_SUBMISSION) == PROD_SUBMISSION_SHA, "production submission changed"
    assert sha256(CHECKPOINT).startswith(CHECKPOINT_SHA_PREFIX), "checkpoint changed"

    X = build_h24_frame()
    y = X["label"].to_numpy()
    titles = X["title_text"].to_numpy()

    # ---- 1. confirmation on fresh seeds --------------------------------- #
    splits = repeated_splits(y, seeds=FRESH_SEEDS)
    base = oof_matrix("fresh_G", build_lr(), X, y, splits)
    c0 = oof_matrix("fresh_C0", build_lr(**C0_KW), X, y, splits)
    g_row = summarise("G (fresh seeds)", base, y, splits, titles)
    c0_row = summarise("C0 (fresh seeds)", c0, y, splits, titles, base_proba=base)
    print_row(g_row)
    print_row(c0_row)
    confirmed = c0_row["passes_gate"]
    report = {"fresh_seeds": list(FRESH_SEEDS), "G": g_row, "C0": c0_row, "confirmed": confirmed}
    print("confirmed on fresh seeds:", confirmed)

    # ---- 2. candidate build --------------------------------------------- #
    test = build_test_frame()
    prod = pd.read_csv(PROD_SUBMISSION)
    prod_s2 = prod[prod["stage"] == 2].reset_index(drop=True)
    assert (prod_s2["id"].to_numpy() == test["id"].to_numpy()).all()

    # the refit G must reproduce the production Stage 2 labels, otherwise the
    # C0 build would not be a like-for-like replacement
    g_test = (build_lr().fit(X, y).predict_proba(test)[:, 1] >= THRESHOLD).astype(int)
    g_reproduces = bool((g_test == prod_s2["label"].to_numpy()).all())
    report["refit_G_reproduces_production_stage2"] = g_reproduces
    print("refit G reproduces production Stage 2:", g_reproduces)

    unseen_icd = sorted(set(test["icd"]) - set(X["icd"]))
    report["test_icd_unseen_in_train"] = unseen_icd
    report["test_rows_with_icd"] = int((test["icd"] != "NA").sum())

    if not (confirmed and g_reproduces):
        report["candidate_written"] = False
        (OUT / "confirm_build.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                           default=float), encoding="utf-8")
        print("no candidate written")
        return

    p_test = build_lr(**C0_KW).fit(X, y).predict_proba(test)[:, 1]
    c0_test = (p_test >= THRESHOLD).astype(int)
    cand = prod.copy()
    cand.loc[cand["stage"] == 2, "label"] = c0_test
    assert (cand[cand["stage"] == 1].to_numpy() == prod[prod["stage"] == 1].to_numpy()).all()
    path = OUT / "submission_candidate_h24_c0.csv"
    cand.to_csv(path, index=False)

    flips = pd.DataFrame({"id": test["id"], "title": test["title_text"].str.split("\n").str[-1],
                          "icd": test["icd"], "G": prod_s2["label"], "C0": c0_test,
                          "C0_proba": p_test.round(4)})
    flips = flips[flips["G"] != flips["C0"]]
    flips.to_csv(OUT / "candidate_flips_vs_production.csv", index=False)
    pd.DataFrame({"id": test["id"], "proba": p_test}).to_csv(OUT / "c0_test_proba.csv", index=False)

    report.update({
        "candidate_written": True,
        "candidate_path": str(path.relative_to(ROOT)),
        "candidate_sha256": sha256(path),
        "stage1_rows_identical_to_production": True,
        "stage2_positive_rate": {"production_G": float(prod_s2["label"].mean()),
                                 "candidate_C0": float(c0_test.mean())},
        "stage2_flips_vs_production": {"total": int(len(flips)),
                                       "0_to_1": int((flips["C0"] == 1).sum()),
                                       "1_to_0": int((flips["C0"] == 0).sum())},
        "production_untouched": sha256(PROD_SUBMISSION) == PROD_SUBMISSION_SHA,
    })
    (OUT / "confirm_build.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                       default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate_path", "stage2_positive_rate",
                                             "stage2_flips_vs_production", "test_icd_unseen_in_train",
                                             "test_rows_with_icd", "production_untouched")},
                     indent=1, ensure_ascii=False))
    print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
