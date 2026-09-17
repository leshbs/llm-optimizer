"""H23c: selective calibration -- adjust only titles the model can actually rank.

``stage2_h23b`` showed why blanket calibration cannot work: within the titles
that produce the false negatives, the model's within-title AUC is 0.39, i.e.
worse than chance. Lifting such a title promotes the wrong rows preferentially.

The refinement this implies is to calibrate **selectively**: estimate each
title's within-title ranking quality on the training fold, and lift only the
titles that both sit below the global mean *and* can be ranked. This is the last
variant the diagnostic leaves open.

Fold safety: the per-title AUC and the per-title offset are both estimated from
inner cross-validated probabilities on the outer fold's training rows only. A
validation row never influences the statistic applied to it.

Run:  python src/stage2_h23c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    build_base_frame,
    load_folds,
    load_h8_features,
    merge_extra_cols,
)
from stage2_h23 import BASELINE_M2, build_pipeline, row, score, title_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h23"


def title_aucs(titles, proba, y) -> dict:
    """Within-title AUC estimated on training-fold data only."""
    out = {}
    df = pd.DataFrame({"t": titles, "p": proba, "y": y})
    for t, sub in df.groupby("t"):
        if sub["y"].nunique() < 2:
            continue
        out[t] = float(roc_auc_score(sub["y"], sub["p"]))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y_arr = base["label"].to_numpy()
    y = base["label"]
    titles = base["title_text"].to_numpy()
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)
    proba = np.load(ROOT / "h19" / "baseline_oof_proba.npy")
    base_folds = score(y, proba, folds)["per_fold"]

    alphas = [0.5, 0.75, 1.0]
    auc_gates = [0.55, 0.60, 0.70]
    variants = {(g, a): np.zeros(len(y_arr)) for g in auc_gates for a in alphas}
    n_lifted = {(g, a): 0 for g in auc_gates for a in alphas}

    for tr, val in folds:
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_proba = cross_val_predict(clone(build_pipeline(1.0)), X.iloc[tr], y_arr[tr],
                                        cv=inner, method="predict_proba")[:, 1]
        offsets, _ = title_offsets(titles[tr], inner_proba, shrink=5.0)
        aucs = title_aucs(titles[tr], inner_proba, y_arr[tr])

        model = clone(build_pipeline(1.0)).fit(X.iloc[tr], y_arr[tr])
        p_val = model.predict_proba(X.iloc[val])[:, 1]

        for g in auc_gates:
            for a in alphas:
                adj = np.array([
                    min(offsets.get(t, 0.0), 0.0) if aucs.get(t, 0.0) >= g else 0.0
                    for t in titles[val]
                ])
                n_lifted[(g, a)] += int((adj < 0).sum())
                variants[(g, a)][val] = np.clip(p_val - a * adj, 0.0, 1.0)

    rows = []
    print("selective one-sided calibration (lift only titles rankable on the training fold)")
    for g in auc_gates:
        for a in alphas:
            res = score(y, variants[(g, a)], folds)
            r = row("auc_gate=%.2f alpha=%.2f" % (g, a), res, base_folds)
            r["n_rows_lifted"] = n_lifted[(g, a)]
            rows.append(r)
            print("  gate=%.2f alpha=%.2f  M2=%.6f  d=%+.6f  FN=%3d FP=%3d  lifted=%3d  folds+=%d"
                  % (g, a, res["m2"], r["delta_vs_baseline"], res["fn"], res["fp"],
                     r["n_rows_lifted"], r["folds_improved"]))

    pd.DataFrame(rows).to_csv(OUT / "selective_calibration.csv", index=False)
    passing = [r for r in rows if r["delta_vs_baseline"] > 0 and r["folds_improved"] >= 4]
    best = max(rows, key=lambda r: r["cv_m2"])
    out = {
        "baseline_m2": BASELINE_M2,
        "rows": rows,
        "best": best["experiment"],
        "best_m2": best["cv_m2"],
        "best_delta": best["delta_vs_baseline"],
        "n_passing": len(passing),
        "passing": [r["experiment"] for r in passing],
    }
    (OUT / "h23c_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("\nbest: %s  M2=%.6f (delta %+.6f)" % (best["experiment"], best["cv_m2"],
                                                 best["delta_vs_baseline"]))
    print("variants meeting the gate: %s" % (out["passing"] or "NONE"))


if __name__ == "__main__":
    main()
