"""H23b: is per-title calibration recoverable at all?

The symmetric centering in ``stage2_h23`` failed, and it failed in a way that
points somewhere specific: at alpha=1.0 false negatives got *worse* (81 -> 92).
Centering is symmetric, so it lifts low-prevalence titles and pushes
high-prevalence ones down, and the high-prevalence titles were doing fine.

Two things are tested here.

1. **One-sided calibration** -- lift a title only when its mean sits below the
   global mean, never push one down. This is the asymmetric version the failure
   mode implies.

2. **The decisive diagnostic: within-title AUC.** Any calibration, threshold or
   ranking scheme can only reorder rows *within* a title. If the model's scores
   carry no within-title signal, no such scheme can recover those false
   negatives, and the whole direction is closed regardless of how it is tuned.
   This measures that directly.

Run:  python src/stage2_h23b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    build_base_frame,
    load_folds,
    load_h8_features,
    merge_extra_cols,
)
from stage2_h23 import (  # noqa: E402
    BASELINE_M2,
    nested_calibrated_oof,
    row,
    score,
    title_offsets,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h23"


def apply_one_sided(titles, proba, offsets, alpha):
    """Lift titles whose mean is below the global mean; never push one down."""
    adj = np.array([min(offsets.get(t, 0.0), 0.0) for t in titles])
    return np.clip(proba - alpha * adj, 0.0, 1.0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]
    y_arr = y.to_numpy()
    titles = base["title_text"].to_numpy()
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)
    proba = np.load(ROOT / "h19" / "baseline_oof_proba.npy")

    report = {}

    # ---- the decisive diagnostic: within-title AUC ----------------------- #
    print("[diagnostic] can anything be recovered by reordering within a title?")
    rows = []
    for t, idx in pd.Series(range(len(y_arr))).groupby(titles).groups.items():
        idx = np.asarray(idx)
        yy = y_arr[idx]
        if len(set(yy)) < 2:
            continue
        rows.append({
            "title": t, "n": len(idx), "n_pos": int(yy.sum()),
            "base_rate": float(yy.mean()),
            "within_auc": float(roc_auc_score(yy, proba[idx])),
            "mean_p": float(proba[idx].mean()),
            "all_positives_missed": bool(((proba[idx] >= 0.5).astype(int)[yy == 1] == 0).all()),
        })
    wt = pd.DataFrame(rows)
    wt.to_csv(OUT / "within_title_auc.csv", index=False)

    collapsed = wt[wt["all_positives_missed"]]
    weighted = float((wt["within_auc"] * wt["n_pos"]).sum() / wt["n_pos"].sum())
    report["within_title_auc"] = {
        "n_titles_scored": len(wt),
        "mean_auc": float(wt["within_auc"].mean()),
        "pos_weighted_auc": weighted,
        "median_auc": float(wt["within_auc"].median()),
        "n_titles_auc_below_05": int((wt["within_auc"] < 0.5).sum()),
        "collapsed_titles": {
            "n": len(collapsed),
            "mean_auc": float(collapsed["within_auc"].mean()) if len(collapsed) else float("nan"),
            "median_auc": float(collapsed["within_auc"].median()) if len(collapsed) else float("nan"),
            "n_positives": int(collapsed["n_pos"].sum()),
        },
    }
    print("  titles with both classes present : %d" % len(wt))
    print("  mean within-title AUC            : %.4f" % wt["within_auc"].mean())
    print("  positive-weighted within-title AUC: %.4f" % weighted)
    print("  titles with AUC < 0.50           : %d / %d" % ((wt["within_auc"] < 0.5).sum(), len(wt)))
    if len(collapsed):
        print("  collapsed titles (%d, %d positives): mean AUC %.4f, median %.4f"
              % (len(collapsed), collapsed["n_pos"].sum(),
                 collapsed["within_auc"].mean(), collapsed["within_auc"].median()))
    print("\n  worst collapsed titles by AUC:")
    print(collapsed.nsmallest(5, "within_auc")[["n", "n_pos", "base_rate", "within_auc"]]
          .round(3).to_string(index=False))

    # ---- one-sided calibration ------------------------------------------ #
    print("\n[one-sided calibration] lift low-prevalence titles only")
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5]
    base_folds = score(y, proba, folds)["per_fold"]
    b_rows = []
    for tw in (1.0,):
        nested = nested_calibrated_oof(X, y, titles, folds, title_weight=tw, alphas=[1.0])
        # recompute offsets per fold for the one-sided form
        onesided = {a: np.zeros(len(y_arr)) for a in alphas}
        for tr, val in nested.get("_folds", folds):
            pass  # placeholder; offsets recomputed below
        # redo the nested loop explicitly so the one-sided form uses the same stats
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from stage2_h23 import build_pipeline
        for tr, val in folds:
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            inner_proba = cross_val_predict(clone(build_pipeline(tw)), X.iloc[tr], y_arr[tr],
                                            cv=inner, method="predict_proba")[:, 1]
            offsets, _ = title_offsets(titles[tr], inner_proba, shrink=5.0)
            model = clone(build_pipeline(tw)).fit(X.iloc[tr], y_arr[tr])
            p_val = model.predict_proba(X.iloc[val])[:, 1]
            for a in alphas:
                onesided[a][val] = apply_one_sided(titles[val], p_val, offsets, a)
        for a in alphas:
            res = score(y, onesided[a], folds)
            b_rows.append(row("one-sided alpha=%.2f" % a, res, base_folds))
            print("  alpha=%.2f  M2=%.6f  d=%+.6f  FN=%3d FP=%3d  pos=%.4f  folds+=%d"
                  % (a, res["m2"], b_rows[-1]["delta_vs_baseline"], res["fn"], res["fp"],
                     res["positive_rate"], b_rows[-1]["folds_improved"]))
    pd.DataFrame(b_rows).to_csv(OUT / "one_sided_calibration.csv", index=False)
    report["one_sided"] = b_rows

    passing = [r for r in b_rows if r["delta_vs_baseline"] > 0 and r["folds_improved"] >= 4]
    report["verdict"] = {
        "baseline_m2": BASELINE_M2,
        "best": max(b_rows, key=lambda r: r["cv_m2"])["experiment"],
        "best_m2": max(r["cv_m2"] for r in b_rows),
        "n_passing": len(passing),
        "passing": [r["experiment"] for r in passing],
    }
    (OUT / "h23b_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("\nvariants meeting the gate: %s" % (report["verdict"]["passing"] or "NONE"))
    print("wrote h23/within_title_auc.csv and h23/one_sided_calibration.csv")


if __name__ == "__main__":
    main()
