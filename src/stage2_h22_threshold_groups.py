"""Phases 8 and 9: threshold analysis and recurrence-dependence stress test.

Phase 8 -- the metric is macro-F2 and Phase 1 found the false negatives packed
just under the cut (median probability 0.336, 25 of 81 above 0.40), so the
threshold is a real lever rather than a rounding detail. Two readings are
produced: the in-sample optimum over the out-of-fold probabilities, which is
optimistic, and a nested one where the threshold is chosen on four folds and
applied to the fifth, which is what an honest estimate looks like.

Phase 9 -- the baseline is re-scored under grouped splits that forbid a protocol
(or title) from appearing on both sides. This does not replace the official
frozen split; it quantifies how much of the baseline's score depends on having
seen the same protocol or template during training.

Nothing here is selected on leaderboard feedback.

Run:  python src/stage2_h22_threshold_groups.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    build_base_frame,
    build_variant_pipeline,
    load_folds,
    load_h8_features,
    merge_extra_cols,
    stage2_score,
)
from stage2_h21 import normalise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h22"
GRID = np.round(np.arange(0.20, 0.81, 0.01), 2)


def sweep(y, proba) -> pd.DataFrame:
    rows = []
    for t in GRID:
        hard = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "m2": stage2_score(y, hard),
            "macro_f2": float(fbeta_score(y, hard, beta=2, average="macro", zero_division=0)),
            "precision_pos": float(precision_score(y, hard, pos_label=1, zero_division=0)),
            "recall_pos": float(recall_score(y, hard, pos_label=1, zero_division=0)),
            "fn": int(((y == 1) & (hard == 0)).sum()),
            "fp": int(((y == 0) & (hard == 1)).sum()),
            "positive_rate": float(hard.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"].to_numpy()
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)
    proba = np.load(ROOT / "h19" / "baseline_oof_proba.npy")

    report = {}

    # ---------------- Phase 8: threshold ---------------------------------- #
    curve = sweep(y, proba)
    curve.to_csv(OUT / "threshold_sweep.csv", index=False)
    best = curve.loc[curve["m2"].idxmax()]
    at_050 = curve[curve["threshold"] == 0.50].iloc[0]

    # nested: pick the threshold on the other four folds, apply to the held-out one
    nested_pred = np.zeros(len(y), dtype=int)
    chosen = []
    for _, val in folds:
        mask = np.ones(len(y), bool)
        mask[val] = False
        inner = sweep(y[mask], proba[mask])
        t = float(inner.loc[inner["m2"].idxmax(), "threshold"])
        chosen.append(t)
        nested_pred[val] = (proba[val] >= t).astype(int)
    nested_m2 = stage2_score(y, nested_pred)

    report["threshold"] = {
        "at_050": at_050.to_dict(),
        "best_in_sample": best.to_dict(),
        "in_sample_gain": float(best["m2"] - at_050["m2"]),
        "nested_thresholds": chosen,
        "nested_threshold_median": float(np.median(chosen)),
        "nested_m2": nested_m2,
        "nested_gain_vs_050": float(nested_m2 - at_050["m2"]),
        "nested_fn": int(((y == 1) & (nested_pred == 0)).sum()),
        "nested_fp": int(((y == 0) & (nested_pred == 1)).sum()),
        "nested_positive_rate": float(nested_pred.mean()),
    }

    print("[Phase 8] threshold sweep on the frozen baseline's OOF probabilities")
    print(curve[curve["threshold"].isin([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])]
          .round(4).to_string(index=False))
    print("\nbest in-sample t=%.2f -> M2 %.6f (gain %+.6f over 0.50, optimistic)"
          % (best["threshold"], best["m2"], report["threshold"]["in_sample_gain"]))
    print("nested thresholds %s -> median %.2f" % (chosen, np.median(chosen)))
    print("nested M2 = %.6f  (gain %+.6f over 0.50)  FN %d -> %d, FP %d -> %d"
          % (nested_m2, report["threshold"]["nested_gain_vs_050"],
             int(at_050["fn"]), report["threshold"]["nested_fn"],
             int(at_050["fp"]), report["threshold"]["nested_fp"]))

    # ---------------- Phase 9: grouped CV --------------------------------- #
    print("\n[Phase 9] recurrence dependence of the frozen baseline")
    groupings = {
        "protocol": base["protocol_text"].astype(str),
        "normalised protocol": base["protocol_text"].astype(str).map(normalise),
        "title": base["title_text"].astype(str),
    }
    grouped = {}
    for name, g in groupings.items():
        codes = pd.factorize(g)[0]
        n_groups = len(set(codes))
        splits = list(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups=codes))
        p = cross_val_predict(build_variant_pipeline(CONTRADICTION_COLS), X, y,
                              cv=splits, method="predict_proba")[:, 1]
        hard = (p >= 0.5).astype(int)
        grouped[name] = {
            "n_groups": int(n_groups),
            "m2": stage2_score(y, hard),
            "macro_f2": float(fbeta_score(y, hard, beta=2, average="macro", zero_division=0)),
            "fn": int(((y == 1) & (hard == 0)).sum()),
            "fp": int(((y == 0) & (hard == 1)).sum()),
            "positive_rate": float(hard.mean()),
        }
        print("  grouped by %-20s (%3d groups): M2 %.6f  (frozen split 0.702387, delta %+.6f)  FN %d FP %d"
              % (name, n_groups, grouped[name]["m2"],
                 grouped[name]["m2"] - 0.7023868247508647,
                 grouped[name]["fn"], grouped[name]["fp"]))
    report["grouped_cv"] = grouped
    report["frozen_split_m2"] = 0.7023868247508647

    (OUT / "threshold_and_group_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nwrote h22/threshold_sweep.csv and h22/threshold_and_group_analysis.json")


if __name__ == "__main__":
    main()
