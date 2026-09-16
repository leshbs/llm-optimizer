"""Hyperparameter sweep + nested-CV honest estimate for the Stage 1 rebuild.

Run:  python src/stage1_tune.py

Two things are established here:

1. A grid over ``C``, the shape-n-gram order and ``min_df``, scored out of fold
   with the official metric. Tuning the decision threshold on the same folds
   that report the score is optimistic, so this grid is used only to *rank*
   configurations, not to state the final number.

2. A nested cross-validation where the threshold is chosen on inner folds and
   applied to a held-out outer fold. That is the honest estimate of what the
   rebuild scores on unseen titles, and it is what the decision to ship should
   rest on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import read_competition_csv  # noqa: E402
from stage1_eval import stage1_score, tune_threshold  # noqa: E402
from stage1_invariant import (  # noqa: E402
    SEED,
    InvariantTokenizer,
    SkeletonFeatures,
    build_stage1_model,
)

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "h14"


def configure(model, ngram_max: int, min_df: int, C: float):
    """Apply a grid point to a freshly built pipeline."""
    model.set_params(
        features__shape_ngrams__tfidf__ngram_range=(1, ngram_max),
        features__shape_ngrams__tfidf__min_df=min_df,
        clf__C=C,
    )
    return model


def oof_proba(model, X, y, cv) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def sweep(X, y, cv) -> pd.DataFrame:
    rows = []
    for ngram_max in (1, 2, 3, 4):
        for min_df in (1, 2, 3):
            for C in (0.3, 1.0, 3.0, 10.0):
                model = configure(build_stage1_model(), ngram_max, min_df, C)
                proba = oof_proba(model, X, y, cv)
                t, s, _ = tune_threshold(y, proba)
                rows.append(
                    {
                        "ngram_max": ngram_max,
                        "min_df": min_df,
                        "C": C,
                        "m1_at_050": stage1_score(y, (proba >= 0.5).astype(int)),
                        "m1_tuned": s,
                        "threshold": t,
                        "posrate": float((proba >= t).mean()),
                    }
                )
                print(
                    "  ngram<=%d min_df=%d C=%-5.1f  M1@0.50=%.4f  M1tuned=%.4f (t=%.2f)"
                    % (ngram_max, min_df, C, rows[-1]["m1_at_050"], s, t)
                )
    return pd.DataFrame(rows).sort_values("m1_tuned", ascending=False)


def nested_cv(X, y, params, n_outer: int = 5, n_inner: int = 5) -> dict:
    """Threshold chosen on inner folds, scored on the held-out outer fold."""
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    inner_thresholds, fold_scores = [], []
    y_pred = np.zeros(len(y), dtype=int)

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold)
        model = configure(build_stage1_model(), **params)
        inner_proba = oof_proba(model, X[tr], y[tr], inner)
        t, _, _ = tune_threshold(y[tr], inner_proba)

        model = configure(build_stage1_model(), **params).fit(X[tr], y[tr])
        pred = (model.predict_proba(X[te])[:, 1] >= t).astype(int)
        y_pred[te] = pred
        inner_thresholds.append(t)
        fold_scores.append(stage1_score(y[te], pred))
        print(
            "  outer fold %d: inner t=%.2f  outer M1=%.4f  posrate=%.4f"
            % (fold, t, fold_scores[-1], pred.mean())
        )

    return {
        "thresholds": inner_thresholds,
        "threshold_median": float(np.median(inner_thresholds)),
        "fold_scores": fold_scores,
        "mean_fold_m1": float(np.mean(fold_scores)),
        "std_fold_m1": float(np.std(fold_scores)),
        "pooled_m1": stage1_score(y, y_pred),
        "pooled_posrate": float(y_pred.mean()),
    }


def main() -> None:
    train = read_competition_csv(DATA_DIR / "train_stage1.csv")
    X, y = train["title_text"].to_numpy(dtype=object), train["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    OUT_DIR.mkdir(exist_ok=True)

    print("[grid sweep] (tuned column is optimistic - used for ranking only)")
    grid = sweep(X, y, cv)
    grid.to_csv(OUT_DIR / "stage1_grid.csv", index=False)
    print("\n[top 5 configurations]")
    print(grid.head(5).to_string(index=False))

    best = grid.iloc[0]
    params = {
        "ngram_max": int(best["ngram_max"]),
        "min_df": int(best["min_df"]),
        "C": float(best["C"]),
    }
    print("\n[nested CV with %s]" % params)
    nested = nested_cv(X, y, params)
    print(
        "  pooled M1=%.4f   per-fold mean=%.4f +/- %.4f   posrate=%.4f"
        % (
            nested["pooled_m1"],
            nested["mean_fold_m1"],
            nested["std_fold_m1"],
            nested["pooled_posrate"],
        )
    )

    (OUT_DIR / "stage1_tuning.json").write_text(
        json.dumps({"seed": SEED, "best_params": params, "nested_cv": nested}, indent=2),
        encoding="utf-8",
    )
    print("\nwrote h14/stage1_grid.csv and h14/stage1_tuning.json")


if __name__ == "__main__":
    main()
