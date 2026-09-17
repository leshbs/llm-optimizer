"""H23: attack the title-prior collapse directly.

H19-H22 established that 72.8% of Stage 2 false negatives come from the model
reproducing a title's base rate instead of discriminating within the title: 75%
of out-of-fold probability variance is between titles, and in 8 titles no
positive ever crosses 0.5. Every feature family tested there failed because it
added per-row information the model could already approximate from the title.

Two structural changes are tested here instead.

**Direction A -- reduce title dominance.** The title TF-IDF block is what lets
the model identify the title and fall back on its prior. Down-weighting it
forces the model to read the narrative. Implemented with ColumnTransformer
``transformer_weights``, swept from 1.0 (baseline) down to 0.0 (title text
dropped entirely).

**Direction B -- per-title calibration.** Re-centre each title's probability
distribution toward the global mean before thresholding, so a low-prevalence
title is not uniformly suppressed and within-title ranking decides.

Fold safety for Direction B is the delicate part, and it is handled nested:
the per-title statistics for an outer fold are computed from *inner* cross
-validated probabilities on that outer fold's training rows only. Using the
outer model's in-sample probabilities on its own training rows would bias the
per-title means (they are fitted, the validation ones are not), so an inner
5-fold split produces unbiased training-side probabilities first. A validation
row never contributes to the statistic that adjusts it, and a title unseen in
the training fold is left unadjusted.

Run:  python src/stage2_h23.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    NUMERIC_FEATURES_S2,
    RANDOM_STATE,
    build_base_frame,
    load_folds,
    load_h8_features,
    merge_extra_cols,
    stage2_score,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h23"
BASELINE_M2 = 0.7023868247508647


# --------------------------------------------------------------- Direction A #
def build_pipeline(title_weight: float = 1.0, extra_numeric_cols=CONTRADICTION_COLS) -> Pipeline:
    """Config G with a tunable weight on the title TF-IDF block.

    ``title_weight=0.0`` drops the title text entirely; the title still reaches
    the model through the contradiction features, which is intentional -- those
    encode *conditions* stated by the title rather than its identity.
    """
    transformers = [
        ("narrative_tfidf", TfidfVectorizer(min_df=2, max_features=20_000), "narrative_lemma"),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES_S2),
        ("gender", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
    ]
    weights = {}
    if title_weight > 0:
        transformers.insert(0, ("title_tfidf", TfidfVectorizer(min_df=2), "title_lemma"))
        weights["title_tfidf"] = title_weight
    if extra_numeric_cols:
        transformers.append(("h8_numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), list(extra_numeric_cols)))
    return Pipeline([
        ("features", ColumnTransformer(transformers, transformer_weights=weights or None)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ])


# --------------------------------------------------------------- Direction B #
def title_offsets(titles: np.ndarray, proba: np.ndarray, shrink: float = 5.0) -> tuple:
    """Per-title mean offset from the global mean, shrunk toward 0 for small titles.

    Returns (offset_by_title, global_mean). ``shrink`` is the pseudo-count: a
    title seen n times gets n/(n+shrink) of its raw offset, so rare titles are
    barely moved.
    """
    g = float(proba.mean())
    df = pd.DataFrame({"t": titles, "p": proba})
    stats = df.groupby("t")["p"].agg(["mean", "size"])
    raw = stats["mean"] - g
    weight = stats["size"] / (stats["size"] + shrink)
    return (raw * weight).to_dict(), g


def apply_centering(titles: np.ndarray, proba: np.ndarray, offsets: dict,
                    alpha: float) -> np.ndarray:
    """p' = p - alpha * offset(title). Unseen titles are left alone."""
    adj = np.array([offsets.get(t, 0.0) for t in titles])
    return np.clip(proba - alpha * adj, 0.0, 1.0)


def nested_calibrated_oof(X, y, titles, folds, title_weight: float, alphas,
                          shrink: float = 5.0, n_inner: int = 5) -> dict:
    """Out-of-fold probabilities plus per-title-centred variants, fold-safely.

    For each outer fold the per-title offsets come from inner-CV probabilities
    on that fold's training rows, so nothing fitted on a validation row informs
    its own adjustment.
    """
    y_arr = np.asarray(y)
    raw = np.zeros(len(y_arr))
    centred = {a: np.zeros(len(y_arr)) for a in alphas}

    for tr, val in folds:
        Xtr, ytr = X.iloc[tr], y_arr[tr]
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
        inner_proba = cross_val_predict(
            clone(build_pipeline(title_weight)), Xtr, ytr, cv=inner, method="predict_proba"
        )[:, 1]
        offsets, _ = title_offsets(titles[tr], inner_proba, shrink=shrink)

        model = clone(build_pipeline(title_weight)).fit(Xtr, ytr)
        p_val = model.predict_proba(X.iloc[val])[:, 1]
        raw[val] = p_val
        for a in alphas:
            centred[a][val] = apply_centering(titles[val], p_val, offsets, a)

    return {"raw": raw, "centred": centred}


# ------------------------------------------------------------------ scoring #
def score(y, proba, folds, threshold: float = 0.5) -> dict:
    y_arr = np.asarray(y)
    hard = (proba >= threshold).astype(int)
    per_fold = [fbeta_score(y_arr[v], hard[v], beta=2, average="macro", zero_division=0)
                for _, v in folds]
    return {
        "m2": stage2_score(y_arr, hard),
        "macro_f2": float(fbeta_score(y_arr, hard, beta=2, average="macro", zero_division=0)),
        "per_fold": [float(v) for v in per_fold],
        "mean_fold": float(np.mean(per_fold)),
        "std_fold": float(np.std(per_fold, ddof=1)),
        "fn": int(((y_arr == 1) & (hard == 0)).sum()),
        "fp": int(((y_arr == 0) & (hard == 1)).sum()),
        "positive_rate": float(hard.mean()),
        "threshold": threshold,
    }


def row(name: str, res: dict, base_folds) -> dict:
    wins = int((np.array(res["per_fold"]) > np.array(base_folds)).sum())
    return {
        "experiment": name, "cv_m2": res["m2"], "delta_vs_baseline": res["m2"] - BASELINE_M2,
        "macro_f2": res["macro_f2"], "mean_fold": res["mean_fold"], "std_fold": res["std_fold"],
        "fn": res["fn"], "fp": res["fp"], "positive_rate": res["positive_rate"],
        "folds_improved": wins, "fold_scores": [round(v, 4) for v in res["per_fold"]],
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]
    titles = base["title_text"].to_numpy()
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)

    results = {}

    # ---- Direction A: title weight sweep -------------------------------- #
    print("[Direction A] reducing title dominance")
    weights = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    a_rows, base_folds = [], None
    oof_by_weight = {}
    for w in weights:
        proba = cross_val_predict(build_pipeline(w), X, y, cv=folds, method="predict_proba")[:, 1]
        oof_by_weight[w] = proba
        res = score(y, proba, folds)
        if base_folds is None:
            base_folds = res["per_fold"]
        a_rows.append(row("title_weight=%.2f" % w, res, base_folds))
        # how much between-title variance survives?
        tv = pd.Series(proba).groupby(titles).transform("mean").var(ddof=0) / proba.var(ddof=0)
        a_rows[-1]["between_title_var_share"] = float(tv)
        print("  w=%.2f  M2=%.6f  d=%+.6f  FN=%3d FP=%3d  pos=%.4f  between-title var=%.3f  folds+=%d"
              % (w, res["m2"], a_rows[-1]["delta_vs_baseline"], res["fn"], res["fp"],
                 res["positive_rate"], tv, a_rows[-1]["folds_improved"]))
    pd.DataFrame(a_rows).to_csv(OUT / "title_weight_sweep.csv", index=False)
    results["direction_A"] = a_rows

    # ---- Direction B: per-title calibration ----------------------------- #
    print("\n[Direction B] per-title calibration (nested, fold-safe)")
    alphas = [0.25, 0.5, 0.75, 1.0]
    b_rows = []
    nested = nested_calibrated_oof(X, y, titles, folds, title_weight=1.0, alphas=alphas)
    res_raw = score(y, nested["raw"], folds)
    print("  alpha=0.00 (raw, nested refit)  M2=%.6f  FN=%3d FP=%3d" %
          (res_raw["m2"], res_raw["fn"], res_raw["fp"]))
    b_rows.append(row("calibration alpha=0.00", res_raw, base_folds))
    for a in alphas:
        res = score(y, nested["centred"][a], folds)
        b_rows.append(row("calibration alpha=%.2f" % a, res, base_folds))
        print("  alpha=%.2f  M2=%.6f  d=%+.6f  FN=%3d FP=%3d  pos=%.4f  folds+=%d"
              % (a, res["m2"], b_rows[-1]["delta_vs_baseline"], res["fn"], res["fp"],
                 res["positive_rate"], b_rows[-1]["folds_improved"]))
    pd.DataFrame(b_rows).to_csv(OUT / "calibration_sweep.csv", index=False)
    results["direction_B"] = b_rows

    # ---- Combined: best title weight + calibration ---------------------- #
    best_w = max(a_rows, key=lambda r: r["cv_m2"])
    w_star = float(best_w["experiment"].split("=")[1])
    print("\n[Combined] best title weight %.2f + calibration" % w_star)
    c_rows = []
    if w_star != 1.0:
        nested_c = nested_calibrated_oof(X, y, titles, folds, title_weight=w_star, alphas=alphas)
        res = score(y, nested_c["raw"], folds)
        c_rows.append(row("w=%.2f, alpha=0.00" % w_star, res, base_folds))
        print("  w=%.2f alpha=0.00  M2=%.6f  FN=%3d FP=%3d" % (w_star, res["m2"], res["fn"], res["fp"]))
        for a in alphas:
            res = score(y, nested_c["centred"][a], folds)
            c_rows.append(row("w=%.2f, alpha=%.2f" % (w_star, a), res, base_folds))
            print("  w=%.2f alpha=%.2f  M2=%.6f  d=%+.6f  FN=%3d FP=%3d  folds+=%d"
                  % (w_star, a, res["m2"], c_rows[-1]["delta_vs_baseline"], res["fn"],
                     res["fp"], c_rows[-1]["folds_improved"]))
        pd.DataFrame(c_rows).to_csv(OUT / "combined_sweep.csv", index=False)
    results["combined"] = c_rows

    # ---- verdict --------------------------------------------------------- #
    everything = a_rows + b_rows + c_rows
    passing = [r for r in everything
               if r["delta_vs_baseline"] > 0 and r["folds_improved"] >= 4]
    best = max(everything, key=lambda r: r["cv_m2"])
    results["verdict"] = {
        "baseline_m2": BASELINE_M2,
        "best_experiment": best["experiment"],
        "best_m2": best["cv_m2"],
        "best_delta": best["delta_vs_baseline"],
        "best_folds_improved": best["folds_improved"],
        "n_passing_gate": len(passing),
        "passing": [r["experiment"] for r in passing],
        "gate": "delta_vs_baseline > 0 AND folds_improved >= 4/5",
    }
    (OUT / "h23_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    np.save(OUT / "h23_best_oof.npy", oof_by_weight[w_star])

    print("\nbest overall: %s  M2=%.6f  (delta %+.6f, %d/5 folds)"
          % (best["experiment"], best["cv_m2"], best["best_delta"]
             if "best_delta" in best else best["delta_vs_baseline"], best["folds_improved"]))
    print("variants meeting the gate: %s" % (results["verdict"]["passing"] or "NONE"))
    print("wrote h23/ artifacts")


if __name__ == "__main__":
    main()
