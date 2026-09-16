"""Does the repaired training file actually improve Stage 1?

Run:  python src/stage1_clean_eval.py

The comparison is exact rather than approximate: the repaired file has the same
1,767 ids in the same order with identical labels, so ``StratifiedKFold`` at
seed 42 produces literally the same folds used by H14 and H15. The nested
protocol is unchanged too -- threshold chosen on inner folds, scored on a
held-out outer fold -- so the three M1 numbers are directly comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_clean import Field, build_clean_features, build_clean_model  # noqa: E402
from stage1_eval import stage1_score, tune_threshold  # noqa: E402
from stage1_invariant import SEED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h17"


def oof_proba(model, X, y, cv) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def nested_cv(build, X, y, n_outer: int = 5, n_inner: int = 5) -> dict:
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    thresholds, fold_scores = [], []
    y_pred = np.zeros(len(y), dtype=int)
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold)
        t, _, _ = tune_threshold(y[tr], oof_proba(build(), X[tr], y[tr], inner))
        model = build().fit(X[tr], y[tr])
        pred = (model.predict_proba(X[te])[:, 1] >= t).astype(int)
        y_pred[te] = pred
        thresholds.append(t)
        fold_scores.append(stage1_score(y[te], pred))
        print("  outer fold %d: inner t=%.2f  outer M1=%.4f  posrate=%.4f"
              % (fold, t, fold_scores[-1], pred.mean()))
    return {
        "thresholds": thresholds,
        "threshold_median": float(np.median(thresholds)),
        "fold_scores": fold_scores,
        "mean_fold_m1": float(np.mean(fold_scores)),
        "std_fold_m1": float(np.std(fold_scores)),
        "pooled_m1": stage1_score(y, y_pred),
        "pooled_posrate": float(y_pred.mean()),
    }


def subset(blocks: tuple, C: float = 1.0) -> Pipeline:
    full = build_clean_features()
    kept = [(n, t) for n, t in full.transformer_list if n in blocks]
    return Pipeline([
        ("features", FeatureUnion(kept)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train_stage1.csv")
    test = pd.read_csv(ROOT / "test_stage1.csv")
    X = train["title_text"].to_numpy(dtype=object)
    y = train["label"].to_numpy()
    Xte = test["title_text"].to_numpy(dtype=object)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report = {"seed": SEED}

    # ---- the defect that started all of this: does the vocabulary transfer? --
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A = vec.fit_transform(Field("subsection").transform(X))
    B = vec.transform(Field("subsection").transform(Xte))
    report["transfer"] = {
        "vocab": int(A.shape[1]),
        "mean_active_features_train": float((A > 0).sum(1).mean()),
        "mean_active_features_test": float((B > 0).sum(1).mean()),
        "test_rows_with_zero_features": int((B.sum(1) == 0).sum()),
        "test_vocab_coverage": float((B > 0).sum(0).astype(bool).sum() / A.shape[1]),
    }
    print("[lexical transfer on the repaired file]")
    for k, v in report["transfer"].items():
        print("  %-30s %s" % (k, round(v, 4) if isinstance(v, float) else v))

    # ---- ablation ---------------------------------------------------------- #
    blocks = {
        "lex_subsection only": ("lex_subsection",),
        "lex_sub + context": ("lex_subsection", "lex_context"),
        "lex_sub + context + char": ("lex_subsection", "lex_context", "char"),
        "full (+ struct)": ("lex_subsection", "lex_context", "char", "struct"),
    }
    print("\n[ablation, 5-fold, threshold tuned in-fold (optimistic, ranking only)]")
    rows = []
    for name, bl in blocks.items():
        p = oof_proba(subset(bl), X, y, cv)
        t, s, _ = tune_threshold(y, p)
        rows.append({"config": name, "m1_at_050": stage1_score(y, (p >= 0.5).astype(int)),
                     "m1_tuned": s, "threshold": t})
        print("  %-26s M1@0.50=%.4f  M1tuned=%.4f (t=%.2f)"
              % (name, rows[-1]["m1_at_050"], s, t))
    pd.DataFrame(rows).to_csv(OUT / "clean_ablation.csv", index=False)
    report["ablation"] = rows

    # ---- C sweep ----------------------------------------------------------- #
    print("\n[C sweep, full model]")
    best = None
    for C in (0.3, 1.0, 3.0, 10.0, 30.0):
        p = oof_proba(build_clean_model(C=C), X, y, cv)
        t, s, _ = tune_threshold(y, p)
        print("  C=%-5.1f  M1tuned=%.4f (t=%.2f)" % (C, s, t))
        if best is None or s > best[1]:
            best = (C, s)
    C_best = best[0]
    report["best_C"] = C_best

    # ---- nested CV --------------------------------------------------------- #
    print("\n[nested CV, full model, C=%.1f]" % C_best)
    nested = nested_cv(lambda: build_clean_model(C=C_best), X, y)
    print("  pooled M1=%.4f   per-fold %.4f +/- %.4f   posrate=%.4f"
          % (nested["pooled_m1"], nested["mean_fold_m1"], nested["std_fold_m1"],
             nested["pooled_posrate"]))
    report["nested_cv"] = nested

    # save OOF probabilities -- H15's were never cached and that blocked the
    # ensemble candidate in the h16 audit; do not repeat the omission.
    np.save(OUT / "stage1_clean_oof_proba.npy",
            oof_proba(build_clean_model(C=C_best), X, y, cv))

    # ---- comparison -------------------------------------------------------- #
    h14 = json.loads((ROOT / "h14" / "stage1_tuning.json").read_text(encoding="utf-8"))["nested_cv"]
    h15 = json.loads((ROOT / "h15" / "stage1_hybrid_eval.json").read_text(encoding="utf-8"))["nested_cv"]
    comp = pd.DataFrame({
        "H14 invariant (corrupted)": {"pooled_m1": h14["pooled_m1"], "sd": h14["std_fold_m1"]},
        "H15 hybrid (corrupted)": {"pooled_m1": h15["pooled_m1"], "sd": h15["std_fold_m1"]},
        "H17 clean text": {"pooled_m1": nested["pooled_m1"], "sd": nested["std_fold_m1"]},
    }).T
    print("\n" + comp.round(4).to_string())
    print("\nper-outer-fold M1")
    for name, s in (("H14", h14["fold_scores"]), ("H15", h15["fold_scores"]),
                    ("H17", nested["fold_scores"])):
        print("  %s: %s" % (name, ["%.4f" % v for v in s]))
    d15 = nested["pooled_m1"] - h15["pooled_m1"]
    report["delta_vs_h15"] = d15
    report["delta_vs_h14"] = nested["pooled_m1"] - h14["pooled_m1"]
    print("\nvs H15: %+.4f M1 = %+.2f metric points" % (d15, 70 * 0.3 * d15))
    print("vs H14: %+.4f M1 = %+.2f metric points"
          % (report["delta_vs_h14"], 70 * 0.3 * report["delta_vs_h14"]))

    (OUT / "stage1_clean_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote h17/stage1_clean_eval.json")


if __name__ == "__main__":
    main()
