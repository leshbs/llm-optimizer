"""Cross-validated comparison of Stage 1 feature spaces + threshold tuning.

Run:  python src/stage1_eval.py

Reports the official ``stage1_score`` (rescaled macro-F0.5) under the frozen
5-fold stratified split, for the production configuration and the
corruption-invariant replacement, so the two are compared like for like.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import read_competition_csv  # noqa: E402
from stage1_invariant import SEED, build_stage1_model, mask_cyrillic  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent
N_SPLITS = 5


# --------------------------------------------------------------------------- #
# Official metric (mirrors the platform formula)
# --------------------------------------------------------------------------- #
def stage1_score(y_true, y_pred) -> float:
    macro_f05 = fbeta_score(y_true, y_pred, beta=0.5, average="macro", zero_division=0)
    return float(np.clip((macro_f05 - 0.5) / 0.45, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Production baseline, reproduced here so the comparison is self-contained
# --------------------------------------------------------------------------- #
class TitleMetaFeaturesS1(BaseEstimator, TransformerMixin):
    """The six meta features used by the production Stage 1 pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X, dtype=object)).astype(str)
        words = s.str.split()
        feats = pd.DataFrame(
            {
                "len_chars": s.str.len(),
                "n_lines": s.str.count("\n"),
                "has_latin": s.str.contains(r"[A-Za-z]{2,}").astype(int),
                "has_digit": s.str.contains(r"\d").astype(int),
                "n_quotes": s.str.count(chr(34)),
                "max_word_len": words.apply(lambda ws: max((len(w) for w in ws), default=0)),
            }
        )
        return feats.to_numpy(dtype=float)


def build_production_model() -> Pipeline:
    """Config F: char tf-idf + meta (FeatureUnion) + LinearSVC."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "char_tfidf",
                            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
                        ),
                        (
                            "meta",
                            Pipeline(
                                [("extract", TitleMetaFeaturesS1()), ("scale", StandardScaler())]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
        ]
    )


# --------------------------------------------------------------------------- #
# Threshold tuning against the official metric (train/validation folds only)
# --------------------------------------------------------------------------- #
def tune_threshold(y_true, proba, grid=None) -> tuple:
    """Pick the decision threshold maximising ``stage1_score`` out of fold."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scored = [(float(t), stage1_score(y_true, (proba >= t).astype(int))) for t in grid]
    best_t, best_s = max(scored, key=lambda kv: (kv[1], -abs(kv[0] - 0.5)))
    return best_t, best_s, scored


def main() -> None:
    train = read_competition_csv(DATA_DIR / "train_stage1.csv")
    test = pd.read_csv(DATA_DIR / "test_stage1.csv")
    X, y = train["title_text"].to_numpy(dtype=object), train["label"].to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    print("train rows %d  positive rate %.4f" % (len(y), y.mean()))
    print("test  rows %d" % len(test))

    # -- how much of the production feature space survives to inference ----- #
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    tr_mat = vec.fit_transform(X)
    te_mat = vec.transform(test["title_text"].to_numpy(dtype=object))
    vocab = vec.get_feature_names_out()
    dead = np.array(["?" in v for v in vocab])
    print(
        "\n[transfer audit] vocab=%d  dead(?-bearing)=%d"
        % (len(vocab), int(dead.sum()))
    )
    print(
        "  train mass on dead n-grams: %.4f   test mass on dead n-grams: %.4f"
        % (tr_mat[:, dead].sum() / tr_mat.sum(), te_mat[:, dead].sum() / te_mat.sum())
    )
    print(
        "  active features per row -- train %.1f  test %.1f"
        % ((tr_mat > 0).sum(1).mean(), (te_mat > 0).sum(1).mean())
    )

    results = {}

    # -- production baseline ------------------------------------------------ #
    pred = cross_val_predict(build_production_model(), X, y, cv=cv, n_jobs=1)
    results["production_F_char_meta_linearsvc"] = {
        "cv_m1": stage1_score(y, pred),
        "oof_positive_rate": float(pred.mean()),
    }

    # -- invariant replacement --------------------------------------------- #
    proba = cross_val_predict(
        build_stage1_model(), X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    at_half = (proba >= 0.5).astype(int)
    results["invariant_logreg_t050"] = {
        "cv_m1": stage1_score(y, at_half),
        "oof_positive_rate": float(at_half.mean()),
    }
    best_t, best_s, scored = tune_threshold(y, proba)
    tuned = (proba >= best_t).astype(int)
    results["invariant_logreg_tuned"] = {
        "cv_m1": best_s,
        "threshold": best_t,
        "oof_positive_rate": float(tuned.mean()),
    }

    print("\n[cross-validated official metric, 5-fold seed %d]" % SEED)
    for name, r in results.items():
        print(
            "  %-34s M1=%.4f  posrate=%.4f%s"
            % (
                name,
                r["cv_m1"],
                r["oof_positive_rate"],
                "  t=%.2f" % r["threshold"] if "threshold" in r else "",
            )
        )

    print("\n[threshold curve, invariant model]")
    for t, s in scored:
        if abs(round(t * 100) % 5) == 0:
            print("  t=%.2f  M1=%.4f  posrate=%.4f" % (t, s, (proba >= t).mean()))

    print("\n[out-of-fold report, invariant model @ t=%.2f]" % best_t)
    print(classification_report(y, tuned, digits=3, zero_division=0))

    out = DATA_DIR / "h14"
    out.mkdir(exist_ok=True)
    (out / "stage1_cv_comparison.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_splits": N_SPLITS,
                "train_positive_rate": float(y.mean()),
                "results": results,
                "threshold_curve": [{"t": t, "m1": s} for t, s in scored],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(out / "stage1_invariant_oof_proba.npy", proba)
    print("\nwrote h14/stage1_cv_comparison.json")


if __name__ == "__main__":
    main()
