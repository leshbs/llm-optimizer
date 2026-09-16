"""Verify that the invariant feature space really does transfer to clean test text.

The rebuild's whole argument is that masking both splits puts them in one
feature space. That is a claim about the data, so it gets checked rather than
asserted: this script measures feature coverage and distribution overlap between
the corrupted train titles and the clean test titles, for the production
pipeline and for the rebuild side by side.

Run:  python src/stage1_transfer_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import read_competition_csv  # noqa: E402
from stage1_invariant import (  # noqa: E402
    SKELETON_FEATURE_NAMES,
    InvariantTokenizer,
    SkeletonFeatures,
    mask_cyrillic,
)

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "h14"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    train = read_competition_csv(DATA_DIR / "train_stage1.csv")
    test = pd.read_csv(DATA_DIR / "test_stage1.csv")
    Xtr = train["title_text"].to_numpy(dtype=object)
    Xte = test["title_text"].to_numpy(dtype=object)

    report = {}

    # -- 1. production char tf-idf ------------------------------------------ #
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    dead = np.array(["?" in v for v in vec.get_feature_names_out()])
    report["production_char_tfidf"] = {
        "vocab": int(len(dead)),
        "train_mass_on_dead_ngrams": float(A[:, dead].sum() / A.sum()),
        "test_mass_on_dead_ngrams": float(B[:, dead].sum() / B.sum()),
        "mean_active_features_train": float((A > 0).sum(1).mean()),
        "mean_active_features_test": float((B > 0).sum(1).mean()),
    }

    # -- 2. rebuild shape n-grams ------------------------------------------- #
    tok = InvariantTokenizer()
    vec2 = TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+", ngram_range=(1, 4), min_df=3, sublinear_tf=True
    )
    A2 = vec2.fit_transform(tok.transform(Xtr))
    B2 = vec2.transform(tok.transform(Xte))
    vocab2 = vec2.get_feature_names_out()
    report["rebuild_shape_tfidf"] = {
        "vocab": int(len(vocab2)),
        "train_mass_on_dead_ngrams": float(
            A2[:, np.array(["?" in v for v in vocab2])].sum() / A2.sum()
        ),
        "test_mass_on_dead_ngrams": float(
            B2[:, np.array(["?" in v for v in vocab2])].sum() / B2.sum()
        ),
        "mean_active_features_train": float((A2 > 0).sum(1).mean()),
        "mean_active_features_test": float((B2 > 0).sum(1).mean()),
        "test_rows_with_zero_features": int((B2.sum(1) == 0).sum()),
        "test_vocab_coverage": float((B2 > 0).sum(0).astype(bool).sum() / len(vocab2)),
    }
    # The "?"-bearing vocabulary here is the AGE= categorical, which survives
    # masking bijectively and is therefore active on both splits by design.

    # -- 3. skeleton numeric features --------------------------------------- #
    sk = SkeletonFeatures()
    Ftr, Fte = sk.transform(Xtr), sk.transform(Xte)
    rows = []
    for i, name in enumerate(SKELETON_FEATURE_NAMES):
        a, b = Ftr[:, i], Fte[:, i]
        pooled_sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        rows.append(
            {
                "feature": name,
                "train_mean": a.mean(),
                "test_mean": b.mean(),
                "std_mean_diff": abs(a.mean() - b.mean()) / pooled_sd,
            }
        )
    skel = pd.DataFrame(rows).sort_values("std_mean_diff", ascending=False)
    skel.to_csv(OUT_DIR / "stage1_skeleton_drift.csv", index=False)
    report["skeleton_drift"] = {
        "max_standardised_mean_diff": float(skel["std_mean_diff"].max()),
        "median_standardised_mean_diff": float(skel["std_mean_diff"].median()),
        "n_features_above_0.5_sd": int((skel["std_mean_diff"] > 0.5).sum()),
    }

    print("[production char tf-idf]")
    for k, v in report["production_char_tfidf"].items():
        print("  %-32s %s" % (k, round(v, 4)))
    print("\n[rebuild shape tf-idf]")
    for k, v in report["rebuild_shape_tfidf"].items():
        print("  %-32s %s" % (k, round(v, 4)))
    print("\n[skeleton numeric drift, train vs test]")
    print(skel.head(8).to_string(index=False))
    print("\n  %-32s %s" % ("features > 0.5 SD apart", report["skeleton_drift"]["n_features_above_0.5_sd"]))

    (OUT_DIR / "stage1_transfer_check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nwrote h14/stage1_transfer_check.json")


if __name__ == "__main__":
    main()
