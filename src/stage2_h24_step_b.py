"""H24 step B: give config G the inputs it never saw, one at a time.

Each candidate is config G with **one** change, scored on the step-A harness
and tested against G split by split. A pure-noise column is included as a
negative control: the gate must reject it, otherwise the gate is too loose.

Run:  python src/stage2_h24_step_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from stage2_baseline import CONTRADICTION_COLS, NUMERIC_FEATURES_S2, RANDOM_STATE  # noqa: E402
from stage2_h24_features import build_h24_frame  # noqa: E402
from stage2_h24_harness import (  # noqa: E402
    frozen_oof, oof_matrix, print_row, repeated_splits, save_rows, summarise,
)


def _numeric():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def build_lr(narrative_col="narrative_lemma", extra_text=(), extra_cat=(), extra_num=(),
             C=1.0) -> Pipeline:
    """Config G, parameterised. The defaults reproduce G exactly."""
    blocks = [
        ("title_tfidf", TfidfVectorizer(min_df=2), "title_lemma"),
        ("narrative_tfidf", TfidfVectorizer(min_df=2, max_features=20_000), narrative_col),
        ("numeric", _numeric(), NUMERIC_FEATURES_S2),
        ("gender", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
        ("h8_numeric", _numeric(), CONTRADICTION_COLS),
    ]
    for col in extra_text:
        blocks.append((f"text_{col}", TfidfVectorizer(min_df=2, max_features=20_000), col))
    if extra_cat:
        blocks.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(extra_cat)))
    if extra_num:
        blocks.append(("extra_num", _numeric(), list(extra_num)))
    return Pipeline([
        ("features", ColumnTransformer(blocks)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced", C=C)),
    ])


CANDIDATES = {
    "G": {},
    "B0 control: +noise column": {"extra_num": ("noise",)},
    "B1 +ICD code": {"extra_cat": ("icd",)},
    "B2 +ICD category": {"extra_cat": ("icd_cat",)},
    "B3 +title x ICD": {"extra_cat": ("title_x_icd",)},
    "B4 +ICD +title x ICD": {"extra_cat": ("icd", "title_x_icd")},
    "B5 +diagnosis text": {"extra_text": ("diagnosis_lemma",)},
    "B6 narrative -> clean full text": {"narrative_col": "full_lemma"},
    "B7 +clean full text": {"extra_text": ("full_lemma",)},
}


def main() -> None:
    X = build_h24_frame()
    X["noise"] = np.random.default_rng(RANDOM_STATE).normal(size=len(X))
    y = X["label"].to_numpy()
    titles = X["title_text"].to_numpy()
    splits = repeated_splits(y)

    base = oof_matrix("G", build_lr(), X, y, splits)
    frozen_g = frozen_oof(build_lr(), X, y)
    assert abs(frozen_g - np.load(Path(__file__).resolve().parent.parent / "h19" /
                                  "baseline_oof_proba.npy")).max() < 1e-9, "G no longer reproduces"

    rows = []
    for name, kw in CANDIDATES.items():
        est = build_lr(**kw)
        key = name.split()[0]
        proba = base if key == "G" else oof_matrix(key, est, X, y, splits)
        r = summarise(name, proba, y, splits, titles,
                      base_proba=None if key == "G" else base,
                      frozen=frozen_g if key == "G" else frozen_oof(est, X, y))
        rows.append(r)
        print_row(r)
    save_rows(rows, "step_b_results")
    print("\npassing the gate:", [r["candidate"] for r in rows if r.get("passes_gate")] or "NONE")


if __name__ == "__main__":
    main()
