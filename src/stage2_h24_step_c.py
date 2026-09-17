"""H24 step C: models that can combine title and patient evidence.

Config G is additive: f(title) + g(protocol). The audit showed the label is an
interaction (62 of 102 repeated protocols carry mixed labels across titles), so
this step tests model structures that can represent one:

* **C0** -- the step-B combination fixed in advance: G + ICD code + diagnosis
  text (the two inputs that improved all 5 repeats). This is the reference for
  the structural candidates.
* **C1** -- C0 with per-title copies of a compact patient representation
  ("frustratingly easy domain adaptation", Daume 2007). The shared weights
  learn what holds across titles; each title gets its own deviation, shrunk
  toward the shared model by the L2 penalty. That is a per-title classifier
  that cannot overfit a 15-row title the way a separate model would.
* **C2** -- CatBoost over categorical title / guideline / ICD / gender plus
  numeric and SVD text features. Trees represent the interactions directly.
* **C3** -- C2 plus the protocol id as a categorical feature. CatBoost's ordered
  target statistics encode it without letting a row see its own label, which is
  the leak-free version of the flawed H21-F experiment.
* **C4** -- the mean of C0 and the better tree model.
* **C5** -- regularisation sensitivity of C0 (C = 0.3, 3). Reported as a
  sensitivity check; picking C on these splits would be selection, so the gate
  result for these is informative only.

Run:  python src/stage2_h24_step_c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import CONTRADICTION_COLS, NUMERIC_FEATURES_S2, RANDOM_STATE  # noqa: E402
from stage2_h24_features import build_h24_frame  # noqa: E402
from stage2_h24_harness import (  # noqa: E402
    frozen_oof, oof_matrix, print_row, repeated_splits, save_rows, summarise,
)
from stage2_h24_step_b import build_lr  # noqa: E402

C0_KW = {"extra_cat": ("icd",), "extra_text": ("diagnosis_lemma",)}
PATIENT_NUMERIC = ["age", "has_icd"] + CONTRADICTION_COLS
CAT_COLS = ["title_text", "guideline", "icd", "icd_cat", "gender"]


def _numeric():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


class PerTitleCopies(BaseEstimator, TransformerMixin):
    """Compact patient features, copied once per title (zero outside that title).

    Expects a frame with ``title_text`` plus the patient columns. Title
    categories and the patient encoder are both fit on the training fold only;
    an unseen title gets no copy and falls back to the shared weights.
    """

    def __init__(self, n_svd: int = 20):
        self.n_svd = n_svd

    def _patient_encoder(self):
        return ColumnTransformer([
            ("icd", OneHotEncoder(handle_unknown="ignore"), ["icd", "icd_cat", "gender"]),
            ("num", _numeric(), PATIENT_NUMERIC),
            ("text", Pipeline([("tfidf", TfidfVectorizer(min_df=2, max_features=20_000)),
                               ("svd", TruncatedSVD(self.n_svd, random_state=RANDOM_STATE))]),
             "full_lemma"),
        ])

    def fit(self, X, y=None):
        self.encoder_ = self._patient_encoder().fit(X)
        self.titles_ = np.array(sorted(X["title_text"].unique()))
        return self

    def transform(self, X):
        z = sp.csr_matrix(self.encoder_.transform(X))
        idx = np.searchsorted(self.titles_, X["title_text"].to_numpy())
        idx = np.clip(idx, 0, len(self.titles_) - 1)
        known = self.titles_[idx] == X["title_text"].to_numpy()
        rows = np.arange(len(X))[known]
        # a sparse (n_rows x n_titles) indicator, then the row-wise Kronecker product
        ind = sp.csr_matrix((np.ones(len(rows)), (rows, idx[known])),
                            shape=(len(X), len(self.titles_)))
        return sp.hstack([z, _row_kron(ind, z)]).tocsr()


def _row_kron(ind: sp.csr_matrix, z: sp.csr_matrix) -> sp.csr_matrix:
    """Row-wise Kronecker product: row i -> ind[i] (x) z[i]."""
    n, t = ind.shape
    d = z.shape[1]
    ind = ind.tocsr()
    out_rows, out_cols, out_vals = [], [], []
    for i in range(n):
        ts = ind.indices[ind.indptr[i]:ind.indptr[i + 1]]
        zc = z.indices[z.indptr[i]:z.indptr[i + 1]]
        zv = z.data[z.indptr[i]:z.indptr[i + 1]]
        for ti in ts:
            out_rows.append(np.full(len(zc), i))
            out_cols.append(ti * d + zc)
            out_vals.append(zv)
    if not out_rows:
        return sp.csr_matrix((n, t * d))
    return sp.csr_matrix((np.concatenate(out_vals),
                          (np.concatenate(out_rows), np.concatenate(out_cols))), shape=(n, t * d))


def build_per_title(C: float = 1.0) -> Pipeline:
    lr = build_lr(**C0_KW, C=C)
    blocks = list(lr.named_steps["features"].transformers)
    blocks.append(("per_title", PerTitleCopies(),
                   ["title_text", "icd", "icd_cat", "gender", "full_lemma"] + PATIENT_NUMERIC))
    return Pipeline([("features", ColumnTransformer(blocks)), ("clf", clone(lr.named_steps["clf"]))])


class CatBoostStage2(BaseEstimator, ClassifierMixin):
    """CatBoost with fold-local text SVD; categorical columns use ordered target stats."""

    def __init__(self, use_protocol_id: bool = False, n_svd: int = 30, iterations: int = 600,
                 depth: int = 4, learning_rate: float = 0.05, thread_count: int = 2):
        self.use_protocol_id = use_protocol_id
        self.n_svd = n_svd
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.thread_count = thread_count

    def _cats(self):
        return CAT_COLS + (["protocol_id"] if self.use_protocol_id else [])

    def _frame(self, X):
        svd = self.text_.transform(X["full_lemma"])
        out = pd.DataFrame(svd, columns=[f"svd{i}" for i in range(svd.shape[1])], index=X.index)
        for c in NUMERIC_FEATURES_S2 + CONTRADICTION_COLS + ["has_icd"]:
            out[c] = X[c].astype(float)
        for c in self._cats():
            out[c] = X[c].astype(str)
        return out

    def fit(self, X, y):
        self.text_ = Pipeline([("tfidf", TfidfVectorizer(min_df=2, max_features=20_000)),
                               ("svd", TruncatedSVD(self.n_svd, random_state=RANDOM_STATE))])
        self.text_.fit(X["full_lemma"])
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0,
            thread_count=self.thread_count, allow_writing_files=False,
        )
        self.model_.fit(self._frame(X), np.asarray(y), cat_features=self._cats())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))


class MeanBlend(BaseEstimator, ClassifierMixin):
    """Unweighted probability average; both members are refit on the same fold."""

    def __init__(self, members=()):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.fitted_], axis=0)
        return np.column_stack([1 - p, p])


def main() -> None:
    X = build_h24_frame()
    X["protocol_id"] = pd.factorize(X["protocol_text"])[0].astype(str)
    y = X["label"].to_numpy()
    titles = X["title_text"].to_numpy()
    splits = repeated_splits(y)
    base = oof_matrix("G", build_lr(), X, y, splits)

    candidates = {
        "C0 G+ICD+diagnosis": build_lr(**C0_KW),
        "C1 C0+per-title copies": build_per_title(),
        "C2 CatBoost": CatBoostStage2(),
        "C3 CatBoost+protocol id": CatBoostStage2(use_protocol_id=True),
    }
    rows, probas = [], {}
    for name, est in candidates.items():
        key = name.split()[0]
        probas[key] = oof_matrix(key, est, X, y, splits)
        r = summarise(name, probas[key], y, splits, titles, base_proba=base,
                      frozen=frozen_oof(est, X, y))
        rows.append(r)
        print_row(r)

    tree = max(("C2", "C3"), key=lambda k: next(r["m2_mean"] for r in rows if r["candidate"].startswith(k)))
    blend = MeanBlend(members=(candidates["C0 G+ICD+diagnosis"],
                               candidates[next(n for n in candidates if n.startswith(tree))]))
    name = f"C4 mean(C0,{tree})"
    p = oof_matrix(f"C4_{tree}", blend, X, y, splits)
    rows.append(summarise(name, p, y, splits, titles, base_proba=base,
                          frozen=frozen_oof(blend, X, y)))
    print_row(rows[-1])

    for c in (0.3, 3.0):
        name = f"C5 C0 at C={c}"
        est = build_lr(**C0_KW, C=c)
        p = oof_matrix(f"C5_C{c}", est, X, y, splits)
        rows.append(summarise(name, p, y, splits, titles, base_proba=base, frozen=frozen_oof(est, X, y)))
        print_row(rows[-1])

    save_rows(rows, "step_c_results")
    print("\npassing the gate:", [r["candidate"] for r in rows if r.get("passes_gate")] or "NONE")


if __name__ == "__main__":
    main()
