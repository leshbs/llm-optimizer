"""Phase 4 (H21): recurrence features, computed fold-safely.

Stage 2 is a many-to-many join -- 690 rows over 221 distinct protocols and 37
distinct titles -- so recurrence is a real structural property. It is also the
easiest place in this project to leak, because the obvious implementation counts
over the whole dataset before cross-validation starts.

**How fold safety is guaranteed here.** Every recurrence statistic lives inside
a scikit-learn transformer whose ``fit`` sees only the training rows of the fold
and their labels, and whose ``transform`` maps a validation row through the
statistics learned on that training fold. The transformer sits inside the
pipeline that ``cross_val_predict`` clones per fold, so a validation label can
never reach a validation feature -- the encoder is refit from scratch on each
training fold. Unseen keys fall back to the training-fold prior, never to a
global value.

Run:  python src/stage2_h21.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    RANDOM_STATE,
    build_base_frame,
    load_folds,
    load_h8_features,
    merge_extra_cols,
    stage2_score,
)

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import fbeta_score  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h21"
NUMERIC_FEATURES_S2 = ["age", "has_structured_header", "negation_count", "protocol_len"]

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Whitespace- and case-normalised form, for approximate recurrence."""
    return _WS.sub(" ", str(text).lower()).strip()


class RecurrenceEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe recurrence and label-conditioned statistics.

    ``families`` selects which blocks to emit:
      ``protocol``  exact protocol count / seen flag
      ``norm``      normalised protocol count / seen flag
      ``title``     title count / seen flag
      ``pair``      title+protocol count / seen flag
      ``label``     smoothed P(y=1 | key) for protocol, normalised protocol, title
    """

    def __init__(self, families=("protocol",), alpha: float = 10.0):
        self.families = families
        self.alpha = alpha

    # -- keys ------------------------------------------------------------- #
    @staticmethod
    def _keys(X):
        title = X["title_text"].astype(str)
        protocol = X["protocol_text"].astype(str)
        return {
            "protocol": protocol,
            "norm": protocol.map(normalise),
            "title": title,
            "pair": title + "||" + protocol,
        }

    def fit(self, X, y=None):
        keys = self._keys(X)
        y = np.asarray(y).astype(float)
        self.prior_ = float(y.mean())
        self.counts_, self.rates_ = {}, {}
        for name, k in keys.items():
            self.counts_[name] = k.value_counts()
            if "label" in self.families:
                grouped = pd.DataFrame({"k": k.to_numpy(), "y": y}).groupby("k")["y"]
                n = grouped.size()
                s = grouped.sum()
                # additive smoothing toward the training-fold prior
                self.rates_[name] = (s + self.alpha * self.prior_) / (n + self.alpha)
        self.feature_names_ = self._names()
        return self

    def _names(self):
        names = []
        for fam in ("protocol", "norm", "title", "pair"):
            if fam in self.families:
                names += ["%s_count" % fam, "%s_seen" % fam]
        if "label" in self.families:
            names += ["rate_protocol", "rate_norm", "rate_title"]
        return names

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    def transform(self, X):
        keys = self._keys(X)
        cols = []
        for fam in ("protocol", "norm", "title", "pair"):
            if fam not in self.families:
                continue
            counts = keys[fam].map(self.counts_[fam]).fillna(0.0).to_numpy()
            cols.append(counts)
            cols.append((counts > 0).astype(float))
        if "label" in self.families:
            for fam in ("protocol", "norm", "title"):
                cols.append(keys[fam].map(self.rates_[fam]).fillna(self.prior_).to_numpy())
        return np.column_stack(cols) if cols else np.empty((len(X), 0))


def build_pipeline(extra_numeric_cols, recurrence_families=None) -> Pipeline:
    transformers = [
        ("title_tfidf", TfidfVectorizer(min_df=2), "title_lemma"),
        ("narrative_tfidf", TfidfVectorizer(min_df=2, max_features=20_000), "narrative_lemma"),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES_S2),
        ("gender", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
    ]
    if extra_numeric_cols:
        transformers.append(("h8_numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), list(extra_numeric_cols)))
    if recurrence_families:
        transformers.append(("recurrence", Pipeline([
            ("encode", RecurrenceEncoder(families=recurrence_families)),
            ("scale", StandardScaler())]), ["title_text", "protocol_text"]))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ])


def run(X, y, folds, extra_cols, families, threshold=0.5) -> dict:
    pipe = build_pipeline(extra_cols, families)
    proba = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")[:, 1]
    hard = (proba >= threshold).astype(int)
    y_arr = np.asarray(y)
    per_fold = [fbeta_score(y_arr[v], hard[v], beta=2, average="macro", zero_division=0)
                for _, v in folds]
    return {
        "oof_proba": proba, "m2": stage2_score(y_arr, hard),
        "macro_f2": float(fbeta_score(y_arr, hard, beta=2, average="macro", zero_division=0)),
        "per_fold": [float(v) for v in per_fold],
        "mean_fold": float(np.mean(per_fold)), "std_fold": float(np.std(per_fold, ddof=1)),
        "fn": int(((y_arr == 1) & (hard == 0)).sum()),
        "fp": int(((y_arr == 0) & (hard == 1)).sum()),
        "positive_rate": float(hard.mean()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)

    experiments = {
        "A. baseline (config G)": None,
        "B. exact protocol recurrence": ("protocol",),
        "C. normalised protocol recurrence": ("norm",),
        "D. title recurrence": ("title",),
        "E. title+protocol pair recurrence": ("pair",),
        "F. label-conditioned rates": ("label",),
        "G. all recurrence families": ("protocol", "norm", "title", "pair", "label"),
    }

    rows, baseline = [], None
    for name, fams in experiments.items():
        res = run(X, y, folds, CONTRADICTION_COLS, fams)
        if baseline is None:
            baseline = res
        wins = int((np.array(res["per_fold"]) > np.array(baseline["per_fold"])).sum())
        rows.append({
            "experiment": name,
            "families": "+".join(fams) if fams else "none",
            "cv_m2": res["m2"], "delta_vs_baseline": res["m2"] - baseline["m2"],
            "macro_f2": res["macro_f2"], "mean_fold": res["mean_fold"],
            "std_fold": res["std_fold"], "fn": res["fn"], "fp": res["fp"],
            "positive_rate": res["positive_rate"],
            "folds_improved": wins,
            "fold_scores": [round(v, 4) for v in res["per_fold"]],
        })
        print("%-34s M2=%.6f  d=%+.6f  FN=%3d FP=%3d  pos=%.4f  folds+=%d"
              % (name, res["m2"], rows[-1]["delta_vs_baseline"], res["fn"], res["fp"],
                 res["positive_rate"], wins))
        np.save(OUT / ("h21_oof_%s.npy" % name.split(".")[0]), res["oof_proba"])

    pd.DataFrame(rows).to_csv(OUT / "recurrence_ablation.csv", index=False)
    (OUT / "h21_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    kept = [r["experiment"] for r in rows[1:]
            if r["delta_vs_baseline"] > 0 and r["folds_improved"] >= 4]
    print("\nvariants meeting the gate (delta > 0 AND >= 4/5 folds): %s" % (kept or "NONE"))
    print("wrote h21/recurrence_ablation.csv")


if __name__ == "__main__":
    main()
