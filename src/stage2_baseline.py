"""Phase 0: the frozen Stage 2 baseline (config G), extracted and reproducible.

The production Stage 2 model lives across a dozen notebook cells that depend on
notebook-global state. Every H19-H22 experiment has to be compared against it
fold for fold, so it is lifted here into a standalone module. Nothing is
redesigned -- the components are ported verbatim so that the reproduction is a
check on the extraction, not a new model:

* ``lemmatize_ru``          from notebook cell 4baa9ae9
* ``ProtocolFieldExtractor`` from cell 995abfbf
* ``NEGATION_PATTERN``      from cell a6e2a13a
* ``stage2_score``          from cell fc27f099
* variant pipeline builder  from cell sech8p2setup
* the 11 promoted contradiction columns, read from the H8 feature cache
* the CV splitter, read from ``frozen/skf2_folds.json`` -- the same frozen folds
  every previous Stage 2 experiment used

Run:  python src/stage2_baseline.py
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

# ---------------------------------------------------------------- lemmatiser #
import pymorphy3  # noqa: E402

_morph = pymorphy3.MorphAnalyzer()
_token_re = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "для", "при",
}


@lru_cache(maxsize=200_000)
def _lemmatize_word(word: str) -> str:
    return _morph.parse(word)[0].normal_form


def lemmatize_ru(text: str) -> str:
    tokens = _token_re.findall(str(text).lower())
    return " ".join(_lemmatize_word(t) for t in tokens if t not in RU_STOPWORDS)


# ------------------------------------------------------- protocol extraction #
NEGATION_PATTERN = re.compile(
    r"\bне\s+\w+|\bнет\s+\w+|отсутств\w*|отрицательн\w*|не\s+выявлен\w*|не\s+определя\w*",
    re.IGNORECASE,
)
NARRATIVE_FIELDS = ["Жалобы", "Анамнез", "Объективный статус"]


def _extract_field(text: str, field: str):
    m = re.search(rf"{re.escape(field)}\s*:\s*(.+)", text)
    return m.group(1).strip() if m else None


def _extract_age(text: str) -> float:
    structured = _extract_field(text, "Возраст")
    if structured is not None:
        m = re.search(r"\d{1,3}", structured)
        if m:
            return float(m.group())
    m = re.search(r"(\d{1,3})\s*[- ]?\s*(лет|года|год)\b", text)
    return float(m.group(1)) if m else np.nan


def _extract_gender(text: str):
    structured = _extract_field(text, "Пол")
    if structured:
        s = structured.lower()
        if s.startswith("ж"):
            return "F"
        if s.startswith("м"):
            return "M"
    return None


def extract_narrative(text: str) -> str:
    chunks = []
    for field in NARRATIVE_FIELDS:
        m = re.search(
            rf"{re.escape(field)}\s*:\s*(.+?)(?=\n[А-ЯЁ][^\n:]{{0,40}}:|\Z)", text, re.DOTALL
        )
        if m:
            chunks.append(m.group(1).strip())
    return " ".join(chunks) if chunks else text


class ProtocolFieldExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).astype(str)
        return pd.DataFrame({
            "gender": s.apply(_extract_gender),
            "age": s.apply(_extract_age),
            "has_structured_header": s.str.contains(r"Пол\s*:", regex=True).astype(int),
            "negation_count": s.apply(lambda t: len(NEGATION_PATTERN.findall(t))),
            "protocol_len": s.str.len(),
            "narrative_text": s.apply(extract_narrative),
        })


# ------------------------------------------------------------------- scoring #
def stage2_score(y_true, y_pred) -> float:
    macro_f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    return float(np.clip((macro_f2 - 0.5) / 0.45, 0.0, 1.0))


# ------------------------------------------------------------- baseline spec #
NUMERIC_FEATURES_S2 = ["age", "has_structured_header", "negation_count", "protocol_len"]
CONTRADICTION_COLS = [
    "age_contradiction", "gender_condition_present", "gender_unknown", "severity_match",
    "severity_contradiction", "severity_unknown", "condition_overlap", "condition_contradiction",
    "condition_unknown", "contradiction_count", "contradiction_density",
]
BASELINE_THRESHOLD = 0.5


def load_folds() -> list:
    """The frozen 5-fold split every Stage 2 experiment in this project has used."""
    raw = json.loads((ROOT / "frozen" / "skf2_folds.json").read_text(encoding="utf-8"))
    return [(np.array(s["train_idx"]), np.array(s["val_idx"])) for s in raw]


def build_base_frame(path=None) -> pd.DataFrame:
    """A Stage 2 frame with lemmatised text and extracted protocol fields attached.

    ``gender_raw`` keeps the un-filled gender, because the contradiction features
    test it for null while the pipeline needs the 'unknown'-filled version.
    """
    train = pd.read_csv(path or ROOT / "train_stage2.csv")
    train["title_lemma"] = train["title_text"].apply(lemmatize_ru)
    extracted = ProtocolFieldExtractor().transform(train["protocol_text"])
    train = pd.concat(
        [train.drop(columns=[c for c in extracted.columns if c in train.columns]), extracted],
        axis=1,
    )
    train["narrative_lemma"] = train["narrative_text"].apply(lemmatize_ru)
    train["gender_raw"] = train["gender"]
    train["gender"] = train["gender"].fillna("unknown")
    return train


def load_h8_features(base: pd.DataFrame = None) -> pd.DataFrame:
    """Contradiction features, regenerated (the parquet cache is unreadable here)."""
    from stage2_features import build_contradiction_features

    if base is None:
        base = build_base_frame()
    src = base.copy()
    src["gender"] = src["gender_raw"]
    return build_contradiction_features(src)


def merge_extra_cols(base_df, h8_df, extra_numeric_cols):
    if not extra_numeric_cols:
        return base_df
    sub = h8_df[["id"] + list(extra_numeric_cols)]
    merged = base_df.merge(sub, on="id", how="left")
    assert len(merged) == len(base_df)
    assert (merged["id"].to_numpy() == base_df["id"].to_numpy()).all()
    return merged


def build_variant_pipeline(extra_numeric_cols=(), extra_text_cols=()) -> Pipeline:
    """The config-G pipeline. ``extra_*`` is how H19-H22 features get added."""
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
            ("scale", StandardScaler()),
        ]), list(extra_numeric_cols)))
    for name in extra_text_cols:
        transformers.append((f"text_{name}", TfidfVectorizer(min_df=2), name))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ])


def evaluate(X, y, folds, extra_numeric_cols=(), extra_text_cols=(),
             threshold: float = BASELINE_THRESHOLD) -> dict:
    """Out-of-fold evaluation on the frozen folds, reported the way the roadmap asks."""
    pipe = build_variant_pipeline(extra_numeric_cols, extra_text_cols)
    proba = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")[:, 1]
    hard = (proba >= threshold).astype(int)
    y_arr = np.asarray(y)
    per_fold = [
        fbeta_score(y_arr[v], hard[v], beta=2, average="macro", zero_division=0)
        for _, v in folds
    ]
    return {
        "oof_proba": proba,
        "oof_pred": hard,
        "m2": stage2_score(y_arr, hard),
        "macro_f2": float(fbeta_score(y_arr, hard, beta=2, average="macro", zero_division=0)),
        "per_fold_macro_f2": [float(s) for s in per_fold],
        "mean_fold": float(np.mean(per_fold)),
        "std_fold": float(np.std(per_fold, ddof=1)),
        "fn": int(((y_arr == 1) & (hard == 0)).sum()),
        "fp": int(((y_arr == 0) & (hard == 1)).sum()),
        "positive_rate": float(hard.mean()),
        "threshold": threshold,
    }


def main() -> None:
    out = ROOT / "h19"
    out.mkdir(exist_ok=True)

    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]

    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)
    res = evaluate(X, y, folds, extra_numeric_cols=CONTRADICTION_COLS)

    reference_m2 = 0.7023868247508647
    drift = res["m2"] - reference_m2
    print("reproduced M2 = %.10f" % res["m2"])
    print("reference  M2 = %.10f" % reference_m2)
    print("drift         = %+.2e  -> %s"
          % (drift, "EXACT" if abs(drift) < 1e-9 else
             "close" if abs(drift) < 1e-4 else "MATERIALLY DIFFERENT -- STOP"))
    print("macro-F2 %.6f  per-fold %s" % (res["macro_f2"], [round(s, 4) for s in res["per_fold_macro_f2"]]))
    print("mean %.6f  std %.6f  FN %d  FP %d  posrate %.4f"
          % (res["mean_fold"], res["std_fold"], res["fn"], res["fp"], res["positive_rate"]))

    # cross-check against the OOF cached by H8
    cached = pd.read_csv(ROOT / "h8" / "variants" / "oof_g_final.csv")
    max_dev = float(np.abs(cached["oof_proba_g"].to_numpy() - res["oof_proba"]).max())
    print("max |proba - cached H8 oof_proba_g| = %.3e" % max_dev)

    manifest = {
        "model_name": "G (frozen C2 structured_preprocessor + 11 promoted contradiction features)",
        "estimator": "LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)",
        "feature_set": {
            "title_tfidf": "TfidfVectorizer(min_df=2) on title_lemma",
            "narrative_tfidf": "TfidfVectorizer(min_df=2, max_features=20000) on narrative_lemma",
            "numeric": NUMERIC_FEATURES_S2,
            "gender": "OneHotEncoder(handle_unknown='ignore')",
            "contradiction": CONTRADICTION_COLS,
        },
        "hyperparameters": {"max_iter": 2000, "class_weight": "balanced",
                            "random_state": RANDOM_STATE, "C": 1.0},
        "threshold": BASELINE_THRESHOLD,
        "cv_splitter": "frozen/skf2_folds.json (5 frozen stratified folds)",
        "input_files": ["train_stage2.csv", "h8/cache/h8_features.parquet",
                        "frozen/skf2_folds.json"],
        "m2": res["m2"],
        "macro_f2": res["macro_f2"],
        "fold_scores_macro_f2": res["per_fold_macro_f2"],
        "mean_fold": res["mean_fold"],
        "std_fold": res["std_fold"],
        "fn": res["fn"],
        "fp": res["fp"],
        "positive_rate": res["positive_rate"],
        "reference_m2": reference_m2,
        "drift_vs_reference": drift,
        "max_abs_deviation_vs_cached_oof": max_dev,
        "real_m2_do_not_use_for_selection": 0.7307714752567694,
    }
    (out / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    np.save(out / "baseline_oof_proba.npy", res["oof_proba"])
    print("wrote h19/baseline_manifest.json")


if __name__ == "__main__":
    main()
