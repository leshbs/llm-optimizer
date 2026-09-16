"""Stage 1 on the repaired (clean UTF-8) ``train_stage1.csv``.

The re-downloaded training file is intact, which retires the entire premise of
sections 129 and 130. Those models read word *shapes* and corpus *recurrence*
because the Cyrillic was gone; with real text available, both are strictly worse
than reading the words. Nothing here masks, decodes or matches against a corpus.

Design follows the evidence already gathered rather than starting over:

* the H15 ablation showed the subsection line and its surrounding context carry
  different signal and both matter, so they stay as separate blocks;
* competition brief section 7 defines Special as "names a restricted patient
  group", so the condition-marker block is kept and now fires on every row
  instead of the ~19% that used to decode;
* lemmatisation matters for Russian morphology (brief section 5.3):
  ``тяжёлой``/``тяжелая``/``тяжелый`` must collapse to one feature;
* char n-grams are retained because they absorb morphological variation and
  out-of-vocabulary medical terms that lemmatisation misses.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from stage1_hybrid import CONDITION_GROUPS, CONDITION_NAMES, lemmatize
from stage1_invariant import SEED, parse_title

_LATIN = re.compile(r"[A-Za-z]{2,}")
_DIGIT = re.compile(r"\d")

STRUCT_FEATURE_NAMES = (
    "depth", "n_lines", "len_title", "len_subsection", "n_words_subsection",
    "n_words_guideline", "has_latin", "has_digit", "n_parens", "n_commas",
    "n_quotes", "n_cond_groups_sub", "n_cond_groups_ctx",
)


def title_fields(text: str) -> tuple:
    """(guideline, age_category, hierarchy_text, subsection) from a clean title."""
    p = parse_title(text)
    return p.guideline, p.age_category, " ".join(p.hierarchy), p.body


class Field(BaseEstimator, TransformerMixin):
    """Select and lemmatise one part of the title."""

    def __init__(self, which: str = "subsection", lemma: bool = True) -> None:
        self.which = which
        self.lemma = lemma

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        out = []
        for text in np.asarray(X, dtype=object):
            guideline, age, hierarchy, subsection = title_fields(text)
            if self.which == "subsection":
                picked = subsection
            elif self.which == "context":
                picked = " ".join((guideline, age, hierarchy))
            else:
                picked = str(text)
            out.append(lemmatize(picked) if self.lemma else picked)
        return np.asarray(out, dtype=object)


class StructFeatures(BaseEstimator, TransformerMixin):
    """Structural and condition-marker counts, now computed on real words."""

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(STRUCT_FEATURE_NAMES, dtype=object)

    def transform(self, X) -> np.ndarray:
        return np.asarray([self._row(t) for t in np.asarray(X, dtype=object)], dtype=float)

    @staticmethod
    def _row(text: str) -> list:
        p = parse_title(text)
        guideline, age, hierarchy, subsection = title_fields(text)
        sub_lemmas = set(lemmatize(subsection).split())
        ctx_lemmas = set(lemmatize(" ".join((guideline, age, hierarchy))).split())
        return [
            p.depth,
            p.n_lines,
            len(str(text)),
            len(subsection),
            len(subsection.split()),
            len(guideline.split()),
            int(bool(_LATIN.search(subsection))),
            int(bool(_DIGIT.search(subsection))),
            subsection.count("("),
            subsection.count(","),
            str(text).count('"'),
            sum(1 for g in CONDITION_NAMES if sub_lemmas & CONDITION_GROUPS[g]),
            sum(1 for g in CONDITION_NAMES if ctx_lemmas & CONDITION_GROUPS[g]),
        ]


def build_clean_features(
    sub_ngram_max: int = 2, sub_min_df: int = 2, ctx_min_df: int = 2, char_min_df: int = 3
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "lex_subsection",
                Pipeline([
                    ("field", Field("subsection")),
                    ("tfidf", TfidfVectorizer(ngram_range=(1, sub_ngram_max),
                                              min_df=sub_min_df, sublinear_tf=True)),
                ]),
            ),
            (
                "lex_context",
                Pipeline([
                    ("field", Field("context")),
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=ctx_min_df,
                                              sublinear_tf=True)),
                ]),
            ),
            (
                "char",
                Pipeline([
                    ("field", Field("subsection", lemma=False)),
                    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              min_df=char_min_df, sublinear_tf=True)),
                ]),
            ),
            (
                "struct",
                Pipeline([("extract", StructFeatures()), ("scale", StandardScaler())]),
            ),
        ]
    )


def build_clean_model(C: float = 1.0, **feature_kwargs) -> Pipeline:
    return Pipeline([
        ("features", build_clean_features(**feature_kwargs)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])
