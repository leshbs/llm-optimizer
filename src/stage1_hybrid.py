"""Stage 1 model over partially decoded text plus the H14 invariant features.

The H14 rebuild reads only what survives the corruption: word shapes, depth,
punctuation, the age line. That transfers perfectly but throws away meaning.
``stage1_decode`` recovers real Cyrillic for roughly half the subsection lines,
which makes genuinely lexical features possible for the first time.

The two are unioned rather than swapped. Decoding covers about half the rows,
so a decoded-only model would be blind on the rest, whereas the union degrades
to exactly the H14 feature set when nothing decodes. Every block is computed
through the same leave-one-out decode on both splits, so the feature space
still matches across train and test by construction.

Three blocks sit on top of the H14 union:

``lex_subsection``  lemmatised word n-grams of the decoded final subsection
                    line -- the line that actually decides General vs Special.
``lex_context``     the same over the guideline title and section hierarchy,
                    which set the interpretation of the subsection.
``conditions``      counts of the patient-specificity markers named in the
                    competition brief: age, gender, severity, form, stage,
                    comorbidity, and the ``при``/``у пациентов с`` framing that
                    introduces a restricted population.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from stage1_invariant import SEED, build_stage1_features

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

try:  # pymorphy3 is the project's existing lemmatiser; degrade gracefully
    import pymorphy3

    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:  # pragma: no cover - only hit when the dict package is absent
    _MORPH = None


@lru_cache(maxsize=100_000)
def lemma(word: str) -> str:
    if _MORPH is None:
        return word
    return _MORPH.parse(word)[0].normal_form


def lemmatize(text: str) -> str:
    return " ".join(lemma(w) for w in _WORD.findall(str(text).lower()))


# -- condition markers ----------------------------------------------------- #
# Lemma sets, so that "тяжёлой"/"тяжелая"/"тяжелый" all collapse to one hit.
CONDITION_GROUPS = {
    "age": {
        "ребёнок", "ребенок", "детский", "взрослый", "новорождённый", "новорожденный",
        "подросток", "год", "старше", "младше", "пожилой", "грудной", "младенец",
    },
    "gender": {
        "женщина", "мужчина", "беременный", "беременность", "кормящий",
        "лактация", "женский", "мужской", "роды", "послеродовой",
    },
    "severity": {
        "тяжёлый", "тяжелый", "лёгкий", "легкий", "среднетяжёлый", "среднетяжелый",
        "степень", "стадия", "выраженный", "критический",
    },
    "form": {"форма", "тип", "вариант", "подтип", "течение", "фенотип"},
    "comorbidity": {
        "сопутствующий", "недостаточность", "осложнение", "осложнённый", "осложненный",
        "сахарный", "диабет", "почечный", "печёночный", "печеночный", "сочетанный",
    },
    "refractory": {"рецидив", "резистентный", "рефрактерный", "повторный", "неэффективность"},
    "restriction": {"при", "случай", "пациент", "группа", "наличие"},
}
CONDITION_NAMES = tuple(sorted(CONDITION_GROUPS))

DECODE_FEATURE_NAMES = (
    tuple("cond_%s" % g for g in CONDITION_NAMES)
    + tuple("cond_ctx_%s" % g for g in CONDITION_NAMES)
    + ("n_cond_groups", "decode_coverage", "sub_decoded", "sub_ambiguous", "n_sub_lemmas")
)


class DecodedTextExtractor(BaseEstimator, TransformerMixin):
    """Pull one field out of pre-decoded titles.

    Input is a sequence of ``DecodedTitle``; decoding happens once up front
    rather than inside each block, since it is the expensive step.
    """

    def __init__(self, field: str = "subsection") -> None:
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        out = []
        for d in X:
            if self.field == "subsection":
                text = d.subsection
            elif self.field == "context":
                text = " ".join((d.guideline, *d.hierarchy))
            else:  # full
                text = d.text
            out.append(lemmatize(text))
        return np.asarray(out, dtype=object)


# The status features below describe *whether* a line decoded, not what it
# said. Leave-one-out decoding makes them symmetric across splits, but they are
# still the block most likely to be a shortcut rather than signal, so ``mode``
# exists to score the model with and without them.
STATUS_FEATURES = ("decode_coverage", "sub_decoded", "sub_ambiguous", "n_sub_lemmas")


class ConditionFeatures(BaseEstimator, TransformerMixin):
    """Patient-specificity markers counted over the decoded text.

    ``mode`` selects which half to emit: ``markers`` keeps only the linguistic
    condition counts, ``status`` keeps only the decode-status indicators, and
    ``all`` keeps both.
    """

    def __init__(self, mode: str = "all") -> None:
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def _keep(self) -> np.ndarray:
        is_status = np.array([n in STATUS_FEATURES for n in DECODE_FEATURE_NAMES])
        if self.mode == "markers":
            return ~is_status
        if self.mode == "status":
            return is_status
        return np.ones(len(DECODE_FEATURE_NAMES), dtype=bool)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(DECODE_FEATURE_NAMES, dtype=object)[self._keep()]

    def transform(self, X) -> np.ndarray:
        rows = np.asarray([self._row(d) for d in X], dtype=float)
        return rows[:, self._keep()]

    @staticmethod
    def _row(d) -> list:
        sub = set(lemmatize(d.subsection).split())
        ctx = set(lemmatize(" ".join((d.guideline, *d.hierarchy))).split())
        sub_hits = [len(sub & CONDITION_GROUPS[g]) for g in CONDITION_NAMES]
        ctx_hits = [len(ctx & CONDITION_GROUPS[g]) for g in CONDITION_NAMES]
        return (
            sub_hits
            + ctx_hits
            + [
                sum(1 for h in sub_hits if h),
                d.coverage,
                float(bool(d.subsection)),
                float(d.subsection_ambiguous),
                float(len(sub)),
            ]
        )


class RawTitleExtractor(BaseEstimator, TransformerMixin):
    """Recover the original title string carried alongside each decode."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.asarray([d.raw for d in X], dtype=object)


def build_hybrid_features(
    lex_min_df: int = 2, lex_ngram_max: int = 2, ctx_min_df: int = 3
) -> FeatureUnion:
    """H14 invariant blocks unioned with the decoded-text blocks."""
    return FeatureUnion(
        [
            ("invariant", Pipeline([("raw", RawTitleExtractor()),
                                    ("feats", build_stage1_features())])),
            (
                "lex_subsection",
                Pipeline(
                    [
                        ("text", DecodedTextExtractor(field="subsection")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, lex_ngram_max),
                                min_df=lex_min_df,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "lex_context",
                Pipeline(
                    [
                        ("text", DecodedTextExtractor(field="context")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 1), min_df=ctx_min_df, sublinear_tf=True
                            ),
                        ),
                    ]
                ),
            ),
            (
                "conditions",
                Pipeline([("extract", ConditionFeatures()), ("scale", StandardScaler())]),
            ),
        ]
    )


def build_hybrid_model(C: float = 1.0, **feature_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("features", build_hybrid_features(**feature_kwargs)),
            (
                "clf",
                LogisticRegression(
                    C=C, class_weight="balanced", max_iter=5000, random_state=SEED
                ),
            ),
        ]
    )
