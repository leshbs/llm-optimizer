"""Corruption-invariant feature space for Stage 1 (title -> General/Special).

Motivation
----------
``train_stage1.csv`` had every Cyrillic character replaced by the literal ASCII
byte ``?`` at some earlier point in the data pipeline. The substitution is
*length preserving* and leaves every non-Cyrillic character intact, so train and
test titles effectively live in two different alphabets.

A lexical vectoriser fit on the corrupted train text therefore learns weights on
features that can never fire at inference time. Measured on the production
Stage 1 pipeline (``char_wb`` 2-5 grams, ``min_df=2``):

    share of TRAIN tf-idf mass on "?"-bearing n-grams : 0.9468
    share of TEST  tf-idf mass on "?"-bearing n-grams : 0.0000

i.e. ~95% of the learned model is dead at inference.

The fix implemented here is to apply the *same* corruption to the test text and
derive features only from what survives it. Masking both sides makes the feature
space provably identical across train and test, which in turn makes
cross-validation an honest estimate of leaderboard transfer.

Everything in this module is deterministic and derived from training data only;
no test labels are used anywhere (competition rule 12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

_CYRILLIC = re.compile(r"[А-яЁё]")
_LATIN_RUN = re.compile(r"[A-Za-z]{2,}")
_DIGIT = re.compile(r"\d")


# --------------------------------------------------------------------------- #
# Corruption model
# --------------------------------------------------------------------------- #
def mask_cyrillic(text: str) -> str:
    """Reproduce the corruption: every Cyrillic character becomes ``?``.

    Applied to *both* splits so that train and test share one alphabet.
    """
    return _CYRILLIC.sub("?", str(text))


@dataclass(frozen=True)
class TitleParts:
    """A hierarchical ``title_text`` split into its semantic lines."""

    guideline: str          # line 0 - clinical guideline name
    age_category: str       # line 1 - "Возрастная категория: ..."
    hierarchy: tuple        # intermediate section lines
    subsection: str         # last line - the subsection itself, with '#' prefix
    depth: int              # heading depth of the subsection ('#' count)
    body: str               # subsection with the '#' prefix stripped

    @property
    def n_lines(self) -> int:
        return 3 + len(self.hierarchy)


def parse_title(text: str) -> TitleParts:
    """Split a (masked or clean) title into its parts.

    The layout is addressed positionally rather than by keyword: line 1 is
    always the age category. That matters because line 0 also begins with a long
    run of '?' once masked, and a regex looking for the age field matches it by
    mistake.
    """
    lines = [ln.strip() for ln in str(text).split("\n")]
    while len(lines) < 2:
        lines.append("")
    subsection = lines[-1]
    depth = len(subsection) - len(subsection.lstrip("#"))
    return TitleParts(
        guideline=lines[0],
        age_category=lines[1],
        hierarchy=tuple(lines[2:-1]),
        subsection=subsection,
        depth=depth,
        body=subsection.lstrip("# ").strip(),
    )


# --------------------------------------------------------------------------- #
# Invariant representations
# --------------------------------------------------------------------------- #
SKELETON_FEATURE_NAMES = (
    "depth", "n_lines", "len_guideline", "len_subsection", "len_body",
    "n_words_body", "mean_word_len", "max_word_len", "min_word_len",
    "n_parens", "n_commas", "n_dashes", "n_colons", "n_digits_body",
    "n_quotes", "has_latin_body", "has_digit_body",
    "n_guideline_parens", "n_guideline_commas", "n_words_guideline",
    "n_long_words", "n_short_words", "long_word_ratio",
)


class SkeletonFeatures(BaseEstimator, TransformerMixin):
    """Numeric features computed from the *masked* title only.

    Every quantity here is a count, a length or a punctuation statistic, so it
    takes an identical value on the corrupted and the clean form of a title.
    """

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(SKELETON_FEATURE_NAMES, dtype=object)

    def transform(self, X) -> np.ndarray:
        rows = [self._row(parse_title(mask_cyrillic(t))) for t in np.asarray(X, dtype=object)]
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _row(p: TitleParts) -> list:
        word_lengths = [len(w) for w in p.body.split()]
        n_words = len(word_lengths)
        n_long = sum(1 for w in word_lengths if w >= 10)
        return [
            p.depth, p.n_lines, len(p.guideline), len(p.subsection), len(p.body),
            n_words,
            float(np.mean(word_lengths)) if word_lengths else 0.0,
            max(word_lengths, default=0),
            min(word_lengths, default=0),
            p.body.count("("), p.body.count(","), p.body.count("-"),
            p.body.count(":"), len(_DIGIT.findall(p.body)), p.body.count(chr(34)),
            int(bool(_LATIN_RUN.search(p.body))),
            int(bool(_DIGIT.search(p.body))),
            p.guideline.count("("), p.guideline.count(","), len(p.guideline.split()),
            n_long, sum(1 for w in word_lengths if w <= 3),
            n_long / n_words if n_words else 0.0,
        ]


def invariant_tokens(text: str) -> str:
    """Render a title as a whitespace-separated bag of invariant tokens.

    Word *shapes* (``w7`` = a seven-character word) stand in for the lexical
    n-grams the corruption destroyed: the sequence of word lengths in a
    subsection title survives masking exactly, and n-grams over that sequence
    recover a useful amount of phrase structure. The age-category line is
    emitted verbatim in masked form so the model can use it as a categorical --
    it survives masking bijectively and its label rate ranges from 0.079 to
    0.292 across its four values, which the production meta-features never
    captured.
    """
    p = parse_title(mask_cyrillic(text))
    tokens = [
        "D%d" % p.depth,
        "AGE=%s" % p.age_category,
        "NW%d" % min(len(p.body.split()), 20),
        "LEN%d" % min(len(p.body) // 10, 15),
        "NL%d" % p.n_lines,
    ]
    if _LATIN_RUN.search(p.body):
        tokens.append("HASLATIN")
    if _DIGIT.search(p.body):
        tokens.append("HASDIGIT")
    # word-length skeleton of the subsection, carrying attached punctuation
    for word in p.body.split():
        shape = "w%d" % len(word)
        if "(" in word:
            shape += "p"
        if "," in word:
            shape += "c"
        if _DIGIT.search(word):
            shape += "d"
        tokens.append(shape)
    return " ".join(tokens)


class InvariantTokenizer(BaseEstimator, TransformerMixin):
    """``title_text`` -> invariant token document (a plain string)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.asarray(
            [invariant_tokens(t) for t in np.asarray(X, dtype=object)], dtype=object
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_stage1_features() -> FeatureUnion:
    """Invariant numeric skeleton + tf-idf over invariant shape n-grams."""
    return FeatureUnion(
        [
            (
                "skeleton",
                Pipeline([("extract", SkeletonFeatures()), ("scale", StandardScaler())]),
            ),
            (
                "shape_ngrams",
                Pipeline(
                    [
                        ("tokens", InvariantTokenizer()),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                token_pattern=r"\S+",
                                ngram_range=(1, 3),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )


def build_stage1_model(C: float = 1.0, class_weight: str = "balanced") -> Pipeline:
    """The Stage 1 estimator.

    ``LogisticRegression`` rather than ``LinearSVC`` because the decision
    threshold is tuned against macro-F0.5, which needs a probability rather than
    an uncalibrated margin.
    """
    return Pipeline(
        [
            ("features", build_stage1_features()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
