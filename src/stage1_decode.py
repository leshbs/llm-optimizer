"""Partial recovery of the Cyrillic text destroyed in ``train_stage1.csv``.

The corruption replaced every Cyrillic character with ``?`` but preserved
length and every non-Cyrillic character. That is not invertible in general --
a pure-Cyrillic word of length 7 masks to ``???????`` and there are only as
many distinct masked word forms as there are word lengths, so no word-level
dictionary can help. What *is* invertible is a line that also occurs somewhere
in a corpus we can still read: masking is a function, so a corrupted line and
a clean line that mask to the same string are candidates for being equal.

The readable corpus is the clean side of the competition data -- the Stage 1
test titles and both Stage 2 title columns. Those are model *inputs*, never
labels, so using them is transductive, not leakage (competition rule 12).

Two properties make this usable rather than dangerous:

*Leave-one-out matching.* A test title is itself in the corpus, so a naive
matcher would decode every test row trivially while decoding only half the
train rows. Decodability is strongly label-correlated (Special rate 0.41 for
subsections absent from the corpus versus 0.04 for those present), so that
asymmetry would hand the model a shortcut that is constant at inference time
-- exactly the failure that sank the pre-H14 Stage 1 pipeline. Every lookup
therefore excludes the querying row's own contribution to the corpus, which
brings the two splits to within 2 points of each other on every line role.

*Intersection over ambiguous candidates.* When several clean lines share a
masked form, any lemma present in *all* of them is in the true line whatever
the right answer is. Those lemmas are emitted as certain; the rest are
dropped. This extracts sound information from ambiguous matches instead of
discarding them or guessing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from io_utils import read_competition_csv  # noqa: E402
from stage1_invariant import mask_cyrillic

DATA_DIR = Path(__file__).resolve().parent.parent

# A source key identifies the row a clean line came from, so that row can be
# excluded when it is the one asking.
SourceKey = tuple


@dataclass(frozen=True)
class DecodedTitle:
    """Result of decoding one corrupted title, line role by line role."""

    guideline: str
    age_category: str
    hierarchy: tuple
    subsection: str
    subsection_ambiguous: bool
    n_lines_decoded: int
    n_lines: int
    raw: str = ""     # the original title, so invariant features stay available

    @property
    def coverage(self) -> float:
        return self.n_lines_decoded / self.n_lines if self.n_lines else 0.0

    @property
    def text(self) -> str:
        """All certainly-recovered text, newline joined."""
        parts = [self.guideline, self.age_category, *self.hierarchy, self.subsection]
        return "\n".join(p for p in parts if p)


class LineDecoder:
    """Masked-line -> clean-line index with leave-one-out lookup."""

    def __init__(self) -> None:
        # masked form -> clean form -> set of source keys supporting it
        self._index: dict = defaultdict(lambda: defaultdict(set))

    def add_title(self, title: str, key: SourceKey) -> None:
        for line in str(title).split("\n"):
            line = line.strip()
            if line:
                self._index[mask_cyrillic(line)][line].add(key)

    def candidates(self, masked_line: str, exclude: SourceKey = None) -> set:
        """Clean lines matching ``masked_line``, ignoring support from ``exclude``.

        A candidate survives only if some source *other* than the excluded row
        attests it, which is what keeps train and test decodable at the same
        rate.
        """
        entry = self._index.get(masked_line)
        if not entry:
            return set()
        return {clean for clean, keys in entry.items() if keys - {exclude}}

    def resolve(self, masked_line: str, exclude: SourceKey = None) -> tuple:
        """Return (certain_text, is_ambiguous).

        A unique candidate is returned whole. Several candidates collapse to
        the words they all share, in the order they appear in the first
        candidate -- those are certainly in the true line. No candidates at all
        yields the empty string.
        """
        cands = self.candidates(masked_line, exclude)
        if not cands:
            return "", False
        if len(cands) == 1:
            return next(iter(cands)), False
        token_sets = [set(_words(c)) for c in cands]
        shared = set.intersection(*token_sets)
        if not shared:
            return "", True
        first = sorted(cands)[0]
        return " ".join(w for w in _words(first) if w in shared), True


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> list:
    return _WORD.findall(text.lower())


def build_decoder(
    test_s1: pd.DataFrame, train_s2: pd.DataFrame, test_s2: pd.DataFrame
) -> LineDecoder:
    """Index every clean title available in the competition inputs."""
    decoder = LineDecoder()
    for i, title in enumerate(test_s1["title_text"]):
        decoder.add_title(title, ("s1_test", i))
    for i, title in enumerate(train_s2["title_text"]):
        decoder.add_title(title, ("s2_train", i))
    for i, title in enumerate(test_s2["title_text"]):
        decoder.add_title(title, ("s2_test", i))
    return decoder


def decode_title(title: str, decoder: LineDecoder, exclude: SourceKey = None) -> DecodedTitle:
    """Decode one title. ``title`` may be corrupted or clean; it is masked first.

    Masking a clean title before decoding is deliberate: inference must run the
    same procedure training ran, or the feature space stops matching.
    """
    lines = [ln.strip() for ln in str(title).split("\n")]
    while len(lines) < 3:
        lines.append("")
    masked = [mask_cyrillic(ln) for ln in lines]

    resolved = [decoder.resolve(m, exclude) for m in masked]
    texts = [t for t, _ in resolved]
    n_decoded = sum(1 for t in texts if t)

    return DecodedTitle(
        guideline=texts[0],
        age_category=texts[1],
        hierarchy=tuple(t for t in texts[2:-1]),
        subsection=texts[-1],
        subsection_ambiguous=resolved[-1][1],
        n_lines_decoded=n_decoded,
        n_lines=len(lines),
        raw=str(title),
    )


def decode_frame(
    frame: pd.DataFrame, decoder: LineDecoder, self_source: str = None
) -> list:
    """Decode a whole split.

    ``self_source`` names this frame's own contribution to the corpus so it can
    be excluded row by row; pass ``None`` for a split that never entered the
    corpus (the corrupted Stage 1 train set).
    """
    return [
        decode_title(
            title, decoder, exclude=(self_source, i) if self_source else None
        )
        for i, title in enumerate(frame["title_text"])
    ]


def load_all() -> dict:
    """Load the four competition frames with their correct encodings."""
    return {
        "train_s1": read_competition_csv(DATA_DIR / "train_stage1.csv"),
        "test_s1": pd.read_csv(DATA_DIR / "test_stage1.csv"),
        "train_s2": pd.read_csv(DATA_DIR / "train_stage2.csv"),
        "test_s2": pd.read_csv(DATA_DIR / "test_stage2.csv"),
    }
