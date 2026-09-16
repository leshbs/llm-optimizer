"""Deterministic post-model corrections for Stage 1.

Only rules that hold with 100% precision on the training data are allowed to
override the model, matching the precision bar already used for the Stage 2
age/gender contradiction detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stage1_invariant import mask_cyrillic


def stage2_title_index(*frames: pd.DataFrame) -> set:
    """Masked ``title_text`` values that appear anywhere in the Stage 2 data.

    Stage 2 is defined over *Special* subsections only: the competition brief
    states the task is to decide whether a "special clinical guideline
    subsection" applies to a patient. A title used in Stage 2 is therefore
    Special by construction. Matching is done on the masked form so that the
    corrupted Stage 1 titles and the clean Stage 2 titles are comparable.
    """
    titles = set()
    for frame in frames:
        titles.update(mask_cyrillic(t) for t in frame["title_text"])
    return titles


def audit_stage2_title_rule(train_s1: pd.DataFrame, s2_titles: set) -> dict:
    """Measure the rule's precision on labelled Stage 1 training rows."""
    masked = train_s1["title_text"].map(mask_cyrillic)
    hit = masked.isin(s2_titles)
    labels = train_s1.loc[hit, "label"]
    return {
        "n_matched": int(hit.sum()),
        "n_special": int((labels == 1).sum()),
        "n_general": int((labels == 0).sum()),
        "precision": float((labels == 1).mean()) if hit.any() else float("nan"),
    }


def apply_stage2_title_rule(
    titles: pd.Series, predictions: np.ndarray, s2_titles: set
) -> tuple:
    """Force Special (1) on any title that also appears in the Stage 2 data."""
    hit = titles.map(mask_cyrillic).isin(s2_titles).to_numpy()
    corrected = predictions.copy()
    corrected[hit] = 1
    return corrected, hit, int((corrected != predictions).sum())
