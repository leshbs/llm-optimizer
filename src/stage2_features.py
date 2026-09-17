"""The 11 promoted contradiction features, regenerated from source.

The H8 cache is a parquet file and both parquet engines are blocked by an
Application Control policy on this machine, so the features are rebuilt from the
notebook code (cells 28.0 and 05c53c40) rather than loaded. The regeneration is
verified end to end: feeding these features through the config-G pipeline must
reproduce ``h8/variants/oof_g_final.csv`` to floating-point tolerance.

One hazard carried over from the original: ``condition_contradiction`` iterates
``list(toks)[:5]`` over a *set*, and CPython randomises string hashing per
process, so rows with more than five shared condition tokens can flip between
runs. ``deterministic=True`` sorts the tokens first, which removes the
nondeterminism; ``deterministic=False`` reproduces the original behaviour for
comparison. The audit in ``main`` reports how many rows are exposed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_baseline import NEGATION_PATTERN

ROOT = Path(__file__).resolve().parent.parent

PREGNANCY_MARKERS = ("беремен", "грудно", "лактац", "кормлен")

SEVERITY_TERMS = {
    "легкий": 1, "лёгкий": 1, "легкая": 1, "лёгкая": 1, "легкое": 1,
    "среднетяжелый": 2, "среднетяжёлый": 2, "умеренный": 2, "умеренная": 2,
    "средний": 2, "средняя": 2,
    "тяжелый": 3, "тяжёлый": 3, "тяжелая": 3, "тяжёлая": 3, "тяжелое": 3,
    "выраженный": 3, "выраженная": 3,
}

GENERIC_STOPWORDS_TITLE = {
    "клинический", "рекомендация", "лечение", "особенность", "пациент", "терапия", "взрослый",
    "категория", "возрастной", "консервативный", "хирургический", "медикаментозный", "группа",
    "локализация", "форма", "период", "снизить", "кроме", "прочий", "другой", "определённый",
}

CONTRADICTION_COLS = [
    "age_contradiction", "gender_condition_present", "gender_unknown", "severity_match",
    "severity_contradiction", "severity_unknown", "condition_overlap", "condition_contradiction",
    "condition_unknown", "contradiction_count", "contradiction_density",
]


def extract_age_bound(title: str) -> float:
    m = re.search(r"до\s*(\d{1,3})\s*лет", str(title), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def gender_requirement(title: str):
    lowered = str(title).lower()
    female = ("эндометриоз", "беремен", "контрацептив", " мгт", "ддмж", "молочн",
              "маточ", "яичник", "гинеколог")
    male = ("простат", "мужского пола")
    if any(k in lowered for k in female):
        return "F"
    if any(k in lowered for k in male):
        return "M"
    return None


def is_pregnancy_title(title: str) -> bool:
    lowered = str(title).lower()
    return any(k in lowered for k in PREGNANCY_MARKERS)


def extract_severity_level(text):
    if not isinstance(text, str) or not text:
        return np.nan
    levels = {SEVERITY_TERMS[t] for t in set(text.split()) if t in SEVERITY_TERMS}
    return max(levels) if levels else np.nan


def title_condition_tokens(title_lemma: str) -> set:
    return {t for t in str(title_lemma).split()
            if len(t) > 3 and t not in GENERIC_STOPWORDS_TITLE}


def negation_window_hit(text: str, token: str, window: int = 40) -> bool:
    """True if ``token`` occurs within ``window`` characters of a negation match."""
    for m in NEGATION_PATTERN.finditer(text):
        start, end = max(0, m.start() - window), min(len(text), m.end() + window)
        if token in text[start:end]:
            return True
    return False


def build_contradiction_features(df: pd.DataFrame, deterministic: bool = True) -> pd.DataFrame:
    """Rebuild the 11 promoted features.

    ``df`` must carry ``id``, ``title_text``, ``title_lemma``, ``narrative_lemma``,
    ``narrative_text``, ``age`` and the *raw* ``gender`` (None where unknown --
    not the 'unknown'-filled column the pipeline uses).
    """
    n = len(df)
    out = pd.DataFrame({"id": df["id"].to_numpy()})

    age_bound = df["title_text"].apply(extract_age_bound)
    patient_age = df["age"]
    has_age_rule = age_bound.notna()
    out["age_contradiction"] = (
        has_age_rule & patient_age.notna() & (patient_age > age_bound)
    ).astype(int).to_numpy()
    out["age_compatible"] = (
        has_age_rule & patient_age.notna() & (patient_age <= age_bound)
    ).astype(int).to_numpy()
    out["age_unknown"] = (has_age_rule & patient_age.isna()).astype(int).to_numpy()

    required_gender = df["title_text"].apply(gender_requirement)
    preg = df["title_text"].apply(is_pregnancy_title)
    patient_gender = df["gender"]
    has_gender_rule = required_gender.notna()
    out["gender_condition_present"] = has_gender_rule.astype(int).to_numpy()
    out["gender_contradiction"] = (
        has_gender_rule & patient_gender.notna() & (patient_gender != required_gender)
    ).astype(int).to_numpy()
    out["gender_unknown"] = (
        has_gender_rule & (patient_gender.isna() | preg)
    ).astype(int).to_numpy()

    title_sev = df["title_lemma"].apply(extract_severity_level)
    narr_sev = df["narrative_lemma"].apply(extract_severity_level)
    has_sev_rule = title_sev.notna()
    out["severity_match"] = (
        has_sev_rule & narr_sev.notna() & (title_sev == narr_sev)
    ).astype(int).to_numpy()
    out["severity_contradiction"] = (
        has_sev_rule & narr_sev.notna() & (title_sev != narr_sev)
    ).astype(int).to_numpy()
    out["severity_unknown"] = (has_sev_rule & narr_sev.isna()).astype(int).to_numpy()

    cond_tokens = df["title_lemma"].apply(title_condition_tokens)
    narr_sets = df["narrative_lemma"].apply(lambda t: set(str(t).split()))
    narr_text = df["narrative_text"].astype(str)

    overlap, contra = [], []
    for i in range(n):
        shared = cond_tokens.iloc[i] & narr_sets.iloc[i]
        overlap.append(int(bool(shared)))
        if not shared:
            contra.append(0)
            continue
        probe = sorted(shared)[:5] if deterministic else list(shared)[:5]
        low = narr_text.iloc[i].lower()
        contra.append(int(any(negation_window_hit(low, tok) for tok in probe)))
    out["condition_overlap"] = overlap
    out["condition_contradiction"] = contra
    out["condition_unknown"] = (
        (np.array(overlap) == 0) & (cond_tokens.apply(len).to_numpy() > 0)
    ).astype(int)

    out["contradiction_count"] = (
        out["age_contradiction"] + out["gender_contradiction"]
        + out["severity_contradiction"] + out["condition_contradiction"]
    )
    applicable = (
        has_age_rule.astype(int).to_numpy() + has_gender_rule.astype(int).to_numpy()
        + has_sev_rule.astype(int).to_numpy() + (cond_tokens.apply(len).to_numpy() > 0).astype(int)
    )
    out["contradiction_density"] = (
        out["contradiction_count"] / pd.Series(applicable).replace(0, np.nan)
    ).fillna(0.0).to_numpy()

    return out


def n_rows_exposed_to_set_order(df: pd.DataFrame) -> int:
    """How many rows have more than five shared condition tokens."""
    cond = df["title_lemma"].apply(title_condition_tokens)
    narr = df["narrative_lemma"].apply(lambda t: set(str(t).split()))
    return int(sum(len(cond.iloc[i] & narr.iloc[i]) > 5 for i in range(len(df))))
