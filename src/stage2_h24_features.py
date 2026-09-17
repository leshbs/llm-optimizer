"""H24 step B: the protocol inputs config G never sees.

The H23 audit found that ``extract_narrative`` keeps a median 18.7% of each
protocol (Жалобы/Анамнез/Объективный статус only). Everything else is dropped,
including two things that carry the label directly:

* ``Код МКБ-10`` -- present in 497/690 train rows and 114/173 test rows. A
  (title, ICD) lookup alone reaches AUC 0.877 against G's 0.888. Example:
  every K50.1 (Crohn's disease of the *large* intestine) row under
  "БК тонкой кишки (кроме терминального илеита)" is labelled 0.
* the ``Диагноз`` line and the treatment/recommendation text.

The bulk of what is dropped is lab boilerplate: repeated
``Показатель: ..., Референсные значения: ..., Результат: ...`` lines, which
make up most of the long protocols and every 32,765-character truncated one.
``clean_protocol`` removes exactly those lines and keeps all prose, taking the
median length from 3,953 to 1,609 characters.

Every function here is row-local (one protocol or one title in, one value
out). No corpus statistics and no labels are used, so the features are
fold-safe by construction. Anything that needs corpus statistics (TF-IDF
vocabularies, one-hot categories) is fitted inside the sklearn pipeline, per
fold.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    build_base_frame,
    lemmatize_ru,
    load_h8_features,
    merge_extra_cols,
)

ROOT = Path(__file__).resolve().parent.parent

ICD_PATTERN = re.compile(r"Код МКБ-10\s*:\s*([A-Z]\d{2}(?:\.\d{1,2})?)")
# Fallback: a protocol without the structured header often still opens its
# diagnosis with the code ("Диагноз: I50.0 Застойная ...").
ICD_IN_DIAGNOSIS = re.compile(r"Диагноз\s*:\s*([A-Z]\d{2}(?:\.\d{1,2})?)")
DIAGNOSIS_FIELDS = ("Диагноз", "Основное заболевание", "Заключение",
                    "Сопутствующие заболевания", "Осложнения")
LAB_LINE = re.compile(r"Показатель:[^\n]*")
BLANK_RUNS = re.compile(r"\s*\n\s*(?:\n\s*)+")
MISSING = "NA"


def icd_code(text: str) -> str:
    """Full ICD-10 code (e.g. ``K50.1``), or ``NA`` when the protocol has none."""
    text = str(text)
    m = ICD_PATTERN.search(text) or ICD_IN_DIAGNOSIS.search(text)
    return m.group(1) if m else MISSING


def icd_category(code: str) -> str:
    """Three-character ICD-10 category (``K50.1`` -> ``K50``)."""
    return code if code == MISSING else code[:3]


def diagnosis_text(text: str) -> str:
    """The diagnosis-bearing fields, joined; empty when the protocol has none."""
    text = str(text)
    chunks = []
    for field in DIAGNOSIS_FIELDS:
        for m in re.finditer(rf"{re.escape(field)}\s*:\s*([^\n]+)", text):
            chunks.append(m.group(1).strip())
    return " ".join(chunks)


def clean_protocol(text: str) -> str:
    """Full protocol minus the lab-value boilerplate lines."""
    return BLANK_RUNS.sub("\n", LAB_LINE.sub("", str(text))).strip()


def guideline_name(title: str) -> str:
    """The guideline a title belongs to (its first line)."""
    return str(title).strip().split("\n")[0]


def build_h24_frame() -> pd.DataFrame:
    """Config G's frame plus the H24 inputs. Row order matches train_stage2.csv."""
    base = build_base_frame()
    frame = merge_extra_cols(base, load_h8_features(base), CONTRADICTION_COLS)
    add_h24_columns(frame)
    return frame


def add_h24_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the H24 columns in place (shared by train and test framing)."""
    protocol = frame["protocol_text"].astype(str)
    frame["icd"] = protocol.map(icd_code)
    frame["icd_cat"] = frame["icd"].map(icd_category)
    frame["guideline"] = frame["title_text"].map(guideline_name)
    # the interaction is a string key; its one-hot categories are fit per fold
    frame["title_x_icd"] = frame["title_text"] + "||" + frame["icd"]
    frame["title_x_icd_cat"] = frame["title_text"] + "||" + frame["icd_cat"]
    frame["diagnosis_lemma"] = protocol.map(diagnosis_text).map(lemmatize_ru)
    # lemmatise each distinct protocol once; protocols repeat across titles
    cleaned = {p: lemmatize_ru(clean_protocol(p)) for p in protocol.unique()}
    frame["full_lemma"] = protocol.map(cleaned)
    frame["has_icd"] = (frame["icd"] != MISSING).astype(int)
    return frame


if __name__ == "__main__":
    f = build_h24_frame()
    print(f[["icd", "icd_cat", "has_icd"]].describe(include="all").to_string())
    print("diagnosis text non-empty:", int((f["diagnosis_lemma"].str.len() > 0).sum()))
    print("full_lemma median tokens:", int(f["full_lemma"].str.split().str.len().median()),
          " narrative median tokens:", int(f["narrative_lemma"].str.split().str.len().median()))
    print(f["icd"].value_counts().head(12).to_string())
