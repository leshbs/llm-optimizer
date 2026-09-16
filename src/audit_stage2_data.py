"""Integrity audit of the Stage 2 data and its loading/inference path.

Analysis only. Looks for the classes of defect that actually bite in this
project: encoding damage like the one that hit train_stage1, silent truncation,
row-order or id misalignment between the test file and the shipped predictions,
duplicate or leaked rows, and structured-field extraction failures.

Run:  python src/audit_stage2_data.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import read_competition_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h18"

FIELDS = {
    "icd": re.compile(r"Код\s*МКБ-10\s*:", re.IGNORECASE),
    "sex": re.compile(r"Пол\s*:", re.IGNORECASE),
    "age": re.compile(r"Возраст\s*:", re.IGNORECASE),
    "date": re.compile(r"Дата\s+приема\s*:", re.IGNORECASE),
    "complaints": re.compile(r"Жалобы\s*:", re.IGNORECASE),
    "anamnesis": re.compile(r"Анамнез\s*:", re.IGNORECASE),
    "status": re.compile(r"Объективный\s+статус\s*:", re.IGNORECASE),
}
MOJIBAKE = re.compile(r"[ÐÑ][\x80-\xbf]|Ã[\x80-\xbf]")


def has(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    # Series.str.contains silently returns all-False for compiled patterns on
    # the pinned pandas; apply the search explicitly.
    return series.apply(lambda s: bool(pattern.search(s)))


def byte_profile(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "replacement_ratio": raw.count(0x3F) / max(len(raw), 1),
        "high_byte_ratio": sum(1 for b in raw if b > 0x7F) / max(len(raw), 1),
        "utf8_clean": _decodes(raw, "utf-8"),
    }


def _decodes(raw: bytes, enc: str) -> bool:
    try:
        raw.decode(enc)
        return True
    except UnicodeDecodeError:
        return False


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = {}

    tr = read_competition_csv(ROOT / "train_stage2.csv")
    te = read_competition_csv(ROOT / "test_stage2.csv")

    # ---- 1. file-level integrity ----------------------------------------- #
    report["byte_profile"] = {
        n: byte_profile(ROOT / n) for n in ("train_stage2.csv", "test_stage2.csv")
    }

    # ---- 2. frame-level integrity ---------------------------------------- #
    def frame_stats(df: pd.DataFrame, name: str) -> dict:
        proto = df["protocol_text"].astype(str)
        title = df["title_text"].astype(str)
        return {
            "shape": list(df.shape),
            "nulls": int(df.isna().sum().sum()),
            "duplicate_ids": int(df["id"].duplicated().sum()),
            "duplicate_pairs": int(df.duplicated(["title_text", "protocol_text"]).sum()),
            "empty_protocol": int((proto.str.strip() == "").sum()),
            "empty_title": int((title.str.strip() == "").sum()),
            "protocol_len_min": int(proto.str.len().min()),
            "protocol_len_median": float(proto.str.len().median()),
            "protocol_len_max": int(proto.str.len().max()),
            "qmark_share_protocol": float(proto.str.count(r"\?").sum() / proto.str.len().sum()),
            "mojibake_rows": int(has(proto, MOJIBAKE).sum()),
            "n_distinct_protocols": int(proto.nunique()),
            "n_distinct_titles": int(title.nunique()),
        }

    report["train"] = frame_stats(tr, "train_stage2.csv")
    report["test"] = frame_stats(te, "test_stage2.csv")

    # ---- 3. structured-field coverage ------------------------------------ #
    cov = {}
    for split, df in (("train", tr), ("test", te)):
        proto = df["protocol_text"].astype(str)
        cov[split] = {k: float(has(proto, p).mean()) for k, p in FIELDS.items()}
    report["field_coverage"] = cov
    report["field_coverage_gap"] = {
        k: abs(cov["train"][k] - cov["test"][k]) for k in FIELDS
    }

    # ---- 4. truncation check --------------------------------------------- #
    # A silently truncated protocol usually ends mid-token rather than on
    # sentence punctuation, and clusters at a round character count.
    def truncation(df: pd.DataFrame) -> dict:
        proto = df["protocol_text"].astype(str)
        lens = proto.str.len()
        common = collections.Counter(lens).most_common(3)
        return {
            "most_common_lengths": [[int(a), int(b)] for a, b in common],
            "share_ending_in_sentence_punct": float(
                proto.str.rstrip().str[-1].isin(list(".!?»\"")).mean()
            ),
            "n_at_exact_power_of_two": int(lens.isin([512, 1024, 2048, 4096, 8192]).sum()),
        }

    report["truncation"] = {"train": truncation(tr), "test": truncation(te)}

    # ---- 5. train/test overlap and leakage ------------------------------- #
    tr_pairs = set(zip(tr["title_text"], tr["protocol_text"]))
    te_pairs = set(zip(te["title_text"], te["protocol_text"]))
    report["overlap"] = {
        "shared_exact_pairs": len(tr_pairs & te_pairs),
        "shared_protocols": len(set(tr["protocol_text"]) & set(te["protocol_text"])),
        "shared_titles": len(set(tr["title_text"]) & set(te["title_text"])),
        "test_titles_unseen_in_train": int(
            (~te["title_text"].isin(set(tr["title_text"]))).sum()
        ),
        "id_collisions_train_test": len(set(tr["id"]) & set(te["id"])),
    }

    # ---- 6. inference-path alignment ------------------------------------- #
    preds = pd.read_csv(ROOT / "final" / "predictions" / "stage2_predictions.csv")
    align = {
        "pred_rows": len(preds),
        "test_rows": len(te),
        "ids_same_order": list(preds["id"]) == list(te["id"]),
        "ids_same_set": set(preds["id"]) == set(te["id"]),
        "labels_binary": bool(preds["label"].isin([0, 1]).all()),
        "positive_rate": float(preds["label"].mean()),
    }
    proba_path = ROOT / "final" / "predictions" / "stage2_probabilities.csv"
    if proba_path.exists():
        proba = pd.read_csv(proba_path)
        col = [c for c in proba.columns if c != "id"][0]
        merged = preds.merge(proba, on="id", suffixes=("", "_p"))
        align["proba_rows"] = len(proba)
        align["proba_ids_same_order"] = list(proba["id"]) == list(te["id"])
        # which threshold reproduces the shipped labels?
        best = None
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            agree = float(((merged[col] >= t).astype(int) == merged["label"]).mean())
            if best is None or agree > best[1]:
                best = (float(t), agree)
        align["threshold_reproducing_labels"] = best[0]
        align["reproduction_agreement"] = best[1]
    report["inference_alignment"] = align

    # ---- 7. submission file ---------------------------------------------- #
    sub = pd.read_csv(ROOT / "final_submission" / "submission.csv")
    s2 = sub[sub["stage"] == 2]
    report["submission"] = {
        "rows": len(sub),
        "stage2_rows": len(s2),
        "stage2_ids_same_order_as_test": list(s2["id"]) == list(te["id"]),
        "stage2_matches_predictions": list(s2["label"]) == list(preds["label"]),
    }

    (OUT / "stage2_data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
