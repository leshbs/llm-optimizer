"""Candidate F: error taxonomy over cached out-of-fold predictions.

Stage 2 reuses ``h8/variants/oof_g_final.csv`` (config G, the production model)
and Stage 1 reuses ``h14/stage1_invariant_oof_proba.npy``. Nothing is refitted;
the regex detectors below are diagnostic labels for the writeup and are never
handed to a model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_decode import build_decoder, load_all  # noqa: E402
from stage1_invariant import mask_cyrillic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h16"

NEGATION = re.compile(
    r"\b(не|нет|без|отрицает|не\s+выявлен\w*|не\s+отмечает\w*|отсутств\w+|"
    r"не\s+отягощен\w*)\b", re.IGNORECASE
)
AGE_RESTRICTION = re.compile(r"(до\s+\d+\s*лет|старше\s+\d+|у\s+дет\w+|у\s+взрослых|"
                             r"новорожд\w+|подростк\w+|пожил\w+)", re.IGNORECASE)
GENDER_RESTRICTION = re.compile(r"(женщин\w*|мужчин\w*|беременн\w*|кормящ\w*|лактац\w*)",
                                re.IGNORECASE)
SEVERITY = re.compile(r"(тяжёл\w*|тяжел\w*|лёгк\w*|легк\w*|среднетяжёл\w*|степен\w*|стади\w*)",
                      re.IGNORECASE)
FIELD_SEX = re.compile(r"Пол\s*:", re.IGNORECASE)
FIELD_AGE = re.compile(r"Возраст\s*:", re.IGNORECASE)
FIELD_ICD = re.compile(r"(МКБ|Код\s+МКБ)", re.IGNORECASE)


def has(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    """Regex membership test.

    ``Series.str.contains`` silently returns all-False for compiled patterns
    under the pinned pandas version, so the search is applied explicitly.
    """
    return series.apply(lambda s: bool(pattern.search(s)))


def stage2_taxonomy(threshold: float = 0.5) -> dict:
    oof = pd.read_csv(ROOT / "h8" / "variants" / "oof_g_final.csv")
    pred = (oof["oof_proba_g"] >= threshold).astype(int)
    y = oof["label"].to_numpy()
    err = pred.to_numpy() != y

    title = oof["title_text"].astype(str)
    proto = oof["protocol_text"].astype(str)

    ambiguous_ids = set(
        pd.read_csv(ROOT / "h9" / "analysis" / "annotation_ambiguity_audit.csv")["id"]
    )
    s2_guidelines = title.map(lambda t: t.split("\n")[0].strip())
    guideline_counts = s2_guidelines.value_counts()

    flags = pd.DataFrame(
        {
            "id": oof["id"],
            "label": y,
            "pred": pred,
            "proba": oof["oof_proba_g"],
            "error": err,
            "kind": np.where(err & (y == 1), "FN", np.where(err & (y == 0), "FP", "correct")),
            "demographic_restriction": has(title, AGE_RESTRICTION) | has(title, GENDER_RESTRICTION),
            "severity_restriction": has(title, SEVERITY),
            "negation_in_protocol": has(proto, NEGATION),
            "missing_sex_field": ~has(proto, FIELD_SEX),
            "missing_age_field": ~has(proto, FIELD_AGE),
            "missing_icd_field": ~has(proto, FIELD_ICD),
            "annotation_ambiguous": oof["id"].isin(ambiguous_ids),
            "rare_guideline": s2_guidelines.map(guideline_counts) <= 2,
            "protocol_len": proto.str.len(),
        }
    )
    flags["missing_structured_fields"] = (
        flags["missing_sex_field"] | flags["missing_age_field"]
    )
    flags.to_csv(OUT / "stage2_error_flags.csv", index=False)

    cats = [
        "demographic_restriction", "severity_restriction", "negation_in_protocol",
        "missing_structured_fields", "missing_icd_field", "annotation_ambiguous",
        "rare_guideline",
    ]
    rows = []
    for c in cats:
        for kind in ("FN", "FP"):
            sel = flags["kind"] == kind
            rows.append(
                {
                    "category": c,
                    "kind": kind,
                    "n_errors_with_flag": int((sel & flags[c]).sum()),
                    "n_errors": int(sel.sum()),
                    "flag_rate_in_errors": float(flags.loc[sel, c].mean()) if sel.sum() else 0.0,
                    "flag_rate_in_correct": float(flags.loc[flags["kind"] == "correct", c].mean()),
                }
            )
    table = pd.DataFrame(rows)
    table["lift"] = table["flag_rate_in_errors"] / table["flag_rate_in_correct"].replace(0, np.nan)
    table.to_csv(OUT / "stage2_error_taxonomy.csv", index=False)

    return {
        "n": int(len(flags)),
        "threshold": threshold,
        "n_FN": int((flags["kind"] == "FN").sum()),
        "n_FP": int((flags["kind"] == "FP").sum()),
        "error_rate": float(err.mean()),
        "mean_protocol_len_FN": float(flags.loc[flags["kind"] == "FN", "protocol_len"].mean()),
        "mean_protocol_len_correct": float(
            flags.loc[flags["kind"] == "correct", "protocol_len"].mean()
        ),
        "taxonomy": table.round(4).to_dict("records"),
    }


def stage1_taxonomy() -> dict:
    f = load_all()
    train = f["train_s1"]
    decoder = build_decoder(f["test_s1"], f["train_s2"], f["test_s2"])
    y = train["label"].to_numpy()

    proba = np.load(ROOT / "h14" / "stage1_invariant_oof_proba.npy")
    threshold = json.loads(
        (ROOT / "h14" / "stage1_tuning.json").read_text(encoding="utf-8")
    )["nested_cv"]["threshold_median"]
    pred = (proba >= threshold).astype(int)
    err = pred != y

    sub = [mask_cyrillic(str(t).split("\n")[-1].strip()) for t in train["title_text"]]
    gl = [mask_cyrillic(str(t).split("\n")[0].strip()) for t in train["title_text"]]
    flags = pd.DataFrame(
        {
            "label": y,
            "pred": pred,
            "error": err,
            "kind": np.where(err & (y == 1), "FN", np.where(err & (y == 0), "FP", "correct")),
            "decoder_failure": [len(decoder.candidates(m, None)) == 0 for m in sub],
            "guideline_undecodable": [len(decoder.candidates(m, None)) == 0 for m in gl],
            "ambiguous_decode": [len(decoder.candidates(m, None)) > 1 for m in sub],
        }
    )
    flags.to_csv(OUT / "stage1_error_flags.csv", index=False)

    rows = []
    for c in ("decoder_failure", "guideline_undecodable", "ambiguous_decode"):
        for kind in ("FN", "FP"):
            sel = flags["kind"] == kind
            rows.append(
                {
                    "category": c,
                    "kind": kind,
                    "n_errors": int(sel.sum()),
                    "flag_rate_in_errors": float(flags.loc[sel, c].mean()),
                    "flag_rate_in_correct": float(flags.loc[flags["kind"] == "correct", c].mean()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "stage1_error_taxonomy.csv", index=False)
    return {
        "threshold": threshold,
        "n_FN": int((flags["kind"] == "FN").sum()),
        "n_FP": int((flags["kind"] == "FP").sum()),
        "error_rate": float(err.mean()),
        "taxonomy": table.round(4).to_dict("records"),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = {"stage2": stage2_taxonomy(), "stage1_h14_oof": stage1_taxonomy()}
    (OUT / "error_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
