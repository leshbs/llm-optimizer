"""Phase 1: the reusable Stage 2 error-analysis table.

Builds one row per out-of-fold Stage 2 prediction with everything later phases
need to categorise an error, then writes ``h19/stage2_error_analysis.csv``.

IMPORTANT -- the recurrence columns here are computed over the *whole* training
set on purpose. They are diagnostic, for reading the error structure, and are
never fed to a model. Any recurrence feature that reaches a model in H21 must be
recomputed fold-safely (fit on the training fold, applied to the validation
fold); the columns in this file are explicitly suffixed ``_analysis_only`` so
they cannot be used by accident.

Run:  python src/stage2_error_analysis.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    NEGATION_PATTERN,
    build_base_frame,
    evaluate,
    load_folds,
    load_h8_features,
    merge_extra_cols,
)
from stage2_features import title_condition_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h19"

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def condition_hits(protocol_low: str, tokens: list) -> list:
    """Character offsets where any condition token appears literally."""
    hits = []
    for tok in tokens:
        start = protocol_low.find(tok)
        while start != -1:
            hits.append((start, tok))
            start = protocol_low.find(tok, start + 1)
    return hits


def negation_spans(protocol_low: str) -> list:
    return [(m.start(), m.end()) for m in NEGATION_PATTERN.finditer(protocol_low)]


def row_diagnostics(title_lemma: str, protocol: str) -> dict:
    """Condition/negation geometry for one (title, protocol) pair."""
    low = str(protocol).lower()
    tokens = sorted(title_condition_tokens(title_lemma))
    hits = condition_hits(low, tokens)
    negs = negation_spans(low)

    matched = sorted({t for _, t in hits})
    out = {
        "condition_tokens": "|".join(tokens),
        "n_condition_tokens": len(tokens),
        "condition_present": int(bool(hits)),
        "condition_present_count": len(hits),
        "condition_tokens_matched": "|".join(matched),
        "n_condition_tokens_matched": len(matched),
        "negation_count_total": len(negs),
    }

    if not hits or not negs:
        out.update({
            "nearest_negation_distance": np.nan,
            "neg_before_condition": 0,
            "neg_after_condition": 0,
            "neg_within_5": 0, "neg_within_10": 0,
            "neg_within_20": 0, "neg_within_50": 0,
            "condition_negated_exact": 0,
        })
        return out

    best = np.inf
    before = after = 0
    for pos, tok in hits:
        end = pos + len(tok)
        for ns, ne in negs:
            if ne <= pos:
                d = pos - ne
                if d < best:
                    best = d
                before = 1 if d <= 50 or before else before
            elif ns >= end:
                d = ns - end
                if d < best:
                    best = d
                after = 1 if d <= 50 or after else after
            else:  # overlapping
                best = 0
                before = 1
    out.update({
        "nearest_negation_distance": float(best),
        "neg_before_condition": int(before),
        "neg_after_condition": int(after),
        "neg_within_5": int(best <= 5),
        "neg_within_10": int(best <= 10),
        "neg_within_20": int(best <= 20),
        "neg_within_50": int(best <= 50),
        "condition_negated_exact": int(best <= 40),   # the H8 window, for continuity
    })
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]
    X = merge_extra_cols(base, h8, CONTRADICTION_COLS)

    res = evaluate(X, y, folds, extra_numeric_cols=CONTRADICTION_COLS)
    proba, pred = res["oof_proba"], res["oof_pred"]
    y_arr = y.to_numpy()

    fold_of = np.full(len(y_arr), -1)
    for k, (_, val) in enumerate(folds):
        fold_of[val] = k

    diag = pd.DataFrame(
        [row_diagnostics(t, p) for t, p in zip(base["title_lemma"], base["protocol_text"])]
    )

    protocol = base["protocol_text"].astype(str)
    title = base["title_text"].astype(str)

    # analysis-only recurrence (whole-corpus counts -- never a model feature)
    pcount = protocol.map(protocol.value_counts())
    tcount = title.map(title.value_counts())
    pair = title + "||" + protocol
    paircount = pair.map(pair.value_counts())

    out = pd.DataFrame({
        "id": base["id"],
        "fold": fold_of,
        "label": y_arr,
        "proba": proba,
        "pred": pred,
        "outcome": np.where(
            (y_arr == 1) & (pred == 1), "TP",
            np.where((y_arr == 0) & (pred == 0), "TN",
                     np.where((y_arr == 0) & (pred == 1), "FP", "FN"))),
        "protocol_len_chars": protocol.str.len(),
        "protocol_n_tokens": protocol.apply(lambda s: len(_WORD.findall(s))),
        "protocol_truncated": (protocol.str.len() == 32765).astype(int),
        "title_len_chars": title.str.len(),
        "title_n_tokens": title.apply(lambda s: len(_WORD.findall(s))),
        "title_text": title,
        "protocol_head": protocol.str[:300],
        "has_structured_header": base["has_structured_header"],
        "age": base["age"],
        "gender": base["gender"],
        "protocol_recurrence_analysis_only": pcount.to_numpy(),
        "title_recurrence_analysis_only": tcount.to_numpy(),
        "pair_recurrence_analysis_only": paircount.to_numpy(),
    })
    out = pd.concat([out, diag], axis=1)
    for c in CONTRADICTION_COLS:
        out[c] = X[c].to_numpy()

    out.to_csv(OUT / "stage2_error_analysis.csv", index=False, encoding="utf-8")

    # ---- summary used to write error_analysis.md ------------------------- #
    fn = out[out["outcome"] == "FN"]
    fp = out[out["outcome"] == "FP"]
    corr = out[out["outcome"].isin(["TP", "TN"])]

    def block(df: pd.DataFrame) -> dict:
        return {
            "n": len(df),
            "condition_present_rate": float(df["condition_present"].mean()),
            "condition_negated_exact_rate": float(df["condition_negated_exact"].mean()),
            "neg_within_20_rate": float(df["neg_within_20"].mean()),
            "median_nearest_negation_distance": float(df["nearest_negation_distance"].median()),
            "mean_protocol_len": float(df["protocol_len_chars"].mean()),
            "mean_title_len": float(df["title_len_chars"].mean()),
            "mean_n_condition_tokens": float(df["n_condition_tokens"].mean()),
            "mean_matched_tokens": float(df["n_condition_tokens_matched"].mean()),
            "truncated_rate": float(df["protocol_truncated"].mean()),
            "mean_protocol_recurrence": float(df["protocol_recurrence_analysis_only"].mean()),
            "mean_title_recurrence": float(df["title_recurrence_analysis_only"].mean()),
            "median_proba": float(df["proba"].median()),
        }

    summary = {
        "baseline": {k: res[k] for k in
                     ("m2", "macro_f2", "per_fold_macro_f2", "mean_fold", "std_fold",
                      "fn", "fp", "positive_rate", "threshold")},
        "FN": block(fn), "FP": block(fp), "correct": block(corr),
        "fn_proba_deciles": [float(v) for v in np.quantile(fn["proba"], np.arange(0, 1.01, 0.1))],
        "fn_proba_above_040": int((fn["proba"] >= 0.40).sum()),
        "fn_proba_below_020": int((fn["proba"] < 0.20).sum()),
        "fn_by_fold": fn["fold"].value_counts().sort_index().to_dict(),
        "fp_by_fold": fp["fold"].value_counts().sort_index().to_dict(),
    }
    (OUT / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("baseline M2=%.6f  FN=%d  FP=%d  posrate=%.4f"
          % (res["m2"], res["fn"], res["fp"], res["positive_rate"]))
    print()
    print(pd.DataFrame({"FN": summary["FN"], "FP": summary["FP"],
                        "correct": summary["correct"]}).round(4).to_string())
    print()
    print("FN probability deciles:", [round(v, 3) for v in summary["fn_proba_deciles"]])
    print("FN with proba >= 0.40: %d / %d" % (summary["fn_proba_above_040"], len(fn)))
    print("FN with proba <  0.20: %d / %d" % (summary["fn_proba_below_020"], len(fn)))
    print("\nwrote h19/stage2_error_analysis.csv and h19/error_analysis_summary.json")


if __name__ == "__main__":
    main()
