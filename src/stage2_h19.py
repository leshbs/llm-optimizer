"""Phase 2 (H19): scoped negation around the title's condition.

The hypothesis is that negation *attached to the title's condition* explains a
meaningful share of the 81 false negatives, in a way that document-level
negation (which fires on 97.5% of protocols) cannot.

Phase 1 already measured the relevant contrast and it is not encouraging -- FN
and FP have near-identical scoped-negation profiles -- but the ablation is run
in full because a measured negative result is worth more than an assumption.

Every feature here is **row-local**: it is computed from one (title, protocol)
pair alone, with no corpus statistics, no label information and no cross-row
lookup. Fold safety is therefore structural rather than something that has to be
arranged, and there is nothing for a training fold to leak into a validation
fold.

Run:  python src/stage2_h19.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import (  # noqa: E402
    CONTRADICTION_COLS,
    build_base_frame,
    evaluate,
    load_folds,
    load_h8_features,
    merge_extra_cols,
)
from stage2_error_analysis import row_diagnostics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h19"

PRESENCE_COLS = [
    "condition_present", "condition_present_count",
    "n_condition_tokens", "n_condition_tokens_matched",
]
SCOPED_NEG_COLS = [
    "negation_count_total", "nearest_negation_distance",
    "neg_before_condition", "neg_after_condition",
    "neg_within_5", "neg_within_10", "neg_within_20", "neg_within_50",
    "condition_negated_exact",
]
TERNARY_COLS = ["condition_affirmed", "condition_negated", "condition_unknown_h19"]


def build_h19_features(base: pd.DataFrame) -> pd.DataFrame:
    rows = [row_diagnostics(t, p) for t, p in zip(base["title_lemma"], base["protocol_text"])]
    df = pd.DataFrame(rows)
    df.insert(0, "id", base["id"].to_numpy())
    df = df.drop(columns=["condition_tokens", "condition_tokens_matched"])

    # A three-state read of the condition, which is what the competition brief's
    # "unknown is not contradiction" principle actually asks for.
    present = df["condition_present"] == 1
    negated = df["condition_negated_exact"] == 1
    df["condition_affirmed"] = (present & ~negated).astype(int)
    df["condition_negated"] = (present & negated).astype(int)
    df["condition_unknown_h19"] = (~present).astype(int)

    # median-impute the distance so the column is usable where nothing matched
    df["nearest_negation_distance"] = df["nearest_negation_distance"].fillna(
        df["nearest_negation_distance"].median()
    )
    return df


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = build_base_frame()
    h8 = load_h8_features(base)
    folds = load_folds()
    y = base["label"]

    h19 = build_h19_features(base)
    h19.to_csv(OUT / "h19_features.csv", index=False)

    merged = merge_extra_cols(base, h8, CONTRADICTION_COLS)
    merged = merge_extra_cols(merged, h19, [c for c in h19.columns if c != "id"])

    experiments = {
        "A. baseline (config G)": list(CONTRADICTION_COLS),
        "B. + condition presence": list(CONTRADICTION_COLS) + PRESENCE_COLS,
        "C. + scoped negation": list(CONTRADICTION_COLS) + SCOPED_NEG_COLS,
        "D. + presence + scoped neg": list(CONTRADICTION_COLS) + PRESENCE_COLS + SCOPED_NEG_COLS,
        "E. + all H19 features": (list(CONTRADICTION_COLS) + PRESENCE_COLS
                                  + SCOPED_NEG_COLS + TERNARY_COLS),
    }

    baseline = None
    rows = []
    for name, cols in experiments.items():
        res = evaluate(merged, y, folds, extra_numeric_cols=cols)
        if baseline is None:
            baseline = res
        rows.append({
            "experiment": name,
            "n_extra_features": len(cols),
            "cv_m2": res["m2"],
            "delta_vs_baseline": res["m2"] - baseline["m2"],
            "macro_f2": res["macro_f2"],
            "mean_fold": res["mean_fold"],
            "std_fold": res["std_fold"],
            "fn": res["fn"],
            "fp": res["fp"],
            "positive_rate": res["positive_rate"],
            "fold_scores": [round(v, 4) for v in res["per_fold_macro_f2"]],
        })
        print("%-28s M2=%.6f  d=%+.6f  FN=%3d FP=%3d  pos=%.4f  folds=%s"
              % (name, res["m2"], rows[-1]["delta_vs_baseline"], res["fn"], res["fp"],
                 res["positive_rate"], rows[-1]["fold_scores"]))
        np.save(OUT / ("h19_oof_%s.npy" % name.split(".")[0]), res["oof_proba"])

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "h19_ablation.csv", index=False)

    # Per-fold win/loss against the baseline -- "reproducible across folds" is
    # the roadmap's gate for keeping H19 at all.
    base_folds = np.array(rows[0]["fold_scores"])
    verdict = {}
    for r in rows[1:]:
        wins = int((np.array(r["fold_scores"]) > base_folds).sum())
        verdict[r["experiment"]] = {
            "delta_m2": r["delta_vs_baseline"],
            "folds_improved": wins,
            "folds_total": len(base_folds),
            "fn_change": r["fn"] - rows[0]["fn"],
            "fp_change": r["fp"] - rows[0]["fp"],
            "keep": bool(r["delta_vs_baseline"] > 0 and wins >= 4),
        }

    (OUT / "h19_results.json").write_text(
        json.dumps({"ablation": rows, "verdict": verdict}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("\nper-fold verdict vs baseline")
    print(pd.DataFrame(verdict).T.to_string())
    kept = [k for k, v in verdict.items() if v["keep"]]
    print("\nH19 variants meeting the gate (delta > 0 AND >= 4/5 folds improved): %s"
          % (kept if kept else "NONE"))
    print("wrote h19/h19_ablation.csv and h19/h19_results.json")


if __name__ == "__main__":
    main()
