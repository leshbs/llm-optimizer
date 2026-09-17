"""Phase 3 (H20): categorise every false negative.

Categories are assigned in priority order so each FN lands in exactly one, and
they are adapted to what Phase 1 actually found rather than to the generic list
in the roadmap -- the dominant structure here is per-title prior collapse, which
the generic taxonomy has no slot for.

Run:  python src/stage2_h20_fn_categories.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "h19"
OUT = ROOT / "h20"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(IN / "stage2_error_analysis.csv")

    # titles in which the model misses *every* positive
    pos = d[d["label"] == 1].groupby("title_text").agg(
        npos=("label", "size"), miss=("pred", lambda x: (x == 0).sum()))
    collapsed_titles = set(pos[pos["miss"] == pos["npos"]].index)

    # titles where the model is below threshold on average
    title_mean_p = d.groupby("title_text")["proba"].mean()
    low_titles = set(title_mean_p[title_mean_p < 0.5].index)

    def categorise(r) -> str:
        if r["title_text"] in collapsed_titles:
            return "title_prior_collapse_total"
        if r["title_text"] in low_titles:
            return "title_prior_low_partial"
        if r["condition_present"] == 0:
            return "condition_absent"
        if r["condition_negated_exact"] == 1:
            return "condition_negated"
        return "condition_affirmed_but_missed"

    fn = d[d["outcome"] == "FN"].copy()
    fn["category"] = fn.apply(categorise, axis=1)

    # did H19's best FN-oriented variant (C, scoped negation) fix it?
    proba_c = np.load(IN / "h19_oof_C.npy")
    fixed_by_h19 = (proba_c[fn.index.to_numpy()] >= 0.5).astype(int)
    fn["fixed_by_h19_C"] = fixed_by_h19

    rows = []
    for cat, sub in fn.groupby("category"):
        rows.append({
            "category": cat,
            "count": len(sub),
            "pct_of_all_fn": 100 * len(sub) / len(fn),
            "proba_median": float(sub["proba"].median()),
            "proba_p10": float(sub["proba"].quantile(0.1)),
            "proba_p90": float(sub["proba"].quantile(0.9)),
            "mean_protocol_len": float(sub["protocol_len_chars"].mean()),
            "mean_title_len": float(sub["title_len_chars"].mean()),
            "condition_present_rate": float(sub["condition_present"].mean()),
            "scoped_negation_rate": float(sub["condition_negated_exact"].mean()),
            "n_fixed_by_h19_C": int(sub["fixed_by_h19_C"].sum()),
        })
    table = pd.DataFrame(rows).sort_values("count", ascending=False)
    table.to_csv(OUT / "fn_categories.csv", index=False)
    fn.to_csv(OUT / "fn_rows_categorised.csv", index=False)

    print(table.round(4).to_string(index=False))
    top = table.iloc[0]
    print("\nlargest actionable concentration: %s -- %d FN (%.1f%%), H19 fixes %d of them"
          % (top["category"], top["count"], top["pct_of_all_fn"], top["n_fixed_by_h19_C"]))
    print("H19 variant C fixes %d of %d FN in total" % (int(fn["fixed_by_h19_C"].sum()), len(fn)))

    (OUT / "fn_categories_summary.json").write_text(
        json.dumps({"categories": rows,
                    "n_fn": len(fn),
                    "n_collapsed_titles": len(collapsed_titles),
                    "total_fixed_by_h19_C": int(fn["fixed_by_h19_C"].sum())},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote h20/fn_categories.csv")


if __name__ == "__main__":
    main()
