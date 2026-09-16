"""Choose C for the clean-text Stage 1 model on the nested estimate.

Run:  python src/stage1_clean_select_C.py

The in-fold sweep in ``stage1_clean_eval.py`` tunes its threshold on the same
folds it scores, so it ranks configurations but overstates all of them. C is
chosen here on the nested estimate instead, and every candidate is written out
so the margin between them stays visible.

Selecting on the nested estimate does make the winner's number mildly
optimistic. The spread across the top candidates is reported for that reason:
if several C values land together, the honest read is the group, not the max.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_clean import build_clean_model  # noqa: E402
from stage1_clean_eval import nested_cv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h17"
C_GRID = (1.0, 3.0, 10.0, 30.0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train_stage1.csv")
    X = train["title_text"].to_numpy(dtype=object)
    y = train["label"].to_numpy()

    results = {}
    for C in C_GRID:
        print("[nested CV, C=%.1f]" % C)
        results[str(C)] = nested_cv(lambda C=C: build_clean_model(C=C), X, y)
        print("  pooled M1=%.4f  per-fold %.4f +/- %.4f\n"
              % (results[str(C)]["pooled_m1"], results[str(C)]["mean_fold_m1"],
                 results[str(C)]["std_fold_m1"]))

    best_C = float(max(results, key=lambda k: results[k]["pooled_m1"]))
    pooled = {c: results[c]["pooled_m1"] for c in results}
    top = sorted(pooled.values(), reverse=True)

    summary = {
        "C_grid": list(C_GRID),
        "nested_by_C": results,
        "best_C": best_C,
        "best_pooled_m1": pooled[str(best_C)],
        "threshold_median": results[str(best_C)]["threshold_median"],
        "spread_top_two": top[0] - top[1],
        "selection_note": (
            "C chosen on the nested estimate, which makes the winner mildly "
            "optimistic; the top two are within %.4f M1 of each other."
            % (top[0] - top[1])
        ),
    }
    (OUT / "stage1_clean_C_selection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame({c: {"pooled_m1": results[c]["pooled_m1"],
                            "sd": results[c]["std_fold_m1"],
                            "threshold_median": results[c]["threshold_median"]}
                        for c in results}).T.round(4).to_string())
    print("\nselected C=%.1f  threshold=%.2f  (top two within %.4f)"
          % (best_C, summary["threshold_median"], summary["spread_top_two"]))


if __name__ == "__main__":
    main()
