"""Candidates G and H: fold disagreement and guideline generalization.

Reuses cached fold scores and the deterministic fold assignment (the splitter is
seeded, so the same folds regenerate without touching a model). Nothing is
trained here.

The guideline group key is the *masked* first line of the title. Masking is
deterministic and the guideline line is constant within a guideline, so the
masked form is an exact group identifier on both splits even though the train
side is corrupted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_decode import build_decoder, load_all  # noqa: E402
from stage1_invariant import SEED, mask_cyrillic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h16"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    f = load_all()
    train, test = f["train_s1"], f["test_s1"]
    decoder = build_decoder(f["test_s1"], f["train_s2"], f["test_s2"])
    y = train["label"].to_numpy()

    guideline = np.array([mask_cyrillic(str(t).split("\n")[0].strip()) for t in train["title_text"]])
    sub_masked = [mask_cyrillic(str(t).split("\n")[-1].strip()) for t in train["title_text"]]
    matched = np.array([len(decoder.candidates(m, None)) > 0 for m in sub_masked])
    depth = np.array([str(t).split("\n")[-1].strip().count("#") for t in train["title_text"]])
    title_len = np.array([len(str(t)) for t in train["title_text"]])

    h14 = json.loads((ROOT / "h14" / "stage1_tuning.json").read_text(encoding="utf-8"))["nested_cv"]
    h15 = json.loads((ROOT / "h15" / "stage1_hybrid_eval.json").read_text(encoding="utf-8"))["nested_cv"]

    report = {}

    # ---- per-fold composition -------------------------------------------- #
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        seen = set(guideline[tr])
        rows.append(
            {
                "fold": k + 1,
                "n": len(te),
                "special_rate": float(y[te].mean()),
                "decoder_coverage": float(matched[te].mean()),
                "n_guidelines": int(len(set(guideline[te]))),
                "pct_rows_guideline_seen_in_train_part": float(
                    np.mean([g in seen for g in guideline[te]])
                ),
                "mean_depth": float(depth[te].mean()),
                "mean_title_len": float(title_len[te].mean()),
                "h14_m1": h14["fold_scores"][k],
                "h15_m1": h15["fold_scores"][k],
                "delta": h15["fold_scores"][k] - h14["fold_scores"][k],
                "h14_threshold": h14["thresholds"][k],
                "h15_threshold": h15["thresholds"][k],
            }
        )
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "fold_composition.csv", index=False)
    report["folds"] = folds.to_dict("records")

    loser = folds.loc[folds["delta"].idxmin()]
    report["worst_fold_for_h15"] = int(loser["fold"])
    report["worst_fold_deltas_vs_mean"] = {
        c: float(loser[c] - folds[c].mean())
        for c in ("special_rate", "decoder_coverage", "mean_depth", "mean_title_len",
                  "pct_rows_guideline_seen_in_train_part")
    }

    # ---- guideline generalization ---------------------------------------- #
    gl_counts = pd.Series(guideline).value_counts()
    report["guideline_structure"] = {
        "n_guidelines_train": int(len(gl_counts)),
        "median_rows_per_guideline": float(gl_counts.median()),
        "max_rows_per_guideline": int(gl_counts.max()),
        "pct_rows_in_guidelines_with_1_row": float((gl_counts[gl_counts == 1].sum()) / len(y)),
    }
    report["stratified_cv_guideline_overlap"] = float(
        folds["pct_rows_guideline_seen_in_train_part"].mean()
    )

    # Under GroupKFold the overlap is 0 by construction; what matters is how
    # much of the signal currently rides on within-guideline information.
    test_gl = set(mask_cyrillic(str(t).split("\n")[0].strip()) for t in test["title_text"])
    report["train_test_guideline_overlap"] = {
        "n_train_guidelines": int(len(set(guideline))),
        "n_test_guidelines": int(len(test_gl)),
        "pct_test_guidelines_seen_in_train": float(
            len(test_gl & set(guideline)) / max(len(test_gl), 1)
        ),
    }

    # ---- decoder dependence on same-guideline corpus sources -------------- #
    # If a subsection only decodes because another row of the *same* guideline
    # sits in the corpus, that decode disappears under a guideline holdout.
    src_guideline = {}
    for name, frame in (("s1_test", test), ("s2_train", f["train_s2"]), ("s2_test", f["test_s2"])):
        for i, t in enumerate(frame["title_text"]):
            src_guideline[(name, i)] = mask_cyrillic(str(t).split("\n")[0].strip())

    same, cross, total = 0, 0, 0
    for g, m in zip(guideline, sub_masked):
        entry = decoder._index.get(m)
        if not entry:
            continue
        total += 1
        keys = set().union(*entry.values())
        if any(src_guideline.get(k) == g for k in keys):
            same += 1
        if any(src_guideline.get(k) != g for k in keys):
            cross += 1
    report["decoder_source_dependence"] = {
        "n_matched_rows": total,
        "pct_supported_by_same_guideline": same / max(total, 1),
        "pct_supported_by_other_guideline": cross / max(total, 1),
        "pct_supported_only_by_same_guideline": (total - cross) / max(total, 1),
    }

    (OUT / "fold_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(folds.round(4).to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
