"""H24 step D: does a zero-shot LLM probability add anything over C0?

The feature comes from ``colab/stage2_llm_feature.ipynb``, which must be run
on a GPU; its two output files go in ``h24/llm/``:

    llm_feature_train.csv   id, p_llm, logit_llm     (690 rows, train order)
    llm_feature_test.csv    id, p_llm, logit_llm     (173 rows, test order)
    llm_manifest.json       model id and revision, prompt sha256, versions

**Fold safety.** The LLM is zero-shot: no training label enters any prompt, so
``logit_llm`` is a row-local feature like age or ICD code. It can be merged
before cross-validation without leaking.

The candidates are tested against **C0** (the promoted model), because the
question is whether the LLM adds anything on top of it. The same gate is used,
on both seed sets (0-4, and the fresh 5-9). A candidate submission is written
only if it passes on **both**.

Run:  python src/stage2_h24_step_d_eval.py [--llm-dir h24/llm]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import stage2_score  # noqa: E402
from stage2_h24_confirm_build import (  # noqa: E402
    PROD_SUBMISSION, PROD_SUBMISSION_SHA, build_test_frame, sha256,
)
from stage2_h24_features import build_h24_frame  # noqa: E402
from stage2_h24_harness import (  # noqa: E402
    OUT, THRESHOLD, _within_title_auc, collapsed_titles, oof_matrix, print_row, repeated_splits,
    save_rows, summarise,
)
from stage2_h24_promote_c0 import CANDIDATE as C0_CANDIDATE  # noqa: E402
from stage2_h24_step_b import build_lr  # noqa: E402
from stage2_h24_step_c import C0_KW  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED_SETS = {"s0": (0, 1, 2, 3, 4), "s5": (5, 6, 7, 8, 9)}
CANDIDATES = {
    "D1 C0+LLM": {**C0_KW, "extra_num": ("logit_llm",)},
    "D2 G+LLM": {"extra_num": ("logit_llm",)},
}


def attach_llm(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    llm = pd.read_csv(path)
    assert len(llm) == len(frame) and (llm["id"].to_numpy() == frame["id"].to_numpy()).all(), \
        f"{path.name}: ids missing or out of order"
    assert llm["logit_llm"].notna().all(), f"{path.name}: missing LLM scores"
    frame = frame.copy()
    frame["p_llm"] = llm["p_llm"].to_numpy()
    frame["logit_llm"] = llm["logit_llm"].to_numpy()
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-dir", default=str(OUT / "llm"))
    ap.add_argument("--tag", default="llm", help="cache/output prefix (use another for dry runs)")
    args = ap.parse_args()
    llm_dir = Path(args.llm_dir)

    X = attach_llm(build_h24_frame(), llm_dir / "llm_feature_train.csv")
    y = X["label"].to_numpy()
    titles = X["title_text"].to_numpy()

    # ---- the LLM on its own (labels used for reporting only) ------------ #
    p = X["p_llm"].to_numpy()
    alone = {
        "auc": float(roc_auc_score(y, p)),
        "m2_at_0.5": stage2_score(y, (p >= 0.5).astype(int)),
        "m2_best_in_sample": max(stage2_score(y, (p >= t).astype(int)) for t in np.arange(0.05, 0.96, 0.01)),
        "collapsed_within_auc": _within_title_auc(y, p, collapsed_titles(), titles),
        "positive_rate_at_0.5": float((p >= 0.5).mean()),
    }
    print("LLM alone:", json.dumps(alone, indent=1))

    rows, passes = [], {}
    for set_name, seeds in SEED_SETS.items():
        splits = repeated_splits(y, seeds=seeds)
        prefix = "" if set_name == "s0" else "fresh_"
        g = oof_matrix(f"{prefix}G", build_lr(), X, y, splits)
        c0 = oof_matrix(f"{prefix}C0", build_lr(**C0_KW), X, y, splits)
        for name, kw in CANDIDATES.items():
            key = f"{args.tag}_{set_name}_{name.split()[0]}"
            proba = oof_matrix(key, build_lr(**kw), X, y, splits)
            r_c0 = summarise(f"{name} vs C0 [{set_name}]", proba, y, splits, titles, base_proba=c0)
            r_g = summarise(f"{name} vs G [{set_name}]", proba, y, splits, titles, base_proba=g)
            rows += [r_c0, r_g]
            print_row(r_c0)
            print_row(r_g)
            passes.setdefault(name, []).append(r_c0["passes_gate"])
    save_rows(rows, f"step_d_{args.tag}_results")

    winners = [n for n, v in passes.items() if n.startswith("D1") and all(v)]
    report = {"llm_alone": alone, "passes_vs_C0": passes, "winner": winners[0] if winners else None}
    manifest_path = llm_dir / "llm_manifest.json"
    if manifest_path.exists():
        report["llm_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if winners and args.tag == "llm":
        test = attach_llm(build_test_frame(), llm_dir / "llm_feature_test.csv")
        p_test = build_lr(**CANDIDATES[winners[0]]).fit(X, y).predict_proba(test)[:, 1]
        c0_sub = pd.read_csv(C0_CANDIDATE)
        cand = c0_sub.copy()
        cand.loc[cand["stage"] == 2, "label"] = (p_test >= THRESHOLD).astype(int)
        path = OUT / "submission_candidate_h24_d1.csv"
        cand.to_csv(path, index=False)
        c0_s2 = c0_sub.loc[c0_sub["stage"] == 2, "label"].to_numpy()
        d1_s2 = (p_test >= THRESHOLD).astype(int)
        report.update({
            "candidate": str(path.relative_to(ROOT)), "candidate_sha256": sha256(path),
            "flips_vs_C0": {"0_to_1": int(((c0_s2 == 0) & (d1_s2 == 1)).sum()),
                            "1_to_0": int(((c0_s2 == 1) & (d1_s2 == 0)).sum())},
            "stage2_positive_rate": float(d1_s2.mean()),
        })
    assert sha256(PROD_SUBMISSION) == PROD_SUBMISSION_SHA, "production submission changed"
    (OUT / f"step_d_{args.tag}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\npasses vs C0 on both seed sets:", winners or "NONE")
    if "candidate" in report:
        print("candidate written:", report["candidate"], report["flips_vs_C0"])


if __name__ == "__main__":
    main()
