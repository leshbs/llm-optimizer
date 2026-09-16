"""Candidates C, D and E: interactions, threshold headroom, ensembling.

All three are assessed from cached artefacts. Nothing is refitted.

C is approached through interaction information: a linear model can only add
feature contributions, so if I(Y; A, B) exceeds I(Y; A) + I(Y; B) there is
synergy an explicit cross term could capture, and if it does not there is
nothing for one to find.

D is bounded by an oracle: the best achievable per-fold threshold is an upper
limit no selection rule can beat, so the gap between the shipped rule and that
oracle is the most threshold work could ever be worth.
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
from stage1_eval import stage1_score  # noqa: E402
from stage1_invariant import SEED, mask_cyrillic, parse_title  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h16"


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi(groups, y):
    base = entropy(y)
    return base - sum((groups == g).mean() * entropy(y[groups == g]) for g in np.unique(groups))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    f = load_all()
    train = f["train_s1"]
    y = train["label"].to_numpy()
    decoder = build_decoder(f["test_s1"], f["train_s2"], f["test_s2"])
    report = {}

    # ================= C: interaction information ========================= #
    parts = [parse_title(mask_cyrillic(t)) for t in train["title_text"]]
    sub_masked = [mask_cyrillic(str(t).split("\n")[-1].strip()) for t in train["title_text"]]
    n_cand = np.array([len(decoder.candidates(m, None)) for m in sub_masked])

    feats = {
        "decode_status": np.where(n_cand == 0, 0, np.where(n_cand == 1, 1, 2)),
        "depth": np.clip([p.depth for p in parts], 0, 4),
        "age_category": pd.factorize(pd.Series([p.age_category for p in parts]))[0],
        "n_words_bucket": np.clip([len(p.body.split()) // 3 for p in parts], 0, 5),
        "has_digit": np.array([int(any(c.isdigit() for c in p.body)) for p in parts]),
    }
    names = list(feats)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = feats[names[i]], feats[names[j]]
            joint = a * (b.max() + 1) + b
            i_a, i_b, i_ab = mi(a, y), mi(b, y), mi(joint, y)
            rows.append(
                {
                    "pair": "%s x %s" % (names[i], names[j]),
                    "mi_a": i_a,
                    "mi_b": i_b,
                    "mi_joint": i_ab,
                    "synergy_bits": i_ab - i_a - i_b,
                    "synergy_pct_of_entropy": (i_ab - i_a - i_b) / entropy(y),
                }
            )
    inter = pd.DataFrame(rows).sort_values("synergy_bits", ascending=False)
    inter.to_csv(OUT / "interaction_audit.csv", index=False)
    report["interactions"] = {
        "label_entropy_bits": entropy(y),
        "max_synergy_bits": float(inter["synergy_bits"].max()),
        "max_synergy_pair": inter.iloc[0]["pair"],
        "n_pairs_with_positive_synergy": int((inter["synergy_bits"] > 0).sum()),
        "n_pairs": len(inter),
        "table": inter.round(4).to_dict("records"),
    }

    # ================= D: Stage 1 threshold headroom ====================== #
    proba = np.load(ROOT / "h14" / "stage1_invariant_oof_proba.npy")
    tuning = json.loads((ROOT / "h14" / "stage1_tuning.json").read_text(encoding="utf-8"))
    shipped_t = tuning["nested_cv"]["threshold_median"]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    curve = [
        {"threshold": float(t), "m1": stage1_score(y, (proba >= t).astype(int))} for t in grid
    ]
    pd.DataFrame(curve).to_csv(OUT / "stage1_threshold_curve.csv", index=False)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    for k, (_, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        scores = [(float(t), stage1_score(y[te], (proba[te] >= t).astype(int))) for t in grid]
        best_t, best_s = max(scores, key=lambda kv: kv[1])
        fold_rows.append(
            {
                "fold": k + 1,
                "oracle_threshold": best_t,
                "m1_at_oracle": best_s,
                "m1_at_shipped": stage1_score(y[te], (proba[te] >= shipped_t).astype(int)),
            }
        )
    fd = pd.DataFrame(fold_rows)
    fd["gap"] = fd["m1_at_oracle"] - fd["m1_at_shipped"]
    fd.to_csv(OUT / "stage1_threshold_folds.csv", index=False)

    best_global = max(curve, key=lambda r: r["m1"])
    report["threshold"] = {
        "shipped_threshold": shipped_t,
        "m1_at_shipped_pooled": stage1_score(y, (proba >= shipped_t).astype(int)),
        "best_global_threshold": best_global["threshold"],
        "m1_at_best_global": best_global["m1"],
        "global_gain_if_oracle": best_global["m1"]
        - stage1_score(y, (proba >= shipped_t).astype(int)),
        "oracle_per_fold_thresholds": fd["oracle_threshold"].tolist(),
        "oracle_threshold_spread": float(fd["oracle_threshold"].max() - fd["oracle_threshold"].min()),
        "mean_per_fold_oracle_gap": float(fd["gap"].mean()),
        "plateau_width_within_0.01_of_best": int(
            sum(1 for r in curve if r["m1"] >= best_global["m1"] - 0.01)
        ),
        "folds": fd.round(4).to_dict("records"),
    }

    # ================= E: ensemble headroom =============================== #
    h14p = pd.read_csv(ROOT / "h14" / "stage1_rebuild_predictions.csv").set_index("id")
    h15p = pd.read_csv(ROOT / "h15" / "stage1_hybrid_predictions.csv").set_index("id")
    joined = h14p.join(h15p, lsuffix="_h14", rsuffix="_h15")
    disagree = joined["label_h14"] != joined["label_h15"]
    n14 = tuning["nested_cv"]["fold_scores"]
    n15 = json.loads(
        (ROOT / "h15" / "stage1_hybrid_eval.json").read_text(encoding="utf-8")
    )["nested_cv"]["fold_scores"]
    report["ensemble"] = {
        "test_disagreement_rate": float(disagree.mean()),
        "n_test_rows_disagreeing": int(disagree.sum()),
        "mean_abs_proba_gap_on_disagreements": float(
            (joined.loc[disagree, "proba_h14"] - joined.loc[disagree, "proba_h15"]).abs().mean()
        ),
        "fold_score_correlation": float(np.corrcoef(n14, n15)[0, 1]),
        "h15_wins_folds": int(sum(b > a for a, b in zip(n14, n15))),
        "h15_oof_proba_cached": (ROOT / "h15" / "stage1_hybrid_oof_proba.npy").exists(),
    }

    (OUT / "interaction_threshold_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("== C: interaction information ==")
    print(inter.round(4).to_string(index=False))
    print("\n== D: threshold ==")
    print(json.dumps({k: v for k, v in report["threshold"].items() if k != "folds"}, indent=2))
    print(fd.round(4).to_string(index=False))
    print("\n== E: ensemble ==")
    print(json.dumps(report["ensemble"], indent=2))


if __name__ == "__main__":
    main()
