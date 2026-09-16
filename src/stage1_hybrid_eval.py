"""Does decoding actually buy anything? Ablation + nested CV against H14.

Run:  python src/stage1_hybrid_eval.py

H14 shipped an honest nested-CV M1 of 0.7111 on features that survive the
corruption. Decoding recovers real text for about half the subsection lines, so
the question this script answers is whether lexical features over that half beat
shape features over all of it. The comparison uses the same folds, the same
seed, the same threshold-on-inner-folds protocol and the same official metric,
so the two numbers are directly comparable.

The ablation exists because "decoding helped" is not a claim worth making
without knowing which block did the work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_decode import build_decoder, decode_frame, load_all  # noqa: E402
from stage1_eval import stage1_score, tune_threshold  # noqa: E402
from stage1_hybrid import build_hybrid_features, build_hybrid_model  # noqa: E402
from stage1_invariant import SEED  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "h15"


def coverage_report(decoded_train: list, decoded_test: list) -> dict:
    """Per-role decode rates on both splits -- the transfer check for H15."""

    def stats(rows: list) -> dict:
        n = len(rows)
        return {
            "n": n,
            "guideline": float(np.mean([bool(d.guideline) for d in rows])),
            "age": float(np.mean([bool(d.age_category) for d in rows])),
            "hierarchy": float(
                np.mean([all(bool(h) for h in d.hierarchy) if d.hierarchy else True for d in rows])
            ),
            "subsection": float(np.mean([bool(d.subsection) for d in rows])),
            "subsection_ambiguous": float(np.mean([d.subsection_ambiguous for d in rows])),
            "mean_coverage": float(np.mean([d.coverage for d in rows])),
        }

    train, test = stats(decoded_train), stats(decoded_test)
    gaps = {
        k: abs(train[k] - test[k]) for k in train if k != "n"
    }
    return {"train": train, "test": test, "max_role_gap": max(gaps.values()), "gaps": gaps}


def subset_model(blocks: tuple, C: float = 1.0) -> Pipeline:
    """A hybrid model restricted to the named feature blocks."""
    full = build_hybrid_features()
    kept = [(name, tr) for name, tr in full.transformer_list if name in blocks]
    return Pipeline(
        [
            ("features", FeatureUnion(kept)),
            (
                "clf",
                LogisticRegression(
                    C=C, class_weight="balanced", max_iter=5000, random_state=SEED
                ),
            ),
        ]
    )


def oof_proba(model, X, y, cv) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def nested_cv(build, X, y, n_outer: int = 5, n_inner: int = 5) -> dict:
    """Threshold chosen on inner folds, scored on the held-out outer fold."""
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    thresholds, fold_scores = [], []
    y_pred = np.zeros(len(y), dtype=int)

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold)
        inner_proba = oof_proba(build(), X[tr], y[tr], inner)
        t, _, _ = tune_threshold(y[tr], inner_proba)
        model = build().fit(X[tr], y[tr])
        pred = (model.predict_proba(X[te])[:, 1] >= t).astype(int)
        y_pred[te] = pred
        thresholds.append(t)
        fold_scores.append(stage1_score(y[te], pred))
        print(
            "  outer fold %d: inner t=%.2f  outer M1=%.4f  posrate=%.4f"
            % (fold, t, fold_scores[-1], pred.mean())
        )

    return {
        "thresholds": thresholds,
        "threshold_median": float(np.median(thresholds)),
        "fold_scores": fold_scores,
        "mean_fold_m1": float(np.mean(fold_scores)),
        "std_fold_m1": float(np.std(fold_scores)),
        "pooled_m1": stage1_score(y, y_pred),
        "pooled_posrate": float(y_pred.mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    frames = load_all()
    decoder = build_decoder(frames["test_s1"], frames["train_s2"], frames["test_s2"])

    # The train split never entered the corpus, so it has nothing to exclude;
    # the test split must exclude itself or it decodes trivially.
    dec_train = decode_frame(frames["train_s1"], decoder, self_source=None)
    dec_test = decode_frame(frames["test_s1"], decoder, self_source="s1_test")

    cov = coverage_report(dec_train, dec_test)
    print("[decode coverage]  train / test")
    for role in ("guideline", "age", "hierarchy", "subsection", "mean_coverage"):
        print("  %-16s %.3f / %.3f" % (role, cov["train"][role], cov["test"][role]))
    print("  max role gap between splits: %.3f" % cov["max_role_gap"])

    X = np.asarray(dec_train, dtype=object)
    y = frames["train_s1"]["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # -- ablation ----------------------------------------------------------- #
    ablation = {
        "invariant_only (= H14)": ("invariant",),
        "+ conditions": ("invariant", "conditions"),
        "+ lex_subsection": ("invariant", "lex_subsection"),
        "+ lex_both": ("invariant", "lex_subsection", "lex_context"),
        "full hybrid": ("invariant", "lex_subsection", "lex_context", "conditions"),
        "decoded_only (no invariant)": ("lex_subsection", "lex_context", "conditions"),
    }
    print("\n[ablation, 5-fold, threshold tuned in-fold (optimistic, for ranking)]")
    rows = []
    for name, blocks in ablation.items():
        proba = oof_proba(subset_model(blocks), X, y, cv)
        t, s, _ = tune_threshold(y, proba)
        rows.append(
            {
                "config": name,
                "blocks": "+".join(blocks),
                "m1_at_050": stage1_score(y, (proba >= 0.5).astype(int)),
                "m1_tuned": s,
                "threshold": t,
            }
        )
        print("  %-28s M1@0.50=%.4f  M1tuned=%.4f (t=%.2f)" % (name, rows[-1]["m1_at_050"], s, t))
    pd.DataFrame(rows).to_csv(OUT_DIR / "stage1_hybrid_ablation.csv", index=False)

    # -- C sweep on the full hybrid ----------------------------------------- #
    print("\n[C sweep, full hybrid]")
    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        proba = oof_proba(build_hybrid_model(C=C), X, y, cv)
        t, s, _ = tune_threshold(y, proba)
        print("  C=%-5.1f  M1tuned=%.4f (t=%.2f)" % (C, s, t))
        if best is None or s > best[1]:
            best = (C, s)
    C_best = best[0]

    # -- nested CV, the shipping estimate ----------------------------------- #
    print("\n[nested CV, full hybrid, C=%.1f]" % C_best)
    nested = nested_cv(lambda: build_hybrid_model(C=C_best), X, y)
    print(
        "  pooled M1=%.4f   per-fold %.4f +/- %.4f   posrate=%.4f"
        % (
            nested["pooled_m1"],
            nested["mean_fold_m1"],
            nested["std_fold_m1"],
            nested["pooled_posrate"],
        )
    )

    h14 = json.loads((DATA_DIR / "h14" / "stage1_tuning.json").read_text(encoding="utf-8"))
    delta = nested["pooled_m1"] - h14["nested_cv"]["pooled_m1"]
    print(
        "\n[verdict] H14 nested M1 %.4f -> H15 nested M1 %.4f   delta %+.4f (%+.2f metric points)"
        % (h14["nested_cv"]["pooled_m1"], nested["pooled_m1"], delta, 70 * 0.3 * delta)
    )

    (OUT_DIR / "stage1_hybrid_eval.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "coverage": cov,
                "ablation": rows,
                "best_C": C_best,
                "nested_cv": nested,
                "h14_nested_m1": h14["nested_cv"]["pooled_m1"],
                "delta_m1": delta,
                "delta_metric_points": 70 * 0.3 * delta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote h15/stage1_hybrid_eval.json")


if __name__ == "__main__":
    main()
