"""H24 step A: an evaluation harness that can actually see a real gain.

Why the single frozen split is not enough (measured in the H23 audit):

* config G re-scored on 10 different 5-fold splits: M2 0.694 +/- 0.019,
  range 0.660-0.711;
* on the frozen split one fold sits at M2 0.544 against ~0.74 for the other
  four, so the old "delta > 0 and >= 4/5 folds" gate mostly measured that fold;
* the inner-CV seed alone moved an H23 variant by +/- 0.011.

Anything below roughly 0.02 M2 was invisible, and some old "rejections" may
be ties.

The protocol here, fixed **before** any H24 candidate was scored:

* 5 x 5 repeated stratified K-fold (seeds 0-4). Every candidate sees exactly
  the same 25 splits, so comparisons are paired.
* A candidate is compared with config G fold by fold on macro-F2, and the
  variance of the 25 paired deltas is inflated with the Nadeau-Bengio
  correction for overlapping training sets:
  ``var * (1/k + n_test/n_train)``.
* **Promotion gate:** mean paired delta > 0, AND corrected one-sided
  p < 0.10, AND the pooled-OOF M2 improves in at least 4 of 5 repeats.
* The frozen split is still scored and reported, for continuity with H8-H23,
  but it is not the gate.
* Threshold is fixed at 0.5 for every candidate (all are class-balanced), so
  no threshold tuning noise enters the comparison.

Out-of-fold probabilities are cached in ``h24/oof/<name>.npy`` with shape
(repeats, rows).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_baseline import load_folds, stage2_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h24"
OOF_DIR = OUT / "oof"
SEEDS = (0, 1, 2, 3, 4)
N_SPLITS = 5
THRESHOLD = 0.5
GATE_P = 0.10
GATE_MIN_REPEATS = 4
BASELINE_NAME = "G"


def repeated_splits(y, seeds=SEEDS) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """``[repeat][fold] -> (train_idx, val_idx)``; identical for every candidate."""
    y = np.asarray(y)
    return [list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=s).split(np.zeros(len(y)), y))
            for s in seeds]


def collapsed_titles() -> list[str]:
    """Titles where config G missed every positive on the frozen split (from H23)."""
    wt = pd.read_csv(ROOT / "h23" / "within_title_auc.csv")
    return wt.loc[wt["all_positives_missed"], "title"].tolist()


def _fit_predict(estimator, X, y, tr, va):
    model = clone(estimator).fit(X.iloc[tr], y[tr])
    return va, model.predict_proba(X.iloc[va])[:, 1]


def oof_matrix(name, estimator, X, y, splits, use_cache=True, n_jobs=6) -> np.ndarray:
    """OOF probabilities for every repeat, fitted in parallel and cached."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = OOF_DIR / f"{name}.npy"
    if use_cache and path.exists():
        return np.load(path)
    y = np.asarray(y)
    jobs = [(r, tr, va) for r, rep in enumerate(splits) for tr, va in rep]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(estimator, X, y, tr, va) for _, tr, va in jobs)
    proba = np.zeros((len(splits), len(y)))
    for (r, _, _), (va, p) in zip(jobs, results):
        proba[r, va] = p
    np.save(path, proba)
    return proba


def frozen_oof(estimator, X, y) -> np.ndarray:
    y = np.asarray(y)
    proba = np.zeros(len(y))
    for tr, va in load_folds():
        _, proba[va] = _fit_predict(estimator, X, y, tr, va)
    return proba


def _fold_f2(y, proba, splits) -> np.ndarray:
    hard = (proba >= THRESHOLD).astype(int)
    return np.array([[fbeta_score(y[va], hard[r, va], beta=2, average="macro", zero_division=0)
                      for _, va in rep] for r, rep in enumerate(splits)])


def _within_title_auc(y, p, mask_titles, titles) -> float:
    aucs = []
    for t in mask_titles:
        m = titles == t
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(aucs)) if aucs else float("nan")


def summarise(name, proba, y, splits, titles, base_proba=None, frozen=None) -> dict:
    """Everything the roadmap asks for, plus the paired test against config G."""
    y = np.asarray(y)
    pooled = np.array([stage2_score(y, (p >= THRESHOLD).astype(int)) for p in proba])
    hard = proba >= THRESHOLD
    coll = collapsed_titles()
    coll_mask = np.isin(titles, coll)
    row = {
        "candidate": name,
        "m2_mean": float(pooled.mean()),
        "m2_sd_repeats": float(pooled.std(ddof=1)),
        "m2_per_repeat": [round(v, 4) for v in pooled],
        "auc_mean": float(np.mean([roc_auc_score(y, p) for p in proba])),
        "fn_mean": float(((y == 1) & ~hard).sum(axis=1).mean()),
        "fp_mean": float(((y == 0) & hard).sum(axis=1).mean()),
        "positive_rate": float(hard.mean()),
        "collapsed_within_auc": float(np.mean([_within_title_auc(y, p, coll, titles) for p in proba])),
        "collapsed_fn_mean": float(((y == 1) & ~hard & coll_mask).sum(axis=1).mean()),
        "collapsed_pos": int(((y == 1) & coll_mask).sum()),
    }
    if frozen is not None:
        row["m2_frozen_split"] = stage2_score(y, (frozen >= THRESHOLD).astype(int))
    if base_proba is not None:
        base_pooled = np.array([stage2_score(y, (p >= THRESHOLD).astype(int)) for p in base_proba])
        d = (_fold_f2(y, proba, splits) - _fold_f2(y, base_proba, splits)).ravel() / 0.45
        k = d.size
        n_test = len(splits[0][0][1])
        n_train = len(y) - n_test
        var = d.var(ddof=1) * (1.0 / k + n_test / n_train)
        t = d.mean() / np.sqrt(var) if var > 0 else 0.0
        p_one_sided = float(1 - stats.t.cdf(t, df=k - 1)) if var > 0 else 1.0
        rep_delta = pooled - base_pooled
        row.update({
            "delta_m2_mean": float(rep_delta.mean()),
            "delta_m2_per_repeat": [round(v, 4) for v in rep_delta],
            "repeats_improved": int((rep_delta > 0).sum()),
            "fold_delta_m2_mean": float(d.mean()),
            "corrected_t": float(t),
            "p_one_sided": p_one_sided,
        })
        row["passes_gate"] = bool(rep_delta.mean() > 0 and p_one_sided < GATE_P
                                  and row["repeats_improved"] >= GATE_MIN_REPEATS)
    return row


def print_row(r: dict) -> None:
    extra = ""
    if "delta_m2_mean" in r:
        extra = ("  d=%+.4f  rep+=%d/5  t=%+.2f p=%.3f  %s"
                 % (r["delta_m2_mean"], r["repeats_improved"], r["corrected_t"],
                    r["p_one_sided"], "PASS" if r["passes_gate"] else "-"))
    print("%-30s M2=%.4f±%.4f  AUC=%.4f  FN=%5.1f FP=%5.1f  coll.AUC=%.3f coll.FN=%4.1f/%d%s"
          % (r["candidate"], r["m2_mean"], r["m2_sd_repeats"], r["auc_mean"], r["fn_mean"],
             r["fp_mean"], r["collapsed_within_auc"], r["collapsed_fn_mean"], r["collapsed_pos"], extra))


def save_rows(rows: list[dict], stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
