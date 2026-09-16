"""Fit the Stage 1 rebuild on full train, predict test, assemble a submission.

Run:  python src/stage1_build.py

Stage 2 predictions are taken unchanged from the current production model
(config G at threshold 0.50) -- five structurally different Stage 2 feature sets
all land at CV M2 ~0.70, so Stage 2 is treated as frozen and only Stage 1 is
rebuilt here.

Guard rails:
  * the frozen checkpoint submission is hashed before and after and must not
    change;
  * no test label is read at any point, and the decision threshold comes from
    inner-fold tuning on train only (competition rules 11 and 12).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import read_competition_csv  # noqa: E402
from stage1_invariant import SEED, build_stage1_model, mask_cyrillic  # noqa: E402
from stage1_rules import (  # noqa: E402
    apply_stage2_title_rule,
    audit_stage2_title_rule,
    stage2_title_index,
)
from stage1_tune import configure  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "h14"
CHECKPOINT = DATA_DIR / "h12" / "checkpoint" / "checkpoint_submission.csv"
STAGE2_PREDICTIONS = DATA_DIR / "final" / "predictions" / "stage2_predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    checkpoint_hash_before = sha256(CHECKPOINT)

    tuning = json.loads((OUT_DIR / "stage1_tuning.json").read_text(encoding="utf-8"))
    params = tuning["best_params"]
    # The median of the inner-fold optima, not the single best out-of-fold
    # threshold: the threshold curve is a broad plateau between 0.70 and 0.80,
    # and the median is the stable point inside it.
    threshold = tuning["nested_cv"]["threshold_median"]
    print("params=%s  threshold=%.2f" % (params, threshold))
    print(
        "nested-CV M1 = %.4f (per-fold %.4f +/- %.4f)"
        % (
            tuning["nested_cv"]["pooled_m1"],
            tuning["nested_cv"]["mean_fold_m1"],
            tuning["nested_cv"]["std_fold_m1"],
        )
    )

    train = read_competition_csv(DATA_DIR / "train_stage1.csv")
    test = pd.read_csv(DATA_DIR / "test_stage1.csv")
    train_s2 = pd.read_csv(DATA_DIR / "train_stage2.csv")
    test_s2 = pd.read_csv(DATA_DIR / "test_stage2.csv")

    X, y = train["title_text"].to_numpy(dtype=object), train["label"].to_numpy()

    # -- rule audit on labelled data --------------------------------------- #
    s2_titles = stage2_title_index(train_s2, test_s2)
    audit = audit_stage2_title_rule(train, s2_titles)
    print(
        "\n[rule audit] stage-2 titles seen in train_stage1: %d matched, "
        "%d Special, %d General -> precision %.4f"
        % (audit["n_matched"], audit["n_special"], audit["n_general"], audit["precision"])
    )
    rule_enabled = audit["n_matched"] > 0 and audit["precision"] == 1.0
    print("  rule %s" % ("ENABLED" if rule_enabled else "DISABLED (below 100% precision)"))

    # -- fit and predict ---------------------------------------------------- #
    model = configure(build_stage1_model(), **params).fit(X, y)
    proba = model.predict_proba(test["title_text"].to_numpy(dtype=object))[:, 1]
    pred = (proba >= threshold).astype(int)
    print(
        "\n[model] test positive rate %.4f  (train prior %.4f)"
        % (pred.mean(), y.mean())
    )

    n_changed = 0
    if rule_enabled:
        pred, hit, n_changed = apply_stage2_title_rule(test["title_text"], pred, s2_titles)
        print(
            "[rule] %d test titles matched a stage-2 title, %d predictions flipped to Special"
            % (int(hit.sum()), n_changed)
        )
    print("[final] test positive rate %.4f" % pred.mean())

    # -- compare against the current production Stage 1 --------------------- #
    prod = pd.read_csv(DATA_DIR / "final" / "predictions" / "stage1_predictions.csv")
    prod = prod.set_index("id").loc[test["id"]].reset_index()
    agreement = float((prod["label"].to_numpy() == pred).mean())
    print(
        "[delta vs production C1] agreement %.4f  production posrate %.4f"
        % (agreement, prod["label"].mean())
    )

    # -- assemble the submission ------------------------------------------- #
    stage2 = pd.read_csv(STAGE2_PREDICTIONS)
    submission = pd.concat(
        [
            pd.DataFrame({"stage": 1, "id": test["id"].to_numpy(), "label": pred}),
            pd.DataFrame(
                {"stage": 2, "id": stage2["id"].to_numpy(), "label": stage2["label"].to_numpy()}
            ),
        ],
        ignore_index=True,
    )
    assert len(submission) == len(test) + len(stage2) == 615, "unexpected submission length"
    assert submission["label"].isin([0, 1]).all(), "labels must be binary"
    assert list(submission.loc[submission["stage"] == 1, "id"]) == list(test["id"]), (
        "stage 1 row order must match test_stage1.csv"
    )

    sub_path = OUT_DIR / "submission_stage1_rebuild.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")
    joblib.dump(model, OUT_DIR / "stage1_invariant_model.joblib")
    pd.DataFrame(
        {"id": test["id"], "proba": proba, "label": pred}
    ).to_csv(OUT_DIR / "stage1_rebuild_predictions.csv", index=False)

    manifest = {
        "seed": SEED,
        "params": params,
        "threshold": threshold,
        "nested_cv": tuning["nested_cv"],
        "stage2_title_rule": {**audit, "enabled": rule_enabled, "n_flipped": n_changed},
        "test_positive_rate": float(pred.mean()),
        "train_positive_rate": float(y.mean()),
        "production_positive_rate": float(prod["label"].mean()),
        "agreement_with_production": agreement,
        "stage2_source": str(STAGE2_PREDICTIONS.relative_to(DATA_DIR)),
        "submission_sha256": sha256(sub_path),
    }
    (OUT_DIR / "stage1_rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    checkpoint_hash_after = sha256(CHECKPOINT)
    assert checkpoint_hash_before == checkpoint_hash_after, (
        "checkpoint_submission.csv was modified -- THIS MUST NEVER HAPPEN"
    )
    print("\ncheckpoint intact: %s" % checkpoint_hash_after[:16])
    print("wrote %s" % sub_path.relative_to(DATA_DIR))


if __name__ == "__main__":
    main()
