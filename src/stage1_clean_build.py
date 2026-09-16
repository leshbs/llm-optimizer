"""Fit the clean-text Stage 1 model and assemble a submission.

Run:  python src/stage1_clean_build.py

Same contract as the H14 and H15 builds: Stage 2 is copied unchanged from the
production model, the frozen checkpoint is hashed before and after, no test
label is read, and the threshold comes from inner-fold tuning on train only.

What is gone relative to H15: no masking, no decoding, no corpus matching, no
recurrence features. The training file is intact, so the model simply reads the
words. The Stage-2-title rule survives -- it never depended on the corruption,
only on the fact that a title used in Stage 2 is Special by construction -- and
is now an exact string match rather than a masked one.
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
from stage1_clean import build_clean_model  # noqa: E402
from stage1_invariant import SEED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h17"
CHECKPOINT = ROOT / "h12" / "checkpoint" / "checkpoint_submission.csv"
STAGE2_PREDICTIONS = ROOT / "final" / "predictions" / "stage2_predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    checkpoint_hash_before = sha256(CHECKPOINT)

    sel = json.loads((OUT / "stage1_clean_C_selection.json").read_text(encoding="utf-8"))
    C, threshold = sel["best_C"], sel["threshold_median"]
    nested = sel["nested_by_C"][str(C)]
    print("C=%.1f  threshold=%.2f" % (C, threshold))
    print("nested-CV M1 = %.4f (per-fold %.4f +/- %.4f)"
          % (nested["pooled_m1"], nested["mean_fold_m1"], nested["std_fold_m1"]))

    train = pd.read_csv(ROOT / "train_stage1.csv")
    test = pd.read_csv(ROOT / "test_stage1.csv")
    train_s2 = pd.read_csv(ROOT / "train_stage2.csv")
    test_s2 = pd.read_csv(ROOT / "test_stage2.csv")
    X = train["title_text"].to_numpy(dtype=object)
    y = train["label"].to_numpy()

    # sanity: the repaired file must actually be readable text
    assert train["title_text"].str.count(r"\?").mean() < 1.0, (
        "train_stage1.csv still looks corrupted -- use the H15 pipeline instead"
    )

    # -- Stage-2-title rule, audited on labelled data ------------------------ #
    s2_titles = set(train_s2["title_text"]) | set(test_s2["title_text"])
    hit_train = train["title_text"].isin(s2_titles)
    audit = {
        "n_matched": int(hit_train.sum()),
        "n_special": int((train.loc[hit_train, "label"] == 1).sum()),
        "n_general": int((train.loc[hit_train, "label"] == 0).sum()),
        "precision": float(train.loc[hit_train, "label"].mean()) if hit_train.any() else float("nan"),
    }
    rule_enabled = audit["n_matched"] > 0 and audit["precision"] == 1.0
    print("\n[rule audit] %d matched, %d Special, %d General -> precision %.4f  -> %s"
          % (audit["n_matched"], audit["n_special"], audit["n_general"],
             audit["precision"], "ENABLED" if rule_enabled else "DISABLED"))

    # -- fit and predict ----------------------------------------------------- #
    model = build_clean_model(C=C).fit(X, y)
    proba = model.predict_proba(test["title_text"].to_numpy(dtype=object))[:, 1]
    pred = (proba >= threshold).astype(int)
    print("\n[model] test positive rate %.4f  (train prior %.4f)" % (pred.mean(), y.mean()))

    n_changed = 0
    if rule_enabled:
        hit = test["title_text"].isin(s2_titles).to_numpy()
        n_changed = int(((pred == 0) & hit).sum())
        pred[hit] = 1
        print("[rule] %d test titles matched, %d predictions flipped" % (int(hit.sum()), n_changed))
    print("[final] test positive rate %.4f" % pred.mean())

    # -- compare against the corruption-era models --------------------------- #
    deltas = {}
    for tag, path in (("h15", ROOT / "h15" / "stage1_hybrid_predictions.csv"),
                      ("h14", ROOT / "h14" / "stage1_rebuild_predictions.csv"),
                      ("production", ROOT / "final" / "predictions" / "stage1_predictions.csv")):
        other = pd.read_csv(path).set_index("id").loc[test["id"]]
        deltas[tag] = {
            "agreement": float((other["label"].to_numpy() == pred).mean()),
            "n_differ": int((other["label"].to_numpy() != pred).sum()),
        }
        print("[delta vs %-10s] agreement %.4f  (%d rows differ)"
              % (tag, deltas[tag]["agreement"], deltas[tag]["n_differ"]))

    # -- assemble ------------------------------------------------------------ #
    stage2 = pd.read_csv(STAGE2_PREDICTIONS)
    submission = pd.concat([
        pd.DataFrame({"stage": 1, "id": test["id"].to_numpy(), "label": pred}),
        pd.DataFrame({"stage": 2, "id": stage2["id"].to_numpy(),
                      "label": stage2["label"].to_numpy()}),
    ], ignore_index=True)
    assert len(submission) == len(test) + len(stage2) == 615, "unexpected submission length"
    assert submission["label"].isin([0, 1]).all(), "labels must be binary"
    assert list(submission.loc[submission["stage"] == 1, "id"]) == list(test["id"]), "row order"
    assert submission.loc[submission["stage"] == 2, "label"].tolist() == stage2["label"].tolist(), (
        "stage 2 must be byte-identical to the frozen production predictions"
    )

    sub_path = OUT / "submission_stage1_clean.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")
    joblib.dump(model, OUT / "stage1_clean_model.joblib")
    pd.DataFrame({"id": test["id"], "proba": proba, "label": pred}).to_csv(
        OUT / "stage1_clean_predictions.csv", index=False)

    manifest = {
        "seed": SEED, "C": C, "threshold": threshold, "nested_cv": nested,
        "stage2_title_rule": {**audit, "enabled": rule_enabled, "n_flipped": n_changed},
        "test_positive_rate": float(pred.mean()),
        "train_positive_rate": float(y.mean()),
        "deltas": deltas,
        "stage2_source": str(STAGE2_PREDICTIONS.relative_to(ROOT)),
        "submission_sha256": sha256(sub_path),
    }
    (OUT / "stage1_clean_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert checkpoint_hash_before == sha256(CHECKPOINT), (
        "checkpoint_submission.csv was modified -- THIS MUST NEVER HAPPEN"
    )
    print("\ncheckpoint intact: %s" % checkpoint_hash_before[:16])
    print("wrote %s" % sub_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
