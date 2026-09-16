"""Fit the H15 decoded-hybrid Stage 1 model and assemble a submission.

Run:  python src/stage1_hybrid_build.py

Same contract as ``stage1_build.py``: Stage 2 is taken unchanged from the
production model, the frozen checkpoint is hashed before and after, no test
label is ever read, and the decision threshold comes from inner-fold tuning on
train only.

The one thing that differs from H14 is that inference has to run the *same*
leave-one-out decode the training rows ran. A test title is clean and sits in
the matching corpus, so decoding it without excluding itself would recover the
whole title every time, while training rows only ever recovered about half.
That mismatch is the defect H14 was built to remove, so it is not reintroduced
here -- the test decode excludes each row's own contribution.
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
from stage1_decode import build_decoder, decode_frame, load_all  # noqa: E402
from stage1_hybrid import build_hybrid_model  # noqa: E402
from stage1_invariant import SEED  # noqa: E402
from stage1_rules import (  # noqa: E402
    apply_stage2_title_rule,
    audit_stage2_title_rule,
    stage2_title_index,
)

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "h15"
CHECKPOINT = DATA_DIR / "h12" / "checkpoint" / "checkpoint_submission.csv"
STAGE2_PREDICTIONS = DATA_DIR / "final" / "predictions" / "stage2_predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    checkpoint_hash_before = sha256(CHECKPOINT)

    evaluation = json.loads((OUT_DIR / "stage1_hybrid_eval.json").read_text(encoding="utf-8"))
    C = evaluation["best_C"]
    threshold = evaluation["nested_cv"]["threshold_median"]
    print("C=%.1f  threshold=%.2f" % (C, threshold))
    print(
        "nested-CV M1 = %.4f (per-fold %.4f +/- %.4f)   H14 was %.4f"
        % (
            evaluation["nested_cv"]["pooled_m1"],
            evaluation["nested_cv"]["mean_fold_m1"],
            evaluation["nested_cv"]["std_fold_m1"],
            evaluation["h14_nested_m1"],
        )
    )

    frames = load_all()
    train, test = frames["train_s1"], frames["test_s1"]
    decoder = build_decoder(frames["test_s1"], frames["train_s2"], frames["test_s2"])
    dec_train = np.asarray(decode_frame(train, decoder, self_source=None), dtype=object)
    dec_test = np.asarray(decode_frame(test, decoder, self_source="s1_test"), dtype=object)
    y = train["label"].to_numpy()

    print(
        "\n[decode] subsection recovered on %.3f of train rows, %.3f of test rows"
        % (
            float(np.mean([bool(d.subsection) for d in dec_train])),
            float(np.mean([bool(d.subsection) for d in dec_test])),
        )
    )

    # -- rule audit on labelled data (unchanged from H14) -------------------- #
    s2_titles = stage2_title_index(frames["train_s2"], frames["test_s2"])
    audit = audit_stage2_title_rule(train, s2_titles)
    print(
        "[rule audit] %d matched, %d Special, %d General -> precision %.4f"
        % (audit["n_matched"], audit["n_special"], audit["n_general"], audit["precision"])
    )
    rule_enabled = audit["n_matched"] > 0 and audit["precision"] == 1.0
    print("  rule %s" % ("ENABLED" if rule_enabled else "DISABLED (below 100% precision)"))

    # -- fit and predict ----------------------------------------------------- #
    model = build_hybrid_model(C=C).fit(dec_train, y)
    proba = model.predict_proba(dec_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print("\n[model] test positive rate %.4f  (train prior %.4f)" % (pred.mean(), y.mean()))

    n_changed = 0
    if rule_enabled:
        pred, hit, n_changed = apply_stage2_title_rule(test["title_text"], pred, s2_titles)
        print("[rule] %d titles matched, %d predictions flipped" % (int(hit.sum()), n_changed))
    print("[final] test positive rate %.4f" % pred.mean())

    # -- compare against H14 and against production -------------------------- #
    h14 = pd.read_csv(DATA_DIR / "h14" / "stage1_rebuild_predictions.csv")
    h14 = h14.set_index("id").loc[test["id"]].reset_index()
    prod = pd.read_csv(DATA_DIR / "final" / "predictions" / "stage1_predictions.csv")
    prod = prod.set_index("id").loc[test["id"]].reset_index()
    agree_h14 = float((h14["label"].to_numpy() == pred).mean())
    agree_prod = float((prod["label"].to_numpy() == pred).mean())
    print(
        "[delta] agreement with H14 %.4f (%d rows differ), with production C1 %.4f"
        % (agree_h14, int((h14["label"].to_numpy() != pred).sum()), agree_prod)
    )

    # -- assemble the submission --------------------------------------------- #
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
    assert submission.loc[submission["stage"] == 2, "label"].tolist() == stage2["label"].tolist(), (
        "stage 2 must be byte-identical to the frozen production predictions"
    )

    sub_path = OUT_DIR / "submission_stage1_hybrid.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")
    joblib.dump(model, OUT_DIR / "stage1_hybrid_model.joblib")
    pd.DataFrame({"id": test["id"], "proba": proba, "label": pred}).to_csv(
        OUT_DIR / "stage1_hybrid_predictions.csv", index=False
    )

    manifest = {
        "seed": SEED,
        "C": C,
        "threshold": threshold,
        "nested_cv": evaluation["nested_cv"],
        "h14_nested_m1": evaluation["h14_nested_m1"],
        "delta_m1": evaluation["delta_m1"],
        "decode_coverage": evaluation["coverage"],
        "stage2_title_rule": {**audit, "enabled": rule_enabled, "n_flipped": n_changed},
        "test_positive_rate": float(pred.mean()),
        "train_positive_rate": float(y.mean()),
        "agreement_with_h14": agree_h14,
        "agreement_with_production": agree_prod,
        "stage2_source": str(STAGE2_PREDICTIONS.relative_to(DATA_DIR)),
        "submission_sha256": sha256(sub_path),
    }
    (OUT_DIR / "stage1_hybrid_manifest.json").write_text(
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
