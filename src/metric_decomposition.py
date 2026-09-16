"""Decompose the leaderboard total into its Stage 1 and Stage 2 halves.

The platform reports one number, ``70 * (0.3 * M1 + 0.7 * M2)``, so a single
score cannot separate the stages. Two facts make the split exact:

* an all-zero Stage 1 prediction forces macro-F0.5 <= 0.5, which the M1 clip
  maps to exactly 0 -- confirmed empirically in ``h13`` to a residual of 8e-6;
* a probe submitted with that trivial Stage 1 therefore reads out
  ``49 * M2`` directly.

That probe scored 35.8078, which pins M2 = 0.73077 as a *measured* value rather
than an estimate. Every later submission that leaves Stage 2 untouched can then
be inverted for its real M1.

Run:  python src/metric_decomposition.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "h18"

M1_WEIGHT = 70 * 0.3   # 21.0 points
M2_WEIGHT = 70 * 0.7   # 49.0 points

# The trivial-Stage-1 probe from h13: Stage 2 = production config G at t=0.50.
TRIVIAL_PROBE_TOTAL = 35.8078022875817

# Leaderboard totals, in submission order.
SUBMISSIONS = {
    "production C1 + G": 49.021,
    "H17 clean-text S1 + G": 55.168,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m2_real = TRIVIAL_PROBE_TOTAL / M2_WEIGHT

    rows = {}
    for name, total in SUBMISSIONS.items():
        rows[name] = {
            "leaderboard_total": total,
            "stage2_points": TRIVIAL_PROBE_TOTAL,
            "stage1_points": total - TRIVIAL_PROBE_TOTAL,
            "M1_real": (total - TRIVIAL_PROBE_TOTAL) / M1_WEIGHT,
        }

    offline = {
        "M1_production_C1_cv": 0.4648142133801812,
        "M1_h14_nested_cv": json.loads(
            (ROOT / "h14" / "stage1_tuning.json").read_text(encoding="utf-8")
        )["nested_cv"]["pooled_m1"],
        "M1_h15_nested_cv": json.loads(
            (ROOT / "h15" / "stage1_hybrid_eval.json").read_text(encoding="utf-8")
        )["nested_cv"]["pooled_m1"],
        "M1_h17_nested_cv": json.loads(
            (ROOT / "h17" / "stage1_clean_C_selection.json").read_text(encoding="utf-8")
        )["best_pooled_m1"],
        "M2_offline_cv": 0.7023868247508647,
    }

    h17_real = rows["H17 clean-text S1 + G"]["M1_real"]
    report = {
        "M2_real_measured": m2_real,
        "M2_offline_cv": offline["M2_offline_cv"],
        "M2_cv_underestimates_by": m2_real - offline["M2_offline_cv"],
        "submissions": rows,
        "offline_reference": offline,
        "M1_h17_real": h17_real,
        "M1_h17_cv_underestimates_by": h17_real - offline["M1_h17_nested_cv"],
        "which_M2_to_use": {
            "score_accounting_and_reporting": "M2_real = 0.7308 (measured, not estimated)",
            "model_selection_between_variants": (
                "M2_offline_cv = 0.7024 -- the only figure available for an unscored "
                "candidate, and competition rule 12 forbids selecting on leaderboard "
                "feedback"
            ),
            "never": (
                "compare an offline M2 against the real M2. Offline runs ~0.028 low, so "
                "that comparison rejects good candidates for free."
            ),
        },
        "bookkeeping_discrepancy": {
            "h12_checkpoint_manifest_records_total": 49.21,
            "user_reported_total": 49.021,
            "implied_M1_if_49_21": (49.21 - TRIVIAL_PROBE_TOTAL) / M1_WEIGHT,
            "implied_M1_if_49_021": (49.021 - TRIVIAL_PROBE_TOTAL) / M1_WEIGHT,
            "note": (
                "h12/checkpoint/checkpoint_manifest.json records 49.21 and derives "
                "M1_real(C1)=0.6382 from it; the reported total is 49.021, which gives "
                "0.6292. Only the baseline delta is affected -- H17's real M1 is "
                "derived from its own total and the measured M2 anchor, so it stands "
                "either way."
            ),
        },
    }

    (OUT / "metric_decomposition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("M2 real (measured via trivial-S1 probe) : %.5f   -> %.3f pts" % (m2_real, TRIVIAL_PROBE_TOTAL))
    print("M2 offline (5-fold CV)                  : %.5f" % offline["M2_offline_cv"])
    print("  CV underestimates real M2 by          : %.5f\n" % report["M2_cv_underestimates_by"])
    for name, r in rows.items():
        print("%-24s total %7.3f = %6.3f (S1) + %6.3f (S2)   M1_real = %.4f"
              % (name, r["leaderboard_total"], r["stage1_points"], r["stage2_points"], r["M1_real"]))
    print()
    print("H17: CV M1 %.4f -> real M1 %.4f  (CV underestimates by %.4f)"
          % (offline["M1_h17_nested_cv"], h17_real, report["M1_h17_cv_underestimates_by"]))
    print("\nremaining headroom: Stage 1 %.2f pts, Stage 2 %.2f pts"
          % (M1_WEIGHT * (1 - h17_real), M2_WEIGHT * (1 - m2_real)))
    print("wrote h18/metric_decomposition.json")


if __name__ == "__main__":
    main()
