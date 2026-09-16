"""Candidate B audit: is graded recurrence worth more than the binary flag?

The shipped H15 model encodes recurrence through two binary features
(``sub_decoded``, ``sub_ambiguous``). This measures how much label information
the raw candidate *count* holds that those flags do not, using mutual
information on the training labels. Analysis only -- no model is fitted.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_decode import build_decoder, decode_frame, load_all  # noqa: E402
from stage1_invariant import mask_cyrillic  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "h16"


def entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(groups: np.ndarray, y: np.ndarray) -> float:
    """I(Y; G) in bits."""
    base = entropy(y)
    cond = 0.0
    for g in np.unique(groups):
        m = groups == g
        cond += m.mean() * entropy(y[m])
    return base - cond


def main() -> None:
    OUT.mkdir(exist_ok=True)
    f = load_all()
    train = f["train_s1"]
    decoder = build_decoder(f["test_s1"], f["train_s2"], f["test_s2"])
    y = train["label"].to_numpy()

    sub_masked = [mask_cyrillic(str(t).split("\n")[-1].strip()) for t in train["title_text"]]
    counts = np.array([len(decoder.candidates(m, None)) for m in sub_masked])
    decoded = decode_frame(train, decoder, self_source=None)

    report = {"base_entropy_bits": entropy(y)}

    # ---- frequency histogram + rarity ------------------------------------ #
    hist = Counter(counts.tolist())
    report["candidate_count_histogram"] = {str(k): int(v) for k, v in sorted(hist.items())}
    # Rarity as the corpus supplies it: how many distinct clean lines attest
    # this masked pattern. 0 = attested nowhere = maximally rare.
    report["rarity_quantiles"] = {
        q: float(np.quantile(counts, float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")
    }
    report["ambiguity_rate"] = float((counts > 1).mean())
    report["decode_entropy_bits_mean"] = float(
        np.mean([np.log2(c) if c > 0 else 0.0 for c in counts])
    )

    # ---- label rate per count ------------------------------------------- #
    rows = []
    for c in sorted(hist):
        m = counts == c
        rows.append({"n_candidates": int(c), "n_rows": int(m.sum()),
                     "special_rate": float(y[m].mean())})
    report["label_rate_per_count"] = rows

    # ---- information content of each encoding ---------------------------- #
    binary = (counts > 0).astype(int)
    shipped = np.array(
        [
            0 if not d.subsection and not d.subsection_ambiguous
            else 1 if d.subsection and not d.subsection_ambiguous
            else 2
            for d in decoded
        ]
    )
    three = np.where(counts == 0, 0, np.where(counts == 1, 1, 2))
    graded = np.where(counts == 0, 0, np.where(counts == 1, 1, np.where(counts <= 3, 2, 3)))
    full = np.minimum(counts, 10)

    encodings = {
        "binary (matched / not)": binary,
        "shipped H15 flags": shipped,
        "3-level (0 / 1 / >1)": three,
        "4-level (0 / 1 / 2-3 / 4+)": graded,
        "full count (capped at 10)": full,
    }
    mi = {name: mutual_information(g, y) for name, g in encodings.items()}
    report["mutual_information_bits"] = mi
    report["mi_gain_over_shipped"] = {
        name: v - mi["shipped H15 flags"] for name, v in mi.items()
    }
    report["mi_as_pct_of_label_entropy"] = {
        name: v / report["base_entropy_bits"] for name, v in mi.items()
    }

    # ---- does count add anything *within* the matched group? ------------- #
    m = counts > 0
    report["within_matched"] = {
        "n": int(m.sum()),
        "entropy_bits": entropy(y[m]),
        "mi_of_count_bits": mutual_information(np.minimum(counts[m], 10), y[m]),
        "special_rate": float(y[m].mean()),
    }

    (OUT / "recurrence_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
