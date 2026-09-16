"""Candidate A audit: how much more could better line matching recover?

Analysis only -- nothing here trains, fits or writes a feature. The decisive
measurement is an offline simulation: the Stage 1 test titles are clean, so for
any test row we know the true text. Running the decoder against a corpus that
excludes that row tells us exactly what a matcher would have retrieved and how
close it came to the truth.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_decode import build_decoder, load_all  # noqa: E402
from stage1_invariant import mask_cyrillic  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "h16"


def edit_distance(a: str, b: str, cap: int = 6) -> int:
    """Levenshtein with an early exit once the cap is exceeded."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def tokens(text: str) -> set:
    return set(w for w in str(text).lower().replace("#", " ").split() if w)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    f = load_all()
    train, test = f["train_s1"], f["test_s1"]
    decoder = build_decoder(f["test_s1"], f["train_s2"], f["test_s2"])
    report = {}

    def subline(t):
        return str(t).split("\n")[-1].strip()

    tr_sub = [subline(t) for t in train["title_text"]]
    te_sub = [subline(t) for t in test["title_text"]]
    y = train["label"].to_numpy()

    tr_masked = [mask_cyrillic(s) for s in tr_sub]
    te_masked = [mask_cyrillic(s) for s in te_sub]

    # ---- 1. candidate-set size = decoder confidence ---------------------- #
    tr_n = np.array([len(decoder.candidates(m, None)) for m in tr_masked])
    te_n = np.array(
        [len(decoder.candidates(m, ("s1_test", i))) for i, m in enumerate(te_masked)]
    )

    def bucket(n):
        return "0 (unmatched)" if n == 0 else "1 (exact)" if n == 1 else "2-3" if n <= 3 else "4+"

    report["confidence_distribution"] = {
        "train": {k: int(v) for k, v in Counter(bucket(n) for n in tr_n).items()},
        "test_loo": {k: int(v) for k, v in Counter(bucket(n) for n in te_n).items()},
    }
    report["matched_subsection_rate"] = {
        "train": float((tr_n > 0).mean()),
        "test_loo": float((te_n > 0).mean()),
    }
    rows = []
    for b in ["0 (unmatched)", "1 (exact)", "2-3", "4+"]:
        m = np.array([bucket(n) == b for n in tr_n])
        if m.sum():
            rows.append({"bucket": b, "n": int(m.sum()), "special_rate": float(y[m].mean())})
    report["label_rate_by_confidence"] = rows

    # ---- 2. duplicate title rate ----------------------------------------- #
    tr_full = [mask_cyrillic(str(t)) for t in train["title_text"]]
    te_full = [mask_cyrillic(str(t)) for t in test["title_text"]]
    sub_counts = Counter(tr_masked)
    report["duplicate_rate"] = {
        "train_full_title_dupes": float(1 - len(set(tr_full)) / len(tr_full)),
        "test_full_title_dupes": float(1 - len(set(te_full)) / len(te_full)),
        "train_subsection_distinct": len(sub_counts),
        "train_rows_whose_subsection_repeats": int(
            sum(1 for m in tr_masked if sub_counts[m] > 1)
        ),
    }

    # ---- 3. unmatched subsection frequency ------------------------------- #
    unmatched = [m for m, n in zip(tr_masked, tr_n) if n == 0]
    uc = Counter(unmatched)
    report["unmatched_subsections"] = {
        "n_rows": len(unmatched),
        "n_distinct": len(uc),
        "rows_in_singleton_patterns": int(sum(1 for m in unmatched if uc[m] == 1)),
        "top_repeat_counts": [int(c) for _, c in uc.most_common(5)],
    }

    # ---- 4. what could fuzzy matching add? ------------------------------- #
    # Structural note for the writeup: if the exact masked form is absent from
    # the corpus then the true line is absent too, because masking is
    # deterministic. Fuzzy matching therefore cannot recover an unmatched line
    # verbatim -- it can only retrieve a near neighbour. The question this
    # simulates is whether that neighbour carries the right words.
    corpus_masked = {}
    for t in (
        list(f["test_s1"]["title_text"])
        + list(f["train_s2"]["title_text"])
        + list(f["test_s2"]["title_text"])
    ):
        for ln in str(t).split("\n"):
            ln = ln.strip()
            if ln:
                corpus_masked.setdefault(mask_cyrillic(ln), set()).add(ln)
    corpus_keys = list(corpus_masked)

    sim_rows = []
    for i, (true_line, masked) in enumerate(zip(te_sub, te_masked)):
        if te_n[i] != 0:
            continue  # exact leave-one-out match already succeeded
        best, best_d = None, 99
        for m in corpus_keys:
            cleans = corpus_masked[m] - {true_line}
            if not cleans:
                continue
            d = edit_distance(masked, m, cap=6)
            if d < best_d:
                best_d, best = d, cleans
        if best is None:
            continue
        truth = tokens(true_line)
        retrieved = max((tokens(c) for c in best), key=lambda s: len(s & truth))
        sim_rows.append(
            {
                "edit_distance": best_d,
                "jaccard": len(retrieved & truth) / max(len(retrieved | truth), 1),
                "recall_of_true_tokens": len(retrieved & truth) / max(len(truth), 1),
                "exact": float(retrieved == truth),
            }
        )

    sim = pd.DataFrame(sim_rows)
    if len(sim):
        sim.to_csv(OUT / "fuzzy_simulation.csv", index=False)
        by_d = (
            sim.groupby(sim["edit_distance"].clip(upper=7))
            .agg(
                n=("jaccard", "size"),
                mean_jaccard=("jaccard", "mean"),
                mean_token_recall=("recall_of_true_tokens", "mean"),
                exact=("exact", "mean"),
            )
            .reset_index()
        )
        report["fuzzy_simulation"] = {
            "n_unmatched_test_rows_simulated": int(len(sim)),
            "overall_mean_jaccard": float(sim["jaccard"].mean()),
            "overall_mean_token_recall": float(sim["recall_of_true_tokens"].mean()),
            "exact_recovery_rate": float(sim["exact"].mean()),
            "by_edit_distance": by_d.to_dict("records"),
        }

    # ---- 5. why exact matching fails: near-miss structure ----------------- #
    near = []
    for m, n in zip(tr_masked, tr_n):
        if n:
            continue
        d = min((edit_distance(m, k, cap=6) for k in corpus_keys), default=7)
        near.append(min(d, 7))
    report["unmatched_train_nearest_corpus_distance"] = {
        str(k): int(v) for k, v in sorted(Counter(near).items())
    }

    (OUT / "decoder_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
