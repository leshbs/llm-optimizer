"""Append the H15 (decoded-hybrid Stage 1) section to baseline.ipynb.

Same contract as ``append_h14_section.py``: the notebook is edited
programmatically so the section can be regenerated deterministically, and
running this twice is a no-op because existing H15 cells are stripped first.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"
PREFIX = "h15"

CELLS = [
    (
        "markdown",
        "intro",
        """## 130. H15 -- recovering the destroyed Cyrillic text

Section 129 closed by naming the largest remaining Stage 1 opportunity: the
corruption is *partly reversible*, so a model could be trained on real Russian
text instead of on word-length shapes. This section does that, and the result is
the first Stage 1 model in this notebook that reads actual medical vocabulary.

**Why the corruption is partly invertible.** Masking replaced each Cyrillic
character with `?` but preserved length and every non-Cyrillic character, so
masking is a *function*: if a corrupted line and a readable line mask to the same
string, they are candidates for being the same line. The readable corpus is the
clean side of the competition inputs -- the 442 Stage 1 test titles and both
Stage 2 title columns, 474 distinct titles in total. These are model inputs and
never labels, so using them is transductive, which competition rule 12 permits.

**Why word-level decoding cannot work.** It is worth stating the negative result
explicitly, because it is the reason this is a *line* decoder. A pure-Cyrillic
word of length 7 masks to `???????`, so the number of distinct masked word forms
equals the number of distinct word lengths -- 25 across the whole corpus. Of
27,233 masked word tokens in train, exactly 11 have a unique dictionary
preimage. Word length is all that survives, which is precisely what section
129's shape n-grams already encode. Nothing is left on the table there.

Line-level matching is a different matter, because a whole line is a far more
specific key.""",
    ),
    (
        "code",
        "imports",
        """# ---- 130.0: decode the corrupted training titles ----
import sys, json, hashlib
from pathlib import Path

SRC_DIR_130 = Path("src").resolve()
if str(SRC_DIR_130) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_130))

from stage1_decode import build_decoder, decode_frame, decode_title, load_all
from stage1_hybrid import (
    CONDITION_GROUPS,
    STATUS_FEATURES,
    ConditionFeatures,
    build_hybrid_features,
    build_hybrid_model,
    lemmatize,
)
from stage1_invariant import mask_cyrillic

H15_DIR = Path("h15")
H15_DIR.mkdir(exist_ok=True)

# The frozen checkpoint must never be touched by anything in this section.
CHECKPOINT_130 = Path("h12") / "checkpoint" / "checkpoint_submission.csv"
_checkpoint_hash_before_130 = hashlib.sha256(CHECKPOINT_130.read_bytes()).hexdigest()

frames_130 = load_all()
decoder_130 = build_decoder(
    frames_130["test_s1"], frames_130["train_s2"], frames_130["test_s2"]
)

# The train split never entered the corpus, so it has nothing to exclude. The
# test split must exclude its own contribution -- see 130.1 for why.
dec_train_130 = decode_frame(frames_130["train_s1"], decoder_130, self_source=None)
dec_test_130 = decode_frame(frames_130["test_s1"], decoder_130, self_source="s1_test")
y_130 = frames_130["train_s1"]["label"].to_numpy()

example = next(d for d in dec_train_130 if d.subsection and not d.subsection_ambiguous)
print("a corrupted training title, decoded:")
print("  corrupted :", example.raw.splitlines()[-1])
print("  recovered :", example.subsection)
print("  guideline :", example.guideline or "(not recoverable)")""",
    ),
    (
        "markdown",
        "loo_md",
        """### 130.1 The trap: decodability is not label-neutral

The obvious way to build this feature set is to decode every row against the
full corpus and move on. That would have been a serious mistake, and the check
below is the reason the decoder carries a leave-one-out rule.

Whether a subsection line decodes at all is *strongly* correlated with the
label, because a subsection title that recurs across several guidelines tends to
be a generic one (`Хирургическое лечение`) while a title unique to one guideline
tends to name a restricted patient group. That is genuine signal. The trap is
that **a test title is itself in the matching corpus**, so a naive decoder
recovers far more of a test row than it ever could of a train row -- every
guideline line and 64% of subsection lines, against 64% and 19% on train. The
model would learn "did not decode ⇒ Special" on a feature that is close to
constant at inference time -- the exact failure mode section 129 was written to remove,
reintroduced through a different door.

The fix is to exclude each querying row's own contribution to the corpus. The
cell below quantifies both halves of this.""",
    ),
    (
        "code",
        "loo_check",
        """# ---- 130.1: the label correlation, and the leave-one-out repair ----
import numpy as np, pandas as pd

sub_any_130 = np.array([bool(d.subsection) for d in dec_train_130])
print("Special rate when the subsection line is ...")
print("  absent from the corpus : %.4f  (n=%d)"
      % (y_130[~sub_any_130].mean(), (~sub_any_130).sum()))
print("  present in the corpus  : %.4f  (n=%d)"
      % (y_130[sub_any_130].mean(), sub_any_130.sum()))
print("  overall                : %.4f" % y_130.mean())

# Naive decoding (no exclusion) versus leave-one-out, on the test split.
naive_test_130 = [decode_title(t, decoder_130) for t in frames_130["test_s1"]["title_text"]]

def _rates(rows):
    return {
        "guideline": np.mean([bool(d.guideline) for d in rows]),
        "age": np.mean([bool(d.age_category) for d in rows]),
        "subsection": np.mean([bool(d.subsection) for d in rows]),
    }

coverage_130 = pd.DataFrame(
    {
        "train": _rates(dec_train_130),
        "test (naive)": _rates(naive_test_130),
        "test (leave-one-out)": _rates(dec_test_130),
    }
).round(3)
print("\\ndecode rate by line role:")
print(coverage_130.to_string())
print("\\nmax train/test gap: naive %.3f -> leave-one-out %.3f"
      % ((coverage_130["train"] - coverage_130["test (naive)"]).abs().max(),
         (coverage_130["train"] - coverage_130["test (leave-one-out)"]).abs().max()))""",
    ),
    (
        "markdown",
        "loo_res",
        """The Special rate is **0.406** among rows whose subsection line is absent from
the corpus and **0.043** among rows where it is present -- a factor of nine. A
naive decoder hands the model that signal in a form that evaporates at
inference: it recovers the guideline line on 100% of test rows against 64% of
train rows, and the subsection line on 64% against 19%, for a worst-case gap of
0.450. (The naive subsection rate is 64% rather than 100% only because a test
title whose masked subsection collides with other corpus lines still resolves to
nothing when those candidates share no words.) Leave-one-out matching closes the
gap to 0.011 on the subsection line and 0.087 at worst across all roles, so the
feature means the same thing on both sides of the split. Everything downstream uses the leave-one-out decode,
including at inference time in `src/stage1_hybrid_build.py`.

**What survives decoding**, per line role, on train: the age line 100%, the
section hierarchy 88.5%, the guideline title 64.2%, and the final subsection
title only 19.3%. The subsection is both the most informative line and the least
recoverable, because 844 of 1,767 train subsections simply do not occur anywhere
in the readable corpus. No decoder can invent those -- that is a hard ceiling,
not an implementation shortfall.

Ambiguity is handled rather than discarded. When several clean lines share a
masked form, any word present in **all** candidates is in the true line whatever
the correct answer is, so those words are emitted as certain and the rest
dropped. That extracts sound information from ambiguous matches without
guessing.""",
    ),
    (
        "markdown",
        "features_md",
        """### 130.2 Feature design: union, not replacement

Decoding covers roughly half the rows, so a decoded-only model would be blind on
the other half. The decoded blocks are therefore **unioned with** section 129's
invariant features rather than replacing them, which means the feature space
degrades gracefully to exactly the H14 representation when nothing decodes.

| block | content |
|---|---|
| `invariant` | the full H14 union -- numeric skeleton + shape n-grams |
| `lex_subsection` | lemmatised word 1-2 grams of the decoded subsection line |
| `lex_context` | lemmatised unigrams of the decoded guideline + hierarchy |
| `conditions` | patient-specificity markers + decode-status indicators |

The `conditions` block is where domain knowledge enters. Section 7 of the brief
defines a Special subsection as one naming a restricted patient group, so the
block counts lemma hits against seven marker sets -- age, gender, severity,
form, comorbidity, refractoriness, and the `при ...`/`у пациентов с` framing
that introduces a restriction. Lemmatisation matters here: `тяжёлой`,
`тяжелая` and `тяжелый` are one marker, not three.""",
    ),
    (
        "code",
        "ablation",
        """# ---- 130.2: which block earns its place? ----
# Regenerate with:  python src/stage1_hybrid_eval.py   (a few minutes)
eval_130 = json.loads((H15_DIR / "stage1_hybrid_eval.json").read_text(encoding="utf-8"))

# Markers are counted over the subsection line and over the guideline+hierarchy
# context separately, because the two decode at very different rates (19% vs
# 64-89%) -- the context is where most marker hits actually come from.
_sub_lemmas = [set(lemmatize(d.subsection).split()) for d in dec_train_130]
_ctx_lemmas = [set(lemmatize(" ".join((d.guideline, *d.hierarchy))).split())
               for d in dec_train_130]
print("condition markers, Special rate where they fire (base %.3f):" % y_130.mean())
for group, lemmas in CONDITION_GROUPS.items():
    row = ["  %-12s" % group]
    for where, bags in (("subsection", _sub_lemmas), ("context", _ctx_lemmas)):
        hit = np.array([bool(b & lemmas) for b in bags])
        row.append("%s n=%-4d rate %s" % (
            where, hit.sum(), "%.3f" % y_130[hit].mean() if hit.sum() else "  -  "))
    print("  ".join(row))

print("\\nablation, 5-fold, threshold tuned in-fold (optimistic -- for ranking only):")
print(pd.DataFrame(eval_130["ablation"])[["config", "m1_at_050", "m1_tuned", "threshold"]]
      .round(4).to_string(index=False))""",
    ),
    (
        "markdown",
        "ablation_res",
        """Reading the ablation:

* `decoded_only` scores **0.6318** against `invariant_only`'s **0.7090**. The
  decoded text alone is *worse* than the shapes it was meant to improve on,
  which is exactly what a 19% subsection recovery rate predicts. This is the
  result that justifies the union design; had the blocks been swapped rather
  than combined, Stage 1 would have gone backwards.
* `lex_subsection` added alone is flat (0.7058). Added together with
  `lex_context` it reaches **0.7447** -- the guideline and hierarchy lines decode
  at 64% and 89%, so they supply the context that makes a sparsely recovered
  subsection interpretable. This matches the brief's warning in section 7 not to
  discard the hierarchy.
* The full union reaches **0.7768**, above every proper subset.""",
    ),
    (
        "markdown",
        "status_md",
        """### 130.3 Auditing the riskiest block

The `conditions` block contains four *decode-status* features -- coverage,
whether the subsection decoded, whether it was ambiguous, how many lemmas came
back. These describe **whether** a line decoded rather than what it said, and
given the 0.406-versus-0.043 label correlation above they are the features most
likely to be a shortcut rather than signal. They are only defensible if they
mean the same thing on both splits, so they get audited separately rather than
being taken on trust.""",
    ),
    (
        "code",
        "status_audit",
        """# ---- 130.3: do the decode-status features transfer? ----
_status = ConditionFeatures(mode="status")
A_130 = _status.transform(np.asarray(dec_train_130, dtype=object))
B_130 = _status.transform(np.asarray(dec_test_130, dtype=object))

drift_130 = []
for i, name in enumerate(_status.get_feature_names_out()):
    a, b = A_130[:, i], B_130[:, i]
    pooled_sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
    drift_130.append({"feature": name, "train_mean": a.mean(), "test_mean": b.mean(),
                      "drift_sd": abs(a.mean() - b.mean()) / pooled_sd})
print(pd.DataFrame(drift_130).round(3).to_string(index=False))

# Contribution decomposition, measured separately in src/stage1_hybrid_eval.py.
print("\\ncontribution of each half of the conditions block (5-fold, tuned in-fold):")
for label, score in [("no conditions block", 0.7590), ("linguistic markers only", 0.7768),
                     ("decode-status only", 0.7842), ("both (shipped)", 0.7832)]:
    print("  %-26s M1 = %.4f" % (label, score))""",
    ),
    (
        "markdown",
        "status_res",
        """Both halves carry independent weight: the linguistic markers lift M1 from
0.7590 to 0.7768 and the decode-status features to 0.7842, with the combination
at 0.7832 -- statistically indistinguishable from status-only, and kept because
the markers are interpretable and directly defensible under the medical-text
criterion.

The status features drift by at most **0.141 SD** between splits (`decode_coverage`)
and under 0.04 SD for the other three, so they transfer. What they actually
encode is *recurrence*: a subsection title attested elsewhere in the corpus is a
generic one. That is a real property of the data, computed identically on both
sides.

One honest caveat: recurrence is a transductive feature, so it depends on the
composition of the test set it is computed against. It is stable here because
train and test are stratified samples of the same guideline pool, but a hidden
evaluation set drawn from different guidelines would weaken it. The invariant
block underneath is unaffected either way.""",
    ),
    (
        "markdown",
        "nested_md",
        """### 130.4 Honest comparison against H14

Every number above tunes its threshold on the same folds it reports, which is
optimistic and fine for ranking configurations but not for deciding what to
ship. The shipping estimate uses the same nested protocol as section 129 --
threshold chosen on inner folds, scored on a held-out outer fold -- with the
same seed and the same folds, so the two sections' numbers are directly
comparable.""",
    ),
    (
        "code",
        "nested",
        """# ---- 130.4: nested CV, H14 versus H15 ----
n14 = json.loads((Path("h14") / "stage1_tuning.json").read_text(encoding="utf-8"))["nested_cv"]
n15 = eval_130["nested_cv"]

print(pd.DataFrame({
    "H14 invariant": {"pooled M1": n14["pooled_m1"], "per-fold mean": n14["mean_fold_m1"],
                      "per-fold SD": n14["std_fold_m1"], "positive rate": n14["pooled_posrate"]},
    "H15 hybrid": {"pooled M1": n15["pooled_m1"], "per-fold mean": n15["mean_fold_m1"],
                   "per-fold SD": n15["std_fold_m1"], "positive rate": n15["pooled_posrate"]},
}).round(4).to_string())

delta = n15["pooled_m1"] - n14["pooled_m1"]
print("\\nper-outer-fold M1")
print("  H14: %s" % ["%.4f" % s for s in n14["fold_scores"]])
print("  H15: %s" % ["%.4f" % s for s in n15["fold_scores"]])
wins = sum(b > a for a, b in zip(n14["fold_scores"], n15["fold_scores"]))
print("  H15 ahead on %d of %d folds" % (wins, len(n15["fold_scores"])))
print("\\ndelta %+.4f M1  =  %+.2f metric points" % (delta, 70 * 0.3 * delta))""",
    ),
    (
        "markdown",
        "final_md",
        """### 130.5 Final build

The build refits on all 1,767 training rows at the median inner-fold threshold,
runs the **same leave-one-out decode** at inference, and reapplies the verified
Stage-2-title rule from section 129.6 unchanged. Stage 2 is copied byte for byte
from the production predictions, and the frozen checkpoint is hashed before and
after.""",
    ),
    (
        "code",
        "build",
        """# ---- 130.5: load the built submission and re-verify it ----
# Regenerate with:  python src/stage1_hybrid_build.py
manifest_130 = json.loads(
    (H15_DIR / "stage1_hybrid_manifest.json").read_text(encoding="utf-8")
)
sub_130 = pd.read_csv(H15_DIR / "submission_stage1_hybrid.csv")
stage2_frozen = pd.read_csv(Path("final") / "predictions" / "stage2_predictions.csv")

s1_130 = sub_130[sub_130["stage"] == 1]
s2_130 = sub_130[sub_130["stage"] == 2]
assert len(sub_130) == 615, "submission must have 615 rows"
assert sub_130["label"].isin([0, 1]).all(), "labels must be binary"
assert list(s1_130["id"]) == list(frames_130["test_s1"]["id"]), "stage 1 row order"
assert list(s2_130["label"]) == list(stage2_frozen["label"]), "stage 2 must be untouched"

print("threshold            %.2f" % manifest_130["threshold"])
print("stage 1 positive rate %.4f   (train prior %.4f)"
      % (manifest_130["test_positive_rate"], manifest_130["train_positive_rate"]))
print("agreement with H14    %.4f" % manifest_130["agreement_with_h14"])
print("agreement with prod   %.4f" % manifest_130["agreement_with_production"])
print("stage-2 title rule    %d matched, precision %.4f, %d flipped"
      % (manifest_130["stage2_title_rule"]["n_matched"],
         manifest_130["stage2_title_rule"]["precision"],
         manifest_130["stage2_title_rule"]["n_flipped"]))

_after_130 = hashlib.sha256(CHECKPOINT_130.read_bytes()).hexdigest()
assert _after_130 == _checkpoint_hash_before_130, "checkpoint modified -- MUST NEVER HAPPEN"
print("\\ncheckpoint intact:", _after_130[:16])
print("submission sha256:", manifest_130["submission_sha256"][:16])""",
    ),
    (
        "markdown",
        "close",
        """### H15 summary

**What was done.** The corruption was partially inverted by masked-line matching
against the readable competition inputs, and the recovered Russian text was used
to build lexical and condition-marker features on top of the H14 invariant
representation.

**Result.** Nested-CV M1 **0.7111 -> 0.7527**, an improvement of **+0.0416 M1 =
+0.87 metric points**, ahead of H14 on 4 of 5 outer folds. The Stage 1 positive
rate moves from 0.1629 to 0.1787 against a training prior of 0.2168. Stage 2 is
untouched, so the expected total moves from roughly 50.7 to roughly **51.6**.

**How much confidence this deserves.** Less than the point estimate suggests.
The per-fold SD is 0.0593 and the two models' fold scores are correlated, so
while the direction is consistent the magnitude is not tightly pinned. The gain
also rests partly on a transductive recurrence signal whose strength depends on
the test set sharing a guideline pool with train. Both submissions are kept:
`h14/submission_stage1_rebuild.csv` is the conservative choice that depends on
nothing but corruption-invariant structure, and
`h15/submission_stage1_hybrid.csv` is the stronger estimate.

**The hard ceiling.** 844 of 1,767 training subsection lines occur nowhere in
the readable corpus and are permanently lost -- 47.8% of the most informative
line in the input. A Stage 1 model trained on uncorrupted data would very likely
beat everything in this notebook; nothing here recovers that, and no method
applied to this data can.

**What is still open.** Stage 2 remains frozen at CV M2 0.7024, and it carries
70% of the metric weight. Five structurally different Stage 2 feature sets all
landed within 0.007 of each other, which suggests the classical lexical approach
has saturated there and that a Russian medical transformer -- the one model
family this notebook never tested -- is the remaining lever worth pulling.""",
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    before = len(nb["cells"])
    nb["cells"] = [c for c in nb["cells"] if not str(c.get("id", "")).startswith(PREFIX)]
    removed = before - len(nb["cells"])

    for cell_type, cid, source in CELLS:
        cell = {
            "cell_type": cell_type,
            "id": "%s_%s" % (PREFIX, cid),
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("removed %d stale H15 cells, appended %d, total now %d"
          % (removed, len(CELLS), len(nb["cells"])))


if __name__ == "__main__":
    main()
