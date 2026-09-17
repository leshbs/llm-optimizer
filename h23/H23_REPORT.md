# H23: per-title calibration and title-dominance reduction

**Recommendation: KEEP CURRENT MODEL.** Both structural directions were implemented
in full and both fail. The reason is now proven rather than suspected.

Baseline: config G, offline M2 **0.7023868248**, FN 81, FP 42, frozen 5 folds.
Gate: delta > 0 **and** >= 4/5 folds improved. Offline-vs-offline only; nothing submitted.

## Direction A -- reduce title dominance

| Variant | CV M2 | Delta | FN | FP | Between-title var | Folds + |
|---|---|---|---|---|---|---|
| title_weight=1.00 | 0.702387 | +0.000000 | 81 | 42 | 0.750 | 0/5 |
| title_weight=0.75 | 0.688197 | -0.014190 | 80 | 46 | 0.730 | 1/5 |
| title_weight=0.50 | 0.653260 | -0.049126 | 91 | 47 | 0.700 | 1/5 |
| title_weight=0.25 | 0.579050 | -0.123337 | 116 | 48 | 0.665 | 0/5 |
| title_weight=0.10 | 0.532903 | -0.169484 | 128 | 51 | 0.655 | 0/5 |
| title_weight=0.00 | 0.532903 | -0.169484 | 128 | 51 | 0.653 | 0/5 |

**Failed, monotonically.** Removing the title block costs 0.169 M2 and drives FN
from 81 to 128.

The important column is the second-to-last. **Dropping the title TF-IDF entirely
only moves between-title variance from 0.750 to 0.653.** Title identity is not
carried mainly by the title block -- the narrative TF-IDF leaks it (protocols
under one guideline share vocabulary) and the contradiction features are computed
*from* the title. Title dominance cannot be removed by removing the title text;
the model reconstructs it from everything else, and loses real signal in the process.

## Direction B -- per-title calibration

Symmetric centring, nested and fold-safe (per-title offsets from inner-CV
probabilities on the training fold only):

| Variant | CV M2 | Delta | FN | FP | Folds + |
|---|---|---|---|---|---|
| calibration alpha=0.00 | 0.702387 | +0.000000 | 81 | 42 | 0/5 |
| calibration alpha=0.25 | 0.688197 | -0.014190 | 80 | 46 | 2/5 |
| calibration alpha=0.50 | 0.682446 | -0.019940 | 79 | 48 | 2/5 |
| calibration alpha=0.75 | 0.664052 | -0.038335 | 81 | 51 | 2/5 |
| calibration alpha=1.00 | 0.599331 | -0.103056 | 92 | 59 | 0/5 |

At alpha=1.0 false negatives get *worse* (81 -> 92), because centring is
symmetric: it lifts low-prevalence titles and pushes high-prevalence ones down,
and the high-prevalence titles were already fine. That failure mode implies a
one-sided form -- lift only, never push down:

| Variant | CV M2 | Delta | FN | FP | Folds + |
|---|---|---|---|---|---|
| one-sided alpha=0.25 | 0.684975 | -0.017412 | 75 | 50 | 1/5 |
| one-sided alpha=0.50 | 0.688601 | -0.013786 | 69 | 53 | 1/5 |
| one-sided alpha=0.75 | 0.696932 | -0.005455 | 58 | 58 | 2/5 |
| one-sided alpha=1.00 | 0.669595 | -0.032791 | 50 | 69 | 2/5 |
| one-sided alpha=1.50 | 0.351154 | -0.351233 | 29 | 146 | 0/5 |

Better, and it moves false negatives a long way (81 -> 50 at alpha=1.0), but
every point of FN reduction is paid for roughly one-for-one in FP, so M2 never
recovers.

## Why no calibration can work: within-title AUC

Any calibration, threshold or ranking scheme can only reorder rows **within** a
title. So the question is whether the model's scores carry within-title signal.

| | value |
|---|---|
| titles with both classes present | 24 |
| mean within-title AUC | 0.6068 |
| positive-weighted within-title AUC | 0.6803 |
| titles with AUC < 0.50 | 9 / 24 |
| **collapsed titles (7 titles, 36 positives)** | **mean AUC 0.3909, median 0.3143** |

**In the titles that produce the false negatives, the model ranks positives
worse than chance** (0.39, with individual titles at 0.00, 0.17, 0.22, 0.31).
Lifting such a title promotes the wrong rows preferentially. This closes the
direction: the information needed is not in the scores, so no post-hoc transform
can extract it.

## Selective calibration -- the last variant the diagnostic allows

Lift only titles that are both below the global mean *and* rankable on the
training fold (within-title AUC above a gate, estimated fold-safely):

| Variant | CV M2 | Delta | FN | FP | Rows lifted | Folds + |
|---|---|---|---|---|---|---|
| auc_gate=0.55 alpha=0.50 | 0.692046 | -0.010341 | 74 | 49 | 166 | 1/5 |
| auc_gate=0.55 alpha=0.75 | 0.700030 | -0.002357 | 68 | 51 | 166 | 3/5 |
| auc_gate=0.55 alpha=1.00 | 0.703520 | +0.001133 | 62 | 54 | 166 | 3/5 |
| auc_gate=0.60 alpha=0.50 | 0.696341 | -0.006046 | 74 | 48 | 152 | 2/5 |
| auc_gate=0.60 alpha=0.75 | 0.701590 | -0.000797 | 69 | 50 | 152 | 3/5 |
| auc_gate=0.60 alpha=1.00 | 0.702379 | -0.000008 | 64 | 53 | 152 | 2/5 |
| auc_gate=0.70 alpha=0.50 | 0.695067 | -0.007320 | 76 | 47 | 127 | 1/5 |
| auc_gate=0.70 alpha=0.75 | 0.694820 | -0.007567 | 73 | 49 | 127 | 3/5 |
| auc_gate=0.70 alpha=1.00 | 0.696054 | -0.006333 | 71 | 50 | 127 | 3/5 |

One variant crosses zero: `auc_gate=0.55, alpha=1.00` at **+0.001133** with FN
81 -> 62. It fails the >= 4/5 fold gate (3/5), and a seed-stability check settles it:

| inner-CV seed | M2 | delta |
|---|---|---|
| 42 | 0.703520 | +0.001133 |
| 1 | 0.681049 | -0.021337 |
| 7 | 0.695269 | -0.007118 |
| 13 | 0.687013 | -0.015374 |
| 99 | 0.707123 | +0.004737 |

**mean delta -0.007592, sd 0.010930, positive in 2 of 5 seeds.** The gain was an
artifact of the inner-CV seed, not an effect.

## Conclusion

Both directions are now closed with evidence rather than opinion:

* **title dominance is not removable** -- it survives deletion of the title block
  because other feature groups encode it;
* **per-title calibration cannot help** -- within the relevant titles the model's
  ranking is worse than chance, so there is nothing correct to promote.

Combined with H19-H22, every post-hoc and feature-level route has now been tested
and rejected. The remaining false negatives require a model that can genuinely
match a title's stated condition against patient evidence *within* a title --
structured condition matching or a semantic model, not another lexical feature or
another calibration of the same scores.

Production artifacts are untouched: Stage 1 remains H17, Stage 2 remains config G
at threshold 0.50, submission ids and ordering unchanged.
