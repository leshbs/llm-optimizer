# H12 Patch Summary -- Decision Router Recovery (Part 7)

## Original H12 assumption (Part 3)

H12 Part 3's `decision_tree.json` (106.0) computed a branch directly from whatever was in 104.0's
manual leaderboard score table, with no check that the probes producing those scores were
themselves valid, and no check on whether the platform's displayed score could be trusted at all.
In practice this was harmless so far only because every score has stayed `None` (Part 3 already
degrades gracefully to `PENDING_INSUFFICIENT_DATA`) -- but the routing logic itself had no explicit
gate stopping it from acting on a real-but-corrupted score, which is exactly the gap this patch closes.

## Integrity findings (Part 5)

`probe_integrity_gate.json`: **PROBE_VALID = True**. All 5
components passed (hash audit, structural validation, stage-difference-vs-expectation, probe
generation audit, impossible-score detector). One real incident was found and disclosed during
that audit: `h12/probes/` was emptied by an earlier verification script's over-broad cleanup and
was restored via H12 Part 1's own unchanged cells, with checksums verified identical.

## Platform findings (Part 6)

`platform_diagnosis.json`: **platform_state = UNKNOWN_PLATFORM_BEHAVIOR**,
`n_scored_probes = 0`. Zero of the 5 named probes (and
the H12 Part 6 canary) have been submitted to the leaderboard yet, so platform behavior on a
differentiated probe file has never actually been observed -- this is an absence of evidence, not
evidence that the platform is behaving correctly.

## Patched decision (Part 7)

`decision_router_guard.json`: **guard_state = WAIT_FOR_PLATFORM**
(Probes are valid, but H12 Part 6's platform_diagnosis reports platform_state='UNKNOWN_PLATFORM_BEHAVIOR', not PLATFORM_OK/PLATFORM_SCORE_VIEW_ONLY: No probe has been submitted and scored yet, so the platform's behavior on differentiated probe files has never been exercised -- absence of evidence is not evidence of correctness.). Because probes are valid but the platform is unverified,
125.0's `updated_decision_matrix.csv` reports all 4 rows as `BLOCKED` (not `PENDING` -- a stronger,
gate-enforced state), and 126.0 activates **no branch** (`activated_branch = None`). This is a
strictly more conservative outcome than Part 3's `PENDING_INSUFFICIENT_DATA`: Part 3 would resolve
to a real branch the moment scores are entered in 104.0, with no check on where those scores came
from; this patch additionally requires Part 5 and Part 6 to both actively confirm validity first.

## Remaining blockers

1. **Zero real probe scores.** None of Probe A, Probe B, Probe C, the LLM-Stage1 probe, or the
   LLM-Stage2 probe has been submitted.
2. **Platform behavior unverified.** The H12 Part 6 canary (`canary_extreme_floor.csv`) has not
   been uploaded, so there is no evidence yet that the platform scores files by their actual content
   rather than showing a cached "best score".
3. **127.0's concrete next action**: submit the 5 probes + the canary, record every real score in
   `h12/platform/submission_history.csv`, then re-run H12 Parts 6 and 7 in that order -- Part 6 will
   resolve `platform_state`, and Part 7's guard will then either move to `CONTINUE` (if
   `PLATFORM_OK`/`PLATFORM_SCORE_VIEW_ONLY`) or `STOP` (if the canary or a probe reveals a genuine
   platform/generation bug). No code changes are required for either outcome.
