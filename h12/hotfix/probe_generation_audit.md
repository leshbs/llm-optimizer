# Probe Generation Audit

**Overall: PASS**

## Notebook Provenance

| filename | dataframe_variable | generating_cell_id | cell_positional_index | file_mtime_utc |
|---|---|---|---|---|
| probe_stage2_only.csv | probe_a_df | sec77code | 338 | 2026-09-15T01:25:53Z |
| probe_stage1_only.csv | probe_b_df | sec78code | 341 | 2026-09-15T01:25:53Z |
| probe_trivial.csv | probe_c_df | sec79code | 344 | 2026-09-15T01:25:53Z |
| probe_llm_stage1.csv | probe_llm_s1_df | sec95code | 401 | 2026-09-15T12:14:36Z |
| probe_llm_stage2.csv | probe_llm_s2_df | sec96code | 404 | 2026-09-15T12:14:36Z |

execution_count is None for every cell in this notebook (never run via a live Jupyter kernel -- only via out-of-band verification harnesses that exec() extracted cell source). cell_positional_index (list order in baseline.ipynb) is used as the execution-order proxy.


**Overwrite incident (disclosed):** h12/probes/ was found empty prior to this audit (probe_llm_stage1.csv/probe_llm_stage2.csv deleted by an earlier verification script's rm -rf h12) and was restored by re-running H12 Part 1's unchanged cells 95.0/96.0; restored checksums matched the originally-documented ones exactly. See file_mtime_utc above for the real restoration timestamps.


`h11_build_probe`/`h12_build_probe` confirmed to construct a fresh DataFrame via `pd.concat` (not mutate a shared one): True.


## Issues found

None. No stale-variable reuse, no filename/manifest mismatch, all 5 probe cells write a freshly-constructed DataFrame to their own uniquely-named output file.
