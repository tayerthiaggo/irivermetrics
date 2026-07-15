# Regression baseline evidence (U7)

**Captured:** 2026-07-10T21:10:00+08:00  
**Method:** read-only CSV inspection + `tests/conftest.py` / `tests/test_integration.py` review

## Reference artifact

| Field | Value |
|---|---|
| Path | `tests/results_iRiverMetrics/metrics/irm_metrics.csv` |
| SHA-256 | `83b622d43c911f6b0f2dc9c39bf57de686a6c2cbba9becae5a2f78e20d013370` |
| Rows | 441 (= 63 dates × 7 sections) |
| Date range | 2018-01-01 → 2020-12-16 |
| Sections | 0–6 |

## Schema (legacy wide CSV)

Columns include dropped/v1.2-forbidden metrics:

| Column | v1.2 status |
|---|---|
| `PF` | dropped (circular) — **present** |
| `PFL` | dropped naming variant — **absent** (tests map `PFL`→`PLF` expecting current code) |
| `AWMPA`, `AWMPL`, `AWMPW` | dropped/replaced — **present** |
| `pp_mean_%` | not occurrence frequency — **present** |
| `LPSEC` | core but often empty when `section_length` missing |

## Provenance / correctness assessment

| Finding | Evidence |
|---|---|
| Naive persistence denominator | `pp_mean_%` is **constant per section** across all 63 dates (one unique value per section) — consistent with full-series mean persistence, **not** `water_obs/valid_obs` |
| Matches legacy code path | `calculate_pixel_persistence()` divides by total timesteps (`spec_compliance.md` A5) |
| Integration test expectation | `test_integration.py` regression compares `wet_area_km2`, `APSEC`, `pp_mean_%` within 5% of reference |
| Fixture path bug | `conftest.py` still points at missing `results_ecofragments/metrics/ecof_metrics.csv` — regression test **broken** even before science review |

## U7 disposition (pending maintainer approval)

| Option | Recommendation | Rationale |
|---|---|---|
| **Retire as v1.2 correctness oracle** | **Recommended** | Would canonise wrong denominators and dropped metrics |
| Keep as historical smoke only | Acceptable with explicit exclusions | Kernel-level area/APSEC checks only; never occurrence/RA/schema |
| Regenerate reference from v1.2 | Future | After validity policy closed and canonical pipeline exists |

**Consequence if wrong:** v1.2 tests fail on correct science, or wrong science ships as “regression passed.”

## Milestone 1 actions (not executed in M0)

- Quarantine CSV from canonical contract tests
- Fix `conftest.py` path or remove broken regression hook
- Add explicit test that canonical v1.2 path **rejects** legacy CSV as oracle
