# Handoff: Milestone 0 → Milestone 1

**Handoff date:** 2026-07-14
**Purpose:** Start a fresh session on Milestone 1 (Characterisation suite and historical baseline quarantine) without re-litigating Milestone 0. Paste the kickoff prompt in §4 into a new session.

---

## 1. Milestone 0 status: CLOSED for M1 purposes

Decision Gate 0 (`docs/audit/decisions.md`) is **not fully closed**, but every row that blocks Milestone 1
is resolved:

| Row | Status | Affects M1? |
|---|---|---|
| U1 — bundled fixture suitability | `approved` | Yes — governs M1 fixture quarantine |
| U7 — legacy regression baseline | `approved` | Yes — governs M1 quarantine of `irm_metrics.csv` |
| Q8 — validation fixtures | `approved` | Yes |
| U4/Q6 — drainage | `approved` | No (M10) |
| Q2 — canonical `WaterCube` input object | `approved` | No (M2/M3) |
| Q4 — DCI scope | `approved` | No (M11) |
| Q5 — legacy compatibility policy | `approved` | No (M1/M7/M8 — informs M1 quarantine framing but not blocking) |
| Q10 — predecessor history | `approved` | No (M13) |
| **U2/Q1 — validity denominator + seasonal estimator** | `pending sign-off only` (evidence complete) | **No** — affected milestones are M3/M5/M9/M10, not M1 |
| Q7 — HY algorithm authority | `pending` (external `hydroseason`) | No (M9, blocked on external dep) |
| Q9 — config hash golden rules | `pending` (test infra) | No (M2/M7) |

**Conclusion: Milestone 1 is not blocked.** Do not re-open U1–U7/Q1–Q10 in the M1 session; treat
`docs/audit/decisions.md` as the source of truth and only re-litigate if new evidence contradicts it.

---

## 2. What M0 already built that M1 must extend, not recreate

M0 pre-built part of M1's own deliverable list while gathering evidence. The M1 session **must inspect
these before writing new code**, since redoing them would duplicate work or regress the seasonal-MNAR
finding:

| Already exists | Path | M1 relevance |
|---|---|---|
| Fixture characterisation tests | `tests/contracts/test_fixture_characterisation.py` (14 tests, all passing) | This is the exact file M1's plan calls for creating — **extend, don't overwrite** |
| Read-only fixture inspector | `tests/contracts/fixture_inspector.py` | Helper module backing the above; extend with same read-only-no-mutation discipline |
| Legacy CSV quarantine evidence | `docs/audit/evidence/regression_baseline.md` (U7 `approved`) | Confirms which baseline columns are legacy-smoke-only vs excluded entirely |
| Bundled fixture inventory | `docs/audit/evidence/fixture_inventory.md` (U1 `approved`) | Confirms `tests/wmask_ts.nc` is legacy-smoke only; Tier A synthetic + Tier C real-zarr fixtures required for v1.2 contract tests |
| Real Tier-C fixtures already in repo | `data/wofs_monthly_masks_1986_2026.zarr` (SHA-256 of `.zmetadata`: `c69f7e8b0706...36e790`), `data/fitzroy_kimberley_drainage.gpkg` (SHA-256: `004442d0a65a...9980a46dc3b3a`), `data/fitzroy_kimberley_aoi.geojson` | Real validation-catchment data for Tier C fixtures — no need to source new data for M1 |
| Seasonal MNAR regression test | `tests/contracts/test_fixture_characterisation.py::test_fitzroy_zarr_missingness_is_seasonal_mnar` | Locks a finding relevant to future M3/M5/M9/M10 metric kernels — do not delete when reorganising, just relocate if needed |

**Action for M1 session:** run `python -m pytest tests/contracts -v` first to confirm the 14 existing tests
still pass, then build M1's additional scope (legacy quarantine, `tests/legacy/`, `tests/fixtures/README.md`,
`docs/testing.md`, expanded analytic fixtures) around what's already there.

---

## 3. Forward note for later milestones (not M1's job, but don't lose it)

`docs/audit/evidence/validity_reliability_report.md` §4 found that missingness in the real water-mask cube
is **seasonal MNAR** (coverage lowest exactly when wetness peaks — Jan–Mar monsoon), causing naive pooled
occurrence ratios to under-estimate wetness by ~6.6% relative on this catchment. The recommended fix
(season-stratified estimator: per-calendar-month ratio, equal-weighted across 12 months) is only pending a
sign-off conversation on wording, not on the underlying evidence. **This is an algorithm-design requirement
for Milestones 3, 5, 9, and 10** (occurrence/RA/recurrence/hydroperiod/dry-down kernels) — carry it forward
when those milestones start; it is out of scope for M1.

---

## 4. Kickoff prompt for the new Milestone 1 session

Paste the following into a new session to start Milestone 1:

```text
You are implementing HydroFragments v1.2 Milestone 1 only: Characterisation suite and historical baseline
quarantine. Use test-driven-development for every implementation change and verification-before-completion
before claiming completion.

Before writing anything, read docs/audit/handoff_m0_to_m1.md in full, then docs/audit/decisions.md and
docs/audit/implementation_plan.md Milestone 1 section. Decision Gate 0 is not blocking Milestone 1 — do not
re-litigate U1-U7/Q1-Q10; treat decisions.md as settled for this milestone's purposes.

Milestone 0 already created tests/contracts/test_fixture_characterisation.py and
tests/contracts/fixture_inspector.py with 14 passing read-only tests covering the legacy tests/wmask_ts.nc
fixture, tests/rcor_extent.shp, tests/results_iRiverMetrics/metrics/irm_metrics.csv, and real Tier-C data
at data/wofs_monthly_masks_1986_2026.zarr and data/fitzroy_kimberley_drainage.gpkg. Run
`python -m pytest tests/contracts -v` first to confirm these still pass, then extend this structure rather
than recreating it.

Your scope for Milestone 1:
- Quarantine tests/results_iRiverMetrics/metrics/irm_metrics.csv so it cannot be used as a v1.2 correctness
  oracle (U7 is approved: retire it as oracle, retain only for approved low-level kernel smoke).
- Create tests/legacy/test_legacy_kernels.py for legacy kernel characterisation only (unchanged low-level
  areas, perimeters, skeleton paths, EDT behavior) — never validate v1.2 occurrence or schema against
  legacy output.
- Add an explicit canonical test proving the legacy CSV is rejected/fails when used as a v1.2 correctness
  baseline.
- Add analytic fixtures for diagonal connectivity, one-pixel noise, empty/full masks, holes, long bars, and
  components crossing future chunk boundaries (2-chunk and 4-chunk crossing).
- Create tests/fixtures/README.md documenting fixture provenance and tiers (Tier A synthetic, Tier C real).
- Create docs/testing.md describing the test suite structure and quarantine rules.
- Modify tests/conftest.py, tests/test_unit_metrics.py, tests/test_integration.py as needed to reflect the
  quarantine.

Acceptance criteria (from implementation_plan.md Milestone 1):
- Fast and slow suites collect successfully.
- Historical CSV limited to smoke comparisons for approved invariant kernels only.
- No canonical test requires dropped metric columns or naive pp_mean_% equivalence.
- Tiny analytic fixtures cover diagonal connectivity, one-pixel noise, empty/full masks, holes, long bars,
  and chunk-crossing components.

Do not implement new v1.2 numerical kernels. Do not touch drainage, connectivity, CUDA, or validity-
denominator implementation code — those are later milestones. Stop and report back if you find a decision
in decisions.md that appears to block this milestone's scope; do not silently override it.
```
