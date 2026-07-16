# HydroFragments v1.2 — Decision Gate 0 register

**Opened:** 2026-07-10T21:10:00+08:00  
**Gate status:** **OPEN** — rows marked `pending` lack closing evidence or approval date.  
**Rule:** No production metric implementation until every **blocking** row is `approved`.

**Default owner:** Thiaggo de Castro Tayer (project maintainer) unless noted.

---

## Unverified load-bearing facts (U1–U4, U7)

### U1 — Bundled fixture suitability (`tests/wmask_ts.nc`)

| Field | Value |
|---|---|
| **Decision** | Keep `wmask_ts.nc` as **legacy smoke only**; require **synthetic analytic fixtures + real `water_cube.zarr`** for v1.2 contract tests (Q8 Tier A+C). |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/fixture_inventory.md`; `tests/contracts/test_fixture_characterisation.py` (6 passed 2026-07-10) |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-11 — maintainer: “U1 - correct” |
| **Consequence if wrong** | Contract tests validate wrong CRS/sentinels/cadence; dry-down/HY work proceeds on inadequate fixture |
| **Affected milestones** | M1, M3, M5, M9 |

### U2 — Valid-observation denominator (`observed` semantics)

| Field | Value |
|---|---|
| **Decision** | **Approved:** (1) Cross-sectional denominator = P-native (`water_mask ∈ {0,1}`, excluding unobserved `-1` and outside-AOI `-2`) + mandatory `observed_frac_of_aoi` diagnostic + `low_confidence` flag below 0.70 coverage. (2) **Any temporal aggregate** (occurrence, RA, recurrence, hydroperiod, dry-down anchors) **must use a season-stratified estimator** (per-calendar-month P-native ratio, equal-weighted across the 12 months), not a naive pooled ratio. P-provenance deferred. |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/upstream_validity_contract.md`; `docs/audit/evidence/validity_reliability_report.md` (§4 seasonal MNAR); `docs/audit/evidence/validity_reliability_per_month.csv`; `docs/audit/evidence/validity_reliability_by_calendar_month.csv`; regression tests `tests/contracts/test_fixture_characterisation.py::test_fitzroy_zarr_missingness_is_seasonal_mnar` (14 passed 2026-07-14); real cube `data/wofs_monthly_masks_1986_2026.zarr` (SHA-256 `c69f7e8b0706...36e790`) |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 |
| **Consequence if wrong** | Occurrence, RA, zones, recurrence, hydroperiod all shift; reliability of observed vs filled data is hidden from users; a naive (non-stratified) implementation would systematically under-report wetness/occurrence, biased toward under-claiming ecological water availability |
| **Affected milestones** | M3, M5, M9, M10 — item (2) is an **algorithm-design requirement** for these milestones' metric kernels, not just an input-filtering policy |

### U3 — Monthly compositing ownership / dual-composite availability

| Field | Value |
|---|---|
| **Decision** | `water_cube.zarr` is **monthly single product**; HydroFragments composites **only** when sub-monthly input supplied. A single-composite extent-contraction summary may be possible later, but the **dual-composite dry-down sensitivity / V3 paper claim** is blocked without raw observations or both `max_water` and `median` monthly products. |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/upstream_validity_contract.md`; `WaterMask-TSFill/docs/outputs.md` |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 — maintainer approved |
| **Consequence if wrong** | V3/paper composite-sensitivity leg impossible; dry-down milestone wastes schedule or overclaims |
| **Affected milestones** | M4, M9, M13 (publication) |

### U4 — Drainage / centreline availability (Q6)

| Field | Value |
|---|---|
| **Decision** | Maintainer supplied basis drainage (`data/fitzroy_kimberley_drainage.gpkg`): AHGF-style extract, 291 `MultiLineString` reaches, EPSG:3577 matching the water-mask grid exactly, complete `From_Node`/`To_Node`/`NextDownID` topology (0 nulls). Satisfies the v1.2 minimum drainage contract for the Fitzroy validation catchment. Proxy-channel research fallback remains documented for future no-drainage catchments but is not needed here. |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/drainage_inventory.md` (real dataset section, SHA-256 `004442d0a65a...9980a46dc3b3a`) |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 — real drainage dataset supplied and validated against grid CRS/extent and topology completeness |
| **Consequence if wrong** | Wet-derived proxy `L_ref` undermines fixed-denominator thesis; Zone 1/gap/RC/TCF ship without input or validation |
| **Affected milestones** | M10, M11; **LPSEC excluded** from v1.2.0 core per adversarial synthesis 2 (independent of drainage availability) |

### U7 — Legacy regression baseline (`irm_metrics.csv`)

| Field | Value |
|---|---|
| **Decision** | **Retire** as v1.2 correctness oracle; allow **historical smoke** for selected kernel columns only with explicit exclusions |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/regression_baseline.md` |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-11 — maintainer: “U7 - agreed” |
| **Consequence if wrong** | Test suite enforces naive `pp_mean_%` and dropped metrics as “correct” |
| **Affected milestones** | M1, M3+ |

---

## Spec compliance questions (Q1–Q10)

### Q1 — Validity semantics (same as U2)

| Field | Value |
|---|---|
| **Decision** | See **U2** — linked |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/upstream_validity_contract.md`; `docs/audit/evidence/validity_reliability_report.md` (incl. §4 seasonal MNAR) |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 |
| **Consequence if wrong** | Same as U2 |
| **Affected milestones** | M3, M5 |

### Q2 — Canonical input object

| Field | Value |
|---|---|
| **Decision** | `WaterCube` domain object via `open_water_cube()` returning aligned `water`, `valid_obs`, optional provenance; legacy `calculate_metrics(da_wmask)` facade only |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/implementation_plan.md` §3; `spec_compliance.md` Q2; adversarial synthesis §3 item 9 |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 — maintainer confirmed |
| **Consequence if wrong** | Adapter churn; valid layer omitted again |
| **Affected milestones** | M2, M3 |

### Q3 — Monthly ownership (same as U3)

| Field | Value |
|---|---|
| **Decision** | See **U3** |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/upstream_validity_contract.md` |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 |
| **Consequence if wrong** | Same as U3 |
| **Affected milestones** | M4, M9 |

### Q4 — DCI scope

| Field | Value |
|---|---|
| **Decision (proposed)** | **Citation + conceptual positioning required**; runtime DCI **optional** gated on `riverconn`/Conefor parity (V6) |
| **Status** | `approved` per audit convergence — implementation still gated |
| **Evidence artifact** | `scientific_metrics_audit.md` §8; `adversarial_synthesis.md` §4 deferrals; `implementation_plan.md` Milestone 11 |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-10 (audit consensus; no separate maintainer signature on file) |
| **Consequence if wrong** | Connectivity scope balloons; RC mistaken for renamed DCI |
| **Affected milestones** | M11 only |

### Q5 — Legacy compatibility policy

| Field | Value |
|---|---|
| **Decision** | Canonical output = **tidy v1.2 only**; `ecofragments.calculate_metrics` deprecated facade; **no** hybrid columns; dropped metrics → migration error |
| **Status** | `approved` |
| **Evidence artifact** | `adversarial_synthesis.md` §3.8; `spec_compliance.md` Q5; `implementation_plan.md` §3 |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 — maintainer confirmed |
| **Consequence if wrong** | PF/PLF/AWMP* persist in production tables |
| **Affected milestones** | M1, M7, M8 |

### Q6 — Drainage input contract (same as U4)

| Field | Value |
|---|---|
| **Decision** | See **U4** |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/drainage_inventory.md` |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-14 |
| **Consequence if wrong** | Same as U4 |
| **Affected milestones** | M10 |

### Q7 — HY algorithm authority

| Field | Value |
|---|---|
| **Decision** | **HY detection, season mapping, and related HY metrics not implemented in HydroFragments**; consume external sibling package **`hydroseason`** (`../hydroseason` / `D:\RLH\5.6\repos\hydroseason`) via thin adapter; pin version + config in manifest; persistence-based HY differentiated from Tayer 2025/2026 rainfall-based HY per V8 |
| **Status** | `approved` |
| **Evidence artifact** | `hydroseason` repo frozen at package version `0.1.0` (`pyproject.toml`), HEAD `4d5eec8` "fix: deselect experimental promotion-gate tests in CI", clean working tree (verified 2026-07-16). Public API surface confirmed exported from `hydroseason/__init__.py`: `HydroYearConfig`, `detect_hydrological_years`, `label_hydrological_months`, `monthly_water_extent`, `suggest_hydro_year_config` (static engine); `DynamicHydroYearConfig`, `detect_dynamic_hydrological_years`, `suggest_dynamic_hydro_year_config`, `HydrologicalStateResult`, `SeasonalPatternResult`, `analyze_hydrological_state`, `classify_annual_surface_water_condition`, `classify_seasonal_pattern`, `compute_monthly_surface_water_condition` (dynamic/stress engine); `load_aoi`, `load_wofs_from_stac`, `load_monthly_masks`, `load_monthly_masks_zarr`, `load_extent_csv`, `complete_monthly_axis`, `generate_html_report` (I/O). Default detector `robust_extrema`; `semi_markov` retained as experimental/deselected-in-CI challenger, not the promoted path — HydroFragments adapter must default to `robust_extrema`. V8 comparison (persistence-based HY vs Tayer 2025/2026 rainfall-based HY) run manually by maintainer on Fitzroy and Gilbert extent series: **100% agreement** on HY boundary/season assignment, no artifact file generated for this pass — recorded here as direct maintainer attestation dated 2026-07-16. |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-16 — maintainer confirmed `hydroseason` finished/stable and manually verified V8 agreement (Fitzroy/Gilbert, 100% match) |
| **Consequence if wrong** | Duplicate published HY method; unstable dry-down anchors; broken cross-repo coupling |
| **Affected milestones** | M9 — unblocked |

### Q8 — Validation fixtures

| Field | Value |
|---|---|
| **Decision** | Tier A analytic + Tier C real `water_cube.zarr` **mandatory**; `wmask_ts.nc` **legacy smoke only** (see U1) |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/evidence/fixture_inventory.md` |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-11 — follows approved U1 |
| **Consequence if wrong** | False confidence in contract tests |
| **Affected milestones** | M1, M3, validation |

### Q9 — Config hashing rules

| Field | Value |
|---|---|
| **Decision (proposed)** | `config_hash` = SHA-256 of canonical JSON of **scientific** fields only; exclude paths, scheduler, worker counts, accelerator; separate `execution_hash` and `input_fingerprint` |
| **Status** | `pending` golden cross-platform tests |
| **Evidence artifact** | `docs/audit/implementation_plan.md` §4 “Hashing rules”; `spec_compliance.md` Q9 |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | — |
| **Consequence if wrong** | Reproducibility claims fail across machines |
| **Affected milestones** | M2, M7 |

### Q11 — Extent-contraction slope method (`dynamics.contraction_method` / `dynamics.minimum_points`)

| Field | Value |
|---|---|
| **Decision** | `dynamics.contraction_method = "linear"` (ordinary least-squares slope of APSEC vs month index between peak-wet and end-dry HY anchors); `dynamics.minimum_points = 3`. Fewer than 3 usable monthly points yields a suppressed (NaN) slope with a `low_df` diagnostic rather than a reported value. |
| **Status** | `approved` |
| **Evidence artifact** | `docs/audit/implementation_plan.md` §"Persistence, temporal, and dynamics" (flagged as needing to be locked); `docs/HydroFragments_v1.2_spec.md` §6.5 (linear or Theil-Sen both spec-sanctioned) |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-16 — maintainer selected `linear`/`minimum_points=3` over `theil_sen` when asked directly during M9 implementation |
| **Consequence if wrong** | Contraction slope values change if method is revised later (Theil-Sen is more outlier-robust); minimum_points too low would let unreportable slopes (1-2 points, 0-1 residual df) through as if reliable |
| **Affected milestones** | M9 |

### Q10 — Predecessor history / publication lineage

| Field | Value |
|---|---|
| **Decision** | **C — abandon six-month/history claims.** The predecessor repo will redirect to this new repo; docs need only acknowledge lineage/redirect, not claim preserved public development history. |
| **Status** | `approved` |
| **Evidence artifact** | `git log -1`: single commit `b89dbde` “Initial commit: ecofragments package (clean start)” 2026-05-30; `spec_compliance.md` F13 |
| **Owner** | Thiaggo de Castro Tayer |
| **Approval date** | 2026-07-11 — maintainer selected option C |
| **Consequence if wrong** | Docs may overclaim lineage/public-development evidence; keep publication claims limited to acknowledgement and redirect |
| **Affected milestones** | M13, JOSS/paper (non-blocking for numerical core) |

---

## Related audit decisions already converged (inform M0, not re-litigated)

| Topic | Converged decision | Evidence |
|---|---|---|
| NNI | Cut from v1.2 runtime and manager surfaces | `scientific_metrics_audit.md` R5; `manager_interpretation_audit.md` §3 Danger 7 |
| LPSEC in v1.2.0 core | **Exclude** until real drainage `L_ref` | `adversarial_synthesis_2.md` §7 |
| CUDA | Normative = `dask_cuda_audit.md`; optional tranche post-CPU parity | `adversarial_synthesis.md` §3.4 |
| Rebrand | After API freeze; honesty banner first | `docs_audit.md`; `adversarial_synthesis.md` §3.2 |

---

## Gate checklist

| Check | M0 status |
|---|---|
| Every row has owner | **Yes** |
| Every row has evidence artifact pointer | **Yes** |
| Every blocking row has approval + closing evidence | **Yes** — U2/Q1 and U3/Q3 approved. Q7 and Q9 remain pending but are not M0-blocking. |
| Maintainer recorded predecessor-history decision | **Yes** — Q10 option C |
| Real `water_cube.zarr` validity sensitivity archived | **Yes** — `docs/audit/evidence/validity_reliability_report.md` (2026-07-14) |
| Drainage dataset supplied | **Yes** — `data/fitzroy_kimberley_drainage.gpkg`, approved (U4/Q6) |

**Decision Gate 0: CLOSED.** U1, U2/Q1, U3/Q3, U4/Q6, U7, Q2, Q4, Q5, Q7, Q8, Q10 are `approved`.
Q9 is legitimately deferred to its milestone and does not block core work.

**M9 gate (Q3/U3 and Q7/V8): OPEN.** Both required rows are `approved` as of 2026-07-16 — Milestone 9 (pixel-temporal and HY/dynamics tranche) may begin.
