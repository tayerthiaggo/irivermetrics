# Milestone 9 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Milestone 9 scientifically guarded, reachable through the public API, and reproducible with hydroseason provenance.

**Architecture:** Keep recurrence/hydroperiod and dynamics kernels small and pure. Convert local HydroConfig hydroyear settings into the pinned hydroseason config at the adapter boundary, then pass adapter provenance through analysis and manifest output. Add only the minimum API records needed to expose pixel-temporal and dynamics results.

**Tech Stack:** Python 3.10+, xarray, pandas, NumPy, hydroseason 0.1.0, pytest.

## Global Constraints

- Temporal aggregates use valid observations and the locked `p_native_season_stratified_v1` policy.
- Extent contraction requires both `max_water` and `median` composites.
- Extent contraction uses linear OLS with minimum 3 usable monthly points and elapsed calendar-month coordinates.
- HY detection and season mapping remain delegated to `hydroseason`; no detector logic is added locally.
- Hydroseason version and exact passed config must be included in scientific provenance.
- No recession-constant, flow, or discharge claim is emitted.

### Task 1: Lock configuration and temporal denominators

**Files:**
- Modify: `hydrofragments/config.py`
- Modify: `hydrofragments/metrics/persistence.py`
- Test: `tests/metrics/test_recurrence_hydroperiod.py`
- Test: `tests/config/test_config.py` or nearest existing config test module

- [x] Add failing tests for default dynamics values (`linear`, `3`) and season-stratified recurrence support.
- [x] Implement defaults and valid-month/year reductions without treating unobserved months as dry.
- [x] Run focused config and persistence tests.

### Task 2: Correct dual-composite contraction and edge handling

**Files:**
- Modify: `hydrofragments/metrics/dynamics.py`
- Test: `tests/metrics/test_extent_contraction.py`

- [x] Add failing tests for median slope output, missing-month elapsed spacing, NaN exclusion, and missing/unsorted end-dry records.
- [x] Return both composite slopes, use elapsed calendar months, count only finite usable records, and reject invalid anchors explicitly.
- [x] Run the contraction test module.

### Task 3: Complete hydroseason boundary and provenance

**Files:**
- Modify: `hydrofragments/temporal/hydroyear.py`
- Modify: `hydrofragments/output/manifest.py`
- Modify: `hydrofragments/api.py`
- Modify: `hydrofragments/models.py` and/or `hydrofragments/schema.py`
- Test: `tests/temporal/test_hydroyear_adapter.py`
- Test: `tests/output/test_manifest_hydroseason.py`

- [x] Add failing tests proving local config is converted to external config and manifest auto-records version/config.
- [x] Implement explicit adapter conversion and serializable provenance payload.
- [x] Normalize HY confidence to canonical output representation while preserving categorical source value in provenance/diagnostics.
- [x] Run adapter and manifest tests.

### Task 4: Wire metrics into public analysis/output

**Files:**
- Modify: `hydrofragments/api.py`
- Modify: `hydrofragments/metrics/registry.py`
- Modify: `hydrofragments/output/tables.py` if required by record shape
- Test: `tests/compat/test_hydrofragments_public_api.py`
- Test: new end-to-end Milestone 9 API test

- [x] Add failing test selecting `pixel_temporal`/`dynamics` profiles and asserting records are emitted or explicitly skipped with reasons.
- [x] Integrate recurrence/hydroperiod AOI summaries and HY dynamics records through the canonical tidy path.
- [x] Correct registry units/value types and preserve explicit skip diagnostics when dual composites or HY anchors are unavailable.
- [x] Run public API and output tests.

### Task 5: Guard deferred metrics, fixtures, docs, and full verification

**Files:**
- Modify: `hydrofragments/metrics/dynamics.py`
- Modify: `tests/fixtures/analytic_masks.py` and Milestone 9 tests
- Create: `docs/metrics/dynamics.md`

- [x] Add failing test that reconnection only searches months after end-dry and carries proxy provenance.
- [x] Move inline tiny masks into approved analytic fixtures.
- [x] Document extent-contraction interpretation, dual-composite requirement, confidence, and unresolved V3 validation status.
- [x] Run focused tests, full pytest, `git diff --check`, and package build.
