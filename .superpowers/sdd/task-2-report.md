# Task 2 Report: Wire `compat.py`'s `compute_apsec` call to pass `valid_obs`/`min_valid_fraction`

## Summary

Closed the gap identified in the brief: `ApsecRecord.low_coverage_flag` was fully
implemented and unit-tested in `compute_apsec` (Task 8), but the only production
call site (`section_compat_rows()` in `hydrofragments/compat.py`) never passed
`valid_obs`/`min_valid_fraction`, so the flag was always `False` in real
`analyze()` output. It is now wired end-to-end and surfaced on the public
`MetricRecord` schema.

## Step 1: Config knob — already existed, no change needed

Per the brief's instruction to check before adding: `ValidityConfig` in
`hydrofragments/config.py` (lines 34-39) already has

```python
min_valid_fraction_month: float = 0.70
```

This is exactly the spec item 5 knob (`docs/HydroFragments_v1.2_spec.md` line 75:
"use `min_valid_fraction_month` for whether a monthly AOI/zone metric is
reportable"). It is already parsed/validated in `HydroConfig.from_mapping` and
round-tripped in `to_mapping()`. **No config.py change was needed.** This
confirms the brief's own caveat ("may already exist").

## Step 2: Investigation — `MetricRecord` schema gap (real, as the brief predicted)

Traced the full path:

- `analyze()` (`hydrofragments/api.py`) already passes `valid_obs=monthly["valid_obs"]`
  into `section_compat_rows()` (this was Task 1's plumbing — confirmed at
  `hydrofragments/api.py:780`).
- `section_compat_rows()` (`hydrofragments/compat.py:215-217`, pre-Task-2) called
  `compute_apsec(monthly, a_ref_m2=..., cell_area_m2=..., config=config)` with
  **no** `valid_obs`/`min_valid_fraction` — the gap the brief described, confirmed
  still present after Task 1.
- `compute_apsec` (`hydrofragments/metrics/extent.py:68-128`) was already correct
  and fully unit-tested (`tests/metrics/test_apsec_coverage_floor.py`, 4 tests, all
  passing pre-Task-2) — untouched by this task, reference only, as the brief said.
- `MetricRecord` (`hydrofragments/models.py`) had **no** `low_coverage_flag` field.
  It does have `valid_fraction_month`/`min_valid_fraction_month` fields already
  (unused/always-`None` on this path today — out of scope for this task, which is
  narrowly about `low_coverage_flag`), but nothing to carry the APSEC coverage
  flag through. `_records_from_compat_rows()` (`hydrofragments/api.py`) builds one
  `MetricRecord` per compat-row column via a fixed `mapping` dict and had no way to
  attach per-record boolean metadata beyond `n_pools`.

**Conclusion (as the brief flagged as a real possibility): extending
`MetricRecord`/the row schema was necessary**, not scope creep. Implemented as:

- `MetricRecord.low_coverage_flag: bool | None = None` — new field, inserted
  immediately after `min_valid_fraction_month` (before `edge_flag`) in
  `hydrofragments/models.py`.
- `OUTPUT_COLUMNS` and `OUTPUT_DTYPES` in `hydrofragments/schema.py` updated in
  the same position (`"boolean"` dtype, matching `is_reportable`/
  `composite_sensitive`'s existing dtype convention).
- `section_compat_rows()` now adds a `"low_coverage_flag"` key to each row dict
  (`None` for months where APSEC wasn't computed, i.e. `want_apsec=False`).
- `_records_from_compat_rows()` picks up `row["low_coverage_flag"]` and passes it
  through `_metric_record()`'s new `low_coverage_flag` kwarg, **only** for the
  `apsec` metric id (every other compat-row metric — `number_of_pools`, `lpi`,
  `awre`, `awmsi`, `occurrence`, `refuge_area` — gets `low_coverage_flag=None`,
  since the flag is APSEC-specific).

This is a frozen-schema change: `tests/contracts/test_schema.py` pins
`EXPECTED_COLUMNS`/`EXPECTED_DTYPES` as an exact tuple/dict. Both were updated in
lockstep (adding `"low_coverage_flag"` / `"boolean"` in the same position) — this
is the expected, intentional cost of the additive schema change the brief called
out, not an accidental break.

## Step 3: Wiring `section_compat_rows()`

`hydrofragments/compat.py`, inside `section_compat_rows()`'s `want_apsec` block:

```python
apsec_records = compute_apsec(
    monthly,
    a_ref_m2=a_ref_m2,
    cell_area_m2=cell_area_m2,
    config=config,
    valid_obs=monthly["valid_obs"],
    min_valid_fraction=config.validity.min_valid_fraction_month,
)
```

`monthly["valid_obs"]` is the same mask `_monthly_dataset()` already builds
(Task 1's `water & valid_obs` unification — either the cube's real `valid_obs`
passed down from `analyze()`, or the all-True default for legacy
`calculate_metrics_compat()` callers, in which case `low_coverage_flag` is
computed against an all-valid mask and is always `False`, correctly matching
"no separate validity concept" legacy semantics).

`config.validity.min_valid_fraction_month` (default 0.70) is now always supplied,
so the coverage floor is live on **every** call to `section_compat_rows()` —
both the canonical `analyze()` path and the legacy `calculate_metrics_compat()`
shim (previously, no caller ever triggered the flag at all; both paths' behavior
before/after only differs in whether `valid_obs` is real vs. all-True).

Per-month, `low_coverage_flag` is read off the corresponding `ApsecRecord` and
threaded into the row dict alongside the existing `apsec_value`.

**Value safety confirmed:** `compute_apsec`'s `value` arithmetic is untouched —
only `low_coverage_flag` on the returned `ApsecRecord` changes based on the new
kwargs. Verified via `test_apsec_value_unchanged_by_coverage_floor` (pre-existing,
passing) and the new integration test's explicit `nunique() == 1` value-invariance
assertion (see below).

## Step 4: Failing integration test written first (TDD)

Added `test_analyze_flags_low_coverage_month_on_apsec_record` to
`tests/compat/test_hydrofragments_public_api.py` — exercises the real `analyze()`
entry point (not `compute_apsec` directly, which Task 8 already covered):

- 6-month cube, 10x10 grid, `open_water_cube(..., valid_obs=..., input_kind="generic_binary")`.
- Water confined to rows 4-9 every month (60 px); rows 0-3 always dry.
- Month index 3 has 40 invalid pixels in the always-dry rows 0-3 -> 60% coverage,
  below the default `min_valid_fraction_month` floor of 0.70. This design
  deliberately keeps the invalid pixels **out of the water region**, so masking
  changes only `valid_obs`, never the `water & valid_obs` count -- isolating the
  flag from the value, matching the existing kernel-level test's design
  intent (`test_apsec_value_unchanged_by_coverage_floor`). (First attempt put
  invalid pixels inside the always-wet region, which correctly changed the
  water count too -- confirming that would have been the wrong fixture for an
  "unaffected value" claim; corrected before finalizing.)
- Confirmed **red** first: `KeyError: 'low_coverage_flag'` (column didn't exist on
  `metrics_table` before the schema change).
- After wiring: **green**. Asserts `low_coverage_flag == True` for month index 3
  only, `False` for all 5 other months, and `apsec_rows["value"].nunique() == 1`
  (proves the flag never perturbed the value).

## Step 5: Regression suite

Brief's exact command:

```
pytest tests/gating/ tests/metrics/test_apsec_coverage_floor.py tests/metrics/test_apsec_vectorized.py tests/compat/ -v
```

Result: **54 passed, 1 skipped** (pre-existing unrelated skip in
`test_package_metadata.py`), 0 failed.

`tests/gating/analyze_snapshot.json` -- confirmed **unchanged** via `git diff`
(zero diff, not even listed in `git status`). This is the numeric-safety
guarantee the brief required for this task (distinct from Task 1's: here zero
numeric drift is required, only new metadata is exposed).

### Full suite

Ran the entire test suite as a final check:

```
pytest tests/ --tb=no -rN
```

Result: **439 passed, 2 skipped, 1 failed**. The 1 failure
(`tests/contracts/test_fixture_characterisation.py::test_fitzroy_zarr_exists_and_checksum_is_stable`)
was verified via `git stash` to fail **identically on the unmodified worktree**
(pre-existing, unrelated environmental/fixture checksum issue -- not caused by
this task). All other tests, including `tests/contracts/test_schema.py` (updated
in lockstep with the new field) and the two skipped tests (pre-existing), pass.

## Additional changes beyond the brief's literal file list

- `tests/contracts/test_schema.py`: updated `EXPECTED_COLUMNS`/`EXPECTED_DTYPES`
  to include `low_coverage_flag` (required -- this is a frozen-schema pin test
  that would otherwise fail after the intentional schema extension).
- `docs/HydroFragments_v1.2_spec.md`: added a `[TASK 8 FOLLOW-UP 2026-07-19]`
  provenance note under §6.17's APSEC formula documenting `low_coverage_flag`'s
  semantics, config source, and the fact it's now wired into the live path
  (matching this repo's established pattern of dated provenance notes, e.g. the
  `[AUDIT FIX 2026-07-19]` notes already in that section). Also added
  `low_coverage_flag` to the §7.1 illustrative output-schema-amendments column
  list. Neither is required by the brief's steps, but both close the exact
  "trap of false confidence" the brief describes (a user reading the spec
  should now find accurate documentation of a flag that is genuinely live).

## Files changed

- `hydrofragments/compat.py` -- wire `valid_obs`/`min_valid_fraction` into the
  `compute_apsec()` call; add `low_coverage_flag` to each row dict.
- `hydrofragments/models.py` -- add `MetricRecord.low_coverage_flag: bool | None`.
- `hydrofragments/schema.py` -- add `low_coverage_flag` to `OUTPUT_COLUMNS` /
  `OUTPUT_DTYPES`.
- `hydrofragments/api.py` -- `_metric_record()` accepts/passes `low_coverage_flag`;
  `_records_from_compat_rows()` sources it from the row for the `apsec` metric id
  only.
- `tests/compat/test_hydrofragments_public_api.py` -- new integration test
  `test_analyze_flags_low_coverage_month_on_apsec_record`.
- `tests/contracts/test_schema.py` -- updated frozen-schema expectations.
- `docs/HydroFragments_v1.2_spec.md` -- provenance note + column list update.

## Config knob

No new config field was added -- `validity.min_valid_fraction_month` (default
`0.70`) already existed and is exactly the spec-item-5 knob the brief pointed
at. `section_compat_rows()` now reads it as `config.validity.min_valid_fraction_month`
and always supplies it to `compute_apsec`.

## Note on this report file

This file (`.superpowers/sdd/task-2-report.md`) previously contained content
from an unrelated, differently-numbered task in this repo's history (a metric
comparison HTML report, dated 2026-07-17). That content has been replaced with
this Task 2 report, since it did not correspond to this task's brief
(`.superpowers/sdd/task-2-brief.md`, the APSEC coverage-floor wiring task).
