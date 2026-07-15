# Testing guide

This document describes the test suite structure introduced/extended at Milestone 1
(Characterisation suite and historical baseline quarantine). It does not describe any
v1.2 numerical kernel — none exists yet; see `docs/audit/implementation_plan.md`
Milestones 2+ for when `hydrofragments/` modules and their contract tests land.

## Suite structure

| Path | Purpose | Speed |
|---|---|---|
| `tests/contracts/` | Read-only fixture characterisation and quarantine/contract tests. Safe to run anywhere; touches no scientific implementation. | fast |
| `tests/legacy/` | Legacy kernel characterisation only: unchanged low-level areas, perimeters, skeleton paths, and EDT behaviour from `ecofragments.utils.calc_metrics`, exercised on Tier A analytic masks. **Never** validates v1.2 occurrence or schema against legacy output. | fast |
| `tests/fixtures/` | Tier A synthetic analytic mask factories (`analytic_masks.py`) plus `README.md` documenting fixture provenance and tiers. Not a test module itself. | n/a |
| `tests/test_unit_metrics.py` | Unit tests for legacy `calc_metrics` helper functions in isolation. Characterises current legacy formulas only (see module docstring). | fast |
| `tests/test_integration.py` | Full legacy pipeline smoke test (`ecofragments.calculate_metrics` end-to-end on bundled `tests/wmask_ts.nc` + `tests/rcor_extent.shp`). Marked `slow`. | slow |

Run the fast suite during normal development:

```bash
pytest -m "not slow"
```

Run the full suite (including the full legacy pipeline) before a release or when
touching `ecofragments/utils/calc_metrics.py`:

```bash
pytest
```

Run only the read-only contract/characterisation suite:

```bash
pytest tests/contracts -v
```

## Quarantine rules (U7, Q5 — approved, `docs/audit/decisions.md`)

`tests/results_iRiverMetrics/metrics/irm_metrics.csv` is a legacy wide-format metrics
export. Evidence (`docs/audit/evidence/regression_baseline.md`) shows it:

- contains v1.2-forbidden dropped/renamed metrics (`PF`, `PFL`/`PLF`, `AWMPA`, `AWMPL`,
  `AWMPW`);
- has a `pp_mean_%` column that is constant per section across all 63 dates, evidence
  it is a total-timestep mean (`calculate_pixel_persistence` divides by total
  timesteps), not the `water_obs / valid_obs` ratio v1.2 requires;
- has no valid-observation support column at all.

**Rule:** this CSV must never be used as a v1.2 correctness oracle. It may only back
historical smoke comparisons for approved, purely-geometric invariant columns that do
not depend on the water mask or any temporal denominator. At Milestone 1 the only such
comparison is `section_area_km2` (`tests/test_integration.py::
test_calculate_metrics_section_area_matches_legacy_geometry_smoke`).

The rule is enforced, not just documented, in two places:

1. `tests/contracts/test_fixture_characterisation.py::
   test_baseline_csv_is_legacy_not_v12_oracle` — asserts the fixture inspector's
   `suitable_as_v12_correctness_oracle` flag is `False`.
2. `tests/contracts/test_legacy_baseline_quarantine.py` — the canonical test. It loads
   the real CSV, runs it through `tests/contracts/legacy_quarantine.py`'s
   `find_v12_correctness_baseline_defects`, and asserts the CSV is explicitly
   *rejected* (non-empty defect list naming the forbidden columns, the missing
   `valid_fraction_month` support column, and the static `pp_mean_%` finding). If this
   test ever passes with an empty defect list, the quarantine has regressed and must be
   re-reviewed before any test may treat the CSV as an oracle again.

Any future v1.2 schema/registry module (Milestone 2) that validates output shape should
reuse or supersede `tests/contracts/legacy_quarantine.py`'s forbidden-column list rather
than re-deriving it.

## Fixture tiers

See `tests/fixtures/README.md` for the full tier table (A synthetic / B legacy smoke /
C real validation-catchment / D deferred) and provenance/checksum pointers into
`docs/audit/evidence/`.

## Adding new tests

- New tiny hand-calculable masks belong in `tests/fixtures/analytic_masks.py`, with a
  docstring stating the ground truth they lock in.
- New legacy kernel characterisation belongs in `tests/legacy/`, restricted to
  low-level areas/perimeters/skeleton/EDT behaviour, and must not import or compare
  against the legacy CSV.
- New real-fixture characterisation belongs in `tests/contracts/`, following the
  read-only, no-mutation discipline of `tests/contracts/fixture_inspector.py`.
- Do not add a `legacy_output=True`-style path or any test that asserts current legacy
  output values as "correct" v1.2 targets. Dropped metrics failing v1.2 requests is the
  expected behaviour (Q5), not a bug to work around in tests.
