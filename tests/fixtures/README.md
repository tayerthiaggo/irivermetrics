# Test fixture provenance and tiers

This repository's fixtures are organised into tiers per `docs/audit/decisions.md`
(U1, Q8 — both approved). See `docs/testing.md` for how these tiers map onto the test
suite structure, and `docs/audit/evidence/fixture_inventory.md` /
`docs/audit/evidence/drainage_inventory.md` for full characterisation evidence and
checksums.

## Tier A — synthetic analytic fixtures

**Location:** `tests/fixtures/analytic_masks.py`

Tiny, hand-built boolean masks with documented, hand-calculable ground truth (component
count, pixel area, connectivity behaviour). They have no provenance beyond this
repository — they exist purely to check kernels against known-correct answers.

| Fixture function | Ground truth it locks in |
|---|---|
| `empty_mask` | 0 connected components, 0 wet pixels |
| `full_mask` | 1 connected component covering every pixel |
| `diagonal_pair_mask` | 1 component under 8-connectivity (2 under 4-connectivity) |
| `one_pixel_noise_mask` | 2 components (1 px + 4 px) before any small-object filter |
| `mask_with_hole` | 1 component, hole does not split it, area = footprint - hole |
| `long_bar_mask` | 1 component, already its own skeleton, longest path = length - 1 |
| `padded_square_mask` | 1 component with real dry background; EDT maximum at centre |
| `chunk_crossing_mask` | 1 component spanning 2 or 4 equal-width chunk boundaries |

**Approved uses:** legacy kernel characterisation today (`tests/legacy/`); the same
ground truth is the reference future v1.2 patch-engine tests (Milestone 6) must
reproduce once real Dask chunking and global-label reconciliation exist.

**Unsuitable uses:** none — these are synthetic and carry no real-world validity
constraints. They do not, by themselves, establish real-catchment plausibility; pair
them with Tier C for that.

## Tier B — bundled legacy fixtures (smoke only)

| Fixture | Path | Status |
|---|---|---|
| Legacy water mask | `tests/wmask_ts.nc` | Legacy smoke only (U1, approved). No `valid_obs` layer; sub-monthly irregular cadence; CRS is EPSG:28351, not equal-area. |
| Legacy corridor AOI | `tests/rcor_extent.shp` | Legacy seven-section smoke AOI. Polygon corridors only — not a drainage centreline. |
| Legacy regression CSV | `tests/results_iRiverMetrics/metrics/irm_metrics.csv` | **Quarantined (U7, approved): retired as v1.2 correctness oracle.** May only back smoke comparisons of approved, purely-geometric invariant columns (currently: `section_area_km2`). Never occurrence, schema, or `pp_mean_%`/APSEC equivalence. See `tests/contracts/test_legacy_baseline_quarantine.py`. |

## Tier C — real validation-catchment fixtures

| Fixture | Path | Status |
|---|---|---|
| Real monthly water-mask cube (Fitzroy) | `data/wofs_monthly_masks_1986_2026.zarr` | Approved for v1.2 contract/sensitivity evidence (U1/Q8). SHA-256 of `.zmetadata` recorded in `docs/audit/evidence/fixture_inventory.md`. |
| Real drainage centreline (Fitzroy Kimberley) | `data/fitzroy_kimberley_drainage.gpkg` | Approved real `L_ref` source (U4/Q6). 291 `MultiLineString` reaches, complete topology. |
| AOI polygon (Fitzroy Kimberley) | `data/fitzroy_kimberley_aoi.geojson` | Supporting AOI geometry for the Tier C catchment. |

**Unsuitable uses (documented, not yet closed):** dual-composite dry-down analysis (no
`method_flag`/`confidence` band delivered); P-provenance policy tests.

## Tier D — deferred

Raw sub-monthly or dual monthly composite products for dry-down/composite-sensitivity
work (U3/Q3). Not present in this repository at Milestone 1; out of scope until a later
milestone supplies them.
