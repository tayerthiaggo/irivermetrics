# Dynamics and Spatial Exports Implementation Plan

> **Status:** audited and rewritten 2026-08-12. The earlier draft was not implementation-ready: it targeted a nonexistent API helper, bypassed the checkpoint-only vector contract, omitted output configuration and orchestration, lost raster georeferencing, and introduced unbounded time-by-space retention.
>
> **For implementers:** use the `superpowers:executing-plans` skill and complete tasks in order. Use test-driven development for every behaviour change. Do not begin the spatial writers until Tasks 1-6 pass.

**Goal:** Add scientifically defensible reconnection lag and refuge-stability metrics, plus optional GIS-ready vector and raster products, without slowing or increasing memory use for the default tabular workflow.

**Primary users:** Hydrologists and GIS analysts who run either `analyze()` on an existing `WaterCube` or `analyze_from_dea()` and need (a) canonical metric rows for downstream statistics and (b) spatial products that open correctly in QGIS/GDAL without manual repair.

**Architecture:** Keep the monthly cube as the single source of truth, but stream it as memory-admitted spatial-window/time blocks rather than full-grid month arrays. Each source block is materialized once for the enabled metric/checkpoint consumers, then released. Additive raster state is chunked/checkpoint-backed, not a collection of full-grid accumulators. Patch analysis reuses globally reconciled labels through a chunked label checkpoint when a window exceeds the in-memory budget. Optional pool vectors consume those same labels; export code never accepts an accumulated run-wide `GeoDataFrame`. Raster outputs carry a validated source-grid contract. A staged bundle is validated completely and committed with one same-filesystem directory rename.

**Performance rule:** Spatial exports are off by default. With them off there must be no polygonization or output serialization. Enabling an export must reuse already-required source/label work rather than add a second source pass. Resident memory is byte-admitted and bounded across both time length and raster area; large counters and labels spill to chunked local checkpoints. One exact morphology crop that individually exceeds the configured worker budget fails early with an actionable resolution/windowing message rather than risking an out-of-memory kill.

**Technology:** Python 3.10-3.13, NumPy, pandas, xarray, Dask, Rasterio, rioxarray, GeoPandas, Shapely, Pyogrio, PyArrow, Zarr 2; optional `h5netcdf` for NetCDF4 output. Preserve the existing HydroFragments output metric schema unless a task explicitly states otherwise.

---

## 1. Audit decisions and scope

### 1.1 Defects the implementation must correct first

The audit found these existing or draft-plan defects. They are part of this plan because they directly affect correctness, usability, or performance:

1. `PROFILES["dynamics"]` selects only `extent_contraction`. Adding new IDs only to the runtime tuple does not make the named profile select them.
2. The old draft targets `_dynamics_profile_records()`, which does not exist. The current helper is `_extent_contraction_records()` and `analyze()` only invokes that helper.
3. Reconnection needs an LPI/LPSEC monthly support series, but a dynamics-only run currently skips patch-section computation. This hidden dependency needs an explicit provider contract.
4. `DynamicsConfig` has no reconnection thresholds. LPI and LPSEC are percentages, so thresholds must be named and validated on the 0-100 scale.
5. The draft supplied neither valid end-dry masks nor a second hydrological year, and its proposed test fixture could not run.
6. The draft converted an empty refuge union to zero. The existing scalar contract defines empty/empty as undefined (`NaN`), not zero.
7. The current monthly engine bounds in-flight futures but returns a list containing every month’s `water_month` and `valid_obs_month`. This is latent `O(time * y * x)` memory retention.
8. The existing output contract deliberately rejects an accumulated patch `GeoDataFrame` and permits vector export only from a durable checkpoint. The old draft violated that contract.
9. Bare `np.ndarray` raster inputs lose CRS, affine transform, coordinates, nodata, and alignment information.
10. `analyze()` writes metadata before the DEA workflow writes tables; the workflow then rewrites the manifest. Spatial artifacts need one finalizer and one manifest publication, last.
11. `HydroResult.write()` passes an unsupported `formats=` argument to `write_output_tables()`.
12. `OutputConfig.formats` accepts arbitrary strings and execution ignores it. `metric_overrides` is parsed and hashed but never applied.
13. `HydroResult.manifest` is a path stub after plain `analyze()` and a full dictionary after the DEA workflow.
14. Provenance hard-codes `git_sha="unknown"` for every row and manifest.
15. The README names output files that are not produced and implies plain `analyze()` writes tables when it currently does not.
16. The test suite currently has three environment-assumption failures in the active Python 3.14/CuPy environment. The project and its pinned HydroSeason/native stack need a truthful supported-Python boundary and environment-independent capability tests before a new baseline is claimed.
17. With `output.output_dir` omitted, `analyze()` currently resolves it to `.` and writes metadata into the caller’s working directory. Library-default analysis must be side-effect-free.

### 1.2 In scope

- Wire `reconnection_timing` and scalar `refuge_spatial_stability` through configuration, registry profiles, API execution, provenance, coverage, and output tables.
- Capture hydrological-year end-dry states from the monthly pass using common-valid-support semantics.
- Add optional, validated spatial products:
  - monthly pool polygons;
  - hydrological zones as a raster and a dissolved polygon layer when zone inputs exist;
  - occurrence, valid-observation-count, refuge-mask, recurrence, hydroperiod, refuge-overlap, and refuge-stability rasters when their prerequisites exist;
  - reach geometry plus a normalized reach-by-month wetness table when channel profiles exist.
- Write tiled, compressed GeoTIFF by default for selected raster products.
- Offer NetCDF4 only when explicitly requested and the `netcdf` extra is installed.
- Retain metric and spatial intermediates in bounded consumers/checkpoints so no source month or connected-component labelling pass is repeated.
- Make output publication failure-safe and inventory every artifact in the final manifest.
- Fix adjacent result/output/config/provenance defects listed above.
- Add measured performance, memory, exactness, cross-platform, and failure-injection gates.
- Correct public documentation and include an offline, repository-owned example.

### 1.3 Explicit non-goals

- Do not implement river-connection (RC), time-connected fraction (TCF), or a new channel-network algorithm. Reconnection uses real LPSEC when its live inputs are available; otherwise it uses LPI and records the proxy.
- Do not reproject during export. Preserve the cube’s actual grid and CRS. The DEA workflow may already provide EPSG:3577; generic cubes may not. Never stamp EPSG:3577 onto unreprojected pixels.
- Do not call a GeoTIFF a COG unless COG layout and validation are separately implemented and tested. This plan promises tiled, compressed GeoTIFF.
- Do not add GPU processing, multi-AOI admission scheduling, a database, a web map, or a tile server.
- Do not change published metric formulae. This work supplies missing dataflow and validity handling around existing kernels.
- Do not make NetCDF the default. Xarray cannot incrementally write NetCDF variables and compressed NetCDF is materially slower than the chunked GeoTIFF/Zarr paths.

### 1.4 User-facing acceptance criteria

1. Selecting the `dynamics` profile produces both new metric IDs when prerequisites are available and explicit coverage reasons otherwise.
2. First-hydrological-year, empty/empty, missing-anchor, missing-month, no-threshold-crossing, and partial-validity cases have specified and tested outcomes.
3. Export-off metric tables are byte-for-byte equal to export-on metric tables for the same run/configuration.
4. Export-off work performs no polygonization, vector-checkpoint write, raster serialization, or extra cube read.
5. Every raster round-trips with correct CRS, affine transform, dimensions, coordinates, dtype, nodata, units, and band/time labels. Equal-shape but shifted grids are rejected.
6. Monthly pool polygon count and rasterized area match the canonical filtered labels used by metrics.
7. Peak resident memory is bounded with respect to both time length and raster area. Long-time and large-spatial synthetic runs stay within the configured byte admission; a single unprocessable component fails before allocation.
8. Every final artifact is reopened and validated before publication. A failed run leaves no final manifest and does not overwrite a pre-existing artifact.
9. `run_manifest.json` is written once, last, and contains a relative artifact inventory with digests and spatial metadata.
10. The default no-export path is no more than 10% slower by median wall-clock time than the pre-change baseline on the controlled benchmark. Enabled-export peak RSS is no more than 125% of the core run unless an evidence-backed limit is approved and recorded.
11. The full suite passes in clean supported environments on Linux and the spatial writer smoke suite passes on Windows.

---

## 2. Scientific and output contracts

Freeze these contracts in tests before implementing producers or writers.

### 2.1 Dynamics configuration

Add to `DynamicsConfig`:

```python
reconnection_lpi_threshold_pct: float = 50.0
reconnection_lpsec_threshold_pct: float = 50.0
```

Both values are percentages and must be finite and within `[0, 100]`. The `50.0` defaults resolve the design specification’s ambiguous mix of `50.0%` and `0.5` in favour of the registry’s actual percent units for both LPI and LPSEC. They belong in `scientific_config()` and therefore affect the scientific configuration hash. Keep the existing extent-contraction configuration unchanged.

Provider precedence for `reconnection_timing`:

1. RC, only if a future runtime supplies a real RC series. RC is not added by this plan.
2. LPSEC when complete live channel-profile prerequisites are available for the searched interval.
3. LPI, computed as internal support even when the user did not request LPI as an output row.

Do not fall back from a valid preferred series merely because it never crosses the threshold. That means “no reconnection according to the preferred provider”, not “try an easier proxy”. Record `proxy_reconnection_flag=True` and `WarningFlag.PROXY_RECONNECTION` for both LPSEC and LPI because both are proxies for RC under the existing scientific contract.

Normalize cube dates and anchors to calendar-month keys before alignment. A detected anchor resolves by year/month to exactly one post-composite cube slice; it does not require identical day-of-month timestamps. Duplicate cube month keys are an error. A cube month absent from the record remains missing. LPSEC is considered available for an interval only when its live profile has a finite value for every cube month present in that interval; otherwise choose LPI for the whole interval and record why.

For each detected hydrological year:

- The search starts after `end_dry` (exclusive).
- It ends before the next hydrological year’s `end_dry` (exclusive), or at the end of the available record for the final year.
- Missing months are not imputed and cannot cross a threshold.
- Lag is the integer calendar-month difference from `end_dry` to the first crossing, not the number of available observations.
- Emit one row per detected hydrological year. Missing anchors and no-crossing cases remain visible as non-reportable rows with `value=None` and a specific edge flag.
- The metric record maps into existing schema fields: `date=end_dry`, `hy`, `hy_anchor`, `hy_confidence`, `connected_wet_metric`, `connected_wet_threshold`, `reconnection_metric_used`, `proxy_reconnection_flag`, and `warning_flags`.

### 2.2 Refuge-stability validity

For an end-dry state, a pixel is eligible only when it is inside `analysis_mask` and valid at that anchor. For a consecutive hydrological-year pair, compare on common valid support:

```text
common_valid = analysis_mask & valid_previous & valid_current
previous_refuge = water_previous & common_valid
current_refuge  = water_current  & common_valid
union = previous_refuge | current_refuge
stability = count(previous_refuge & current_refuge) / count(union)
```

`common_valid_fraction = count(common_valid) / count(analysis_mask)`. An empty analysis mask is invalid input, not a zero-support stability row.

- The first hydrological year has no previous state: non-reportable, `value=None`.
- If the two years are not consecutive, do not form a pair.
- If common-valid fraction is below `validity.min_valid_fraction_month`, the pair is non-reportable.
- If `union` is empty, the scalar is undefined (`NaN`/non-reportable), not zero.
- Map common-valid fraction to `valid_fraction_month`, common-valid count to `n_valid_pixels`, and union-pixel count to `n_water_pixels`. The row’s `date` is the current end-dry date and `hy` is the current hydrological year; raster band metadata carries both dates in the pair.
- Keep previous/current states and counters in chunked checkpoint arrays; only the currently admitted chunks are resident. Do not keep a dictionary of annual full-resolution masks.

The per-pixel stability raster is a different product from scalar Jaccard stability:

```text
stable_count[p]   = number of eligible pairs where p is wet in both years
eligible_union[p] = number of eligible pairs where p is wet in either year
frequency_pct[p]  = 100 * stable_count[p] / eligible_union[p]
```

Pixels with `eligible_union == 0` are nodata/`NaN`. Name and metadata must not imply this percentage is the scalar Jaccard metric.

Add explicit string `EdgeFlag` values without adding metric-table columns: `missing_HY_anchor`, `no_previous_HY`, `nonconsecutive_HY`, `low_common_valid_support`, `empty_refuge_union`, and `no_threshold_crossing`. These make non-reportable rows machine-readable and require the explicit metric-schema `1.1.0` bump in Section 2.6 because the enum is part of the frozen contract.

### 2.3 Spatial grid contract

Create one immutable grid value object and require it at every raster/vector boundary:

```python
@dataclass(frozen=True)
class SpatialGrid:
    crs: CRS
    transform: Affine
    height: int
    width: int
    y_dim: str
    x_dim: str
    y: np.ndarray
    x: np.ndarray

    @classmethod
    def from_dataarray(cls, data: xr.DataArray) -> "SpatialGrid": ...

    def validate_dataarray(self, data: xr.DataArray) -> None: ...
```

Validation must compare CRS, affine transform, coordinate values/order, dimensions, and shape. Shape equality alone is insufficient. Spatial export requests against a cube without resolvable CRS/transform fail early with an actionable error; tabular analysis remains allowed.

### 2.4 Raster products

Use these exact data contracts:

| Product | dtype | nodata | units / codes | dimensions |
|---|---:|---:|---|---|
| occurrence | `float32` | `NaN` | percent, 0-100 | y, x |
| valid observation count | `uint32` | `4294967295` | months | y, x |
| refuge mask | `uint8` | `255` | 0 false, 1 true | y, x |
| hydrological zone | `uint8` | `0` | 0 outside/no zone, configured zone codes | y, x |
| recurrence | `float32` | `NaN` | percent, 0-100 | y, x |
| recurrence valid-year count | `uint16` | `65535` | calendar years | y, x |
| hydroperiod | `float32` | `NaN` | fraction, 0-1 | calendar year, y, x |
| hydroperiod valid-month count | `uint8` | `255` | months, 0-12 | calendar year, y, x |
| refuge overlap | `uint8` | `255` | 0 dry, 1 lost, 2 new, 3 stable, 255 unsupported | HY pair, y, x |
| refuge stability frequency | `float32` | `NaN` | percent, 0-100 | y, x |
| refuge union-pair count | `uint16` | `65535` | valid HY pairs wet in either year | y, x |

Each multi-band GeoTIFF has deterministic band descriptions/tags for calendar year (hydroperiod) or hydrological-year pair (refuge overlap) and the applicable end-dry dates. Every file carries CRS, affine transform, units/codes, algorithm version, scientific config hash, and nodata metadata.

### 2.5 Vector products

Monthly pool polygons come only from the canonical globally filtered label raster created during metric measurement. Exact layer schema:

```text
date: datetime64[ns]
pool_id: string              # YYYY-MM-DD:<window_id>:<label_id>
label_id: int32
n_pixels: int32
area_m2: float64
perimeter_m: float64
major_axis_length_m: float64
width_m: float64 nullable
elongation_ratio: float64 nullable
shape_index: float64 nullable
geometry: Polygon/MultiPolygon in source CRS
```

Do not put aggregate monthly metrics such as AWRE or AWMSI on every pool feature. Polygon area/count must agree with the measured label properties within raster/vector tolerance.

Other layers:

- `zones`: one dissolved feature per zone with `zone_id`, `zone_name`, `area_km2`, `source`, and geometry.
- `reaches`: one geometry per reach with stable `reach_id` and static attributes.
- `reach_wet_monthly`: a non-spatial table keyed by `reach_id,date` with `is_wet`, `length_m`, and `lpsec_contribution_pct`. Do not repeat reach geometry for every month.

If a requested product lacks runtime prerequisites, fail preflight with `SpatialProductUnavailable` and an actionable message. Never silently omit a requested artifact.

### 2.6 Configuration and version boundaries

Keep four versions distinct:

- package version;
- accepted configuration schema version;
- metric row schema version (`SCHEMA_VERSION`);
- run-manifest schema version.

Introduce additive configuration schema `1.1.0`; continue accepting `1.0.0`, mapping omitted new fields to export-off defaults. Bump metric row schema to `1.1.0` because the new machine-readable `EdgeFlag` values expand the frozen enum contract even though no columns change. Writers emit run-manifest schema `1.1.0` for the artifact inventory. Table/bundle readers continue to accept legacy metric/manifest `1.0.0` under their original contracts, reject mixed row-schema versions in one dataset, and never rewrite old user artifacts. Do not call this work “v1.3.0” without a separate release decision.

The raw parser `config_schema_version` must not make scientifically identical `1.0.0` and `1.1.0` inputs hash differently after normalization. Replace it in the canonical scientific payload with an explicit `scientific_hash_schema_version`; keep the parser version separately in resolved config/manifest provenance. Adding the threshold fields intentionally updates the golden scientific-hash contract once. Record the hash-algorithm version so legacy manifests remain interpretable rather than silently recomputing their hashes under new rules.

Extend `OutputConfig` with typed, validated values:

```python
spatial_products: tuple[
    Literal[
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
    ], ...
] = ()
raster_formats: tuple[Literal["geotiff", "netcdf"], ...] = ("geotiff",)
```

Retain `include_vectors` as a deprecated compatibility alias for `monthly_pools` for one configuration-schema cycle, rejecting contradictory settings. Validate table `formats` rather than accepting arbitrary strings. Output selections belong in `execution_config()`, not `scientific_config()`.

Apply `metric_overrides.add` and `.remove` before dependency resolution. Unknown IDs, contradictory add/remove entries, and removing a required provider without an allowed hidden-support path must produce clear validation errors. A removed output metric may still be computed internally as support when its selected dependent requires it; it must not be emitted as a metric row.

### 2.7 Stable output layout

Use deterministic relative paths so scripts and GIS projects do not need manifest-specific guessing:

`output_dir` names the final run directory, not a general shared output root. Its parent may exist; the final directory must be absent or empty so the bundle can be committed with one directory rename. This intentional safety rule and migration from pre-created target directories must be documented.

```text
<output_dir>/
  config.json
  metrics/                              partitioned canonical Parquet dataset
  metrics.csv                           only when CSV is selected
  metric_coverage.csv
  vectors/spatial.gpkg
    monthly_pools                       optional spatial layer
    zones                               optional spatial layer
    reaches                             optional spatial layer
    reach_wet_monthly                   optional non-spatial table
  rasters/occurrence.tif
  rasters/valid_observation_count.tif
  rasters/refuge_mask.tif
  rasters/zones.tif
  rasters/recurrence.tif
  rasters/recurrence_valid_year_count.tif
  rasters/hydroperiod_by_year.tif
  rasters/hydroperiod_valid_month_count_by_year.tif
  rasters/refuge_overlap_by_hy.tif
  rasters/refuge_stability_frequency.tif
  rasters/refuge_stability_union_pair_count.tif
  rasters/spatial.nc                    optional consolidated NetCDF
  run_manifest.json                     always last
```

Only selected/applicable paths are created. Empty but valid selected vector products create their named zero-feature layer with the exact schema. A temporal/refuge raster requiring unavailable years fails preflight when that absence is knowable, or records a precise runtime-unavailable failure and publishes no bundle when hydrological-year detection itself establishes the absence.

---

## 3. HydroSeason patterns to reuse

The sibling repository contains processing patterns worth porting, not copying blindly:

1. **Per-block materialization:** HydroSeason commit `75ef0ee` and `hydroseason/hydro_year.py` materialize time blocks instead of a whole time stack. Keep HydroFragments’ bounded in-flight scheduling, but consume completed month arrays immediately rather than returning them all.
2. **Batch shared Dask work:** `_io_wofs_zarr.py` batches several related lazy arrays into one `dask.compute`, preventing repeated upstream reads. Compute water, validity, metric inputs, labels, and enabled sidecar counters from the same monthly materialization.
3. **Sidecars in the write pass:** commits `70b43b4` and `cdee122` update compact count products while data are already materialized. Update refuge `uint32` counters and end-dry state in the same consumer pass.
4. **Atomic verified artifacts:** `_historical_water_mask.py` and commit `adea1ef` write a temporary artifact, reopen and validate it, atomically replace the final path, and publish the index last. Apply this to TIFF/GPKG/NetCDF and publish `run_manifest.json` last.
5. **Atomic reports:** `_report_export.py` and commit `189eac6` use same-directory temporary files plus `replace`, which is portable to Windows when handles are closed.
6. **Output-inclusive benchmarks:** `scripts/benchmark_wofs_cache.py` measures subprocess wall time through persistence and readback, repeated medians, peak RSS, and exactness. Extend HydroFragments’ benchmark harness similarly.
7. **Worker tuning only with evidence:** commit `1f8d2c7` added an I/O worker knob, but the resulting benchmark notes show default Dask scheduling can outperform forced overrides. Do not add a new worker setting unless the benchmark demonstrates a repeatable win.

Use compact state: `uint8` for overlap categories, `uint32` for counters, and `float32` for percentage rasters. Avoid `dict[str, np.ndarray]` followed by `np.stack`.

Do not make HydroFragments depend on unpublished sibling source paths or HydroSeason private functions. The patterns above are design evidence. If implementation needs a new HydroSeason public API, release and test that version first, then update HydroFragments’ exact dependency pin and manifest-version test in a separate, explicit dependency commit.

---

## 4. File map

Expected primary changes:

```text
hydrofragments/
  api.py                              execution and dynamics orchestration
  config.py                           schema 1.1, thresholds, output products
  models.py                           result/output contract
  section_analysis.py                 ordered monthly consumer integration
  analysis/
    __init__.py                       new internal analysis package
    window_stream.py                  byte-admitted window/time consumer loop
  metrics/
    dynamics.py                       existing kernels, validity helpers as needed
    patches.py                        measured labels + properties bundle
    registry.py                       dynamics profile and override resolution
  patches/
    labels.py                         chunked globally reconciled label checkpoint
    components.py                     admitted component-crop iteration
  output/
    tables.py                         formats fix; checkpoint contract preserved
    rasters.py                        georeferenced raster builders/writers
    vectors.py                        checkpoint-to-GPKG writers
    checkpoints.py                    bounded spatial checkpoint consumers
    bundle.py                         one output finalizer
    manifest.py                       atomic manifest + artifact inventory
    spatial.py                        SpatialGrid and product metadata contracts
  spatial/
    active_windows.py                  budgeted analysis-mask window planning
    zones.py                          georeferenced zone result
  workflow.py                         call core analysis then one finalizer

tests/
  api/test_public_api.py
  analysis/test_window_stream.py
  contracts/test_config.py
  contracts/test_hashing.py
  contracts/test_provenance.py
  contracts/test_registry.py
  contracts/test_schema.py
  metrics/test_dynamics_edges.py
  metrics/test_dynamics_pipeline.py
  metrics/test_patch_metrics.py
  output/test_tables.py
  output/test_spatial_grid.py
  output/test_rasters.py
  output/test_vectors.py
  output/test_bundle.py
  output/test_manifest.py
  output/test_manifest_hydroseason.py
  integration/test_spatial_exports.py
  benchmarks/test_spatial_export_baseline.py
  release/test_package_metadata.py
  compute/test_capabilities.py

docs/
  spatial_exports.md
  metrics/dynamics.md                  canonical dynamics documentation
  superpowers/specs/2026-08-12-dynamics-and-spatial-export-design.md
README.md
CHANGELOG.md
pyproject.toml
.github/workflows/ci.yml
examples/spatial_exports.py
```

Names may follow existing repository conventions, but keep grid validation, checkpoint production, serialization, and bundle finalization as separate modules. Do not fold all output logic into `api.py` or `workflow.py`.

---

## 5. Implementation tasks

### Task 1: Restore a trustworthy release/test baseline

**Files:**

- Modify: `pyproject.toml`
- Modify: `tests/release/test_package_metadata.py`
- Modify: `tests/compute/test_capabilities.py`
- Modify: `tests/benchmarks/test_cpu_baseline.py`
- Modify: `.github/workflows/ci.yml`

**Step 1: Write failing environment-independent tests**

- Parse requirements with `packaging.Requirement` and assert CuPy is absent from mandatory dependencies, not from every optional extra.
- Mock `cupy` capability probes for CPU-only behaviour; do not assert that the developer’s physical machine has no GPU/CuPy installation.
- Assert reported CUDA availability, device count, and selected execution path are internally consistent.
- Assert package Python compatibility matches the pinned HydroSeason/native stack.

Run:

```powershell
python -m pytest tests/release/test_package_metadata.py tests/compute/test_capabilities.py tests/benchmarks/test_cpu_baseline.py -q
```

Expected before fix: the three current environment-assumption failures reproduce in the active Python 3.14/CuPy environment.

**Step 2: Implement the narrow fixes**

- Set HydroFragments’ supported Python range to `>=3.10,<3.14` unless dependency verification proves Python 3.14 support.
- Keep CuPy optional.
- Make capability tests use controlled probes, while retaining one separately marked hardware smoke test.
- Extend Linux CI to Python 3.10, 3.11, 3.12, and 3.13. Add a Windows 3.13 job for output-writer smoke tests later in Task 12.

**Step 3: Establish the baseline**

Run the full suite in a fresh supported environment and save:

- commit SHA;
- dependency lock/freeze;
- test counts and skips;
- benchmark wall time and peak RSS;
- fixture dimensions/month count.

Do not use the current 3.14/CuPy failures as a performance baseline and do not hard-code a collected test count into future assertions.

**Step 4: Commit**

```bash
git add pyproject.toml tests/release/test_package_metadata.py tests/compute/test_capabilities.py tests/benchmarks/test_cpu_baseline.py .github/workflows/ci.yml
git commit -m "test: make capability baseline environment independent"
```

### Task 2: Repair public result and output contracts

**Files:**

- Modify: `hydrofragments/models.py`
- Modify: `hydrofragments/output/tables.py`
- Modify: `hydrofragments/api.py`
- Modify: `tests/api/test_public_api.py`
- Modify: `tests/output/test_tables.py`
- Create: `tests/contracts/test_provenance.py`

**Step 1: Add failing public API tests**

Cover:

- `HydroResult.write(path, formats=("parquet",))` succeeds and creates the documented table layout;
- unknown table formats fail during validation;
- the method never accepts or synthesizes an in-memory patch `GeoDataFrame`;
- `HydroResult.manifest` has the same full-dictionary type for both public workflows;
- `manifest_path` is explicit and optional rather than encoded in a stub dictionary.
- omitted `output_dir` performs no filesystem writes and returns `output_dir=None`;

**Step 2: Define the narrow write contract**

`HydroResult.write()` writes metric tables and coverage only. It does not promise spatial export after analysis because canonical labels/checkpoints are deliberately not retained in the result object. Spatial products are requested in `OutputConfig` and written by the run finalizer.

Change `HydroResult.output_dir` to `Path | None`. When no `output_dir` is configured, perform no filesystem writes and return a complete in-memory manifest dictionary with an empty artifact inventory plus `manifest_path=None`. When output is configured, return the same dictionary plus the final path. Do not vary the manifest value’s type by entry point and never use `.` as an implicit output directory.

Translate validated table formats to existing writer flags/signatures. Preserve the checkpoint-only vector guard in `write_output_tables()` and its regression tests.

**Step 3: Resolve revision once per run**

Add a small provenance helper with precedence:

1. explicit build environment variable set by CI;
2. installed package metadata containing a revision;
3. local Git `HEAD` when available;
4. literal `unknown` only when none is resolvable.

Resolve once and reuse the same value for every metric row and the manifest. Test equality, not the local SHA value.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/api/test_public_api.py tests/output/test_tables.py tests/contracts/test_provenance.py -q
```

```bash
git add hydrofragments/models.py hydrofragments/output/tables.py hydrofragments/api.py tests/api/test_public_api.py tests/output/test_tables.py tests/contracts/test_provenance.py
git commit -m "fix: align public result and table output contracts"
```

### Task 3: Add configuration, registry, and override contracts

**Files:**

- Modify: `hydrofragments/config.py`
- Modify: `hydrofragments/metrics/registry.py`
- Modify: `tests/contracts/test_config.py`
- Modify: `tests/contracts/test_hashing.py`
- Modify: `tests/contracts/test_registry.py`

**Step 1: Write failing config tests**

Test:

- config schema `1.0.0` loads with all spatial products disabled;
- `1.1.0` accepts only the product/format literals in Section 2.6;
- non-empty spatial products require an explicit `output_dir`;
- the deprecated `include_vectors` alias maps once and contradictory values fail;
- NetCDF selection is accepted at config parse time but missing optional runtime support gives an actionable preflight error;
- dynamics percentage thresholds reject `NaN`, infinity, and values outside 0-100;
- threshold changes alter the scientific hash;
- semantically equivalent `1.0.0` and `1.1.0` inputs normalize to the same scientific hash;
- output product/format changes alter the execution hash but not the scientific hash;
- unknown metric override IDs, duplicates, and add/remove contradictions fail.

**Step 2: Write failing registry tests**

Assert:

- `PROFILES["dynamics"]` includes `extent_contraction`, `reconnection_timing`, and `refuge_spatial_stability`;
- `all_available` includes both new metrics;
- the runtime-supported tuple includes both IDs and obsolete skip reasons are removed;
- overrides are applied before final selection/dependency validation;
- removing LPI suppresses its row but does not suppress an explicitly selected reconnection metric’s internal LPI support.

**Step 3: Implement config and selection**

Add the fields/contracts from Section 2. Wire the resolved metric selection into both configuration validation and execution; remove direct calls that ignore `metric_overrides`.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/contracts/test_config.py tests/contracts/test_hashing.py tests/contracts/test_registry.py -q
```

```bash
git add hydrofragments/config.py hydrofragments/metrics/registry.py tests/contracts/test_config.py tests/contracts/test_hashing.py tests/contracts/test_registry.py
git commit -m "feat: configure dynamics metrics and spatial products"
```

### Task 4: Freeze the spatial grid and product data model

**Files:**

- Create: `hydrofragments/output/spatial.py`
- Modify: `hydrofragments/spatial/zones.py`
- Modify: `hydrofragments/models.py`
- Create: `tests/output/test_spatial_grid.py`
- Modify: `tests/spatial/test_zones.py`

**Step 1: Write grid-contract tests**

Use small analytic `xarray.DataArray` fixtures and test:

- ascending and descending coordinates produce the correct affine transform;
- source CRS is preserved;
- missing CRS or non-regular coordinates fail only when spatial output is requested;
- equal-shaped arrays with a shifted transform fail alignment;
- swapped dimensions or reversed coordinates fail alignment;
- a round-trip grid reconstructed from Rasterio metadata equals the source contract.

**Step 2: Implement `SpatialGrid`**

Centralize CRS/transform/dimension extraction and validation. Do not let individual writers infer or override georeferencing independently.

**Step 3: Make zone results georeferenced**

Replace the bare zone-mask array boundary with a grid-bearing `DataArray` or a `ZoneResult` that includes and validates `SpatialGrid`. Extend `AnalysisInputs` with optional zone input/source so generic `analyze()` can accept explicit zones and `analyze_from_dea()` can pass the zones it already owns.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/output/test_spatial_grid.py tests/spatial/test_zones.py tests/api/test_public_api.py -q
```

```bash
git add hydrofragments/output/spatial.py hydrofragments/spatial/zones.py hydrofragments/models.py tests/output/test_spatial_grid.py tests/spatial/test_zones.py tests/api/test_public_api.py
git commit -m "feat: define validated spatial grid contract"
```

### Task 5: Stream analysis under a spatial and temporal byte budget

**Files:**

- Create: `hydrofragments/analysis/__init__.py`
- Create: `hydrofragments/analysis/window_stream.py`
- Modify: `hydrofragments/section_analysis.py`
- Modify: `hydrofragments/metrics/patches.py`
- Modify: `hydrofragments/patches/labels.py`
- Modify: `hydrofragments/patches/components.py`
- Modify: `hydrofragments/spatial/active_windows.py`
- Create: `tests/analysis/test_window_stream.py`
- Modify: `tests/analysis/test_parallel_monthly_patch_analysis.py`
- Modify: `tests/analysis/test_monthly_dataset_lazy_materialization.py`
- Modify: `tests/metrics/test_patch_metrics.py`
- Modify: `tests/spatial/test_active_windows.py`

**Step 1: Add temporal and spatial memory regressions**

Instrument synthetic arrays with finalizers/weak references and byte counters. Test both a 480-month small grid and a short, large, Dask-backed grid with sparse active windows. Assert that completed block arrays become collectible, no full-grid month is returned, and admitted live bytes never exceed the policy derived from `target_chunk_bytes`, `worker_memory_fraction`, and worker count.

Also assert:

- out-of-order completion is reduced deterministically by window/date key;
- pending futures plus reorder buffers never exceed the byte bound, even when block sizes differ;
- an exception cancels pending futures and closes consumers;
- a single exact morphology crop whose estimated live bytes exceed the worker budget raises `MemoryBudgetExceeded` before materialization and names the crop/window and mitigation;
- export-disabled processing never constructs a vector or export-only checkpoint consumer; a scientifically required bounded spill store is allowed and must be deleted after scalar finalization.

**Step 2: Introduce the admitted block-consumer interface**

Use a compact payload whose arrays cover one admitted spatial window/block, not a whole raster:

```python
@dataclass
class WindowMonthResult:
    time_index: int
    date: pd.Timestamp
    window_id: str
    row_slice: slice
    col_slice: slice
    estimated_live_bytes: int
    metric_partials: Mapping[str, MetricPartial]
    water: np.ndarray
    valid_obs: np.ndarray
    patch_bundle: MeasuredPatchBundle | None = None

class WindowMonthConsumer(Protocol):
    def consume(self, block: WindowMonthResult) -> None: ...
    def finalize(self) -> object: ...
    def abort(self) -> None: ...
```

Use independent active windows for patch work and storage-aligned blocks for additive raster work. For an analysis mask within budget, retain today’s local planner. For a larger Dask-backed mask, derive exact globally connected component bounds through the same chunked-label/checkpoint mechanism; do not let `independent_active_windows()` call `np.asarray()` or `scipy.ndimage.label()` on the full mask. Schedule a small time block for one spatial window under byte admission, batch its related lazy arrays into one `dask.compute`, drain results deterministically, update consumers/checkpoints, and release it. Reduce small per-window metric partials into canonical monthly rows after the spatial pass. Do not return full-grid arrays or all month payloads from `_run_month_rows()`.

**Step 3: Expose canonical labels without full-raster materialization or a second pass**

Keep the current NumPy label path for masks that fit the budget. For a larger window, retain `dask-image` cross-chunk reconciliation, compute filtering/row-major normalization statistics blockwise, and persist normalized `int32` labels to a completed Zarr label checkpoint instead of calling `np.asarray()` on the full label raster. Scan label chunks for component counts/bounds, then read component crops through byte-admitted batches. Keep `measure_patch_properties()` as a compatibility wrapper for small direct callers.

`MeasuredPatchBundle` contains properties plus either an admitted in-memory label block or a validated label-checkpoint reference. Exact area/perimeter/moment reductions are chunkable. Width/morphology may materialize one component crop; enforce the single-component budget guard rather than silently approximating it.

Requirements:

- connected-component labelling occurs once per processed month/window set;
- metric aggregation and pool checkpoints use the same properties/labels;
- no in-memory labels are retained after consumer delivery; checkpoint labels are released/deleted according to the lifecycle contract in Task 7;
- disabled spatial export follows the current cheapest path and does not construct geometry.

**Step 4: Run correctness/read-count tests**

```powershell
python -m pytest tests/analysis/test_window_stream.py tests/analysis/test_parallel_monthly_patch_analysis.py tests/analysis/test_monthly_dataset_lazy_materialization.py tests/metrics/test_patch_metrics.py tests/patches/test_labels.py tests/patches/test_component_crops.py tests/spatial/test_active_windows.py -q
```

Include spies proving one source-chunk materialization per fused computation and one globally reconciled label graph per month/window set that needs patch support. Compare small-grid output byte-for-byte with the current eager reference.

**Step 5: Commit**

```bash
git add hydrofragments/analysis hydrofragments/section_analysis.py hydrofragments/metrics/patches.py hydrofragments/patches/labels.py hydrofragments/patches/components.py hydrofragments/spatial/active_windows.py tests/analysis tests/metrics/test_patch_metrics.py tests/patches/test_labels.py tests/patches/test_component_crops.py tests/spatial/test_active_windows.py
git commit -m "perf: stream analysis under a spatial byte budget"
```

### Task 6: Wire reconnection and scalar refuge stability

**Files:**

- Modify: `hydrofragments/api.py`
- Modify: `hydrofragments/section_analysis.py`
- Modify: `hydrofragments/metrics/dynamics.py`
- Modify: `hydrofragments/schema.py`
- Modify: `hydrofragments/output/tables.py`
- Modify: `tests/metrics/test_reconnection_refuge_stability.py`
- Modify: `tests/metrics/test_dynamics_edges.py`
- Create: `tests/metrics/test_dynamics_pipeline.py`
- Modify: `tests/api/test_public_api.py`
- Modify: `tests/contracts/test_schema.py`
- Modify: `tests/output/test_tables.py`
- Modify: `tests/gating/test_analyze_row_snapshot.py`
- Modify: `tests/gating/analyze_snapshot.json`

**Step 1: Build a valid analytic fixture**

Use at least 36 monthly observations, two consecutive hydrological years, valid temporal configuration, numeric monthly `hydroyear_extent`, explicit anchors/end-dry dates, partial invalid pixels, and a known post-end-dry threshold crossing. Import all fixture dependencies, including `xarray`.

**Step 2: Add failing end-to-end metric tests**

Cover:

- the `dynamics` profile emits extent contraction, reconnection timing, and refuge stability;
- LPI support is computed for reconnection even when the LPI metric row is not selected;
- live LPSEC is preferred when complete channel inputs exist;
- a preferred LPSEC series with no crossing does not fall back to LPI;
- threshold equality counts as crossing;
- the search window does not leak into the next hydrological year;
- missing months use calendar lag and are not imputed;
- no crossing produces a non-reportable row with a precise reason;
- LPI provider sets proxy warning/provenance;
- first HY, nonconsecutive HY, low common validity, and empty union are non-reportable;
- partial invalidity is excluded through common support, not treated as dry;
- emitted records include `date`, `hy`, `hy_anchor`, `hy_confidence`, support fraction, provider, and threshold;
- new writers stamp metric schema `1.1.0`, readers still accept a homogeneous legacy `1.0.0` dataset, and readers reject mixed-schema rows.

**Step 3: Implement one real dynamics record builder**

Replace the contraction-only orchestration with a helper whose name and signature match actual dataflow, for example:

```python
def _dynamics_profile_records(
    *,
    selected_metric_ids: frozenset[str],
    hy_anchors: HyAnchorResult,
    monthly_support: DynamicsSupport,
    refuge_states: Iterable[EndDryState],
    config: HydroConfig,
    run_context: Mapping[str, object],
) -> list[MetricRecord]: ...
```

Use the existing `HyAnchorResult` from `hydrofragments.temporal.hydroyear`; do not invent a `HydroYearResult` type. Do not create this function until its other input value objects are real. `DynamicsSupport` owns aligned LPI/LPSEC series and provider metadata. `EndDryState` carries water, validity, grid identity, anchor confidence, and date. The monthly consumer supplies both from the existing pass. `run_context` passes the same run/aoi/source/resolution/CRS/revision values already needed by `_record()`; do not invent another provenance schema.

Call the existing scientific kernels after constraining inputs to the defined HY search window/common-valid support. Extend kernels only when needed to express the frozen semantics.

**Step 4: Preserve deterministic outputs**

Compare the new metric table from `max_workers=1` and `max_workers>1`. Ordering, values, coverage, warnings, and provenance must match exactly.

**Step 5: Run tests and commit**

```powershell
python -m pytest tests/metrics/test_reconnection_refuge_stability.py tests/metrics/test_dynamics_edges.py tests/metrics/test_dynamics_pipeline.py tests/api/test_public_api.py tests/contracts/test_schema.py tests/output/test_tables.py tests/gating/test_analyze_row_snapshot.py -q
```

```bash
git add hydrofragments/api.py hydrofragments/section_analysis.py hydrofragments/metrics/dynamics.py hydrofragments/schema.py hydrofragments/output/tables.py tests/metrics/test_reconnection_refuge_stability.py tests/metrics/test_dynamics_edges.py tests/metrics/test_dynamics_pipeline.py tests/api/test_public_api.py tests/contracts/test_schema.py tests/output/test_tables.py tests/gating/test_analyze_row_snapshot.py tests/gating/analyze_snapshot.json
git commit -m "feat: wire reconnection and refuge stability metrics"
```

### Task 7: Build bounded raster accumulators and checkpoints

**Files:**

- Create: `hydrofragments/output/checkpoints.py`
- Modify: `hydrofragments/output/rasters.py`
- Create: `tests/output/test_checkpoints.py`
- Modify: `tests/output/test_rasters.py`

**Step 1: Write failing accumulator tests**

For occurrence, hydroperiod, recurrence, and refuge stability:

- compare incremental output to a tiny eager reference implementation;
- test 3+ hydrological years, alternating refuge state, dry/dry, invalid pixels, missing anchor, and nonconsecutive years;
- assert counter dtypes and overflow guards;
- assert final arrays are grid-bearing `DataArray`/`Dataset` objects;
- assert a large-spatial fixture retains only admitted chunks plus small metadata, not 12 or 24 full-grid counter arrays.

**Step 2: Implement same-pass accumulation**

Update compact, chunked Zarr counters while each admitted source block is already materialized:

- 12 calendar-month wet/valid strata for the season-stratified occurrence/recurrence estimator;
- calendar-year wet and valid counts, flushing hydroperiod on calendar-year completion;
- recurrence values plus valid-calendar-year support count;
- refuge stable and eligible-union counts;
- checkpoint-backed current/previous end-dry state, with only their active chunks resident.

Use exact integer count accumulation (`uint32`, with overflow checks) and finalize occurrence/recurrence one storage block at a time across the 12 calendar strata. Write per-calendar-year/per-HY-pair slices to the checkpoint as soon as complete. Calculate scalar metrics in their existing precision; cast only exported percentage/fraction rasters to `float32`. Batch related lazy source arrays into the same Dask computation as patch/dynamics work. Never keep the full counter set in RAM, construct a run-wide `dict[str, np.ndarray]`, or call `np.stack` on all years.

Scientific persistence metrics need these counters even when exports are off. In that case use a run-owned spill store, finalize scalar rows, and delete it; the export-on path retains the same completed store for serialization. The default-path benchmark must include this spill cost and meet the 10% gate.

**Step 3: Define checkpoint metadata**

Every checkpoint stores:

- `SpatialGrid` identity;
- scientific config hash and algorithm version;
- dtype/nodata/units/codebook;
- calendar-year/HY/HY-pair coordinates;
- completion marker written last;
- expected chunk inventory.

Put spatial sidecars under the configured checkpoint root when one exists; otherwise use a run-owned temporary checkpoint directory. A completed checkpoint with matching grid, input fingerprint, scientific hash, product set, and algorithm version may be reused after validation. Mismatched checkpoints are never reused or overwritten silently. Incomplete checkpoints are invalid and cannot be exported. Preserve valid checkpoints after an output-only failure so a retry can skip source computation; remove run-owned temporary checkpoints only after successful finalization unless the user requested retention.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/output/test_checkpoints.py tests/output/test_rasters.py -q
```

```bash
git add hydrofragments/output/checkpoints.py hydrofragments/output/rasters.py tests/output/test_checkpoints.py tests/output/test_rasters.py
git commit -m "feat: accumulate spatial rasters in the monthly pass"
```

### Task 8: Create checkpoint-only vector export

**Files:**

- Modify: `hydrofragments/output/checkpoints.py`
- Create: `hydrofragments/output/vectors.py`
- Create: `tests/output/test_vectors.py`
- Modify: `tests/output/test_tables.py`

**Step 1: Write failing checkpoint tests**

Test one and multiple windows/months:

- checkpoint records use date-scoped stable pool IDs;
- window offsets are applied to geometry coordinates;
- boundary-spanning components are not duplicated;
- globally filtered labels are the labels polygonized;
- total pixel count and rasterized polygon area match patch properties;
- a selected all-dry record creates an empty `monthly_pools` layer with the exact schema;
- memory remains bounded across many months;
- the writer rejects an in-memory run-wide `GeoDataFrame` and an incomplete checkpoint.

**Step 2: Serialize per-month pool partitions**

Only construct `PoolCheckpointConsumer` when `monthly_pools` is selected. Polygonize the canonical measured labels during consumer delivery and write one durable GeoParquet/Arrow partition per month with the schema in Section 2.5. Close the partition, validate row count/schema/CRS, then release labels and geometry objects.

Do not keep a list of per-month `GeoDataFrame`s.

**Step 3: Stream checkpoints into GeoPackage**

Write to a same-directory temporary `.gpkg`. Create the first layer with an explicit schema. Read consecutive checkpoint partitions into bounded row/estimated-byte batches, append each batch only after exact schema/CRS validation, and release it before loading the next; do not pay one GDAL open/transaction per feature or accumulate the run. Start with a documented internal 64 MiB batch target and change it only from benchmark evidence. Close all GDAL handles, reopen with Pyogrio, validate layers, counts, bounds, CRS, IDs, and sample geometry validity, then atomically replace the final path.

Add zones and normalized reach outputs from their orchestration-owned sources. Avoid a repeated reach geometry per month.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/output/test_vectors.py tests/output/test_tables.py -q
```

```bash
git add hydrofragments/output/checkpoints.py hydrofragments/output/vectors.py tests/output/test_vectors.py tests/output/test_tables.py
git commit -m "feat: export vectors from bounded checkpoints"
```

### Task 9: Implement verified GeoTIFF and opt-in NetCDF writers

**Files:**

- Modify: `hydrofragments/output/rasters.py`
- Modify: `pyproject.toml`
- Modify: `tests/output/test_rasters.py`

**Step 1: Write failing round-trip tests**

For every product in Section 2.4, write a tiny analytic artifact and reopen it with Rasterio or Xarray. Assert:

- exact CRS and affine transform;
- width, height, dimension order, and coordinate direction;
- dtype, nodata, units/codebook, compression, tiling, and band descriptions;
- calendar-year/HY-pair labels and applicable end-dry dates;
- exact categorical values and numeric tolerance for `float32` values;
- failure on shifted-grid input, missing CRS, truncated files, and mismatched checkpoint hashes.

**Step 2: Implement windowed GeoTIFF writes**

Use Rasterio windows aligned to the file’s storage blocks. Default profile:

- tiled, normally 256x256 (clamped for tiny rasters);
- DEFLATE compression;
- predictor chosen by dtype;
- BigTIFF `IF_SAFER`;
- explicit nodata and band descriptions/tags.

Read checkpoint chunks and write corresponding windows. Do not load a whole multi-year cube to produce a TIFF.

**Step 3: Implement verified temporary publication**

Write a same-directory temporary file, close it, reopen and validate it against the product contract, compute its SHA-256 and size, then `Path.replace()` the final path. Preflight fails if a final artifact already exists; never overwrite an unrelated file.

**Step 4: Add opt-in NetCDF4**

Add:

```toml
[project.optional-dependencies]
netcdf = ["h5netcdf>=1.4"]
```

NetCDF export reads completed checkpoint chunks into a CF-style `Dataset`, includes CRS/grid mapping and compression/chunk encoding, writes once through `h5netcdf`, then reopens and validates. If the extra is missing, raise an actionable installation message. Do not simulate incremental appends with repeated `mode="a"` calls.

**Step 5: Run tests and commit**

```powershell
python -m pytest tests/output/test_rasters.py -q
```

```bash
git add hydrofragments/output/rasters.py pyproject.toml tests/output/test_rasters.py
git commit -m "feat: write verified georeferenced raster products"
```

### Task 10: Build one atomic bundle finalizer and manifest

**Files:**

- Create: `hydrofragments/output/bundle.py`
- Modify: `hydrofragments/output/manifest.py`
- Create: `tests/output/test_bundle.py`
- Modify: `tests/output/test_manifest.py`
- Modify: `tests/output/test_manifest_hydroseason.py`

**Step 1: Freeze output transaction semantics**

Treat configured `output_dir` as one final run directory. Before expensive work, require the target to be absent or verified empty; reject any non-empty target. Create a same-filesystem sibling staging directory named with the target, run ID, and `.staging`, and write an ownership/transaction record there first. Writers create and validate every table, raster, and vector entirely inside staging. The partitioned metrics dataset is one directory artifact, not unrelated files.

Write `config.json` in staging, build the artifact inventory, then write `run_manifest.json` exactly once and last inside staging. Reopen and validate the complete staged bundle and close every file/GDAL handle. If the configured target was pre-existing and verified empty, remove only that empty directory at commit time. Commit the entire bundle with one same-filesystem directory rename. A crash before that rename leaves no final output directory; a crash after it leaves the complete manifest-bearing bundle.

On startup, inspect only sibling staging directories carrying HydroFragments’ ownership record for this exact target. A matching, complete staged transaction may be validated and committed; a matching incomplete transaction may be safely resumed from valid checkpoints or removed. A mismatched/unknown directory is never deleted automatically. Final output is never overwritten.

**Step 2: Add failure-injection tests**

Inject failure:

- halfway through a TIFF window loop;
- between GPKG layer appends;
- during NetCDF close;
- during artifact validation;
- immediately before manifest publication;
- in a subprocess immediately before and after the directory-commit boundary.

Assert ordinary exceptions clean owned staging state, killed pre-commit processes leave no final target, killed post-commit processes leave a fully valid bundle, no handle prevents Windows directory rename, no pre-existing file is overwritten, and deterministic retry/recovery works.

**Step 3: Extend the manifest inventory**

For every artifact include:

```text
relative_path
media_type
byte_size
sha256
producer/algorithm version
scientific_config_hash
execution_config_hash
```

For a directory artifact such as partitioned Parquet, `byte_size` is the sum of regular-file sizes and `sha256` is a deterministic tree digest over sorted relative file paths plus each file’s SHA-256. Hash each final artifact in the same sequential validation pass where practical and include that cost in output timing.

For spatial artifacts also include applicable CRS, affine transform, shape, dtype, nodata, layers, band/time labels, feature/band counts, hydrological-year pairs, and checkpoint identity. `validate_result_bundle()` verifies existence, digest, config hashes, and selected spatial metadata.

Include phase timings and peak RSS fields without claiming benchmark comparability inside the manifest.

Add a legacy-bundle test proving manifest `1.0.0` still validates without the new digest/spatial fields. New writers always produce `1.1.0`; validation must not rewrite an older user bundle.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/output/test_bundle.py tests/output/test_manifest.py tests/output/test_manifest_hydroseason.py -q
```

```bash
git add hydrofragments/output/bundle.py hydrofragments/output/manifest.py tests/output/test_bundle.py tests/output/test_manifest.py tests/output/test_manifest_hydroseason.py
git commit -m "feat: publish validated result bundles atomically"
```

### Task 11: Integrate both public workflows

**Files:**

- Modify: `hydrofragments/api.py`
- Modify: `hydrofragments/workflow.py`
- Modify: `hydrofragments/models.py`
- Create: `tests/integration/test_spatial_exports.py`
- Modify: `tests/api/test_public_api.py`
- Modify: `tests/integration/test_dea_workflow.py`

**Step 1: Separate core analysis from finalization**

Refactor into an internal core that returns metrics, coverage, provenance, timings, and completed optional checkpoints without publishing a manifest. Both public workflows call this core. The DEA workflow attaches its zone result and spatial context before invoking the same bundle finalizer when an output directory is configured. Without one, build the equivalent in-memory manifest and perform no output writes.

There must be one owner of:

- table/coverage writes;
- spatial writes;
- artifact validation/inventory;
- manifest publication.

Delete the current metadata-before-tables and manifest-rewrite sequence.

**Step 2: Add preflight validation**

Before expensive analysis:

- resolve selected metric IDs plus hidden support;
- resolve selected products and their prerequisites;
- validate source grid for any spatial request;
- validate optional writer availability;
- calculate and collision-check final paths.

`analyze()` can produce zones/reaches only when those inputs are explicitly supplied. `analyze_from_dea()` passes the already computed zone result and `SpatialContext`; neither path fabricates missing products.

**Step 3: Add end-to-end tests**

Run a small offline synthetic cube through both entry points and assert:

- expected table, GPKG, TIFF, optional NetCDF, config, and manifest products exist;
- the manifest inventories every artifact and validates;
- metrics are byte-identical with exports off and on;
- export-off spies see no vector/export checkpoint, polygon, or raster writer calls and no extra source reads; any bounded scientific spill is run-owned and removed;
- requested unavailable products fail before analysis;
- deterministic single-worker and multi-worker outputs match;
- a failed output prevents manifest publication and leaves the input/checkpoint intact.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/integration/test_spatial_exports.py tests/integration/test_dea_workflow.py tests/api/test_public_api.py -q
```

```bash
git add hydrofragments/api.py hydrofragments/workflow.py hydrofragments/models.py tests/integration/test_spatial_exports.py tests/integration/test_dea_workflow.py tests/api/test_public_api.py
git commit -m "feat: finalize complete outputs from both workflows"
```

### Task 12: Prove speed, memory, exactness, and portability

**Files:**

- Create: `tests/benchmarks/test_spatial_export_baseline.py`
- Modify: `hydrofragments/benchmarks/end_to_end_workflow.py`
- Modify: `hydrofragments/benchmarks/_e2e_worker.py`
- Modify: `tests/benchmarks/test_end_to_end_workflow.py`
- Modify: `.github/workflows/ci.yml`
- Create: `benchmarks/results/dynamics_spatial_exports.json`
- Create: `benchmarks/results/dynamics_spatial_exports.md`

**Step 1: Extend the existing subprocess benchmark**

Use a repository-owned driver and one copied/read-only local monthly Zarr fixture so network/STAC latency is outside the promotion gate. Run the frozen base commit and candidate from separate wheels or worktrees with matched dependency environments; write each trial to a new empty output directory on the same filesystem. Record at least five isolated runs after one warm-up, interleaving base/candidate order when practical:

- input shape, months, chunks, worker count, dependency versions, and commit;
- core analysis wall time;
- output/finalization wall time;
- total wall time through artifact reopen/validation;
- peak RSS of the subprocess tree;
- source read count and connected-component label count;
- output bytes by product;
- exact metric/coverage equality and raster/vector parity checks.

Report median and spread. The driver and every fixture path must be repository-owned and portable; do not depend on a developer-private scratch runner.

The existing live DEA benchmark may remain as a separately labelled acquisition benchmark, but its network-dependent numbers cannot pass or fail this output-processing gate.

**Step 2: Benchmark these scenarios**

1. Current/base commit, default outputs off.
2. Candidate, default outputs off.
3. Candidate with persistence/temporal/refuge GeoTIFFs.
4. Candidate with monthly pool GPKG.
5. Candidate with all appropriate spatial products.
6. Candidate with NetCDF opt-in.
7. Long synthetic record (for example 480 small months) to prove memory does not scale with retained months.
8. Short, large-spatial, chunked record with sparse and single-large-component variants to prove byte admission and the explicit morphology guard.
9. Output-only retry from a valid completed spatial checkpoint to prove source reads and labelling are skipped.

Run at `max_workers=1` and the documented default. Tune workers only if results justify it.

**Step 3: Enforce gates**

- Export-off median total time: `candidate <= baseline * 1.10`.
- Export-off peak RSS: no material regression beyond measurement noise; record an explicit byte/MiB tolerance from repeat variance.
- Enabled all-product peak RSS: `candidate <= core_peak_rss * 1.25` on the controlled fixture.
- Large-spatial incremental peak RSS above idle interpreter baseline: no more than `1.25 *` the configured live-byte admission, except that the single-component over-budget case must fail before its crop is materialized.
- Source reads: no increase caused solely by enabling exports.
- Label passes: one per month/window set requiring patch support.
- Metric and coverage outputs: exact equality on/off and single/multi-worker.
- Rasters/vectors: analytic parity and successful reopen validation.

If a gate fails, profile and fix it; do not weaken the threshold without recording the evidence and user-visible trade-off in the benchmark report.

**Step 4: Add Windows writer CI**

On Windows/Python 3.13, run the grid, raster, vector, bundle, and small integration suites. This specifically covers GDAL handle closure, path replacement, and file-lock behaviour. Keep the full Linux matrix from Task 1.

**Step 5: Commit**

```bash
git add hydrofragments/benchmarks/end_to_end_workflow.py hydrofragments/benchmarks/_e2e_worker.py tests/benchmarks/test_end_to_end_workflow.py tests/benchmarks/test_spatial_export_baseline.py .github/workflows/ci.yml benchmarks/results/dynamics_spatial_exports.json benchmarks/results/dynamics_spatial_exports.md
git commit -m "perf: gate dynamics and spatial export costs"
```

### Task 13: Make the feature discoverable and honest

**Files:**

- Modify: `README.md`
- Create: `docs/spatial_exports.md`
- Modify: `docs/metrics/dynamics.md`
- Modify: `docs/final_metrics_covered.md`
- Modify: `docs/superpowers/specs/2026-08-12-dynamics-and-spatial-export-design.md`
- Create: `CHANGELOG.md` if this repository has not added one before implementation
- Create: `examples/spatial_exports.py`
- Modify: `tests/docs/test_examples.py`

**Step 1: Correct the existing README contract**

Document actual paths and responsibilities:

- partitioned metrics directory rather than a fictitious `metrics.parquet`;
- `metric_coverage.csv` rather than coverage Parquet unless implemented;
- `run_manifest.json` rather than `manifest.json`;
- when plain `analyze()` writes outputs and when `HydroResult.write()` is table-only.

**Step 2: Add a product-selection guide**

For each spatial product document:

- scientific meaning and units/codes;
- prerequisites and which workflow supplies them;
- config example with exports off/default and opt-in examples;
- output path/layer/band schema;
- CRS/grid preservation and absence of automatic reprojection;
- nodata/undefined semantics, especially dry/dry refuge pairs;
- expected performance/storage cost;
- QGIS/GDAL/Python opening examples;
- the checkpoint-only vector design.

Include a small decision table: “Need statistical modelling → tables; need cartography → GeoTIFF/GPKG; need multidimensional scientific exchange → opt-in NetCDF.”

**Step 3: Add an offline executable example**

Use a deterministic synthetic cube and write a small result bundle. The example must run from the repository without DEA credentials or private scratch files, validate its manifest, and show one plot/readback. Add it to CI if runtime is modest.

**Step 4: Reconcile spec and changelog**

Update the design specification to match the frozen contracts in this audited plan. Clearly label package/config/metric/manifest versions. Record the user-facing performance result, not an unsupported “faster” claim.

**Step 5: Run docs/example checks and commit**

```powershell
python -m pytest tests/docs tests/integration/test_spatial_exports.py -q
```

```bash
git add README.md docs CHANGELOG.md examples tests/docs
git commit -m "docs: explain dynamics metrics and spatial outputs"
```

### Task 14: Final verification and release evidence

**Files:**

- Modify only files required by verified failures.
- Update: `benchmarks/results/dynamics_spatial_exports.md`

**Step 1: Run static and focused checks**

Use the repository’s configured formatter/linter/type checker. Then run the focused test files from Tasks 1-13.

**Step 2: Run the full suite in clean supported environments**

```powershell
python -m pytest -q
```

Run at least one clean Linux environment from the supported matrix and the Windows spatial-output job. Record exact pass/skip/fail counts in release evidence; do not bake the count into the implementation.

**Step 3: Inspect generated artifacts manually once**

- Open the GeoPackage and GeoTIFFs in QGIS or `gdalinfo`/`ogrinfo`.
- Confirm visible alignment of zones, pool polygons, reaches, and rasters.
- Inspect nodata, categorical legend, band names, and HY labels.
- Validate the same bundle with `validate_result_bundle()`.

This is a release check, not a substitute for automated round-trip tests.

**Step 4: Re-run the controlled benchmark**

Confirm every gate in Task 12. Commit the final report containing raw commands, machine/environment details, summary tables, and exactness evidence.

**Step 5: Review the diff and commit**

```bash
git status --short
git diff --check
git diff --stat
```

Inspect every changed file, ensure no private paths or generated output bundles are staged, then commit any evidence-only update.

---

## 6. Product availability matrix

| Product / metric | `analyze()` with cube only | `analyze()` with explicit optional inputs | `analyze_from_dea()` |
|---|---:|---:|---:|
| extent contraction | unavailable: no HY/dual-composite inputs | yes, with HY extent and dual composites | yes when derived |
| reconnection via LPI | unavailable: no HY anchors | yes, with HY extent; hidden LPI support | yes |
| reconnection via LPSEC | unavailable: no HY/channel inputs | yes, with HY extent and channel profiles/context | yes when profiles exist |
| scalar refuge stability | unavailable: no HY anchors | yes, with HY extent and >=2 valid anchors | yes |
| persistence/temporal rasters | yes, with a valid grid | yes | yes |
| refuge-stability rasters | unavailable: no HY anchors | yes, with HY extent and >=2 valid anchors | yes |
| monthly pool polygons | yes, when selected before analysis | yes | yes |
| zone raster/polygons | unavailable | yes, with zone input | yes |
| reach layers/monthly table | unavailable | yes, with spatial context/profile matrix | yes |

Unavailable requested products fail preflight. They do not disappear silently and do not produce empty placeholder files.

---

## 7. Risks and mitigations

| Risk | Mitigation / proof |
|---|---|
| Dynamics-only selection skips LPI support | registry dependency test plus end-to-end dynamics-profile test |
| Invalid observations treated as dry | common-valid-support contract and partial-validity analytic tests |
| Time or raster size exhausts memory | weakref/byte-admission tests, 480-month and large-spatial benchmarks, checkpoint-backed counters/labels, single-component fail-fast guard |
| Polygon export repeats labelling | measured-label bundle, read/label spies, raster-vector parity |
| GPKG append silently changes schema | explicit first-layer schema, per-partition validation, reopen test |
| Shape-equal shifted raster is accepted | `SpatialGrid` affine/coordinate equality regression |
| NetCDF path becomes slow/default | explicit opt-in extra; report output time separately |
| Partial bundle appears valid after exception or process death | sibling-directory staging, full staged validation, manifest last, one directory commit, subprocess-kill tests |
| Windows cannot replace an open GDAL file | close-before-replace invariant and Windows CI smoke job |
| Output settings alter scientific identity | separate scientific and execution hashes |
| New features contradict public docs | README correction and executable offline example |
| “Fast” claim lacks evidence | controlled output-inclusive benchmark and stored result report |

---

## 8. Primary implementation references

- Rasterio windowed I/O: https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html — write datasets larger than RAM and align windows with storage blocks.
- Pyogrio writing: https://pyogrio.readthedocs.io/en/latest/introduction.html#writing — GeoPackage support and exact-schema requirement for appending.
- Xarray I/O: https://docs.xarray.dev/en/stable/user-guide/io.html — NetCDF writes are not incremental; compressed NetCDF has important performance constraints.
- Local HydroSeason evidence: `hydroseason/hydro_year.py`, `hydroseason/_io_wofs_zarr.py`, `hydroseason/_historical_water_mask.py`, `hydroseason/_report_export.py`, and `scripts/benchmark_wofs_cache.py` in the sibling repository.

---

## 9. Definition of done

This work is done only when:

- the scientific edge cases in Section 2 pass;
- no-export and export-enabled results are metrically identical;
- the monthly pipeline’s memory is bounded by concurrency/fixed accumulators;
- all requested products are present, valid, inventoried, and open in standard GIS tools;
- unavailable requested products fail before expensive work;
- failure injection never leaves a final manifest or overwrites pre-existing artifacts;
- controlled benchmark gates pass and the evidence is committed;
- clean supported Linux tests and Windows spatial writer tests pass;
- public docs match the actual filesystem/API contract;
- there are no private paths, untracked generated bundles, or unverified speed claims in the release diff.
