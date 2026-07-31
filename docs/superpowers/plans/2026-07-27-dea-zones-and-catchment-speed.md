# Fast DEA Acquisition-to-Metrics Implementation Plan

> **Supersession note (2026-07-31):** HydroSeason's DEA-zones APIs described
> here landed for the first public remote-sensing release, versioned
> `0.1.0`, not `0.1.1`. `0.1.1` was an unpublished coordination bump during
> development and was never released. Historical ledger text below that
> narrates the `0.1.1` target is preserved as-written; active install/release
> instructions in this doc now read `hydroseason==0.1.0`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Minimise elapsed time from catchment/date input to final metrics
table while preserving exact scientific outputs and calculating every
runtime-wired metric whose dependencies are available.

**Architecture:** `hydroseason` remains acquisition/cache owner;
HydroFragments remains metric owner. One native DEA-statistics read produces
exact 30 m zones plus a conservative coarse planning footprint. Hydroseason
uses that footprint to prune remote WOfS work and populate a resumable sparse
cache; HydroFragments processes only independent active windows, reuses shared
intermediates, auto-selects available runtime metrics, and writes one table plus
one metric-coverage table.

**Tech Stack:** Python 3.10+, hydroseason 0.1.1, xarray, Dask, odc-stac,
pystac-client, Rasterio/rioxarray, Zarr v2, NumPy, SciPy, scikit-image,
GeoPandas/Shapely, pandas, pytest.

Plan date: 2026-07-27

## Global Constraints

- Branch base: `development`.
- Exact native-zone and final-metric parity against an unpruned 30 m run.
- `count_wet > 0` planning footprint must cover 100% of native wet pixels.
- APSEC/LPI/reference-area denominators remain full catchment `aoi_mask`.
- Monthly coverage denominator is approved as conservative potential-water
  `analysis_mask`, not full catchment.
- Dates not covered by a proven mask fail open to full-AOI acquisition.
- Never use nearest/mode/mean to downsample a pruning mask; aligned max/any only.
- Cache identity includes both masks, grid, temporal coverage, product
  provenance, aggregation factor, safety halo, and composite bundle.
- Every default run attempts all runtime-wired metrics; missing dependencies
  produce `skipped (reason)`, never fabricated inputs or a failed run.
- Keep remote acquisition resumable and bounded by one annual graph plus
  bounded spatial/month work.
- Preserve deterministic row order and byte-identical values across worker
  counts.
- Existing explicit narrow metric profiles remain supported for expert runs.
- No HydroFragments imports inside hydroseason; dependency direction stays
  `HydroFragments -> hydroseason`.

---

Five workstreams:

1. Define full-catchment versus analysis-footprint semantics.
2. Read DEA statistics once for exact zones and conservative acquisition mask.
3. Prune/resume WOfS acquisition in hydroseason and expose its canonical cache.
4. Process active windows and every available runtime-wired metric by default.
5. Ship one plain-language notebook ending in checkable metric and coverage CSVs.

## File and ownership map

### `hydroseason` acquisition repository

- `hydroseason/_io_dea_stats.py`: native statistics loader and conservative
  planning-footprint builder; no metric logic.
- `hydroseason/_spatial_plan.py`: source/storage-aligned active acquisition
  windows.
- `hydroseason/_io_geo.py`: one shared annual WOfS graph, primary max-water
  monthly mask, and secondary median extent counts from shared observations.
- `hydroseason/_io_wofs_acquire.py`: mask resolution, cache identity,
  fail-open coverage checks, annual resume, diagnostics.
- `hydroseason/_io_wofs_zarr.py`: sparse canonical masks, dual-composite extent
  counts, public lazy cache reader.
- `hydroseason/io.py` and `hydroseason/__init__.py`: public acquisition and
  cache interfaces consumed by HydroFragments.

### HydroFragments metric repository

- `hydrofragments/io/dea.py`: thin conversion from hydroseason DEA statistics
  to `WoStatistics`/zones; no duplicate STAC loader.
- `hydrofragments/workflow.py`: user-input-to-table orchestration only.
- `hydrofragments/models.py`: aligned `aoi_mask` and `analysis_mask` contracts,
  optional advanced inputs, result coverage table.
- `hydrofragments/metrics/registry.py`: capability-driven `all_available`
  default over runtime-wired metrics only.
- `hydrofragments/spatial/active_windows.py`: independent analysis windows and
  native-grid safety proof.
- `hydrofragments/compat.py` and `hydrofragments/metrics/patches.py`: bounded
  month/window materialisation and one combined patch-properties reduction.
- `hydrofragments/api.py`: dependency discovery, shared metric DAG, zones, and
  output assembly.
- `hydrofragments/output/tables.py` and `hydrofragments/output/manifest.py`:
  metric table, coverage table, mask/composite provenance.
- `benchmarks/end_to_end_workflow.py`: cold/warm input-to-table benchmark.
- `examples/end_to_end_workflow.ipynb`: default user workflow.

## Approved end-to-end data flow

1. Validate AOI, dates, resolution, optional drainage, cache location, and
   compute budget.
2. Call public `hydroseason.open_wo_statistics` once at native 30 m for
   `count_wet` and `count_clear`; derive frequency percentage lazily.
3. Build native DEA zones. Separately max-pool `count_wet > 0` using the
   benchmark-selected aligned factor (initial candidates: 90/120 m), expand by
   safety halo, and prove the chosen mask covers every native wet pixel. The
   benchmark selects the default once; production runs do not race candidates.
4. Convert connected planning-mask regions to source/storage-aligned active
   windows. If requested dates exceed proven statistics coverage, use full AOI.
5. `hydroseason.acquire_wofs_cache` builds one source graph per year,
   reads only selected windows, writes sparse monthly max-water masks, and
   writes max-water plus median APSEC counts from shared source observations.
6. Open completed masks lazily. Rasterize full catchment to `aoi_mask`; expand
   planning footprint to native grid as `analysis_mask`.
7. Process each month once across independent active windows. Concatenate patch
   properties across windows, then reduce once so LPI/AWRe/AWMSI/width match
   full-mask results. Accumulate temporal sufficient statistics while data is
   resident.
8. Resolve `all_available`: compute every runtime-wired metric whose dependency
   evidence exists; record every unavailable runtime metric with exact reason.
9. Finalize temporal, hydro-year, dynamics, and channel rows. Emit the exact DEA
   zones as a separate output; optional zone summaries remain configuration-only.
10. Write tidy metrics, wide metrics, metric coverage, config, manifest, and
    timing diagnostics atomically.

## Decisions locked before writing this plan

- **Zones from DEA, metrics stay native.** `build_zones()` consumes the DEA
  multi-year `frequency` + `count_clear` as the *zoning surface only*.
  `occurrence` / `refuge_area` / `recurrence` keep the season-stratified
  P-native estimator. Rationale: DEA `frequency` is a **pooled**
  `count_wet/count_clear` over 1987-2025 and does not correct the seasonal MNAR
  pattern documented in `docs/audit/evidence/validity_reliability_report.md` §4,
  which Decision Gate 0 (U2/Q1) binds us to. Reported persistence numbers must
  not regress.
- **Both zoning and metric-loop optimisation are in scope.**
- **Default means all runtime-wired metrics supported by available inputs.**
  Core, recurrence, and hydroperiod run from a normal water/valid cube;
  pool-width, channel, and dynamics metrics auto-enable only when their locked
  dependencies exist. Kernel-only, validation-disabled, and runtime-deferred
  metrics remain explicit skips.
- **Two spatial denominators are intentional.** `aoi_mask` remains the fixed
  reference for APSEC/LPI. `analysis_mask` is the conservative potential-water
  footprint used for monthly validity coverage, acquisition, and processing.

## Performance discoveries from `hydroseason` (2026-07-28)

`D:\RLH\5.6\repos\hydroseason` now has a working DEA-statistics wet-mask
path (`hydroseason/_io_dea_stats.py`, commits `011c06f` through `b37ce3b`).
Useful findings:

- **Do not coarsen the zoning surface.** Keep `frequency`, `count_clear`, and
  `count_wet` on the native 30 m DEA grid for zone assignment. Coarsening
  frequency changes threshold membership and can erase narrow channels.
- **A separate coarse planning mask is worthwhile.** It may gate remote reads,
  chunk selection, or polygonisation only. Derive `count_wet > 0` at 30 m,
  then aggregate aligned integer blocks with boolean `any` / numeric `max`.
  When expanded back to 30 m, every coarse wet cell becomes wholly included.
  Never downsample this mask with nearest, mode, or mean: an isolated 30 m wet
  pixel can disappear. Benchmark 90 m (3x) and 120 m (4x); do not use an
  unaligned 100 m target as the correctness reference.
- **Coarsening alone does not speed the HydroFragments metric loop.** A full
  shape boolean mask still makes `scipy.ndimage.label` scan the full
  catchment. The mask only pays if it prevents source chunks from being read,
  reduces the array to safe active windows, or avoids expensive vectorisation.
- **Vector geometry can become the bottleneck.** On `gilbert_river_qld`, the
  hydroseason branch records about 24 s for DEA fetch+union, versus about
  490 s for a 150 m Shapely close and 760 s for a 300 m Shapely buffer.
  Prefer a raster/chunk predicate. If geometry is required, use
  `rasterio.features.shapes` followed by `shapely.union_all`; keep close/buffer
  off by default and benchmark them independently.
- **Wet-mask correctness is fail-open.** Use `count_wet > 0`, union sources,
  reject empty/partial/failed masks, and fall back to full coverage. A mask
  used for pruning must be a superset of every wet pixel in the analysed
  period. The current multi-year product ends in 2025, so it cannot by itself
  prune a cube containing 2026. Union a current local ever-wet summary or do
  not prune that uncovered period.
- **Cache mask provenance.** Include source collection/version/item IDs,
  resolution, aggregation factor, safety dilation, time span, grid transform,
  and a stable mask digest in cache identity. Pruned and full-coverage caches
  must never collide.
- **Reuse the hardened hydroseason path.** Keep `DEAStatsUnavailable`, the
  Windows-safe source deadline, stable wet-mask digest, annual cache resume,
  atomic publication, diagnostics callback, and lazy
  `open_completed_mask_cache`. Replace only the unsafe scientific/planning
  coupling: the current `max(resolution, 100)` plus nearest-resampled polygon
  must become one native statistics read followed by conservative max pooling
  and storage-aligned windows.
- **Prove source-lineage compatibility before pruning.** The planning statistic
  must be the archive summary for the same `ga_ls_wo_3` classification lineage
  used by acquisition, and the requested dates must lie inside its recorded
  coverage. A version/lineage mismatch, missing year, or newer date fails open
  to full AOI; it must not be treated as an empty dry footprint.
- **Bound and measure concurrency.** Hydroseason gained more from lazy,
  storage-aligned 512 px work and bounded batches than from assuming a worker
  count. HydroFragments already has `config.compute.workers` (default 1); reuse
  it. Do not add a second `n_workers` field or default to CPU count.

Decision: **native 30 m DEA statistics remain authoritative for zoning; a
conservative coarse mask is the default performance-only planning artifact
when its temporal coverage and superset proof pass.** Otherwise workflow fails
open to full AOI. Promotion still requires exact native-zone parity, exact
metric checksums, and a measured wall-clock/RSS win on a real catchment.

## Product facts (verified 2026-07-28)

Source: [DEA Water Observations Statistics product page](https://knowledge.dea.ga.gov.au/data/product/dea-water-observations-statistics-landsat/).

| Field | Value |
|---|---|
| Multi-year product | `ga_ls_wo_fq_myear_3` (v2.1.0), 1987-2025 |
| Calendar year | `ga_ls_wo_fq_cyear_3`, 1986-2025 |
| Seasonal | `ga_ls_wo_fq_nov_mar_3`, `ga_ls_wo_fq_apr_oct_3` |
| Bands | `count_wet` (int16), `count_clear` (int16), `frequency` (float32) |
| Resolution / CRS | 30 m, EPSG:3577 (GDA94 Australian Albers) |
| Access | STAC `https://explorer.sandbox.dea.ga.gov.au/stac`, S3 `data.dea.ga.gov.au`, OWS |

`frequency = count_wet / count_clear`, so `count_clear` is the native support
counter and maps directly onto our `valid_count` / `min_valid_obs` floor. This
is the key alignment: we are not inventing a support proxy.

For this implementation, load only `count_wet` and `count_clear`; derive the
canonical percentage lazily as
`100 * count_wet / count_clear.where(count_clear > 0)`. The published
`frequency` band is redundant and its documentation uses both ratio-like and
percentage language. Deriving from counts removes that scale ambiguity and one
remote asset read while preserving the documented statistic.

## Why this is faster

Today zoning needs the whole `(time, y, x)` cube reduced by
`_season_stratified_occurrence` — a `groupby("time.month").sum()` over every
month. At catchment scale that is the dominant read. DEA multi-year is **two
2-D rasters already reduced by GA**. Zoning cost drops from `O(T·Y·X)` to
`O(Y·X)` with no local compute.

---

## Workstream 1 — DEA multi-year zoning

### 1.1 Promote one native DEA-statistics loader from hydroseason

**Files:**

- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_io_dea_stats.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\io.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\__init__.py`
- Create: `hydrofragments/io/dea.py`
- Test: `D:\RLH\5.6\repos\hydroseason\tests\test_io_dea_stats.py`
- Test: `tests/io/test_dea.py`

**Interfaces:**

- Produces in hydroseason:

```text
open_wo_statistics(
    aoi: Any,
    *,
    product: str = "ga_ls_wo_fq_myear_3",
    stac_url: str = "https://explorer.sandbox.dea.ga.gov.au/stac",
    resolution: float = 30.0,
    crs: str = "EPSG:3577",
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset
```

- Produces in HydroFragments: frozen `WoStatistics` adapter containing
  `frequency` (0-100 float32, lazily derived), `count_wet`, `count_clear`,
  `product`, `version`, `crs`, `time_span`, and `provenance`.

Requirements:
- Reuse hydroseason's scoped unsigned-COG/GDAL configuration (equivalent to
  `AWS_NO_SIGN_REQUEST=YES`); do not permanently mutate the caller's process
  environment.
- `odc.stac.load` with explicit `crs`/`resolution` so we never resample
  implicitly.
- Request only `count_wet` and `count_clear`; derive `frequency` lazily and
  record that derivation in provenance.
- Return dask-backed arrays; **do not** `.load()` in the loader.
- Keep zoning arrays at native 30 m. `resolution` selects the native grid
  explicitly; it must not become a scientific-resolution knob for
  `zones_from_wo_statistics`.
- Hard-fail if the returned CRS is geographic — `guard_area_metric_crs`
  already encodes that rule, so call it rather than duplicating the check.
- Record `product`, version, and the STAC item IDs in `provenance`; this must
  reach the run manifest (see 1.4).
- Set explicit STAC connect/read timeouts plus an overall load deadline. A
  zoning-source failure must leave the local-cube zoning path available.

- [ ] **Step 1: Write loader-contract tests.** Mock STAC items and assert one
  search, exactly two requested bands, native 30 m grid, Dask-backed outputs,
  derived 0-100 frequency, nodata handling, and provenance.
- [ ] **Step 2: Run the focused tests and verify failure.**

```powershell
python -m pytest tests/test_io_dea_stats.py -q
python -m pytest tests/io/test_dea.py -q
```

- [ ] **Step 3: Split current private wet-polygon loader into public native
  statistics loading plus downstream planning conversion.** Remove the current
  implicit `max(resolution, 100)` behavior from the scientific data loader.
- [ ] **Step 4: Export the public loader and add the HydroFragments adapter.**
- [ ] **Step 5: Run both focused suites and the existing hydroseason I/O suite.**
- [ ] **Step 6: Commit independently in each repository.**

```powershell
git commit -m "feat: expose native DEA water statistics"
git commit -m "feat: adapt DEA statistics for zoning"
```

### 1.2 Adapt `build_zones()` for external support counts

`hydrofragments/spatial/zones.py:18` currently takes `occurrence` (percent),
`max_wet_mask`, `valid_count`. The signature already fits — it needs no
breaking change:

- `occurrence` <- canonical `frequency` percentage (already 0-100)
- `valid_count` <- `count_clear`
- `max_wet_mask` <- `count_wet > 0`

Add a thin adapter so callers can't wire it wrong:

```python
def zones_from_wo_statistics(stats: WoStatistics, *, config, drainage_mask=None) -> ZoneResult
```

It maps the fields above, passes `config.zones.t_persist` / `t_season` and
`config.validity.min_valid_obs`, and stamps `ZoneResult` with the source
product. Add `source: str` to `ZoneResult` (default `"occurrence"`) so
downstream can tell DEA-derived zones from cube-derived ones.

**Threshold semantics check.** `build_zones` compares `frequency` against
`t_persist`/`t_season` which `config.py` validates as *fractions* in [0,1],
but `compute_occurrence` returns *percent* and the existing call sites pass
percent. That inconsistency is latent today. Pin it down in this workstream:
normalise on percent at the `build_zones` boundary, assert
`0 <= t_season < t_persist <= 1` fractions are scaled once, and add a test
that a 45% pixel lands in zone 3 for the default `t_persist=0.50`. Do not
leave this ambiguous — it silently mislabels every zone if wrong.

- [ ] **Step 1: Add failing percentage-boundary and adapter tests.** Cover
  9.9%, 10%, 45%, 50%, 50.1%, support floor, nodata, and no-wet pixels.
- [ ] **Step 2: Normalize thresholds once at `build_zones` boundary and add
  `ZoneResult.source`.**
- [ ] **Step 3: Run zone and occurrence suites; commit.**

```powershell
git commit -m "fix: normalize DEA zone threshold units"
```

### 1.3 Preserve the persistence-zone circularity guard

`docs/HydroFragments_v1.2_spec.md` locks zones as configuration strata, never
persistence strata. `guard_persistence_zone`
(`hydrofragments/guards/scientific.py:30`) therefore continues to forbid
`occurrence`/`refuge_area`/`recurrence` rows for zones 1-4. A DEA product is
external to this run, but it measures the same wet-frequency phenomenon; source
provenance must not silently waive the scientific guard.

Keep the fastest default as one AOI/channel calculation of every available
runtime-wired metric and emit the exact DEA zone raster as a separate artifact.
Do not add per-zone metric multiplication to this default workflow. If an
explicit zone-summary path is added later, it may use only the configuration-
metric allowlist; persistence metrics remain refused for both local- and
DEA-derived zones.

- [ ] **Step 1: Add tests proving persistence metrics remain AOI/channel-only
  for both local and DEA zone provenance.**
- [ ] **Step 2: Add a workflow regression test proving DEA zones are emitted
  without multiplying the default metric rows.**
- [ ] **Step 3: Document the unchanged scientific contract and commit.**

```powershell
git commit -m "test: preserve persistence zone guard"
```

### 1.4 Manifest provenance

`write_run_metadata` already takes `input_fingerprint`. Add the DEA product id,
version, STAC item IDs, and the zone thresholds so any zone raster is
reproducible from the manifest alone.

- [ ] **Step 1: Add manifest round-trip tests for DEA and both mask digests.**
- [ ] **Step 2: Persist product/grid/time-span/threshold/coverage provenance.**
- [ ] **Step 3: Commit.**

```powershell
git commit -m "feat: record DEA zone and planning provenance"
```

### 1.5 Conservative coarse wet planning footprint (performance-only semantics)

**Files:**

- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_io_dea_stats.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_spatial_plan.py`
- Test: `D:\RLH\5.6\repos\hydroseason\tests\test_io_dea_stats.py`
- Test: `D:\RLH\5.6\repos\hydroseason\tests\test_spatial_plan.py`

Add a distinct artifact; never pass it to `build_zones()` as frequency or
support. Public interface:

```python
@dataclass(frozen=True)
class WetPlanningFootprint:
    native_mask: xr.DataArray
    coarse_mask: xr.DataArray
    active_windows: Sequence[GridWindow]
    factor: int
    safety_cells: int
    digest: str
    covered_years: Sequence[int]
    source_collection: str
    source_version: str
    source_lineage: str
    geometry: geopandas.GeoDataFrame | None = None

build_wet_planning_footprint(
    stats: xr.Dataset,
    *,
    factor: int = 4,
    safety_cells: int = 1,
    requested_years: Collection[int],
) -> WetPlanningFootprint
```

Contract:

- Start from native `count_wet > 0`; use aligned
  `coarsen(y=factor, x=factor, boundary="pad").max()` only.
- Preserve edge cells with padding rather than dropping partial blocks.
- Expand/reproject conservatively; apply one coarse-cell safety dilation when
  grids are not exactly aligned.
- Assert every native wet pixel is covered after round-trip expansion in tests.
- Use as a chunk/window predicate where possible. Do not create polygons
  unless the consumer requires them.
- Do not change AOI/reference-area denominators. Pixels skipped for I/O still
  represent dry/outside the planning footprint, not a smaller catchment.
- Fail open when product dates do not cover the analysed cube.
- Fail open when the statistics and daily-observation lineage/version contract
  is absent or incompatible.

Benchmark native/90 m/120 m planning masks on Fitzroy and one thin/braided
catchment. Record mask build time, polygon time if used, selected chunks,
bytes read, total runtime, peak RSS, and native-wet coverage. Promote only if
native-wet coverage is 100%, metric outputs are byte-identical, and end-to-end
runtime improves.

- [ ] **Step 1: Write failing superset/property tests.** Include isolated
  one-pixel water, thin diagonal/orthogonal channels, partial edge blocks,
  shifted-grid rejection, empty mask, missing requested year, and deterministic
  digest cases.
- [ ] **Step 2: Run focused tests and confirm failures.**
- [ ] **Step 3: Implement aligned max pooling, edge padding, coarse-cell halo,
  coarse vectorisation, and storage-aligned active windows.** No Shapely close
  or buffer in the default path.
- [ ] **Step 4: Add a round-trip proof:** expand the coarse mask to native grid
  and assert `native_mask <= expanded_coarse_mask` for every accepted plan.
- [ ] **Step 5: Add temporal fail-open:** incomplete requested-year coverage
  raises `DEAStatsUnavailable`; acquisition catches it and uses full AOI.
- [ ] **Step 6: Run unit suites and offline Fitzroy/Gilbert geometry benchmark.**
- [ ] **Step 7: Commit.**

```powershell
git commit -m "perf: build conservative DEA wet planning windows"
```

---

## Workstream 2 — pruned, resumable acquisition

### 2.1 Reuse one planning footprint through acquisition and analysis

**Files:**

- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_io_wofs_acquire.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_io_geo.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\_io_wofs_zarr.py`
- Modify: `D:\RLH\5.6\repos\hydroseason\hydroseason\io.py`
- Test: `D:\RLH\5.6\repos\hydroseason\tests\test_io_wofs_acquire.py`
- Test: `D:\RLH\5.6\repos\hydroseason\tests\test_io_wofs_zarr.py`

**Interface:**

```text
acquire_wofs_cache(
    stac_url: str,
    collection: str,
    aoi: Any,
    start_date: str,
    end_date: str,
    *,
    cache_root: str | Path,
    crs: int | str = 3577,
    resolution: float = 30.0,
    chunk_x: int = 512,
    chunk_y: int = 512,
    time_chunk: int = 12,
    majority: bool = True,
    offline: bool = False,
    force: bool = False,
    progress: bool = False,
    progress_desc: str | None = None,
    progress_position: int | None = None,
    diagnostics_callback: Callable[[dict[str, int]], None] | None = None,
    wet_aoi: Any = None,
    wet_mask: Literal["off", "dea_stats"] = "off",
    planning_footprint: WetPlanningFootprint | None = None,
    composite_bundle: Literal["legacy", "hydrofragments_v1"] = "legacy",
    compute_batch_size: int = 16,
    read_workers: int | None = None,
    resampling_policy: Literal["categorical_safe", "native_aligned"] =
        "categorical_safe",
    year_workers: int = 1,
) -> WOfSCacheHandle
```

`legacy` preserves every existing hydroseason result and cache identity.
`hydrofragments_v1` uses primary `max_water`, writes secondary median APSEC
counts, and records analysis-footprint semantics. Passing a prepared footprint
must never trigger another DEA-statistics query.

- [ ] **Step 1: Write failing query-count and pruning tests.** Assert one DEA
  statistics query before acquisition, zero duplicate stats queries, one STAC
  item query per uncached request, one shared load graph per year, and only
  active storage windows passed to the writer.
- [ ] **Step 2: Write failing cache-identity tests.** Factor, safety halo,
  footprint digest, covered years, and composite bundle must each change the
  identity; worker count must not.
- [ ] **Step 3: Run focused tests and verify failure.**
- [ ] **Step 4: Thread `planning_footprint` through planner, annual graph, pixel
  clip, writer, progress diagnostics, and manifest.** Keep explicit legacy
  `wet_aoi` compatibility but reject both arguments together.
- [ ] **Step 5: Publish `open_completed_mask_cache` from `hydroseason.io` and
  package `__init__`; remove its current internal-facade description.**
- [ ] **Step 6: Run hydroseason acquisition/Zarr suites.**
- [ ] **Step 7: Commit.**

```powershell
git commit -m "feat: reuse DEA planning footprints in WOfS acquisition"
```

### 2.2 Produce dual-composite extent counts during the source pass

HydroFragments' primary monthly mask remains `max_water`. Extent contraction
also needs median APSEC, but not a second full median raster cache. Compute both
water-pixel counts from the same classified daily observations while the
annual graph is resident; persist one compact record per month.

**Produced cache artifact:** `years/<year>/dual_extent_counts.json` with dates,
full-AOI pixel count, analysis-mask pixel count, max-water count, median-water
count, valid count over analysis mask, schema version, and content digest.

- [ ] **Step 1: Write a hand-traceable failing test** with daily wet/dry/invalid
  observations where max-water and median differ.
- [ ] **Step 2: Assert `hydrofragments_v1` builds one source graph, not one graph
  per composite.**
- [ ] **Step 3: Implement shared reductions and atomic JSON publication during
  annual writes.**
- [ ] **Step 4: Add `open_completed_dual_extent_counts(handle, start, end)` and
  export it publicly.**
- [ ] **Step 5: Verify legacy cache output and digests remain unchanged.**
- [ ] **Step 6: Commit.**

```powershell
git commit -m "perf: persist dual extent counts during WOfS acquisition"
```

### 2.3 Separate full AOI from analysis footprint

Hydroseason currently writes `-2` both outside user AOI and outside wet
footprint. That is sufficient for sparse storage but insufficient to recover
the two denominators later. Persist full-AOI and analysis-footprint geometry,
grid transform, pixel counts, and digests in root metadata. HydroFragments
re-rasterizes and verifies both masks on cache open.

- [ ] **Step 1: Test that pruned and unpruned caches retain identical
  `aoi_pixel_count` while `analysis_pixel_count` may differ.**
- [ ] **Step 2: Test tampered geometry/digest rejection.**
- [ ] **Step 3: Persist canonical WKB/CRS plus counts and digests atomically.**
- [ ] **Step 4: Run cache validation/resume tests.**
- [ ] **Step 5: Commit.**

```powershell
git commit -m "feat: distinguish AOI and analysis footprints in cache metadata"
```

---

## Workstream 3 — catchment-scale metric speed

Profiled from the actual call path: `analyze()` -> `section_compat_rows`
(`hydrofragments/compat.py:135`) -> per-month `analyze_patch_bundle`.

### 3.1 Model AOI/analysis masks and independent active windows

**Files:**

- Modify: `hydrofragments/models.py`
- Modify: `hydrofragments/api.py`
- Create: `hydrofragments/spatial/active_windows.py`
- Modify: `hydrofragments/metrics/extent.py`
- Test: `tests/spatial/test_active_windows.py`
- Test: `tests/metrics/test_analysis_mask_coverage.py`

Add optional aligned 2-D `aoi_mask` and `analysis_mask` fields to `WaterCube`.
All existing constructors default both to all-true over the spatial grid, so
current unpruned behavior stays unchanged. `aoi_mask` supplies fixed area;
`analysis_mask` supplies monthly validity coverage and active processing extent.

Define:

```text
independent_active_windows(
    analysis_mask: xr.DataArray,
    *,
    connectivity: Literal[4, 8],
    halo_pixels: int = 1,
    align_pixels: int = 512,
) -> Sequence[AnalysisWindow]
```

Distinct windows are valid only when no possible retained wet component can
cross between them under configured connectivity. Merge overlapping halos.

- [ ] **Step 1: Write failing mask-alignment and denominator tests.** Pin full
  catchment APSEC/LPI denominators and analysis-mask coverage denominator.
- [ ] **Step 2: Write failing window-equivalence property tests** over random
  masks plus thin/diagonal channels: concatenated window patch properties must
  equal full-mask properties.
- [ ] **Step 3: Add backward-compatible masks to `WaterCube` and validation.**
- [ ] **Step 4: Implement connected analysis regions, halo, alignment, merge,
  and independence assertions.**
- [ ] **Step 5: Run focused tests and existing public API snapshots.**
- [ ] **Step 6: Commit.**

```powershell
git commit -m "feat: separate AOI and active analysis masks"
```

### 3.2 The per-month loop is serial — benchmark bounded parallelism

`compat.py:223` iterates months one at a time, and each iteration does a full
label + measure pass. The months are **completely independent**. Nothing is
shared across iterations except read-only config.

- Extract the loop body into a module-level `_month_row` function (must be
  top-level to be picklable).
- Reuse existing `config.compute.workers` (default `1`). Validate it is at
  least 1. `1` keeps today's serial path; do not add `n_workers`.
- Do not pass a Dask/xarray graph or remote raster handle through a Windows
  process pool. Materialise aligned `water` + `valid_obs` month data together,
  then send bounded NumPy payloads to workers. Keep at most
  `2 * config.compute.workers` months in flight so memory is bounded.
- Benchmark serial, thread, and process execution at workers 1/2/4. Promote a
  parallel default only after catchment evidence; Windows spawn, graph
  serialization, memory multiplication, and I/O saturation can erase CPU
  gains.
- Preserve output order by index, not completion order — rows are keyed by
  `time_index` and must sort back deterministically.
- Determinism is a hard requirement here: the same input must give byte-identical
  output regardless of worker count. Add a test asserting
  `workers=1` and `workers=4` produce identical frames.

Target: useful speedup for patch families without peak RSS exceeding the
benchmark gate. Near-linear scaling is not assumed.

- [ ] **Step 1: Extract `_month_payload` and `_month_patch_properties` with
  serial behavior unchanged; pin current row snapshot.**
- [ ] **Step 2: Add deterministic serial/thread/process parity tests.**
- [ ] **Step 3: Implement bounded producer/consumer execution using
  `config.compute.workers`; validate workers >= 1 in `config.py`.**
- [ ] **Step 4: Run worker matrix on real catchment benchmark and retain serial
  default unless a parallel mode passes time/RSS gates.**
- [ ] **Step 5: Commit.**

```powershell
git commit -m "perf: bound parallel monthly patch analysis"
```

### 3.3 `_monthly_dataset()` loads the whole cube eagerly

`compat.py:118` calls `.load()` on the full `(time, y, x)` feature array, and
`compat.py:164` does the same for `valid_obs`. Comment says "section clips are
bounded" — true for a reach, **false for a catchment**. This is the most likely
OOM at the scale you want.

Fix: keep the array lazy and materialise bounded month payloads. Compute
`water` and caller-supplied `valid_obs` together for each month, then perform
the `water => valid_obs` check on that already-materialised payload. Do not run
a separate cube-wide `.any().compute()` pre-pass: that rereads the same source
before patch work starts. Compute 1-D coverage reductions in bounded batches,
or derive them from the same payload when patches already require it.

- [ ] **Step 1: Add a Dask source counter proving no full-cube `.load()` and no
  separate validity pre-read.**
- [ ] **Step 2: Implement fused
  `xr.Dataset({water, valid_obs}).isel(time=time_index).compute()`
  over bounded months/windows.**
- [ ] **Step 3: Assert source reads scale with selected active chunks, not full
  AOI chunks.**
- [ ] **Step 4: Run OOM-regression and output-parity tests.**
- [ ] **Step 5: Commit.**

```powershell
git commit -m "perf: materialize only active monthly windows"
```

### 3.4 Drop `dask-image` relabel for in-memory masks

`labels.py:42` routes to `dask_image.ndmeasure.label` whenever the mask is a
dask array, then immediately `.compute()`s it. Cross-chunk reconciliation is
real work and it is pure overhead when a single month's mask fits in memory —
which at 30 m it does for most catchments.

Add a size threshold: below it, `np.asarray(mask)` then `scipy.ndimage.label`
(already the non-dask branch). Above it, keep the dask-image path. The two
paths must agree — `_filter_and_normalize` already normalises label IDs by
first row-major occurrence precisely so chunk layout can't change output, so
add a parity test pinning that both branches give identical labels.

- [ ] **Step 1: Add eager/Dask/chunk-layout parity tests.**
- [ ] **Step 2: Add configurable byte threshold using existing compute memory
  policy, not a second independent memory setting.**
- [ ] **Step 3: Use SciPy below threshold; retain dask-image above threshold.**
- [ ] **Step 4: Benchmark and commit.**

```powershell
git commit -m "perf: use local labels for bounded monthly masks"
```

### 3.5 `medial_axis` per patch is the width hot spot

`morphology.py:92` calls `skimage.morphology.medial_axis` per crop, only when
`include_width`. It is the most expensive per-patch op by a wide margin.

We only consume `(2*dist[axis]).max()` — the maximum inscribed diameter. That
is exactly `2 * max(EDT)`, obtainable from `scipy.ndimage.distance_transform_edt`
alone without computing the medial axis at all. The medial axis is a strict
superset of what we use.

Replace with the EDT max. **This must be proven equivalent, not assumed** —
the medial axis is guaranteed to contain the EDT maximum, so the max over the
axis equals the global max, but assert it on the real Fitzroy data before
switching. Guard with a parity test over many random masks.

Note `pool_width` is `secondary`/width-floor-gated, so this only pays off when
width is requested. Do it anyway — it is a small, well-tested change.

- [ ] **Step 1: Add random-mask and Fitzroy parity tests for medial-axis versus
  global EDT maximum.**
- [ ] **Step 2: Replace the width kernel only after parity passes.**
- [ ] **Step 3: Record benchmark delta and commit.**

```powershell
git commit -m "perf: compute maximum pool width from EDT"
```

### 3.6 Cheap kernel wins

- `_pixel_edge_perimeter` (`morphology.py:37`) pads the full crop and does two
  full comparisons. Fine per-crop, but it runs per patch per month. Vectorise
  across a bucket of crops.
- `_bulk_major_axis_lengths` (`morphology.py:44`) builds a block-diagonal
  composite sized `sum(heights) x max(width)`. For many small patches plus one
  large one this wastes a lot of zeros. Bucket crops by similar width before
  compositing.
- `_filter_and_normalize` uses `np.minimum.at` (`labels.py:73`), which is a
known-slow ufunc.at path. When labels come from `scipy.ndimage.label` (3.4),
  IDs are already assigned in row-major first-occurrence order, so the
  reordering is a no-op and can be skipped entirely. Keep it for the dask
  branch where reconciliation can permute IDs.

- [ ] **Step 1: Benchmark each kernel change independently on identical patch
  properties.**
- [ ] **Step 2: Implement width buckets, perimeter batching, and SciPy-label
  normalization bypass behind parity tests.**
- [ ] **Step 3: Keep only changes with measurable end-to-end benefit.**
- [ ] **Step 4: Commit retained changes.**

```powershell
git commit -m "perf: reduce patch morphology allocation"
```

### 3.7 Benchmark gate

`benchmarks/cpu_baseline.py` and `benchmarks/results/cpu_baseline.json` already
exist as the perf evidence contract, and `docs/performance.md` documents
regeneration. Every change in this workstream must:

1. Record a before/after timing in that harness.
2. Leave the committed **checksums unchanged** — these are optimisations, not
   respecifications. A changed checksum means a bug, except for 3.5 where the
   parity test carries the argument.

Add a catchment-scale case to the baseline. The current largest case is only
`(12, 128, 128)` and the baseline does not time patch morphology, so it cannot
surface the OOM in 3.3 or validate the claimed 3.2 speedup. Add per-stage patch
timings, bytes read, process/thread mode, worker count, and peak RSS. Gate peak
RSS at no more than 125% of serial unless a larger limit is explicitly chosen.

Add `benchmarks/end_to_end_workflow.py` for the real goal: user input to final
table. Run isolated subprocesses and record:

- native stats/planning seconds;
- STAC query and remote acquisition seconds;
- source items, selected/total chunks, loaded pixels, bytes read;
- local metrics and output-write seconds;
- total cold and warm seconds;
- peak RSS, worker mode/count, mask factor, output digests.

Required cases: thin/braided Gilbert, compact Fitzroy, and one large catchment.
Compare full-AOI versus factor 3 versus factor 4, workers 1/2/4, and serial /
thread / process metric modes. Promotion gates:

- exact metrics table and per-metric value equality;
- `n_water` equality every month;
- native-wet mask coverage exactly 100%;
- cold Gilbert median at least 30% faster than full AOI;
- warm rerun at least 80% faster than cold full AOI and zero STAC calls;
- compact Fitzroy regression no worse than 10%;
- peak RSS no more than 125% of serial baseline;
- fastest passing candidate becomes default; otherwise keep safer setting.

- [ ] **Step 1: Add deterministic output digest and phase schema tests.**
- [ ] **Step 2: Implement isolated cold/warm runner without privileged cache
  flushing.**
- [ ] **Step 3: Run candidate matrix three times per real case.**
- [ ] **Step 4: Commit machine-readable and Markdown summaries, excluding raw
  caches.**
- [ ] **Step 5: Update defaults only from passing evidence and commit.**

---

## Workstream 4 — capability-driven default metrics and orchestration

### 4.1 Make `all_available` the default profile

**Files:**

- Modify: `hydrofragments/metrics/registry.py`
- Modify: `hydrofragments/config.py`
- Modify: `hydrofragments/api.py`
- Modify: `hydrofragments/models.py`
- Test: `tests/gating/test_all_available_profile.py`

`all_available` contains runtime-wired metrics only:

```python
RUNTIME_WIRED_METRIC_IDS = (
    "occurrence", "refuge_area", "apsec", "number_of_pools", "lpi",
    "awre", "awmsi", "recurrence", "hydroperiod", "extent_contraction",
    "lpsec", "inter_pool_gap", "pool_width",
)
```

It deliberately excludes `mesh` (validation-gated and not emitted),
`reconnection_timing`/`refuge_spatial_stability` (kernel-only, not wired), and
`realised_connectivity`/`tcf` (runtime-deferred). Existing named profiles remain
available and override default selection when explicitly supplied.

- [ ] **Step 1: Write failing default-resolution tests.** With only water and
  validity, default selects core + recurrence + hydroperiod and records exact
  skip reasons for channel/dynamics/width metrics.
- [ ] **Step 2: Add dependency fixtures incrementally.** Width floor unlocks
  pool width; real channel inputs unlock LPSEC/gap; HY + dual APSEC unlock
  contraction. Assert each metric computes exactly once.
- [ ] **Step 3: Add one `_available_dependencies` helper used by both
  validation and execution; remove duplicated dependency assembly.**
- [ ] **Step 4: Change config default to `("all_available",)` while preserving
  explicit profile behavior and metric overrides.**
- [ ] **Step 5: Add backward-compatible
  `HydroResult.metric_coverage: pd.DataFrame` using `default_factory`.** Include
  one row for every registry metric, with columns `metric`, `runtime_wired`,
  `status`, `rows`, `reportable_rows`, and `reason`. Runtime-wired metrics are `computed` or
  `skipped (missing dependency)`; other registry entries are explicitly
  `skipped (not runtime wired)`, `skipped (validation disabled)`, or
  `skipped (runtime deferred)`. `computed` means the kernel ran even when data
  quality leaves zero reportable rows; put that quality reason in `reason`
  instead of falsely calling the metric dependency-skipped.
- [ ] **Step 6: Update snapshots/docs and run the complete gating suite.**
- [ ] **Step 7: Commit.**

```powershell
git commit -m "feat: compute every available runtime metric by default"
```

### 4.2 Add one user-input-to-table entry point

**Files:**

- Create: `hydrofragments/workflow.py`
- Modify: `hydrofragments/__init__.py`
- Modify: `hydrofragments/output/tables.py`
- Modify: `hydrofragments/output/manifest.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/integration/test_dea_workflow.py`

```text
analyze_from_dea(
    aoi: str | Path | geopandas.GeoDataFrame,
    start_date: str,
    end_date: str,
    *,
    aoi_id: str,
    drainage: str | Path | geopandas.GeoDataFrame | None = None,
    config: HydroConfig | None = None,
    cache_dir: str | Path = "output/wofs_cache",
) -> HydroResult
```

Orchestrator calls public hydroseason APIs, creates verified `aoi_mask` and
`analysis_mask`, opens canonical cache, derives hydro-year/dual-composite inputs
automatically, creates channel inputs when drainage is supplied, calls
`analyze()` once, and writes final artifacts. It contains no metric formula and
no acquisition internals.

- [ ] **Step 1: Write an offline fake-hydroseason integration test** pinning
  call order, argument forwarding, mask semantics, dependency discovery, and
  output files.
- [ ] **Step 2: Implement orchestration with explicit progress phase timings:**
  DEA planning, WOfS query, acquisition, local metric processing, output write,
  total input-to-table.
- [ ] **Step 3: Ensure cache hit makes zero STAC calls and still emits same
  metrics/coverage tables.**
- [ ] **Step 4: Add failure tests:** stats unavailable falls open; incomplete
  cache resumes; invalid mask digest fails; unavailable optional metrics skip.
- [ ] **Step 5: Run integration and public API suites.**
- [ ] **Step 6: Update HydroFragments to the released
  `hydroseason==0.1.0`, refresh the lock, build both wheels, and run the
  installed-wheel integration test so sibling checkout imports cannot mask a
  packaging error.**
- [ ] **Step 7: Commit.**

```powershell
git commit -m "feat: add DEA-to-metrics workflow"
```

### 4.3 Reuse active-window intermediates across metric families

Refactor patch code to expose measured properties, then concatenate properties
from independent windows and reduce once:

```text
measure_patch_properties(
    mask: Any,
    *,
    pixel_size_m: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
    include_width: bool = False,
) -> Sequence[PatchProperties]

reduce_patch_properties(
    properties: Sequence[PatchProperties],
    *,
    pixel_size_m: float,
    a_total_m2: float,
    include_mesh: bool = False,
    include_width: bool = False,
    resolution_floor_pixels: float | None = None,
) -> tuple[PatchMetricResult, PoolWidthDistribution | None]
```

During each bounded month payload, also accumulate APSEC counts, calendar-month
wet/valid counts, yearly wet/valid counts, and channel wet profiles. Finalizers
consume these sufficient statistics; no metric family rereads the same month.

- [ ] **Step 1: Pin `analyze_patch_bundle` output before refactor.**
- [ ] **Step 2: Split measure/reduce and prove full-mask versus active-window
  property/output parity.**
- [ ] **Step 3: Add a source counter proving one materialisation per selected
  month/window regardless of number of enabled metrics.**
- [ ] **Step 4: Verify default all-available run has identical outputs to the
  union of explicit individual-profile runs.**
- [ ] **Step 5: Commit.**

```powershell
git commit -m "perf: share active-window metric intermediates"
```

---

## Workstream 5 — all-working-metrics notebook

`examples/end_to_end_workflow.ipynb` already exists (17 cells, untracked) and
covers `contracts_core` well in the right plain-language register. **Extend it,
don't rewrite it** — and keep its voice.

Ground truth for what actually works is `docs/final_metrics_covered.md`:

**Core (already in the notebook):** `occurrence`, `refuge_area`, `apsec`,
`number_of_pools`, `lpi`, `awre`, `awmsi`.

**To add, with their real gating:**

| Metric | What the notebook must supply |
|---|---|
| `recurrence`, `hydroperiod` | `pixel_temporal` profile. No extra inputs. Easy. |
| `pool_width` | `secondary` + `config.patches.width_resolution_floor_pixels` set |
| `lpsec`, `inter_pool_gap` | real `SpatialContext` w/ `has_real_channel`, plus `channel_wet_profiles` and `channel_segment_lengths_m`. `data/fitzroy_kimberley_drainage.gpkg` exists — use it. |
| `extent_contraction` | `dynamics` + HY anchors + **both** `max_water_apsec` and `median_apsec` |
| DEA zones | Workstream 1, emitted as an exact native-grid companion artifact; not multiplied into default metric rows |

Notebook must use default `analyze_from_dea`, not manually make one
`analyze()` call per profile. One call demonstrates acquisition pruning,
automatic capability discovery, cache reuse, all-available metric execution,
and final coverage reporting. A short advanced cell may show how an explicit
narrow profile reduces work.

**Explicitly document as not-yet-wired** (don't fake them):
`reconnection_timing` and `refuge_spatial_stability` are kernel-only, not
profile-wired in `analyze()`. `mesh` is validation-gated off (LPI redundancy).
`realised_connectivity` / `tcf` are runtime-deferred. State this plainly in a
short "what is not in this table and why" section — that honesty is worth more
than a longer table.

**Final CSV.** The notebook ends with one CSV you can open and check:

- One default `analyze_from_dea()` call produces the tidy long frame (`date`,
  `aoi_id`, `metric`, `value`, `unit`, `is_reportable`).
- Then a wide pivot: one row per date, one column per metric, so the whole
  metric set is visible at a glance.
- Write **both**, plus the registry-wide coverage table with runtime-wiring and
  `computed` / `skipped (reason)` status — so the CSV is self-auditing against
  both the runtime registry and `final_metrics_covered.md`.
- Print the last rows inline so the values are checkable without leaving the
  notebook.

- [ ] **Step 1: Replace manual profile orchestration with the one-call default
  workflow and a cached rerun demonstration.**
- [ ] **Step 2: Assert notebook coverage lists every registry metric once and
  every dependency-satisfied runtime-wired metric is `computed`.**
- [ ] **Step 3: Write tidy, wide, coverage, and timing CSVs; display final rows.**
- [ ] **Step 4: Fix markdown mojibake and run notebook end-to-end on Fitzroy.**
- [ ] **Step 5: Commit notebook and generated-output exclusions.**

```powershell
git commit -m "docs: demonstrate fast all-available DEA workflow"
```

Also fix the UTF-8 mojibake already in the notebook's markdown cells (`U+FFFD`
where em-dashes and `km²` belong) — cells 1, 4, 8 at minimum.

---

## Sequencing

1. **W1.1 + W1.5** — public native statistics and proven conservative planning
   footprint in hydroseason.
2. **W2.1 + W2.3** — reuse footprint through acquisition, publish cache reader,
   preserve both denominator masks.
3. **W3.1 + W3.3 + W3.4** — mask semantics, bounded lazy windows, local-label
   fast path. Establish correct bounded serial baseline.
4. **W4.1 + W4.3** — all-available default and shared active-window metric DAG.
5. **W2.2** — dual-composite counts from existing acquisition source pass.
6. **W1.2-W1.4** — exact zones, strict persistence-zone guard, provenance.
7. **W3.2 + W3.5 + W3.6** — worker matrix and kernel wins, each benchmark-gated.
8. **W4.2** — one public DEA-to-table orchestrator after constituent APIs pass.
9. **W3.7** — full cold/warm real-catchment benchmark and default selection.
10. **W5** — notebook last, demonstrating one-call default plus cached rerun.

## Risks

- **Threshold fraction-vs-percent in `build_zones` (W1.2).** Latent bug today.
  If it is wrong, every zone raster is mislabelled and it will not be obvious.
  Pin with a test first.
- **Persistence-zone circularity (W1.3).** DEA provenance does not waive the
  locked guard. Persistence metrics remain AOI/channel-only, and the fast
  default emits zones separately without multiplying metric work.
- **Parallel determinism (W3.2).** Must be byte-identical across worker counts
  or the perf-baseline checksums become meaningless.
- **DEA temporal mismatch.** Multi-year covers 1987-2025; the local cube is
  1986-2026. Zones and metrics therefore rest on slightly different windows.
  Record both spans in the manifest and state it in the notebook — do not
  silently align them.
- **Unsafe mask downsampling.** Nearest/mode/mean can drop narrow or isolated
  wet pixels. Only aligned max/any aggregation is acceptable for a pruning
  mask; zoning remains native 30 m.
- **Mask without pruning.** Applying a coarse boolean mask to a full-shape
  array does not avoid chunk reads or the full label scan. W2 must prove source
  chunks are skipped and W3 must prove independent active-window processing.
- **Process-pool amplification.** Each worker can hold a catchment month plus
  label/crop intermediates. Bound in-flight work, keep default workers at 1,
  and measure RSS on Windows.
- **Network dependency.** STAC access makes the zoning path fail differently
  from the offline path. Keep the local-cube zoning path working as a fallback.
- **Validity contract change.** Monthly coverage now uses `analysis_mask`, while
  fixed-area metrics use `aoi_mask`. Persist both masks/counts/digests and name
  the coverage footprint in every manifest/table; never silently reuse old
  full-AOI coverage values as equivalent.
- **Independent-window aggregation.** Per-window aggregate metrics cannot be
  averaged. Concatenate patch properties, then calculate LPI, AWRe, AWMSI,
  width distribution, and counts once across all windows.
- **Nested parallelism.** Do not run unconstrained Dask read threads inside an
  unconstrained process pool. One coordinator owns bounded I/O payloads; CPU
  workers receive NumPy only.
- **Cross-repository release.** HydroFragments pins `hydroseason==0.1.0`,
  the first public remote-sensing release version (superseding this plan's
  original `0.1.1` coordination target, which was never published). Publish
  the required hydroseason API/version first, update the pin and lock
  file in the same HydroFragments change, and test installed wheels rather than
  relying only on sibling-repository imports.
