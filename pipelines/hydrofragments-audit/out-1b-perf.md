# Efficiency Audit output

## Task

Audit HydroFragments for the performance axis only: identify bottlenecks, CPU/CUDA parity gaps, and benchmark harness coverage without re-verifying metric formulas.

## What this stage did

Read the current `hydrofragments` source directly, especially:

- `hydrofragments/compute/`: `policy.py`, `chunks.py`, `capabilities.py`, `backends/cpu.py`, `backends/cuda.py`
- `hydrofragments/pipeline.py` and `hydrofragments/api.py`
- `hydrofragments/compat.py`
- `hydrofragments/patches/`: `labels.py`, `components.py`, `morphology.py`
- `hydrofragments/metrics/`: `extent.py`, `persistence.py`, `patches.py`, `connectivity.py`, `clustering.py`, `dynamics.py`
- `hydrofragments/spatial/`: `windows.py`, `zones.py`, `connectivity_context.py`
- `hydrofragments/temporal/`: `composites.py`, `cadence.py`, `hydroyear.py`
- `hydrofragments/benchmarks/cpu_baseline.py`
- Prior audit docs: `docs/audit/dask_cuda_audit.md` and `docs/audit/dask_cuda_audit_adversarial.md`

The prior Dask/CUDA audit is only partly applicable to the current package. Its old `ecofragments/utils/calc_metrics.py` nested-delayed findings do not map directly to current `hydrofragments`, because the current code now has explicit modules for monthly products, labels, component crops, and capability gating. However, the central warning remains true in a new form: public analysis still crosses eager CPU/materialization boundaries before most hot metrics run.

## Findings

### 1. Public `analyze()` bypasses the lazy monthly pipeline and materializes through compatibility code

- Location `hydrofragments/api.py:597`, `hydrofragments/api.py:603`, `hydrofragments/compat.py:116`, `hydrofragments/compat.py:118`
- What's slow + why: `analyze()` creates a simple `xr.Dataset` from `cube.water`/`cube.valid_obs`, then immediately calls `section_compat_rows()`. That helper calls `_monthly_dataset()`, which does `da_feature.load()`. This forces the whole time-y-x cube into memory at the compatibility boundary, instead of using `pipeline.assemble_monthly_pipeline()` / `run_monthly_pipeline()` and the explicit checkpoint policy.
- Cost + scaling: O(T * Y * X) eager memory before metric selection. For large satellite cubes this defeats Dask chunking, removes opportunity for monthly checkpoint reuse, and can duplicate I/O with later optional profiles.
- Fix, concrete: make `analyze()` resolve metric IDs before core work, build monthly products through `run_monthly_pipeline()` when cadence/composite policy requires it, and avoid `section_compat_rows()` as the canonical compute path. Keep compatibility facade separate for legacy calls only.
- Risk + effort: high impact, med effort. Main risk is output-row parity because `_records_from_compat_rows()` currently provides the canonical row bridge for retained core metrics.

### 2. Core metrics are computed before profile filtering, so skipped metrics can still cost full patch analysis

- Location `hydrofragments/api.py:603`, `hydrofragments/api.py:626`, `hydrofragments/api.py:656`, `hydrofragments/compat.py:137`, `hydrofragments/compat.py:146`
- What's slow + why: `section_compat_rows()` computes occurrence, refuge area, APSEC, and patch morphology for every month. Only after that does `analyze()` compute `selected_ids` and filter records. A run that requested a narrow profile can still pay for full core patch labeling/morphology.
- Cost + scaling: O(T * label(YX) + T * component_morphology) wasted work whenever patch metrics are not selected. Patch-heavy masks dominate runtime.
- Fix, concrete: resolve `selected_ids` before invoking metric kernels. Dispatch only required metric families. For example, compute APSEC only when `apsec` selected, run `analyze_patch_metrics()` only when one of `number_of_pools`, `lpi`, `awre`, `awmsi`, or `mesh` is selected, and compute occurrence/refuge only for persistence outputs.
- Risk + effort: low/med effort, low science risk. Mostly orchestration refactor plus snapshot tests for row presence/order.

### 3. APSEC is recomputed one month at a time instead of one vectorized temporal reduction

- Location `hydrofragments/compat.py:146`, `hydrofragments/compat.py:155`, `hydrofragments/metrics/extent.py:87`, `hydrofragments/metrics/extent.py:91`
- What's slow + why: `section_compat_rows()` loops over months and calls `compute_apsec(monthly.isel(time=[time_index]))` each iteration. `compute_apsec()` itself can already reduce across all months, but the caller prevents a single Dask/xarray reduction over the time axis.
- Cost + scaling: still O(T * Y * X) arithmetic, but with T separate xarray calls, T small reductions, and avoidable scheduler/Python overhead. If `.load()` is removed, this becomes a much bigger Dask graph overhead bug.
- Fix, concrete: call `compute_apsec(monthly, ...)` once, map returned records by timestamp, and reuse inside row construction.
- Risk + effort: low effort, low risk.

### 4. Dask-backed connected-component labels are globally materialized per month

- Location `hydrofragments/patches/labels.py:42`, `hydrofragments/patches/labels.py:44`, `hydrofragments/metrics/patches.py:225`
- What's slow + why: Dask-backed 2-D masks use `dask_image.ndmeasure.label()` but immediately call `labels.compute()` and convert to NumPy. That is an explicit boundary, reasonable for exact CPU morphology, but it means every month needs a complete int label raster resident on host before component metrics.
- Cost + scaling: O(Y * X) host memory per active month, with int labels typically 4 bytes/pixel plus temporaries. For large AOIs, label materialization becomes a peak-memory limiter even before region properties or width work starts.
- Fix, concrete: make this boundary explicit in the pipeline as a per-month label checkpoint, not an incidental `compute()` inside a helper. Add chunk/shape diagnostics for label materialization. If scaling beyond one full 2-D month is required, replace with a staged distributed label plan that persists global int32 labels and then consumes component crops in bounded tasks.
- Risk + effort: med/high effort. Exact cross-chunk component identity must remain unchanged.

### 5. Label normalization uses global sort/search work over every pixel

- Location `hydrofragments/patches/labels.py:54`, `hydrofragments/patches/labels.py:55`, `hydrofragments/patches/labels.py:75`
- What's slow + why: `_filter_and_normalize()` flattens the label raster, calls `np.unique(..., return_index=True, return_counts=True)`, then uses `np.searchsorted()` over the full flattened raster to remap labels. This is robust but sort-heavy and allocates large full-raster intermediates.
- Cost + scaling: at least O(P log P) sort-like work over P pixels, plus O(P) remap memory. Highly fragmented scenes increase label count and pressure metadata arrays.
- Fix, concrete: use integer-label reductions: `np.bincount(flat)` for counts, `np.minimum.at()` or a sentinel array for first occurrence, build a lookup table indexed by raw label ID, then remap `lookup[raw_labels]`. This is O(P + K), where K is raw labels, and avoids sorting all pixels.
- Risk + effort: med effort, low/med risk. Need tests for deterministic row-major ID ordering and sparse raw labels from `dask-image`.

### 6. Core patch metrics and pool-width metrics repeat the same label/crop pass

- Location `hydrofragments/metrics/patches.py:225`, `hydrofragments/metrics/patches.py:253`, `hydrofragments/api.py:317`, `hydrofragments/api.py:322`, `hydrofragments/api.py:680`
- What's slow + why: `analyze_patch_metrics()` labels and crops each monthly mask for core patch metrics. If `pool_width` is selected, `_pool_width_records()` loops through the same months and calls `analyze_pool_width_distribution()`, which labels and crops again before computing width.
- Cost + scaling: up to ~2x label/crop work for runs that include both fragmentation/morphology core metrics and pool width. Width work also adds medial-axis and EDT cost per component.
- Fix, concrete: introduce a per-month `PatchAnalysis` bundle: label once, create component crops once, measure components once with `include_width` toggled when any width statistic is selected, then emit both core and pool-width records from the shared properties.
- Risk + effort: med effort, low/med risk. Must preserve existing warning flags and width-resolution-floor suppression semantics.

### 7. Width measurement likely computes the distance transform twice

- Location `hydrofragments/patches/morphology.py:55`, `hydrofragments/patches/morphology.py:56`
- What's slow + why: when width is enabled, code calls `medial_axis(mask)` and then `distance_transform_edt(mask)`. `medial_axis` implementations commonly compute the distance map internally and can return it with the skeleton. Current code likely pays for a second EDT over the same crop.
- Cost + scaling: O(sum crop pixels) extra EDT work for every measured component, significant for large pools and width-heavy profiles.
- Fix, concrete: switch to the distance-returning medial-axis API where available: compute skeleton and distance together, then use the returned distance on skeleton pixels. Pin behavior with regression tests before changing dependency versions.
- Risk + effort: low effort, low/med risk. Needs exact output parity test for width pixels.

### 8. Region properties are computed through one skimage object per component

- Location `hydrofragments/patches/morphology.py:53`, `hydrofragments/patches/morphology.py:71`
- What's slow + why: every component crop calls `regionprops(mask.astype(np.uint8))` separately to obtain major-axis length. For highly fragmented masks, Python function/object overhead and per-crop array casts dominate.
- Cost + scaling: O(K) Python/skimage calls for K retained patches, plus O(sum crop pixels) numeric work. Worst case is many tiny patches.
- Fix, concrete: evaluate a bulk path using `regionprops_table()` on a labeled raster or a vectorized component-moment reducer. If exact skimage semantics must remain, bucket crops and measure in worker tasks large enough to amortize overhead.
- Risk + effort: med effort, med science-adjacent risk because major-axis parity must be certified.

### 9. Fixed-graph construction scans node pairs to find adjacent reaches

- Location `hydrofragments/metrics/connectivity.py:74`, `hydrofragments/metrics/connectivity.py:80`, `hydrofragments/metrics/connectivity.py:82`
- What's slow + why: `build_fixed_graph()` builds edges by nested loops over kept nodes and checks `to_node_a == from_node_b`. Drainage topology is sparse and keyed, so pairwise scanning is unnecessary.
- Cost + scaling: O(V^2) comparisons for V wet-capable reaches. Large drainage networks make this avoidable CPU overhead.
- Fix, concrete: build a dictionary from `From_Node` to reach IDs, then for each kept `node_a`, look up children whose `From_Node` equals `To_Node` for `node_a`. Emit stable ordered edges with the existing ordering rule.
- Risk + effort: low effort, low risk.

### 10. RC reachable-pair calculation is O(V^2) after union-find

- Location `hydrofragments/metrics/connectivity.py:163`, `hydrofragments/metrics/connectivity.py:164`, `hydrofragments/metrics/connectivity.py:166`
- What's slow + why: after active edges are unioned, `compute_realised_connectivity()` counts reachable pairs by checking every pair of node roots. Once component IDs are known, pair count can be computed from component sizes.
- Cost + scaling: O(V^2) per monthly RC snapshot. With M months, O(M * V^2).
- Fix, concrete: count roots with `collections.Counter`, then compute `sum(comb(size, 2) for size in component_sizes)` and divide by `comb(V, 2)`.
- Risk + effort: low effort, low risk. Formula stays identical.

### 11. Reach/water context keeps one full raster mask per reach and intersects each month against every reach

- Location `hydrofragments/spatial/connectivity_context.py:77`, `hydrofragments/spatial/connectivity_context.py:84`, `hydrofragments/spatial/connectivity_context.py:88`, `hydrofragments/spatial/connectivity_context.py:91`, `hydrofragments/spatial/connectivity_context.py:94`, `hydrofragments/spatial/connectivity_context.py:95`
- What's slow + why: `reach_wet_any_month()` rasterizes a full boolean buffer mask per reach, then for each month materializes the full water mask, skeletonizes the full frame, and checks `skeleton & buffer_mask` for every not-yet-wet reach.
- Cost + scaling: memory O(R * Y * X) for R reach buffer masks, runtime O(T * skeleton(YX) + T * R * Y * X) boolean intersections. This can dominate connectivity runs.
- Fix, concrete: rasterize reach buffers into a single integer/multilabel raster or sparse per-pixel reach list, skeletonize each month once, then identify wet reaches by indexing/unique-counting labels under skeleton pixels. If overlapping buffers matter, store a sparse coordinate-to-reach mapping rather than R dense masks.
- Risk + effort: med/high effort. Main risk is overlapping buffers and exact reach attribution semantics.

### 12. Temporal AOI summaries trigger separate scalar materializations

- Location `hydrofragments/api.py:438`, `hydrofragments/api.py:439`, `hydrofragments/api.py:458`, `hydrofragments/api.py:460`, `hydrofragments/metrics/persistence.py:178`, `hydrofragments/metrics/persistence.py:183`, `hydrofragments/metrics/persistence.py:200`
- What's slow + why: recurrence and hydroperiod build separate groupby reductions, then `.item()` is called for the recurrence AOI mean and once per year for hydroperiod means. On Dask data this can create repeated small graph executions.
- Cost + scaling: recurrence O(T * Y * X), hydroperiod O(T * Y * X), plus scheduler overhead per scalar/year. The arithmetic is valid, but orchestration is fragmented.
- Fix, concrete: compute requested temporal summaries in one dataset and call `.compute()` once at an explicit boundary, then extract all scalar AOI rows from the concrete result.
- Risk + effort: low/med effort, low risk.

### 13. Chunk policy is inspected but not used to choose or preserve chunks in public input opening

- Location `hydrofragments/api.py:67`, `hydrofragments/api.py:71`, `hydrofragments/api.py:76`, `hydrofragments/compute/chunks.py:45`
- What's slow + why: `open_water_cube()` accepts `chunks` but discards it, and Zarr opening does not pass caller chunk intent. `validate_chunk_budget()` can reject unsafe existing chunks, but there is no public path to set or repair chunk layout.
- Cost + scaling: bad input chunks can cause excessive scheduler overhead, huge live chunks, or unnecessary rechunking outside the visible policy. Users cannot tune large runs without bypassing the API.
- Fix, concrete: honor `chunks` in `open_zarr()`/array wrapping, record chosen chunks in manifest, and add an explicit rechunk-planning step that either preserves storage chunks or writes a planned monthly checkpoint with safe chunks.
- Risk + effort: med effort, low/med risk. Needs input-adapter tests for Dask-backed and eager inputs.

### 14. CUDA scaffold is truthful but incomplete; no runtime path uses GPU kernels

- Location `hydrofragments/compute/policy.py:28`, `hydrofragments/compute/capabilities.py:28`, `hydrofragments/compute/capabilities.py:150`, `hydrofragments/compute/capabilities.py:197`, `hydrofragments/compute/backends/cuda.py:26`, `hydrofragments/api.py:749`
- What's slow + why: `CUDABackend` implements simple CuPy reductions, and capability detection lists candidate stages, but `enabled_cuda_stages` is deliberately empty and `ComputePolicy(accelerator="cuda")` rejects CUDA for the M4 pipeline. `analyze()` records actual backend as CPU. There is no silent full-GPU claim, but also no accelerated production path.
- Cost + scaling: all heavy morphology, labels, connectivity, and public API analysis run CPU/host. CUDA cannot improve current user runs except as future scaffold.
- Fix, concrete: keep CUDA disabled until benchmark evidence exists. First candidate tranche should be pixelwise masks, valid counts, monthly reductions, and occurrence on CuPy-backed Dask arrays. Keep morphology/connectivity CPU until parity and transfer-cost gates pass.
- Risk + effort: high effort. CPU/CUDA parity tests and transfer-cost benchmarks are prerequisite.

### 15. Benchmark harness does not measure the real hot path for user-ready analysis

- Location `hydrofragments/benchmarks/cpu_baseline.py:45`, `hydrofragments/benchmarks/cpu_baseline.py:130`, `hydrofragments/benchmarks/cpu_baseline.py:146`, `hydrofragments/benchmarks/cpu_baseline.py:161`, `hydrofragments/benchmarks/cpu_baseline.py:165`, `hydrofragments/benchmarks/cpu_baseline.py:171`
- What's slow + why: benchmark cases are small synthetic arrays up to `(12, 128, 128)`, and measured stages cover monthly pipeline assembly, monthly reduction, occurrence, and APSEC. The harness never runs `analyze()`, `section_compat_rows()`, patch labels/morphology, pool width EDT/medial axis, connectivity context, or RC/TCF graph work.
- Cost + scaling: current baseline can look healthy while missing dominant O(T * label(YX)), O(K) component, O(T * R * YX) reach-context, and O(V^2) connectivity costs.
- Fix, concrete: add benchmark stages for:
  - public `analyze()` core profile,
  - patch-only monthly masks with low/medium/high fragmentation,
  - pool-width profile with large pools,
  - connectivity context with realistic reach counts,
  - RC graph snapshots across many months,
  - chunk stress cases with scheduler/memory metrics.
- Risk + effort: med effort, low risk.

## Handoff to next stage

Ranked by impact/effort:

1. Stop using `section_compat_rows()` inside canonical `analyze()`. Resolve selected metrics first, dispatch only requested kernels, and route monthly compositing through `run_monthly_pipeline()` when appropriate. This attacks the whole-cube `.load()` and wasted profile work.
2. Share per-month patch analysis between core metrics and pool width. Label once, crop once, measure once. This is the highest-leverage patch-path fix when width metrics are enabled.
3. Replace O(V^2) connectivity code: adjacency lookup in `build_fixed_graph()` and component-size pair counting in `compute_realised_connectivity()`. Small patch, big win for large drainage graphs.
4. Optimize label normalization from `np.unique`/`searchsorted` to bincount/lookup-table remapping. Important for highly fragmented masks.
5. Rework `reach_wet_any_month()` to avoid R dense buffer rasters and per-reach full-frame intersections. Use a single label/sparse reach raster and per-month skeleton label extraction.
6. Improve width path: avoid duplicate EDT by using the distance returned by medial-axis where parity holds.
7. Honor `chunks` in `open_water_cube()` and expose a real chunk plan/checkpoint path in the manifest.
8. Expand `cpu_baseline.py` so benchmark evidence covers public `analyze()`, patch morphology, width, and connectivity.

Likely science-change conflicts:

- Replacing `regionprops()` or `medial_axis()` internals can alter major-axis/width numerics. Treat those as science-adjacent and require exact parity fixtures before merging.
- Distributed labels and CUDA/cupy morphology can change component identity at chunk boundaries unless global reconciliation is proven. Keep CPU label/morphology as the reference until parity is certified.

## Open questions / risks

- Need a real profiler run on representative AOIs to rank patch labeling vs morphology vs reach-context cost. Source reading shows scaling risks, not wall-clock proof.
- Need memory profiling for large monthly label rasters and pool-width crops; current benchmark reports `peak_rss_bytes: None`.
- Need user/product decision on whether canonical `analyze()` should remain legacy-compatible internally or move fully to the newer modular pipeline with a legacy facade only.
- CUDA should stay advertised as candidate/incomplete, not production acceleration, until benchmark rows prove parity and transfer-cost benefit for each enabled stage.
- `docs/audit/dask_cuda_audit.md` and `dask_cuda_audit_adversarial.md` should be updated or superseded: their conclusions about old `ecofragments` nested delayed code are stale, but their warnings about eager materialization, CPU morphology, chunk policy, and CUDA gating remain useful in the current package.
