# Dask-first scalability and CUDA-readiness audit

**Date:** 2026-07-10  
**Scope:** first-phase architecture/performance audit only  
**Source changes:** none; this report is the only deliverable  
**Verdict:** current pipeline is Dask-wrapped, not Dask-first. CPU results can be preserved, but large-section spatial morphology cannot presently scale across spatial chunks. CUDA can accelerate selected array stages; it cannot honestly accelerate the full pipeline.

## Executive finding

Current code builds lazy xarray/Dask objects, then passes them through coarse `dask.delayed` tasks that synchronously materialise arrays and run NumPy, SciPy, scikit-image, Pandas, GeoPandas, Rasterio, Shapely, and igraph code. Three structural problems dominate:

1. [`pre_process_layer()`](../../ecofragments/utils/calc_metrics.py#L911) declares `y` and `x` as `apply_ufunc` core dimensions. Xarray requires each core dimension to occupy one Dask chunk. A genuinely spatially chunked section therefore raises, or must be rechunked to one whole-section array. This is the opposite of out-of-core spatial scaling.
2. [`process_feature_batch()`](../../ecofragments/utils/calc_metrics.py#L356) receives Dask collections inside a delayed task and calls [`np.asarray()` three times](../../ecofragments/utils/calc_metrics.py#L968). Those calls start nested synchronous computations. Label calculation can be repeated for the label, skeleton, and EDT branches instead of shared once.
3. Patch rows, skeleton paths, polygons, lines, and points are accumulated into in-memory Pandas/GeoPandas objects before final output. Memory therefore scales with total patch count and vector complexity, not only raster chunk size.

CPU must remain default and reference-correct. Recommended CUDA scope is initially limited to pixelwise mask operations, valid-observation reductions, monthly compositing, and persistence/occurrence reductions on CuPy-backed Dask blocks. Local connected components and EDT have CUDA implementations, but no current design here makes them exact across arbitrary spatial Dask chunks. Skeletonization, igraph path work, Rasterio polygonization, Shapely/GeoPandas construction, and GDAL I/O remain CPU stages.

This is a source audit, not a performance run. Active environment lacks declared `dask_image`, `dask_regionprops`, scikit-image, igraph, NetCDF backends, and CUDA packages. Structural claims were checked from source; capability claims were checked against current official documentation. Benchmark results must not be invented from this audit.

## Mandatory intake gate for the next phase

Before doing any analysis or implementation, next phase must:

1. enumerate every `docs/audit/*.md` file present at phase start;
2. read each file in full, including this report and files created after it;
3. record an intake manifest listing filenames and unresolved cross-audit decisions;
4. stop if audit findings conflict on input validity, compositing ownership, metric semantics, or implementation order, until conflict is resolved.

A summary or selected excerpts do not satisfy this gate.

## Current processing graph summary

Legend: `L` lazy Dask/xarray; `B` explicit materialisation barrier; `C` synchronous CPU; `N` nested compute inside a delayed task.

```text
calculate_metrics
|
+-- validate
|   +-- GeoTIFF directory: open_rasterio + chunk(auto)                [L]
|   +-- per-file rio.reproject when grids differ                     [C]
|   +-- xr.concat(...).chunk("auto")                                 [L]
|   +-- GeoPandas read/reproject and Shapely validation              [C]
|   +-- projected raster: chunk("auto")                             [L]
|
+-- preprocess
|   +-- AOI intersects/dissolve                                      [C]
|   +-- all-null timestep reduction -> compute                       [B]
|   +-- corridor rasterize to full 2-D NumPy mask                    [C]
|   +-- >=70% validity reduction -> compute                          [B]
|   +-- temporal map_overlap(depth=time:2)                           [L]
|   +-- >=95% post-fill reduction -> compute                         [B]
|
+-- one delayed preprocess_feature per section
|   +-- clip + Rasterio feature mask                                 [C/L]
|   +-- persistence mean and refuge area via .values.item()          [N]
|   +-- apply_ufunc label -> skeleton -> EDT                         [L, CPU kernels]
|
+-- one delayed process_feature_batch per section x date batch
|   +-- Python loop over dates
|   +-- np.asarray(label), np.asarray(EDT), np.asarray(skeleton)      [N]
|   +-- skimage regionprops_table                                    [C]
|   +-- Pandas pixel table + Python edge construction + igraph       [C]
|   +-- Pandas DataFrames and concat                                 [C]
|
+-- compute all section summaries                                   [B]
+-- client-side Pandas concat + groupby.apply                        [C]
+-- write one CSV                                                    [C]
|
+-- optional export phase: second compute call
    +-- recompute upstream delayed graph unless explicitly persisted
    +-- nested delayed polygon tasks + nested compute                [N]
    +-- Rasterio shapes -> Shapely -> GeoPandas                      [C]
    +-- concatenate all vectors in memory; write Shapefiles          [C]
```

No monthly compositing node exists. Every retained timestamp is treated as a metric period. No aligned native valid-observation array enters the graph.

## Chunking strategy audit

### Current behaviour

- GeoTIFFs are opened with `band=-1, x="auto", y="auto"`, concatenated, then rechunked again with `.chunk("auto")` in [`validate_input_folder()`](../../ecofragments/utils/calc_metrics.py#L102).
- Every DataArray is rechunked with `.chunk("auto")` in [`validate_data_array_cm()`](../../ecofragments/utils/calc_metrics.py#L521), regardless of storage chunks or downstream algorithm.
- Dask's automatic target is commonly 128 MiB, but shape selection is generic. It does not know that temporal filling needs time overlap, monthly reductions need observations grouped by month, or morphology needs 2-D topology.
- Temporal `map_overlap` has depth two. Small auto-generated time chunks can be rechunked automatically; large time chunks increase per-task spatial memory. No invariant checks chunk length against overlap depth.
- Label, skeleton, and EDT wrap `y,x` as gufunc core dimensions. A local reproduction with two chunks along each spatial dimension raises Xarray's documented `ValueError`; `allow_rechunk=True` would only hide the problem by creating whole-section chunks and raising peak memory.
- Storage alignment is not preserved across the two auto-chunk calls. Misaligned GeoTIFF/NetCDF/Zarr chunks can amplify reads and graph size.

### Required chunk contracts

One universal chunk layout is wrong. Use named stage contracts and validate them at boundaries.

| Stage | Required logical layout | Reason |
|---|---|---|
| Input normalization | storage-aligned `time,y,x`; target bytes configurable | Avoid read amplification; retain out-of-core input |
| Validity, compositing, occurrence | moderate time chunks spanning several observations; spatial chunks sized to worker/GPU memory | Pixelwise work and reductions parallelize cleanly |
| Monthly materialization | `time=1` or a small month batch; storage-aligned spatial chunks | Natural checkpoint and bounded later reads |
| Distributed connected components | one 2-D month at a time, spatially chunked | Cross-chunk label reconciliation must see only spatial neighbours, never connect adjacent months |
| Component morphology | one component crop or bounded bucket per task | Exact skeleton/EDT need complete component context |
| Tabular output | partition by section/month, not one client DataFrame | Bound memory and allow incremental writes |

Choose chunk bytes from memory budget, not fixed pixel dimensions:

```text
target_chunk_bytes <= worker_memory * usable_fraction / concurrent_chunks_per_worker
spatial_pixels <= target_chunk_bytes / (time_chunk * dtype_bytes * live_array_multiplier)
```

Start benchmark search in Dask's documented 10 MiB-1 GiB range, usually 64-256 MiB here. Morphology has a high live-array multiplier; input-sized chunks safe for an `int8` mask may be unsafe once `int32` labels, boolean temporaries, and `float32` EDT coexist.

## Eager and CPU choke points

| Priority | Stage and evidence | What materialises/runs on CPU | Scaling effect | CUDA truth |
|---|---|---|---|---|
| Critical | Spatial gufuncs, [`pre_process_layer()`](../../ecofragments/utils/calc_metrics.py#L911) | Whole `y,x` slice required per time | Multi-chunk spatial input errors or whole-section memory | Backend swap does not solve global chunk topology |
| Critical | Batch summary, [`summarize_block()`](../../ecofragments/utils/calc_metrics.py#L949) | Three separate `np.asarray()` calls | Nested compute; repeated label ancestry; worker blocking | Forces host NumPy even if blocks started as CuPy |
| Critical | Skeleton loop, [`skeletonize_label()`](../../ecofragments/utils/calc_metrics.py#L898) | `np.unique`, then full-raster boolean mask and scikit-image call per label | Approximately full-image scan/allocation per component; severe fragmentation penalty | Current exact function is CPU; cuCIM alternatives are not proven equivalent |
| Critical | Path graph, [`compute_length_single_graph()`](../../ecofragments/utils/calc_metrics.py#L1032) | `argwhere`, Pandas rows, Python coordinate map/edge loops, igraph graph/subgraphs | High Python/object memory per skeleton pixel; weak threaded scaling | igraph is CPU; no CuPy dispatch |
| High | Validity filters, [`preprocess()`](../../ecofragments/utils/calc_metrics.py#L282), [`update_nodata_in_rcor_extent()`](../../ecofragments/utils/calc_metrics.py#L671), [`fill_nodata_darray()`](../../ecofragments/utils/calc_metrics.py#L779) | Three reductions followed by `.compute()` | Three scheduler round trips and repeated source reads; graph fusion ends | Reductions are GPU candidates, barriers still need consolidation |
| High | Persistence scalars, [`calculate_pixel_persistence_metrics()`](../../ecofragments/utils/calc_metrics.py#L876) | Two `.values.item()` calls inside delayed section task | Synchronous worker computes; shared persistence reduction may run twice per section | Reduction can use CuPy; scalar transfer is expected only at explicit graph boundary |
| High | Region statistics, [`compute_area_and_perimeter_df()`](../../ecofragments/utils/calc_metrics.py#L1196) | scikit-image `regionprops_table` on whole label image | Full 2-D host array plus property temporaries | Area is easy on GPU; exact Crofton/property parity needs certification |
| High | Patch tables and metrics, [`main.py`](../../ecofragments/main.py#L69) | Pandas concat at batch, section, global levels; client `groupby.apply` | Peak RAM grows with every patch row; imported Dask DataFrame is unused | cuDF would not remove geometry/graph CPU islands and is not justified initially |
| High | Polygon export, [`extract_polygons_map_blocks()`](../../ecofragments/utils/calc_metrics.py#L1290) | Nested compute, Rasterio `shapes`, Shapely objects, GeoDataFrames | Dynamic inner graph; all per-time frames concatenated in memory | Rasterio/GDAL, Shapely, and GeoPandas path is CPU |
| High | Export orchestration, [`main.py`](../../ecofragments/main.py#L102) | Separate second `compute()` for exports | Summary and preprocessing graph can run again | CUDA does not prevent recomputation |
| Medium | AOI handling, [`match_input_extent()`](../../ecofragments/utils/calc_metrics.py#L639) and [`create_mask()`](../../ecofragments/utils/calc_metrics.py#L859) | GeoPandas/Shapely intersection/dissolve and Rasterio rasterization | Client CPU; one full mask plus one mask per section | Keep CPU; geometry count usually small relative to rasters |
| Medium | GeoTIFF alignment, [`process_images()`](../../ecofragments/utils/calc_metrics.py#L29) | `rio.reproject`, powered by Rasterio/GDAL | Current method is not Dask reprojection; can materialise large rasters | CPU I/O/reprojection boundary |
| Medium | Persistence GeoTIFF, [`main.py`](../../ecofragments/main.py#L121) | `rio.to_raster(..., compute=True default)` | Immediate synchronous write; no shared output graph | GDAL write remains CPU |

Unused imports give false confidence: `dask_image.ndmeasure`, `dask_regionprops.regionprops`, and `dask.dataframe` are imported in [`calc_metrics.py`](../../ecofragments/utils/calc_metrics.py#L7) but do not participate in executed kernels.

## Dask risks

| Risk | Consequence | Required correction |
|---|---|---|
| Dask collections created inside delayed return values | Outer scheduler cannot optimize inner array graph as one static graph | Build array graph outside delayed tasks; use Dask collections directly or convert explicit blocks with `.to_delayed()` |
| Nested `compute()`/implicit `__array__` on workers | Blocking, scheduler recursion, poor diagnostics, repeated work | One top-level compute/persist per stage; kernel functions accept concrete NumPy/CuPy blocks only |
| Label ancestry consumed in three independent computes | Connected components may be recalculated for label, skeleton, and EDT | Fuse per-component work or persist/checkpoint the label product once |
| Fixed batch heuristic (`6` dates for <3 sections, otherwise `36`) | Task duration/memory changes for unrelated feature count; no hardware/data sensitivity | Batch by measured bytes and target task duration, with config and benchmark tuning |
| Python-heavy graph construction under threaded scheduler | GIL-bound loops limit parallelism; worker threads can appear busy without throughput | Vectorize edge construction; use process workers for CPU object work; cap native-library threads |
| Overlapping AOIs | Same source chunks, persistence reductions, and morphology may repeat by section | Composite/checkpoint globally; schedule spatial windows with shared persisted source; document overlap cost |
| Full patch path stored in object column even when export is off | Large, non-columnar memory and serialization cost | Make path geometry an opt-in side product; aggregate metrics before returning default table |
| Client-side final concat/groupby | Driver OOM despite bounded workers | Return compact metric rows; use Dask DataFrame/partitioned Parquet for large patch tables |
| Optional vector export shares metric graph | Export can recompute all metrics and retains large GeoDataFrames | Separate export DAG consuming a persisted/checkpointed monthly or label product |
| No checkpoint between input/fill/morphology | Retry or export repeats expensive source work | Prefer Zarr stage checkpoints with provenance; allow ephemeral persist only when cluster memory supports it |
| `int16` component labels | More than 32,767 labels can overflow and corrupt component identity | Use `int32` labels and test highly fragmented masks |
| Millions of very small component tasks | Scheduler overhead can dominate useful work | Bucket component crops by section/month and estimated pixel work; target tasks over 100 ms |

Dask documents task overhead around 0.2-1 ms and recommends useful task durations above roughly 100 ms. Current date batches are coarse at outer level, but their hidden inner graphs and per-component Python work prevent reliable granularity control.

## Memory risk

For a 500-step, 10,000 x 10,000 raster, raw logical sizes alone are:

| Array | Dtype | Logical size |
|---|---:|---:|
| Water mask | `int8` | 46.6 GiB |
| Labels | `int16` current | 93.1 GiB |
| Skeleton labels | `int16` current | 93.1 GiB |
| EDT | `float32` | 186.3 GiB |
| Combined, before temporaries |  | 419.1 GiB |

One 10,000 x 10,000 timestep is already about 95 MiB mask + 191 MiB labels + 191 MiB skeleton + 382 MiB EDT = 859 MiB. Boolean masks, SciPy/scikit-image temporaries, label copies, Dask task inputs, and igraph/Pandas objects add more. A whole-section gufunc can therefore exceed 1-2 GiB per active task before graph construction.

Worst memory is not necessarily the dense raster. Highly fragmented water creates many labels, Pandas rows, Python dictionary entries, igraph vertices/edges, object-array paths, Shapely geometries, and GeoDataFrame rows. These objects commonly cost many times their packed numeric representation. Benchmark water density and fragmentation separately.

## Spatial-kernel assessment

| Operation | Exact Dask-first CPU route | CUDA-capable route | Required caveat |
|---|---|---|---|
| Connected components | `dask_image.ndmeasure.label` provides distributed labeling across chunks; run separately per 2-D month | `cupyx.scipy.ndimage.label` works on one device array | Current dask-image capability table does not mark label as GPU-supported. Local CuPy label is not cross-chunk reconciliation |
| Small-object removal | Count pixels per global label with chunked reductions; filter labels lazily | cuCIM provides GPU `remove_small_objects` for local arrays | Filtering before cross-chunk reconciliation can delete valid components split across tiles |
| Skeletonization | After global labels, crop each component and run current scikit-image semantics in bounded CPU tasks | cuCIM documents `thin` and `medial_axis`, not a certified drop-in for current `skeletonize` result | Algorithm changes can alter path length/AWRe. Keep CPU until scientific parity is accepted |
| Exact EDT | Run SciPy EDT per complete component crop padded by background, or whole section when bounded | `cupyx.scipy.ndimage.distance_transform_edt` works on one device array | dask-image does not implement distributed EDT. Fixed halos are not generally exact because required distance can exceed halo |
| Area/count | Dask bincount/reductions over global labels | CuPy reductions/bincount | Strong first CUDA morphology candidate |
| Crofton perimeter/regionprops | CPU component crop or boundary-aware reduction | cuCIM has measurement functions, including region properties/perimeter primitives | Verify requested property, dtype, connectivity, and numerical parity for pinned versions |
| Longest skeleton path | CPU component task; igraph or a more compact compiled/vectorized implementation | No current CuPy path; cuGraph would be a separate algorithm/dependency project | Current code builds graph through Pandas/Python, which is the immediate problem |
| Polygonization | Stream Rasterio `shapes` per partition; Shapely/GeoPandas CPU | None in current stack | Keep optional and outside core metric DAG |

Component crops provide a useful exactness boundary: once global component IDs are correct, a component plus a one-cell background pad contains the context needed for its binary EDT, skeleton, and local properties. This avoids full-section per-label scans. Crop extraction and task bucketing still need careful graph-size control.

## Valid-observation and monthly compositing design

Current pipeline infers validity from mask values, uses hard-coded 70% and 95% timestep thresholds, fills across time, converts every non-`1` value to dry, and divides persistence by total timesteps. It has no monthly compositor. This is both a scientific and scaling blocker.

Required Dask-first order:

1. Load a dataset contract containing at least aligned `water` and `valid_obs`; retain confidence/method provenance when supplied.
2. Normalize sentinel values lazily. Never cast upstream `uint8` sentinels to `int8` before decoding them.
3. Validate CRS, transform, shape, dimension names, cadence, and chunk metadata without reading full arrays.
4. Compute native valid counts and water-among-valid counts with Dask reductions. Temporal filling must not rewrite the native-observation denominator.
5. Build explicit monthly products from valid observations:
   - `valid_count_month`: count of native valid observations per pixel;
   - `max_water`: any confidently wet valid observation in month;
   - `median_water`: median/thresholded secondary composite required by dry-down analysis;
   - `valid_fraction_month`: AOI/zone valid fraction and low-valid flag.
6. Write chosen monthly products and diagnostic counts to a provenance-bearing Zarr checkpoint, or persist them when dataset is demonstrably memory-safe.
7. Run patch morphology on monthly masks, not on every raw acquisition.
8. Compute occurrence as `water_valid_count / valid_count`, mask below `min_valid_obs`, and materialise only final rasters/compact diagnostics.

Month grouping metadata may be eager because it is small. Raster values must remain lazy. Consolidate the current three full-raster validity scans into one shared reduction graph and one explicit compact diagnostic boundary.

## I/O stack audit

| Format | Current support | Scaling/CUDA finding | Recommended contract |
|---|---|---|---|
| Zarr | Path not accepted; canonical multivariable Dataset is rejected unless caller reshapes it | Best fit for chunked intermediate/canonical data. Decode and storage I/O remain CPU; array kernels may become CuPy after load | First-class `open_zarr` adapter; consolidated metadata; preserve source chunks where suitable; explicit rechunk plan; checkpoint monthly products |
| NetCDF/HDF5 | Only works when caller opens a DataArray/Dataset first; no path adapter | Lazy with a Dask-capable backend, but file locking and storage chunk layout can limit concurrency. No GPU decode | First-class adapter with engine/lock policy, decoded CRS, chosen variables, and chunk inspection; avoid one giant unchunked variable |
| GeoTIFF stack | Directory loader exists; one file per time; per-file alignment may call non-Dask `rio.reproject` | Rasterio/GDAL and compression are CPU. File count, locks, and misaligned chunks can dominate. `rio.to_raster` computes immediately by default | Open tiled rasters with storage-aligned chunks and explicit locks; use odc-geo for Dask reprojection; stage long stacks to Zarr; write tiled/windowed outputs through a separate DAG |
| Shapefile/vector | GeoPandas eager read and write | CPU object processing; Shapefile is single-output and poor for large temporal feature sets | Small AOI inputs are acceptable on CPU. Prefer partitioned GeoParquet for large optional exports; retain Shapefile only as compatibility output |

[`process_images()`](../../ecofragments/utils/calc_metrics.py#L29) uses `rio.reproject`; rioxarray's official documentation directs Dask reprojection users to odc-geo or pyresample. Current dependency set already includes odc-geo and uses it only for the geographic-CRS validation branch.

## CUDA-ready design proposal

### One policy, one capability registry

Place detection and dispatch in one compute module, not inside scientific functions. Suggested public policy:

```text
accelerator = "none" | "auto" | "cuda"     # default: "none"
cuda_strict = false                          # strict mode fails on unsupported requested kernels
device_memory_fraction = configurable
```

At run start, create an immutable capability record containing:

- CuPy import/version and compatible CUDA runtime;
- visible device count and selected device;
- tiny allocation plus kernel smoke test;
- available device memory;
- optional cuCIM availability/version;
- certified kernel list for this package version;
- planned backend and actual backend for each stage.

Do not import optional GPU packages at package import time. Put them in an optional extra, not core dependencies. `accelerator="cuda"` should fail clearly when CUDA cannot initialize. CPU-only stages may still run in a mixed pipeline, but run metadata must name them; no log or result may claim full-GPU execution.

### Capability matrix

| Stage | CUDA status | Decision |
|---|---|---|
| Elementwise mask/sentinel normalization | Supported by CuPy-backed Dask Array | Eligible first |
| Valid counts, wet counts, sums/max/means | Supported by CuPy-backed Dask Array | Eligible first |
| Monthly group/resample | Conditional: xarray metadata stays CPU; block reductions may stay CuPy | Certify exact operations with tests before enabling |
| Rechunk/shuffle | Supported by Dask/CuPy, but can exhaust VRAM or transfer heavily | Benchmark and enforce memory limits |
| Raster decode, GDAL reprojection, GeoTIFF write | CPU | Keep CPU boundary |
| Global distributed connected components | CPU dask-image route available; current dask-image table has no GPU support | CPU baseline. Local CuPy label only for one-device complete arrays/crops |
| Exact EDT | Local CuPy supported; distributed Dask EDT absent | GPU only for complete component crops/whole bounded sections |
| Small-object removal | Local cuCIM candidate | Enable only after global label and parity tests |
| Skeletonization | No proven equivalent for current output | CPU |
| Region area/count | CuPy candidate | Eligible after label plan |
| Crofton perimeter/regionprops | cuCIM candidate, property-specific | Experimental until parity certified |
| igraph path length | CPU | CPU |
| Pandas/GeoPandas/Shapely/Rasterio polygon export | CPU | CPU |

### Transfer boundary

Initial useful mixed design is:

```text
CPU storage/decode -> CuPy Dask blocks for validity/compositing/reductions
                   -> monthly Zarr or host monthly masks
                   -> CPU exact morphology/graph/vector stages
```

This reduces many raw observations to monthly masks before host morphology. Moving every component repeatedly between host and device is unlikely to help. Local GPU label/EDT should be enabled only when a benchmark proves that transfer plus kernel time beats CPU and preserves outputs.

## CPU fallback design

CPU is not an alternate code path; it is the reference backend implementing the same interfaces and output schema.

1. Dask/xarray performs input normalization, valid-observation accounting, monthly composites, occurrence, and aggregate raster metrics.
2. Monthly products are checkpointed or persistently shared so sections do not reread/recompute raw acquisitions.
3. Connected components use a cross-chunk CPU algorithm with `int32` labels, separately per section-month.
4. Label counts remove small components only after cross-chunk reconciliation.
5. Component bounding boxes feed batched delayed CPU tasks. Each task computes skeleton, EDT, region properties, and path summary from concrete NumPy crops; it returns compact numeric rows by default.
6. Worker configuration distinguishes numeric kernels from Python/object kernels. Use processes for GIL-heavy graph/vector work, limit native thread pools, and benchmark threads versus processes.
7. Metric rows write incrementally to partitioned Parquet/CSV. Paths and geometries are opt-in side products.
8. Vector export consumes checkpoints in a separate bounded DAG and streams partitions. Failure to export must not force metric recomputation.

## Benchmark plan

### Datasets

| ID | Dataset | Purpose | Required variants |
|---|---|---|---|
| B0 | Tiny analytic masks, 32-512 pixels per side | Exact truth and chunk-boundary tests | Empty/full, diagonal connectivity, component crossing 2/4 chunks, one-pixel neck, holes, long bar for EDT, nodata, multiple observations/month |
| B1 | Bundled `tests/wmask_ts.nc` plus seven sections | Legacy regression and end-to-end smoke | Tests document 63 dates x 7 sections; fixture is about 6.1 MB |
| B2 | Synthetic temporal cube, 24/120/500 times and 2,048/8,192/10,000 pixels per side | Time, space, and graph scaling | Generate lazily; do not require every largest combination to materialise |
| B3 | Fragmentation stress masks | Object-memory and scheduler stress | About 1%, 10%, and 50% wet; few large components versus many tiny components at same wet fraction |
| B4 | AOI stress | Section reuse and overlap | 1, 10, and 100 non-overlapping sections; separate overlapping-section case |
| B5 | Equivalent Zarr, chunked NetCDF, tiled GeoTIFF/COG stack | I/O and storage-chunk sensitivity | At least two chunk layouts per format: aligned and deliberately misaligned |
| B6 | Real Gilbert validation catchment | Representative science/performance and dual-composite result | Raw valid observations plus `max_water` and `median` monthly products; size/provenance recorded, not invented here |

### Environments and modes

- Same Dask-based code in every run.
- CPU reference: one worker/one thread, then local distributed multi-process and multi-thread variants on recorded hardware.
- GPU-available: same host plus one recorded NVIDIA GPU; CuPy-only eligible stages first, then any experimental certified kernels.
- CPU-only environment must install and pass without CuPy, cuCIM, dask-cuda, RAPIDS, or CUDA toolkit.
- Repeat each timed run after one untimed warm-up; report median and dispersion from at least three measured runs.
- Record package versions, scheduler, worker/thread counts, RAM, storage type, CPU, GPU, driver, CUDA runtime, chunk layout, compression, and cold/warm cache state.

### Metrics

| Category | Measurements |
|---|---|
| Correctness | Monthly masks; valid counts; label equivalence after label-ID normalization; component counts/areas; perimeters; skeleton/path outputs; EDT widths; final metric rows and flags |
| Time | End-to-end and per-stage wall time; graph-construction time; first-result latency; pixel-observations/s; section-months/s |
| Dask | Task count; serialized graph bytes; median/p95 task duration; scheduler overhead; worker occupancy; retries; spill bytes; shuffle/transfer bytes |
| Memory | Client and per-worker peak RSS; managed/unmanaged memory; peak GPU VRAM; host-device transfer bytes/time; output table/vector peak memory |
| I/O | Bytes read/written; effective bandwidth; requests/file opens; read amplification where measurable; output file count and size |
| Scaling | Speedup and efficiency over workers; slope versus pixels, timesteps, sections, component count, and skeleton pixels |

Use Dask performance reports/task streams plus process RSS sampling. GPU runs also use CuPy memory-pool statistics and device telemetry. Wall time alone is insufficient.

### Expected outputs and acceptance gates

Benchmark harness must emit machine-readable JSON/Parquet rows and a human-readable report. Each row includes dataset ID, stage, backend planned/actual, hardware, versions, chunks, task/memory/I/O metrics, correctness result, and output checksum.

Acceptance gates:

1. CPU-only install and all correctness tests pass without GPU packages.
2. CPU is canonical. GPU eligible array outputs are exact for integer/count products and within declared dtype-aware tolerance for floating reductions.
3. Label comparison ignores numeric label IDs but requires identical component membership and count.
4. CPU-only skeleton/igraph/vector stages produce identical outputs in CPU and mixed runs.
5. Experimental cuCIM morphology remains disabled unless path-length, width, perimeter, and downstream metric parity pass adversarial shapes and real data.
6. No benchmark promises GPU speedup. Enable a CUDA kernel only when end-to-end or stage-level time improves after transfer cost, without larger failure rate or unacceptable VRAM.
7. Median heavy-task duration should normally exceed 100 ms; scheduler overhead target below 10% of end-to-end time.
8. Peak managed memory stays below configured worker limits without avoidable disk spill; GPU peak stays below configured device fraction.
9. Scaling runs must finish without client accumulation proportional to total patch geometry when vector export is disabled.

## Implementation sequence

1. **Phase intake gate:** read every prior audit markdown in full and reconcile contract decisions. Do not start code from this report alone.
2. **Baseline harness:** add analytic truth datasets, stage timers, Dask diagnostics, checksums, and current CPU baseline before changing kernels.
3. **Input contract:** define aligned water/valid/provenance dataset and explicit Zarr/NetCDF/GeoTIFF adapters. Decode sentinels before dtype conversion.
4. **Chunk policy:** introduce named stage chunk contracts, byte-budget validation, storage alignment inspection, and worker-memory configuration. Remove unconditional `.chunk("auto")` calls.
5. **Temporal Dask graph:** implement valid counts, monthly `max_water`/`median`, occurrence, valid-fraction flags, and one deliberate Zarr/persist boundary. Remove scattered preprocessing computes.
6. **Static orchestration graph:** stop returning Dask collections from delayed functions and stop nested `compute()`/`np.asarray()` calls. Build one visible graph per bounded stage.
7. **CPU global labels:** use cross-chunk 2-D connected components with `int32`, global small-component filtering, and correctness tests at chunk boundaries.
8. **Component work units:** derive bounding boxes, bucket crops into >100 ms tasks, fuse skeleton/EDT/properties/path work, and return compact numeric summaries. Eliminate full-raster scan per label.
9. **Tabular scaling:** aggregate before client collection; partition patch tables by section/month; make path/object data opt-in; use Dask DataFrame/Parquet only where it reduces memory rather than as decoration.
10. **Export isolation:** consume monthly/label checkpoints, stream CPU polygon/line/point partitions, and avoid recomputing metrics. Prefer GeoParquet; keep Shapefile compatibility optional.
11. **Backend registry:** add central CPU capability interface and metadata first. Add optional CuPy detection without changing default results or dependencies.
12. **CUDA tranche 1:** certify pixelwise normalization, validity, compositing, and reductions. Benchmark transfer boundaries and mixed-pipeline truthfulness.
13. **CUDA tranche 2, conditional:** test local component label/EDT and selected cuCIM properties. Keep disabled unless semantic and performance gates pass. Do not replace skeletonization by `thin`/`medial_axis` silently.
14. **Release gate:** run CPU-only, multi-worker CPU, and GPU-available benchmark matrix; publish raw rows, environment manifest, failures, and actual per-stage backend use.

## Sources for capability claims

- [Dask GPU support and CuPy-backed arrays](https://docs.dask.org/en/stable/gpu.html)
- [Dask chunk sizing and storage alignment](https://docs.dask.org/en/latest/array-chunks.html)
- [Dask delayed best practices](https://docs.dask.org/en/latest/delayed-best-practices.html)
- [Xarray `apply_ufunc` Dask/core-dimension constraints](https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html)
- [Xarray duck arrays and CuPy support](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html)
- [dask-image function and GPU coverage](https://image.dask.org/en/stable/coverage.html)
- [dask-image distributed connected-components API](https://image.dask.org/en/v2024.5.2/dask_image.ndmeasure.html)
- [CuPy `cupyx.scipy.ndimage` functions](https://docs.cupy.dev/en/v13.3.0/reference/scipy_ndimage.html)
- [cuCIM stable API](https://docs.rapids.ai/api/cucim/stable/api/)
- [Rasterio feature extraction/rasterization API](https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html)
- [rioxarray reprojection and Dask write behaviour](https://corteva.github.io/rioxarray/stable/rioxarray.html)
- [python-igraph CPU/C-core description](https://python.igraph.org/en/latest/index.html)

Capability documentation is newer than some exact pins in [`requirements.txt`](../../requirements.txt). Implementation must pin and test a compatible matrix; documentation availability alone is not proof that a project-pinned environment supports a path.
