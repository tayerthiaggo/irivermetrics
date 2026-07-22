# HydroFragments v1.2 — First-Pass Repository Triage

**Date:** 2026-07-10  
**Branch:** development  
**Scope:** identify high-signal risks before the deeper spec-compliance, scientific, Dask/CUDA, and documentation audits.  
**Constraint:** no source files edited.

---

## 1. Project overview

The repository is a transitional codebase with three competing identities:

| Identity | Where it appears | v1.2 target |
|---|---|---|
| `iRiverMetrics` / `irivermetrics` | [`README.md`](README.md), [`docs/module2.md`](docs/module2.md), example notebooks, test data directory (`results_iRiverMetrics/`) | `HydroFragments` |
| `EcoFragments` / `ecofragments` | [`pyproject.toml`](pyproject.toml#L2), package directory [`ecofragments/`](ecofragments/), [`docs/architecture.md`](docs/architecture.md) | `hydrofragments` |
| `HydroFragments` | [`docs/HydroFragments_v1.2_spec.md`](docs/HydroFragments_v1.2_spec.md) | final name |

The only implemented execution path is [`calculate_metrics()`](ecofragments/main.py#L10) in [`ecofragments/main.py`](ecofragments/main.py), backed by a single 1,300-line utility module [`ecofragments/utils/calc_metrics.py`](ecofragments/utils/calc_metrics.py). It computes 16 section-level metrics per timestep from a binary water-mask time series and a polygon section layer, writes a wide CSV, and optionally exports shapefiles and a pixel-persistence raster.

The v1.2 specification is a scientific and architectural migration, not a patch: it requires source-agnostic binary-mask + valid-observation input, a tidy long output schema with config hashing, fixed-denominator metrics, zonation, hydrological-year dynamics, a connectivity module, and Dask-first (CUDA-ready) processing. Current implementation is roughly at the pre-migration starting point.

---

## 2. Main execution pathways

```
calculate_metrics(da_wmask, rcor_extent, ...)
├── calc_metrics.validate()
│   ├── directory → DataArray loader (GeoTIFF stack)
│   ├── coerce_water_mask_dataarray()
│   ├── CRS check / UTM reprojection
│   └── setup_directories_cm()
├── calc_metrics.preprocess()
│   ├── match_input_extent()
│   ├── drop empty timesteps
│   ├── update_nodata_in_rcor_extent()   # 70% valid threshold
│   └── fill_nodata_darray()             # ±2 temporal fill, 95% threshold
├── per section (dask.delayed):
│   ├── preprocess_feature()
│   │   ├── clip to section
│   │   ├── calculate_pixel_persistence_metrics()   # full-series static
│   │   └── pre_process_layer()                     # label / skeleton / EDT
│   └── process_feature_batch() for each date batch
│       └── summarize_block()                       # area, length (igraph BFS), width
├── dask.compute(*summary_tasks)
├── groupby(date, section).apply(process_metrics)   # 16 metrics
└── write ecof_metrics.csv + optional shapefiles + Pixel_Persistence.tif
```

Only one public entry point exists. There is no CLI, no config object, and no separate I/O, zonation, hydrology-year, or connectivity modules.

---

## 3. Top 20 risks, ordered by severity

### Critical — wrong science or broken upstream integration

1. **WaterMask-TSFill sentinels are treated as valid pixels.** Current nodata logic recognises `NaN`, `-1`, and the temporary fill sentinel `2`; upstream Zarr uses `255` = invalid/unresolved and `254` = outside AOI. These values will be counted as water or valid background, corrupting every area, persistence, and morphology metric.
2. **No valid-observation layer is consumed.** Occurrence frequency, Refuge Area, and all per-pixel temporal metrics need `water_obs / valid_obs`. The pipeline currently divides by total timesteps (`calculate_pixel_persistence`), so cloudy or low-revisit pixels are systematically biased.
3. **Dropped circular metrics are still emitted.** The code outputs `PF`, `PLF`, `AWMPA`, `AWMPL`, and `AWMPW`, which v1.2 removes or replaces (`PF`/`PLF` circular; `AWMPA` redundant; `AWMPW` replaced by unweighted width distribution). Keeping them risks downstream users continuing to use invalid indices.
4. **Cannot parse the upstream multi-variable Zarr/Dataset.** [`coerce_water_mask_dataarray()`](ecofragments/utils/calc_metrics.py#L507) raises an assertion error on datasets with more than one variable unless one is named `water`. WaterMask-TSFill outputs `water_mask`, `confidence`, `method_flag`, and `observed`.
5. **Pixel persistence is a naive temporal mean, not occurrence frequency.** [`calculate_pixel_persistence()`](ecofragments/utils/calc_metrics.py#L490) computes `sum(wet) / total_obs`. v1.2 requires `sum(wet) / valid_obs` with a per-pixel valid-obs floor.

### High — structural v1.2 mismatch

6. **Core v1.2 metrics are missing.** No `LPI`, no `dry_down_rate`, and no source-agnostic `occurrence` raster. The existing core set is incomplete.
7. **No zonation module and no no-drainage fallback.** There is no Zone 1/2/3/4 logic, no persistence-proxy fallback, and no circularity guard preventing persistence metrics from being stratified by zones.
8. **Output is a wide CSV without config hash or metadata.** [`ecof_metrics.csv`](ecofragments/main.py#L96) lacks `run_id`, `config_hash`, `crs`, `resolution_m`, `monthly_composite`, `water_threshold`, and the tidy long schema required by §7 of the spec.
9. **AWRe length method is not locked and has no fallback.** [`compute_length_single_graph()`](ecofragments/utils/calc_metrics.py#L1032) always uses skeleton BFS length. The spec requires skeleton length when a channel skeleton exists, `regionprops` major-axis length otherwise, and recording `awre_length_method`.
10. **CRS handling defaults to UTM, not a mandatory equal-area projection.** [`validate_data_array_cm()`](ecofragments/utils/calc_metrics.py#L521) reprojects geographic inputs to `estimate_utm_crs()`. v1.2 mandates equal-area (default EPSG:3577 for Australia) or per-pixel area arrays, and requires documenting length-distortion caveats.
11. **Large secondary/exploratory metric set is absent.** Missing: `MESH`, unweighted pool-width distribution, inter-pool gap, reconnection timing, refuge spatial stability, `TCF`, `DCI`, `RC`, pixel recurrence, and hydroperiod/seasonality.

### Medium — scalability, testing, and documentation

12. **Nested synchronous computes inside `dask.delayed` tasks.** [`preprocess_feature()`](ecofragments/utils/calc_metrics.py#L321) calls [`calculate_pixel_persistence_metrics()`](ecofragments/utils/calc_metrics.py#L885), which uses `.values.item()` inside the delayed graph. [`process_feature_batch()`](ecofragments/utils/calc_metrics.py#L356) calls `np.asarray()` on Dask arrays in [`summarize_block()`](ecofragments/utils/calc_metrics.py#L968). These block Dask workers and defeat the scheduler.
13. **Multiple eager `.compute()` calls break graph fusion.** Pre-fill (`preprocess`), corridor masking (`update_nodata_in_rcor_extent`), and post-fill (`fill_nodata_darray`) each compute boolean masks eagerly. For large time series this adds scheduler round-trips and materialises intermediate arrays.
14. **Third-party CPU libraries break laziness throughout the pipeline.** `scikit-image` skeletonization, `scipy.ndimage` label/EDT, `igraph` BFS, `rasterio` rasterization, `GeoPandas` clipping/dissolving, and final `pandas` groupby are all synchronous, in-memory CPU operations. The current architecture is Dask-at-the-edges, not Dask-first.
15. **No CUDA-ready abstraction.** There is no feature-detection layer, no optional CuPy path, and no separation of CPU-only steps (skeletonization, igraph) from array steps that could run on GPU. Adding GPU later will require refactoring the same choke points.
16. **Package and directory names are wrong.** The installed package is `ecofragments`, docs say `iRiverMetrics`, and tests reference `results_ecofragments`. v1.2 requires `hydrofragments` everywhere.
17. **Regression test is broken.** [`tests/conftest.py`](tests/conftest.py#L34) points to `results_ecofragments/metrics/ecof_metrics.csv`; the actual path is [`tests/results_iRiverMetrics/metrics/irm_metrics.csv`](tests/results_iRiverMetrics/metrics/irm_metrics.csv). `test_calculate_metrics_regression` will fail with `FileNotFoundError`.
18. **Tests do not cover v1.2 contracts.** There are no tests for Zarr parsing, sentinel handling, zonation, fixed-denominator metrics, edge flags, config hashing, CRS guards, or the new metrics.
19. **Documentation advertises dropped metrics and wrong imports.** [`docs/module2.md`](docs/module2.md#L110) documents `AWMPA`/`AWMPL`/`AWMPW`/`PF`/`PLF` and imports `irivermetrics.irm_main`, which does not exist. The metric register and API signature are out of sync with the spec.
20. **Architecture document claims a generic patch-dynamics tool.** [`docs/architecture.md`](docs/architecture.md) frames EcoFragments as domain-agnostic (aquatic, terrestrial, urban). v1.2 deliberately restricts scope to river/surface-water systems. This mismatch will confuse reviewers and future contributors.

---

## 4. Evidence table

| Finding | File / function | Why it matters | How to verify | Likely fix area | Priority |
|---|---|---|---|---|---|
| `255`/`254` sentinels not treated as nodata | [`calc_metrics.update_nodata_in_rcor_extent()`](ecofragments/utils/calc_metrics.py#L704) | Upstream invalid/outside-AOI pixels become valid, corrupting all metrics | Feed a synthetic Zarr with `255`/`254`; inspect output wet counts | I/O + preprocessing sentinel map | Critical |
| No valid-observation layer consumed | [`calc_metrics.validate()`](ecofragments/utils/calc_metrics.py#L226), [`calculate_pixel_persistence()`](ecofragments/utils/calc_metrics.py#L490) | Occurrence/RA denominators are wrong; low-obs pixels over-represented | Check function signatures and preprocess flow | Input contract / `io/` module | Critical |
| PF/PLF/AWMPA/AWMPL/AWMPW still emitted | [`calc_metrics.process_metrics()`](ecofragments/utils/calc_metrics.py#L426) | Circular/redundant metrics v1.2 explicitly drops/replaces | Run pipeline; inspect CSV columns | Metric register refactor | Critical |
| Multi-var Dataset rejected | [`calc_metrics.coerce_water_mask_dataarray()`](ecofragments/utils/calc_metrics.py#L507) | WaterMask-TSFill Zarr cannot load | Pass a 4-variable Dataset to `calculate_metrics` | `io/` loader + variable selection | Critical |
| Pixel persistence = naive mean | [`calc_metrics.calculate_pixel_persistence()`](ecofragments/utils/calc_metrics.py#L490) | Replaces occurrence frequency incorrectly | Compare formula to spec §6.17 | Preprocessing / persistence module | Critical |
| LPI and dry-down missing | [`calc_metrics.process_metrics()`](ecofragments/utils/calc_metrics.py#L426) | Core v1.2 metrics absent | Search codebase for `largest_patch_index`, `dry_down` | New metrics modules | High |
| No zonation module | entire codebase | v1.2 zone schema and circularity guard absent | Search for Zone 1/2/3/4 logic | `zones/` module | High |
| Wide CSV, no config hash | [`ecofragments/main.py`](ecofragments/main.py#L96) | Output schema violates §7 tidy-long requirement | Inspect output format | Output writer / schema | High |
| AWRe length method not locked | [`calc_metrics.compute_length_single_graph()`](ecofragments/utils/calc_metrics.py#L1032) | Spec requires skeleton/major-axis fallback | Check `process_metrics` AWRe path | `metrics/patches.py` | High |
| UTM default, not equal-area | [`calc_metrics.validate_data_array_cm()`](ecofragments/utils/calc_metrics.py#L521) | Area/length metrics may be distorted | Check CRS reprojection branch | CRS guard + config | High |
| MESH/width-distribution/gap/reconnection/TCF/DCI/RC/recurrence/hydroperiod absent | [`calc_metrics.process_metrics()`](ecofragments/utils/calc_metrics.py#L426) | Large secondary metric set missing | Search codebase for metric names | `metrics/` submodules | High |
| `.values.item()` inside delayed task | [`calc_metrics.calculate_pixel_persistence_metrics()`](ecofragments/utils/calc_metrics.py#L885) | Synchronous nested compute blocks Dask workers | Read `preprocess_feature` call tree | Preprocessing refactor | Medium |
| `np.asarray()` on Dask arrays in delayed task | [`calc_metrics.summarize_block()`](ecofragments/utils/calc_metrics.py#L968) | Forces eager evaluation inside graph | Read `process_feature_batch` → `summarize_block` | Batch processing refactor | Medium |
| Eager `.compute()` in preprocessing | [`calc_metrics.preprocess()`](ecofragments/utils/calc_metrics.py#L282), [`fill_nodata_darray()`](ecofragments/utils/calc_metrics.py#L779), [`update_nodata_in_rcor_extent()`](ecofragments/utils/calc_metrics.py#L704) | Breaks graph fusion; extra scheduler round-trips | Grep for `.compute()` | Preprocessing refactor | Medium |
| CPU-only libraries break laziness | [`pre_process_layer()`](ecofragments/utils/calc_metrics.py#L911), [`compute_length_single_graph()`](ecofragments/utils/calc_metrics.py#L1032), [`match_input_extent()`](ecofragments/utils/calc_metrics.py#L644) | True Dask scalability not achievable with current design | Profile a medium-sized run | Processing architecture | Medium |
| No CUDA-ready design | entire codebase | GPU acceleration impossible without later refactor | Search for `cupy`, `cuda`, feature flags | `compute/` abstraction | Medium |
| Package name mismatch | [`pyproject.toml`](pyproject.toml#L2), [`README.md`](README.md#L2), [`docs/module2.md`](docs/module2.md#L3) | Imports and docs will break after rebrand | Grep for `ecofragments`, `iRiverMetrics`, `irivermetrics` | Rebrand pass | Medium |
| Regression fixture path wrong | [`tests/conftest.py`](tests/conftest.py#L34) | `test_calculate_metrics_regression` fails | Run `pytest tests/test_integration.py` | Test data path fix | Medium |
| No v1.2 contract tests | [`tests/test_unit_metrics.py`](tests/test_unit_metrics.py), [`tests/test_integration.py`](tests/test_integration.py) | New schema/guards/metrics untested | Review test coverage | New test modules | Medium |
| Docs list dropped metrics / wrong imports | [`docs/module2.md`](docs/module2.md#L81-L110), [`README.md`](README.md#L49) | Users will try to import non-existent packages and use invalid metrics | Read docs | Docs rewrite | Medium |
| Architecture doc claims generic tool | [`docs/architecture.md`](docs/architecture.md#L1-L12) | Contradicts v1.2 river-focused scope | Read intro | Docs rewrite | Low-Medium |

---

## 5. Suggested deeper-audit questions

The next audit phases should answer these before any implementation starts.

### Scientific / spec compliance
1. Should the v1.2 refactor keep the current `calculate_metrics()` signature as a compatibility shim, or break it immediately?
2. What is the smallest credible v1.2 core set? Spec lists 8 core metrics; current code covers ~3.5 (APSEC, LPSEC, N, partial AWRe).
3. Is the current skeleton/BFS length algorithm acceptable for v1.2 AWRe, or should a channel-aware skeleton be required before `awre_length_method=skeleton` is recorded?
4. How should the pipeline behave when `valid_obs_frac` is below the per-pixel floor: mask the pixel, flag it, or drop the timestep?
5. Does the existing test NetCDF (`tests/wmask_ts.nc`) contain enough variability to validate dry-down rate, refuge stability, and recurrence, or do we need a new validation dataset?

### Input / upstream
6. Is WaterMask-TSFill the only upstream source we must support for v1.2, or do we also need a generic GeoTIFF/NetCDF loader with manual valid-obs layer pairing?
7. Should the pipeline raise on grid misalignment between mask and valid-obs layer, or silently resample? (Spec §14 says raise.)
8. How should probabilistic masks be thresholded, and where does `water_threshold`/`threshold_method` live in the config?

### Dask / CUDA
9. Which operations are truly parallelisable at scale? Component labelling, skeletonization, EDT, and igraph BFS are currently per-timestep CPU; can any move to Dask chunk-level `apply_ufunc` or GPU?
10. What is the memory footprint of the current pipeline on a 500-timestep, 10k×10k raster? Is the date-batch strategy sufficient, or do we need spatial tiling too?
11. Where is the boundary between Dask-array lazy computation and Dask-delayed per-section tasks? The current mix of both is the main scalability risk.

### Output / reproducibility
12. What is the canonical output schema: tidy long table only, or long table plus legacy wide CSV behind a flag?
13. Which metadata columns are mandatory for v1.2 MVP versus nice-to-have? (Spec §7.1 lists many; some may be deferred.)
14. How is `config_hash` computed so that scientifically identical runs produce identical hashes across platforms?

### Testing / validation
15. Can the existing `irm_metrics.csv` reference output be mapped to the new metric set for a regression smoke test, given that PF/PLF/AWMPA/AWMPL/AWMPW are removed?
16. What is the validation plan for the asserted-but-not-demonstrated claims in §6.18 (AWRe⊥AWMSI, LPI vs MESH redundancy, NNI instability, pool width as morphology not depth)?

### Documentation / audience
17. Which current docs should be preserved, rewritten, or deleted during the rebrand? `architecture.md` in particular conflicts with v1.2 scope.
18. What is the minimum viable `docs/input_format.md` and `docs/for-managers.md` for v1.2 release?

---

## 6. Uncertainties

- The exact content of `tests/wmask_ts.nc` was not inspected beyond the fixture loader; its dimensionality, CRS, and time coverage are assumed to match the current pipeline.
- GPU/CuPy capability was assessed by absence of code only; no runtime benchmark was performed.
- WaterMask-TSFill upstream output was referenced via the evidence packet; a direct read of its Zarr contract was not repeated here.
- The degree to which the current skeletonization/igraph path is numerically equivalent to the v1.2-required inter-pool-gap and connectivity modules is unknown without a dedicated algorithmic audit.
