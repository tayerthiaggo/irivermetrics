# HydroFragments v1.2 Audit — Evidence Packet

This evidence packet documents the current state, mismatches, and architectural constraints of the HydroFragments repository as of 2026-07-10. It is compiled for downstream ingestion by a senior auditor.

---

## 1. Repository Structure Summary

The current repository contains the following files and directories:

*   **[`ecofragments/`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments)**: The main Python package directory.
    *   **[`__init__.py`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/__init__.py)**: Package initializer exposing public API.
    *   **[`main.py`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/main.py)**: Orchestrates data loading, validation, preprocessing, parallel processing, and final output writing.
    *   **[`utils/calc_metrics.py`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py)**: Core utility module containing spatial logic, validation checks, and metric computation functions.
*   **[`tests/`](file:///d:/RLH/5.6/repos/HydroFragments/tests)**: Testing module.
    *   **[`conftest.py`](file:///d:/RLH/5.6/repos/HydroFragments/tests/conftest.py)**: Pytest fixtures for test water masks, river corridors, and regression outputs.
    *   **[`test_unit_metrics.py`](file:///d:/RLH/5.6/repos/HydroFragments/tests/test_unit_metrics.py)**: Unit tests for core utility helper functions.
    *   **[`test_integration.py`](file:///d:/RLH/5.6/repos/HydroFragments/tests/test_integration.py)**: End-to-end integration tests with regression checks against historical outputs.
    *   **[`wmask_ts.nc`](file:///d:/RLH/5.6/repos/HydroFragments/tests/wmask_ts.nc)**: NetCDF time series of binary water masks used as test input.
    *   **[`rcor_extent.shp`](file:///d:/RLH/5.6/repos/HydroFragments/tests/rcor_extent.shp)** (and sidecars): Shapefile representing 7 river sections.
*   **[`docs/`](file:///d:/RLH/5.6/repos/HydroFragments/docs)**: Documentation.
    *   **[`architecture.md`](file:///d:/RLH/5.6/repos/HydroFragments/docs/architecture.md)**: Architectural design, data flow, and algorithm details.
    *   **[`module2.md`](file:///d:/RLH/5.6/repos/HydroFragments/docs/module2.md)**: Usage guide and metric definitions for `calculate_metrics`.
    *   **[`HydroFragments_v1.2_spec.md`](file:///d:/RLH/5.6/repos/HydroFragments/docs/HydroFragments_v1.2_spec.md)**: Spec and refactor contract for v1.2.
*   **[`pyproject.toml`](file:///d:/RLH/5.6/repos/HydroFragments/pyproject.toml)**: Project packaging configuration, dependencies, and test configurations.
*   **[`README.md`](file:///d:/RLH/5.6/repos/HydroFragments/README.md)**: Repository overview, setup, and usage examples.

---

## 2. Main Public API and Execution Path

### Public API
The main entry point is [`calculate_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/main.py#L10):
```python
def calculate_metrics(da_wmask, 
                      rcor_extent=None, 
                      outdir=None, 
                      section_length=None,
                      section_name_col=None, 
                      min_patch_size=2, 
                      img_ext='.tif', 
                      export_shp=False, 
                      export_PP=False, 
                      fill_nodata=True):
```

### Execution Path
1.  **Validation**: [`calc_metrics.validate`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L226) checks that the input is a valid directory or `xarray.DataArray`/`Dataset`, loads raster files if a directory is passed, setups output directories, creates default bounding box polygons if `rcor_extent` is missing, and validates/reprojects coordinate reference systems.
2.  **Preprocessing**: [`calc_metrics.preprocess`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L282) clips spatial extents to the intersection of the mask and the corridor. It filters invalid timesteps using a 70% valid pixel threshold, fills missing/cloud data using a $\pm 2$ temporal interpolation window (if `fill_nodata=True`), and drops remaining timesteps below a 95% valid pixel threshold after filling.
3.  **Parallel Tasks Delineation**:
    *   Calls [`calc_metrics.preprocess_feature`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L321) (wrapped in `dask.delayed`) per corridor section. This clips the mask, calculates pixel persistence and refuge area, and runs labeling, skeletonization, and distance transform operations on the section mask.
    *   Generates batches of dates and triggers [`calc_metrics.process_feature_batch`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L356) (wrapped in `dask.delayed`) to compute metric snapshots per timestep using [`calc_metrics.summarize_block`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L949).
        *   [`summarize_block`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L949) calls [`compute_area_and_perimeter_df`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1196) (region properties), [`compute_length_single_graph`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1032) (igraph BFS length), and [`process_edt_width`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1170) (medial axis width).
4.  **Computation and Aggregation**:
    *   `dask.compute(*summary_tasks)` triggers parallel execution.
    *   Groupby aggregation runs [`calc_metrics.process_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L426) to compute the final 16 metrics.
5.  **Output Export**: Writes `ecof_metrics.csv` to disk, optionally exports ESRI shapefiles (`export_shp=True`), and exports the pixel persistence geotiff (`export_PP=True`).

---

## 3. Current Metric Outputs

The current codebase computes and writes the following 16 metrics per `(date, section)` row in the output CSV:

| Metric Name | Formula / Definition | Unit | Category |
| :--- | :--- | :--- | :--- |
| `n_patches` | Count of connected components using 8-neighbour connectivity | count | Summary |
| `wet_area_km2` | $\sum a_i$ (sum of individual patch areas) | $\text{km}^2$ | Summary |
| `wet_length_km` | $\sum l_i$ (sum of geodesic lengths of skeletonized patches) | $\text{km}$ | Summary |
| `wet_perimeter_km` | $\sum p_i$ (sum of individual patch perimeters) | $\text{km}$ | Summary |
| `section_area_km2` | Area of the corridor section polygon | $\text{km}^2$ | Context |
| `section_length_km` | Drainage line length of the corridor section | $\text{km}$ | Context |
| `AWMSI` | Area-Weighted Mean Shape Index: $\sum_i \left( 0.25 \cdot \frac{p_i}{\sqrt{a_i}} \right) \cdot \left( \frac{a_i}{\sum a_i} \right)$ | dimensionless | Morphology |
| `AWRe` | Area-Weighted Elongation Ratio: $\sum_i \left( \frac{2\sqrt{a_i / \pi}}{l_i} \right) \cdot \left( \frac{a_i}{\sum a_i} \right)$ | dimensionless | Morphology |
| `AWMPA` | Area-Weighted Mean Pool Area: $\sum (a_i^2) / \sum a_i$ | $\text{km}^2$ | Morphology |
| `AWMPL` | Area-Weighted Mean Pool Length: $\sum (l_i \cdot a_i) / \sum a_i$ | $\text{km}$ | Morphology |
| `AWMPW` | Area-Weighted Mean Pool Width: $\sum (w_i \cdot a_i) / \sum a_i$ | $\text{km}$ | Morphology |
| `PF` | Pool Fragmentation: $N / \sum a_i$ | $\text{pools} / \text{km}^2$ | Fragmentation |
| `PLF` | Pool Longitudinal Fragmentation: $N / \sum l_i$ | $\text{pools} / \text{km}$ | Fragmentation |
| `APSEC` | Wet Area Percentage of Section: $\left( \frac{\sum a_i}{\text{section\_area}} \right) \cdot 100$ | % | Coverage |
| `LPSEC` | Wet Length Percentage of Section: $\left( \frac{\sum l_i}{\text{section\_length}} \right) \cdot 100$ | % | Coverage |
| `pp_mean_%` | Mean Pixel Persistence where PP > 25% (static over time series) | % | Persistence |
| `ra_area_km2` | Refuge Area where PP > 90% (static over time series) | $\text{km}^2$ | Persistence |

---

## 4. v1.2 Required Metric Outputs

The v1.2 specification modifies the metric register as follows:

### Core Set
*   **Occurrence Frequency**: Pixel-level temporal mean wet fraction, replacing the PP implementation.
*   **Refuge Area (`ra_area_km2`)**: Area of pixels with occurrence frequency $> t_{\text{refuge}}$ (default 90%).
*   **APSEC**: Wetted area fraction normalized by a fixed reference area (`A_ref`).
*   **LPSEC**: Wetted length fraction normalized by a fixed reference length (`L_ref`).
*   **Number of Pools (`n_pools`)**: Count of connected water bodies.
*   **Largest Patch Index (LPI)**: Ratio of the largest patch area to the total AOI/landscape area.
*   **AWRe**: Area-weighted elongation ratio, with lengths computed using a skeleton-based approach where a channel is present, falling back to major axis length otherwise.
*   **Dry-down Rate**: Slope of APSEC contraction over the recession limb per hydrological year (HY), verified across both `max_water` and `median` composites.

### Secondary Set
*   **AWMSI**: Area-weighted boundary complexity.
*   **MESH**: Effective mesh size: $\sum (a_i^2) / A_{\text{total}}$, utilizing a fixed landscape area denominator.
*   **Pool Width Distribution**: Distributional summaries (`mean`, `median`, `max`, `cv`) of unweighted pool widths computed using Euclidean Distance Transform (EDT).
*   **Inter-pool Gap**: Spacing/dry gap between pools along the channel skeleton (metric of record for clustering).
*   **Reconnection Timing**: Lag from dry-minimum to network re-merging, based on graph-based realized connectivity or DCI thresholds.
*   **Refuge Spatial Stability**: Inter-annual Jaccard overlap of the end-dry refuge footprint.
*   **TCF (Temporal Connectivity Frequency)**: Formerly PCF; percentage of timesteps a fixed pool node remains connected.
*   **DCI (Dendritic Connectivity Index)**: Fragment-size based connectivity index utilizing reach-length node weights.
*   **Pixel Recurrence**: Inter-annual reliability: years wet / total years.
*   **Seasonality / Hydroperiod**: Fraction of months wet within a hydrological year.
*   **Realised Connectivity (RC)**: Realised fraction of possible network links on a fixed graph.
*   **Graph Components and LCC**: Component count and largest component fraction based on graph edge rules.

### Exploratory
*   **NNI (Clark-Evans)**: Nearest Neighbor Index. Demoted to exploratory fallback for planar AOIs where no channel skeleton is available.

---

## 5. Expected Upstream Input from WaterMask-TSFill

Inspecting the upstream repository [`WaterMask-TSFill`](file:///D:/RLH/5.6/repos/WaterMask-TSFill) at [`watermask_tsfill/contracts.py`](file:///D:/RLH/5.6/repos/WaterMask-TSFill/watermask_tsfill/contracts.py#L46-L81) reveals the following canonical Zarr schema and semantics for the `water_cube.zarr` output:

*   **Four variables** sharing `(time, y, x)` dimensions:
    1.  `water_mask` (`uint8`): The reconstructed semantic mask.
        *   `0` = dry
        *   `1` = water
        *   `254` = outside AOI
        *   `255` = invalid/unresolved gap
    2.  `confidence` (`uint8`): Posterior confidence $[0, 100]$. `255` represents N/A (outside AOI or unfilled).
    3.  `method_flag` (`uint8`): Identifies the gap-filling source (e.g. `observed=0`, `temporal_sim=2`, `unresolved=10`, `outside_aoi=255`).
    4.  `observed` (`bool`): `True` if the pixel was natively observed.
*   **Coordinate Systems**:
    *   `time`: `datetime64[ns]`, monthly cadence.
    *   `y` / `x`: `float64`, projected coordinate system (EPSG:3577, metres).

---

## 6. Explicit Mismatches Between Current Code and v1.2 Spec

1.  **Upstream Input Parsing (Structure and Naming)**:
    *   [`coerce_water_mask_dataarray`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L507) expects a single-variable dataset or a variable named `"water"`. It raises an `AssertionError` if a multi-variable dataset is passed. WaterMask-TSFill outputs a multi-variable Zarr store containing `"water_mask"`, `"confidence"`, `"method_flag"`, and `"observed"`.
2.  **Pixel Sentinel Semantics**:
    *   Current preprocessing ([`update_nodata_in_rcor_extent`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L704)) assumes NoData is marked as `NaN` or `-1`, and uses `2` as a temporary nodata sentinel.
    *   Upstream Zarr uses `255` for invalid/unresolved data and `254` for outside AOI. Current code fails to recognize `255` and `254` as nodata values, causing them to be incorrectly processed as valid pixels, which will corrupt spatial and morphological metrics.
3.  **Missing Metric Infrastructure**:
    *   No spatial zonation logic (Zone 1/2/3/4) is implemented. Fallback logic when a drainage layer is missing is completely absent.
    *   No graph connectivity module is present (missing `RC`, `TCF`, `DCI`, `LCC` metrics).
    *   Dynamics metrics like `dry_down` rate, Jaccard-based `refuge spatial stability`, and `reconnection timing` are missing.
    *   `LPI` and `MESH` (with its fixed landscape area denominator) are not implemented.
    *   Unweighted pool width distribution (mean, median, max, cv) is missing.
    *   Inter-pool dry gap along the skeleton is missing.
4.  **Metric Computation Logic and Rebranding**:
    *   `AWRe` length method: The current code computes pool length via the skeleton's longest path for all patches ([`compute_length_single_graph`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1032)). The spec requires choosing between skeleton length and regionprops major-axis length based on whether a drainage layer skeleton is available.
    *   The current code computes static Pixel Persistence metrics over the entire time series ([`calculate_pixel_persistence_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L876)). The spec requires pixel-temporal occurrence, recurrence, and hydroperiod, and HY-based refuge spatial stability.
    *   Historically "invalid" circular-denominator metrics (`PF`, `PLF`, `AWMPA`, `AWMPL`, `AWMPW`) are still computed, but must be dropped in v1.2.
5.  **Namespace Naming**:
    *   The package and directories are named `ecofragments` (or `iRiverMetrics` in README), but must be renamed to `hydrofragments` namespace and package.

---

## 7. Current Dask Usage: Laziness and Compute Triggers

### Lazy Pipelines (Correct Dask Usage)
*   **Gap Filling**: [`fill_nodata_darray`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L792) uses `dask.array.map_overlap` to lazily apply a temporal gap-fill across chunks.
*   **Feature Operations**: [`pre_process_layer`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L911) uses `xarray.apply_ufunc(..., dask='parallelized')` to lazily map components, skeletonization, and distance transforms.
*   **Delayed Task Graph**: Public entry points create delayed lists of task batches via `dask.delayed`.

### Explicit Computes (Blocking Execution)
*   **Timestep Filtering**: [`preprocess`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L289) calls `.compute()` on `valid_time_mask` to discard empty timesteps.
*   **Pre-fill QA Thresholding**: [`update_nodata_in_rcor_extent`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L724) calls `.compute()` on `valid_time_mask` to enforce the 70% valid pixel threshold.
*   **Post-fill QA Thresholding**: [`fill_nodata_darray`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L819) calls `.compute()` on `valid_time_mask` to enforce the 95% valid pixel threshold.
*   **Nested Compute in Delayed Tasks (Dask Anti-pattern)**:
    *   Inside [`preprocess_feature`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L321) (which is decorated with `@delayed`), [`calculate_pixel_persistence_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L885-L888) calls `.values.item()` on `pp_mean` and `ra_area` data arrays. This triggers a synchronous compute call inside a delayed execution tree, causing worker thread blocking.
    *   Inside [`process_feature_batch`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L356) (which is decorated with `@delayed`), [`summarize_block`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L968-L970) calls `np.asarray()` on Dask arrays representing labeled, skeleton, and distance transform blocks. This forces immediate synchronous evaluation of the array blocks.

### Laziness Broken by Third-party Libraries
*   **GeoPandas & Shapely**: Spatial operations (loading shapefiles, clipping extents, dissolving geometries) are computed synchronously in memory using GeoPandas.
*   **Rasterio**: Rasterizing shapefile geometries is computed in memory via `rasterio.features.rasterize`.
*   **Scikit-Image & SciPy**: Component labeling (`scipy.ndimage.label`), small object removal (`skimage.morphology.remove_small_objects`), skeletonization (`skimage.morphology.skeletonize`), and distance transforms (`scipy.ndimage.distance_transform_edt`) are evaluated synchronously using CPU memory.
*   **igraph**: Longest-path graph algorithms are built by mapping coordinate sets into `igraph.Graph` structures and performing BFS walks in CPU memory ([`compute_length_single_graph`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1032)).
*   **Pandas**: The final grouping and aggregation of metrics ([`groupby().apply(process_metrics)`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/main.py#L91)) is executed entirely on a standard Pandas DataFrame in memory.

---

## 8. Current Tests and What They Prove

The existing test suite is organized into two files:

### [`test_unit_metrics.py`](file:///d:/RLH/5.6/repos/HydroFragments/tests/test_unit_metrics.py)
*   **`TestCoerceWaterMaskDataarray`**: Verifies that input datasets are successfully coerced into `DataArray` objects if a single variable or a variable named `"water"` is present, and raises an `AssertionError` if ambiguous variables exist.
*   **`TestBatchDateList`**: Proves that dates are split into correct batch sizes (with remainder handling) while preserving original temporal ordering.
*   **`TestClipData`**: Proves that spatial clipping restricts coordinate bounds to matching bounding boxes.
*   **`TestCalculatePixelPersistence`**: Verifies that pixel persistence percentages are calculated correctly for homogeneous arrays (all-wet, all-dry, half-wet).
*   **`TestProcessMetrics`**: Checks area-weighted metric calculations, division safety, and correct handling of zero area (returns NaNs/zeros) or missing section lengths.

### [`test_integration.py`](file:///d:/RLH/5.6/repos/HydroFragments/tests/test_integration.py)
*   **`test_calculate_metrics_shape`**: Confirms the execution pipeline finishes and yields rows corresponding to all dates and corridor sections.
*   **`test_calculate_metrics_columns`**: Asserts that the output contains all expected historical metric columns.
*   **`test_calculate_metrics_numeric_range`**: Verifies that key metric values are non-negative, finite, or NaN.
*   **`test_calculate_metrics_csv_written`**: Checks that the final metrics CSV is exported to disk.
*   **`test_calculate_metrics_regression`**: Runs a regression check on deterministic area-based metrics (`wet_area_km2`, `APSEC`, `pp_mean_%`), asserting that the median relative error against historical reference outputs is under 5%. **Failed in active environment due to FileNotFoundError in `conftest.py` (see Section 10).**

---

## 9. Missing Tests Required by v1.2

The following test suites must be developed to validate the v1.2 specification:

1.  **Zarr Contract Parsing Tests**: Validate that the input loading pipeline successfully parses the multi-variable WaterMask-TSFill output, handles the `water_mask` variable name, and correctly maps the `255` and `254` sentinels to invalid and outside-AOI masks.
2.  **Zonation and Fallback Tests**: Validate static 4-zone mask generation. Test the persistence-proxy fallback logic when the `drainage_layer` is missing, and verify the circularity guard (ensuring persistence metrics are not stratified by Zone 2/3/4).
3.  **New Metric Unit/Regression Tests**:
    *   **Core**: `LPI`, `dry_down` rate (recession-limb slope with `max_water` vs `median` composite discrepancy check), and `occurrence` frequency.
    *   **Secondary**: `MESH` (using fixed landscape area), `pool width distribution` (unweighted statistics), and `inter-pool gap` spacing along the skeleton.
    *   **Pixel-Temporal**: `recurrence` and `seasonality/hydroperiod` rasters.
4.  **Connectivity Module Tests**: Validate graph-building, realized connectivity (`RC`), temporal connectivity frequency (`TCF`), and `DCI` (reach-length-weighted fragment connectivity).
5.  **Validation and Constraint Guards**: Tests checking that the pipeline rejects geographic CRS datasets without projection, checks resolution mismatches, and flags `composite_sensitive` anchors.

---

## 10. Documentation and Test Drift

1.  **Package Namespace**: The codebase defines the package folder as [`ecofragments`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments) and specifies `name = "ecofragments"` in [`pyproject.toml`](file:///d:/RLH/5.6/repos/HydroFragments/pyproject.toml#L2). However, [`README.md`](file:///d:/RLH/5.6/repos/HydroFragments/README.md#L2) and [`module2.md`](file:///d:/RLH/5.6/repos/HydroFragments/docs/module2.md#L3) refer to the package as `iRiverMetrics` / `irivermetrics`. The v1.2 spec locks the final package name as `HydroFragments`.
2.  **API Signatures & Metrics Register**:
    *   [`docs/module2.md`](file:///d:/RLH/5.6/repos/HydroFragments/docs/module2.md#L110) lists metrics that are marked to be dropped (`AWMPA`, `AWMPL`, `AWMPW`, `PF`, `PLF`). It does not list any of the new v1.2 metrics (`LPI`, `MESH`, `DCI`, `RC`, `TCF`, inter-pool gap, etc.).
    *   The documented example code imports from `irivermetrics.irm_main` ([`README.md:49`](file:///d:/RLH/5.6/repos/HydroFragments/README.md#L49), [`module2.md:81`](file:///d:/RLH/5.6/repos/HydroFragments/docs/module2.md#L81)), which does not exist in the current folder structure.
3.  **Notebook Examples**: The Jupyter notebook [`examples/irm_example.ipynb`](file:///d:/RLH/5.6/repos/HydroFragments/examples/irm_example.ipynb) contains imports and execution blocks using the legacy `irivermetrics` package name and signatures.
4.  **Test Data Path Discrepancy (Bug in `conftest.py`)**:
    *   [`tests/conftest.py:34`](file:///d:/RLH/5.6/repos/HydroFragments/tests/conftest.py#L34) references `"results_ecofragments" / "metrics" / "ecof_metrics.csv"` for regression checks.
    *   The actual directory in the workspace is named [`results_iRiverMetrics`](file:///d:/RLH/5.6/repos/HydroFragments/tests/results_iRiverMetrics) and the CSV file is [`irm_metrics.csv`](file:///d:/RLH/5.6/repos/HydroFragments/tests/results_iRiverMetrics/metrics/irm_metrics.csv).
    *   This causes a `FileNotFoundError` during test collection/execution of `test_calculate_metrics_regression`.

---

## 11. File and Function References

*   **Public API Entry Point**: [`ecofragments/main.py:calculate_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/main.py#L10)
*   **Validation Functions**: [`ecofragments/utils/calc_metrics.py:validate`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L226) and [`validate_data_array_cm`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L521)
*   **Input Coercion**: [`ecofragments/utils/calc_metrics.py:coerce_water_mask_dataarray`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L507)
*   **Preprocessing Pipeline**: [`ecofragments/utils/calc_metrics.py:preprocess`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L282)
*   **Nodata Alignment & Corridor Masking**: [`ecofragments/utils/calc_metrics.py:update_nodata_in_rcor_extent`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L671)
*   **Nodata Gap-Fill (map_overlap)**: [`ecofragments/utils/calc_metrics.py:fill_nodata_darray`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L779)
*   **Section Preprocessing (delayed)**: [`ecofragments/utils/calc_metrics.py:preprocess_feature`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L320)
*   **Dask Layer Delineation**: [`ecofragments/utils/calc_metrics.py:pre_process_layer`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L911)
*   **Batch Date Processing (delayed)**: [`ecofragments/utils/calc_metrics.py:process_feature_batch`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L355)
*   **Timestep Summary Calculation**: [`ecofragments/utils/calc_metrics.py:summarize_block`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L949)
*   **Longest path BFS via igraph**: [`ecofragments/utils/calc_metrics.py:compute_length_single_graph`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1032)
*   **EDT Width Extraction**: [`ecofragments/utils/calc_metrics.py:process_edt_width`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1170)
*   **Regionprops Extraction**: [`ecofragments/utils/calc_metrics.py:compute_area_and_perimeter_df`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L1196)
*   **Metric Computations**: [`ecofragments/utils/calc_metrics.py:process_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L426)
*   **Pixel Persistence Calculations**: [`ecofragments/utils/calc_metrics.py:calculate_pixel_persistence_metrics`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L876)
*   **Shapefile Exporter (delayed)**: [`ecofragments/utils/calc_metrics.py:export_shapefiles`](file:///d:/RLH/5.6/repos/HydroFragments/ecofragments/utils/calc_metrics.py#L408)
*   **Upstream Schema Contract**: [`watermask_tsfill/contracts.py:ZarrSchema`](file:///D:/RLH/5.6/repos/WaterMask-TSFill/watermask_tsfill/contracts.py#L46)
