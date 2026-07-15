# HydroFragments Dask & CUDA Scalability Audit

## 1. Current Processing Graph Summary
*   **Input**: `xarray.DataArray` / `Dataset`.
*   **Pre-process**: Drop empty timesteps. Fill nodata lazily (`dask.array.map_overlap`). Drop post-fill empty timesteps.
*   **Task Gen**: `dask.delayed` loop over corridor sections.
*   **Section-level**: Mask clip, pixel persistence calc. Map morphology ops (`xarray.apply_ufunc(..., dask='parallelized')`).
*   **Batch-level**: `dask.delayed` batches of timesteps for metric extraction.
*   **Execute**: `dask.compute(*summary_tasks)`.
*   **Aggregate**: Pandas `groupby().apply()` in memory.

## 2. Eager/CPU Choke Points

| Module/Op | Choke Point | Reason |
| :--- | :--- | :--- |
| `preprocess` | `valid_time_mask.compute()` | Blocks pipeline building. Forces full array scan before delayed task graph complete. |
| `update_nodata_in_rcor_extent` | `valid_time_mask.compute()` | Sync execution blocks worker. |
| `fill_nodata_darray` | `valid_time_mask.compute()` | Sync execution blocks worker. |
| `calculate_pixel_persistence_metrics` | `.values.item()` inside `@delayed` task | Triggers nested compute. Blocks worker thread. Dask anti-pattern. |
| `summarize_block` | `np.asarray()` inside `@delayed` task | Forces eager eval of chunk. Defeats lazy graph. |
| GeoPandas/Shapely | Extent clip, spatial ops | CPU single-thread. |
| Rasterio | `rasterio.features.rasterize` | CPU single-thread. |
| Scikit-Image | `label`, `remove_small_objects`, `skeletonize` | CPU bound. Requires NumPy array conversion. |
| SciPy | `distance_transform_edt` | CPU bound. |
| igraph | BFS longest path | CPU bound. Requires building graph object per patch. |
| Pandas | Final `groupby` aggregation | Sync memory limit. |

## 3. Dask Risks
*   **Nested Compute**: `@delayed` functions calling `.compute()` or `.values` freeze workers. Risk deadlocks.
*   **Memory Bloat**: `np.asarray()` in delayed tasks forces chunks into RAM simultaneously. High risk OOM for long time series.
*   **Task Granularity**: Batching timesteps inside `@delayed` loops creates opaque mega-tasks. Dask scheduler cannot optimize inner graph.
*   **Graph Size**: If batch size small, delayed loop creates too many tasks. Scheduler overhead dominates.

## 4. CUDA-Ready Design Proposal
Goal: GPU path where beneficial, zero hard GPU dependency.

*   **Array Backend**: Use array API standard (`__array_namespace__`) or `cupy.asarray` via feature flag. Keep `xarray` wrapping Dask+CuPy.
*   **Morphology (scikit-image/SciPy)**: Replace with `cucim.skimage.morphology` and `cupyx.scipy.ndimage`.
    *   *Supported*: `label`, `remove_small_objects`, `skeletonize` (if implemented in cuCIM, else custom kernel/CuPy).
    *   *EDT*: `cupyx.scipy.ndimage.distance_transform_edt`.
*   **Graph (igraph)**: Replace with `cugraph` for BFS/longest path. Requires CSR format conversion.
*   **Vector (GeoPandas)**: Vector clip/rasterize runs once per section. Keep CPU. Move to GPU only if section count extreme (`cuspatial`).
*   **Final Aggregation**: Use `dask.dataframe` or `cudf` instead of Pandas.

## 5. CPU Fallback Design
*   **Dynamic Dispatch**: `get_array_module(x)` returns `numpy` or `cupy`.
*   **Feature Flag**: `USE_CUDA = bool(cupy_available)`.
*   **Module Aliasing**:
    ```python
    if USE_CUDA:
        import cupy as xp
        from cucim.skimage import morphology
        import cupyx.scipy.ndimage as ndimage
    else:
        import numpy as xp
        from skimage import morphology
        import scipy.ndimage as ndimage
    ```
*   **Graphs**: Fallback to `scipy.sparse.csgraph` or NetworkX/igraph if `cugraph` missing.

## 6. Benchmark Plan
*   **Datasets**:
    *   *Small*: 1 section, 12 timesteps, 1000x1000 pixels.
    *   *Medium*: 5 sections, 120 timesteps, 5000x5000 pixels.
    *   *Large*: 20 sections, 360 timesteps (30 years), 10000x10000 pixels (Stress test).
*   **Environments**:
    *   CPU-only: 8 cores, 16GB RAM.
    *   CPU-Heavy: 32 cores, 128GB RAM.
    *   GPU: 1x NVIDIA L4 or T4, 16GB VRAM.
*   **Metrics**:
    *   Task graph generation time.
    *   Peak RAM/VRAM usage.
    *   Total execution time.
    *   Throughput (pixels/second).
*   **Expected**: GPU crushes Medium/Large on morphology ops. CPU matches GPU on Small due to VRAM transfer overhead.

## 7. Implementation Sequence
1.  **Purge Nested Computes**: Remove `.values.item()` and `np.asarray()` inside `@delayed` tasks. Let Dask manage execution.
2.  **Fix Eager Filtering**: Rework `preprocess` thresholds to use lazy boolean indexing or compute only the 1D time mask, not full array chunks.
3.  **Array Dispatch**: Abstract `numpy`/`scipy`/`skimage` imports behind array namespace alias.
4.  **CuPy Integration**: Implement CuPy path for basic array math and masks.
5.  **cuCIM/cuGraph Integration**: Add GPU path for morphology, EDT, and graph longest-path.
6.  **Benchmark**: Run plan, tune Dask chunk sizes to fit VRAM.
