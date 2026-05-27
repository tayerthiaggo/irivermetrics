# Architecture

## Package layout

```
irivermetrics/
├── irivermetrics/
│   ├── __init__.py          # Public API: calculate_metrics, waterdetect_batch
│   ├── irm_main.py          # Top-level orchestration functions
│   └── utils/
│       ├── __init__.py
│       ├── wd_batch.py      # Module 1 helpers (validation, image loading, WaterDetect wrapper)
│       ├── calc_metrics.py  # Module 2 helpers (validation, preprocessing, metric computation)
│       └── WaterDetect.ini  # Default WaterDetect configuration (bundled package data)
├── tests/
│   ├── conftest.py          # Session-scoped fixtures (wmask_ts.nc, rcor_extent.shp)
│   ├── test_unit_metrics.py # Unit tests — no pipeline required
│   ├── test_integration.py  # Integration tests (marked slow)
│   ├── wmask_ts.nc          # 63-timestep water mask test NetCDF
│   └── rcor_extent.shp      # 7-section river corridor shapefile (EPSG:28351)
├── docs/
├── examples/
│   ├── irm_example.ipynb    # calculate_metrics demo
│   └── STAC_query.ipynb     # STAC/odc-stac data loading demo
└── pyproject.toml
```

## Data-flow diagram

```
Module 1: waterdetect_batch
─────────────────────────────────────────────────────────────────────────────
 Satellite images (dir / DataArray)
   │
   ▼ wd_batch.validate_inputs()       CRS/band/date checks; buffer r_lines
   │
   ▼ wd_batch.change_ini()            Select spectral indices for n_bands
   │
   ▼ Dask Client (workers = CPUs)     Scatter config; submit one task/image
   │
   ▼ wd_batch.process_image_parallel()  WaterDetect → binary mask (uint8)
   │
   ▼ wd_batch.concatenate()           xr.concat → sorted DataArray (time,y,x)
   │
   ▼ return da_wmask or write .tif

Module 2: calculate_metrics
─────────────────────────────────────────────────────────────────────────────
 da_wmask (DataArray / dir / NC)  +  rcor_extent (.shp / GeoDataFrame)
   │
   ▼ calc_metrics.validate()          Input type/dim/CRS checks; UTM reproject
   │
   ▼ calc_metrics.preprocess()
       ├─ match_input_extent()        Clip DataArray to section union bbox
       ├─ update_nodata_in_rcor_extent()
       │     Rasterize section mask → nodata→2; QA: drop timesteps <70% valid
       └─ fill_nodata_darray()        Temporal fill (±2 obs); drop <95% post-fill
   │
   ▼ for each section:
       ├─ preprocess_feature()  [@delayed]
       │     clip_data() → calculate_pixel_persistence_metrics()
       │     pre_process_layer():
       │         find_connected_components()   label/remove_small_objects
       │         skeletonize_label()           skimage.skeletonize
       │         distance_transform()          scipy.ndimage.distance_transform_edt
       │
       └─ process_feature_batch()  [@delayed, batched by date]
             summarize_block() per timestep:
                 compute_area_and_perimeter_df()  regionprops (area, perimeter_crofton)
                 compute_length_single_graph()    igraph BFS longest-path
                 process_edt_width()              EDT values along path
   │
   ▼ dask.compute(*summary_tasks)     Execute all section×batch tasks
   │
   ▼ process_metrics() per (date, section)
       Compute 16 metrics from per-pool area/length/width/perimeter
   │
   ▼ Export irm_metrics.csv  [+ optional shapefiles / PP raster]
```

## Algorithm details

### Pool length — igraph double-BFS

The skeleton of each labeled water pool is loaded as an undirected graph where:
- **Nodes** = skeleton pixels (row, col)
- **Edges** = 8-neighbour connectivity (diagonal pixels connected)

The longest path in the graph (pool length) is found via double-BFS:
1. Start BFS from the highest-degree node → find farthest node A
2. Start BFS from A → find farthest node B
3. The shortest path A→B is the longest path (exact for trees; good approximation for cyclic graphs)

This replaces the original OpenCV convolution + `MCP_Geometric` approach (steps 7–12 in [Tayer et al., 2023](https://doi.org/10.1016/j.jhydrol.2023.129087)), delivering >95% speedup for large water bodies (NESP Technical Report).

### QA thresholds

Applied inside `calc_metrics.preprocess()`:

| Stage | Threshold | Action |
|-------|-----------|--------|
| Pre-fill | <70% valid pixels in section mask | Drop timestep |
| Post-fill | <95% valid pixels in section mask | Drop timestep |

"Valid pixel" = not NaN, not −1 (no-data), not 2 (cloud sentinel), and inside the section mask.

### Pixel persistence and refuge area

- **PP (Pixel Persistence)** — for each pixel: `(wet observations / total observations) × 100`
- **PP mean** — mean PP of pixels with PP > 25% (soil-contaminated pixels discarded)
- **RA (Refuge Area)** — total area of pixels where PP > 90%

These are computed once over the full time series, not per-timestep.

### Metric formulas

| Metric | Formula | Unit |
|--------|---------|------|
| AWMSI | `Σ[(0.25 × pᵢ/√aᵢ) × (aᵢ/Σaᵢ)]` | dimensionless |
| AWRe | `Σ[(2√(aᵢ/π)/lᵢ) × (aᵢ/Σaᵢ)]` | dimensionless |
| AWMPA | `Σ(aᵢ²)/Σaᵢ` | km² |
| AWMPL | `Σ(lᵢ × aᵢ)/Σaᵢ` | km |
| AWMPW | `Σ(wᵢ × aᵢ)/Σaᵢ` | km |
| APSEC | `(Σaᵢ / sa) × 100` | % |
| LPSEC | `(Σlᵢ / sl) × 100` | % |
| PF | `N / Σaᵢ` | pools/km² |
| PLF | `N / Σlᵢ` | pools/km |
| PP | `(WP / T) × 100` per pixel | % |
| RA | `Σ(pixels where PP > 90%) × pixel_area` | km² |

**Notation:** i = pool index; pᵢ = perimeter; aᵢ = area; lᵢ = length; wᵢ = mean width; N = total pools; WP = wet observations; T = total observations; sa = section area; sl = section drainage length

## Design decisions

| Decision | Rationale |
|----------|-----------|
| `igraph` BFS for pool length | 95%+ faster than MCP_Geometric; no OpenCV dependency |
| Dask lazy eval + `@delayed` batches | Avoids loading full 4D array into RAM; scales to multi-year WOfS time series |
| Relative imports inside package | Allows `pip install -e .` without path manipulation |
| UTM reproject at validate step | Ensures metre-based area/distance calculations regardless of input CRS |
| QA before and after fill | Matches NESP report protocol; prevents false confidence from heavily filled timesteps |
| 8-neighbour connectivity | Matches original paper; prevents splitting pools at diagonal junctions |

## Key dependencies

| Library | Role |
|---------|------|
| `xarray` / `rioxarray` | Labelled array operations with CRS |
| `dask` | Lazy parallel computation |
| `igraph` | BFS longest-path for pool length |
| `scikit-image` | Connected components, skeletonization |
| `scipy.ndimage` | Distance transform, connected component labelling |
| `geopandas` / `shapely` | Vector geometry operations |
| `rasterio` | Rasterization, polygon extraction |
| `waterdetect` | Spectral clustering water detection (Module 1) |
| `odc-stac` / `pystac-client` | STAC cloud data loading |

[Back to docs index](./index.md)
