# EcoFragments - Architecture

## Vision

EcoFragments is a Python toolkit for quantifying **spatial patch dynamics** from binary
classification time series. Originally developed for intermittent river surface-water analysis,
the generalised architecture supports any domain where patches change shape, fragment, merge,
or persist over time, including:

- **Aquatic**: river pools, waterholes, floodplain wetlands
- **Terrestrial**: vegetation patches, fire scars, cropping extents
- **Urban**: impervious surface expansion, greenspace fragmentation

---

## Package Structure

```
ecofragments/
|-- __init__.py          # Public API -- exposes calculate_metrics
|-- main.py              # Entry-point functions
+-- utils/
    |-- __init__.py
    +-- calc_metrics.py  # Spatial aggregation engine (16 metrics)

examples/
|-- irm_example.ipynb    # calculate_metrics usage walkthrough
+-- STAC_query.ipynb     # Loading cloud data (DEA / Planetary Computer)

tests/
|-- conftest.py          # Shared fixtures (water mask, corridor shapefile)
|-- test_unit_metrics.py # 16 unit tests (fast)
|-- test_integration.py  # 4 integration tests (marked @slow)
|-- wmask_ts.nc          # 63-timestep test water mask
+-- rcor_extent.shp      # 7-section test corridor

docs/
|-- architecture.md      # This document
|-- index.md             # Quick start
|-- module2.md           # calculate_metrics reference
|-- paper-summary.md     # Scientific foundations
+-- project-overview.md  # Context and validated study areas
```

---

## Current Module: `ecofragments.spatial` (`calculate_metrics`)

### Role

Compute 16 section-level patch metrics per timestep from any binary classification mask.

### Inputs

| Argument | Type | Description |
|---|---|---|
| `da_wmask` | `xr.DataArray` or `str` (directory) | Binary mask time series (1=patch, 0=background) |
| `rcor_extent` | `gpd.GeoDataFrame` or `str` (.shp) | Polygon boundaries defining analysis regions/sections |
| `outdir` | `str` | Output directory |
| `section_length` | `float` | Optional: total drainage line length per section (km) |
| `min_patch_size` | `int` | Minimum patch size in pixels (default 2) |
| `fill_nodata` | `bool` | Fill cloud/shadow gaps via temporal interpolation |

### Outputs

`ecof_metrics.csv` -- one row per `(date, section)` with 16 metrics:

| Metric | Formula | Unit | Category |
|---|---|---|---|
| `n_patches` | count of labeled connected components | -- | Summary |
| `wet_area_km2` | sum(a_i) | km2 | Summary |
| `wet_length_km` | sum(l_i) | km | Summary |
| `wet_perimeter_km` | sum(p_i) | km | Summary |
| `AWMSI` | sum[(0.25*p_i/sqrt(a_i))*(a_i/sum(a_i))] | -- | Morphology |
| `AWRe` | sum[(2*sqrt(a_i/pi)/l_i)*(a_i/sum(a_i))] | -- | Morphology |
| `AWMPA` | sum(a_i^2)/sum(a_i) | km2 | Morphology |
| `AWMPL` | sum(l_i*a_i)/sum(a_i) | km | Morphology |
| `AWMPW` | sum(w_i*a_i)/sum(a_i) | km | Morphology |
| `PF` | N/sum(a_i) | patches/km2 | Fragmentation |
| `PLF` | N/sum(l_i) | patches/km | Fragmentation |
| `APSEC` | (sum(a_i)/sa)*100 | % | Section coverage |
| `LPSEC` | (sum(l_i)/sl)*100 | % | Section coverage |
| `pp_mean_%` | temporal mean pixel persistence | % | Persistence |
| `ra_area_km2` | area where PP > 90% | km2 | Persistence |

Shapefiles `ecof_Polygons.shp`, `ecof_Lines.shp`, `ecof_Points.shp` are also written
when `export_shp=True`.

### Algorithm Overview

```
Input: da_wmask (time, y, x) + rcor_extent (polygon sections)
  |
validate()
  |-- type / dimension / CRS checks
  +-- reproject to UTM (ensures metre-based area and distance)
  |
preprocess()
  |-- match spatial extents
  |-- QA pre-fill:  drop timesteps with < 70% valid pixels
  |-- temporal gap-fill: +/-2 timestep interpolation
  +-- QA post-fill: drop timesteps with < 95% valid after fill
  |
For each section  [parallel via Dask @delayed]:
  preprocess_feature()
    |-- clip data to section geometry
    |-- calculate_pixel_persistence()  -> PP per pixel (temporal %)
    |-- refuge_area()                  -> pixels where PP > 90%
    +-- pre_process_layer()
          |-- find_connected_components() -> label patches (8-neighbour)
          |-- skeletonize_label()         -> 1-pixel centreline per patch
          +-- distance_transform()        -> EDT from patch boundary (for width)
  |
  process_feature_batch()  [batched by date, @delayed]
    For each timestep:
      summarize_block()
        |-- compute_area_and_perimeter_df()  -> regionprops
        +-- compute_length_single_graph()
               igraph BFS double-BFS longest path
  |
process_metrics()  -> aggregate 16 metrics per (date, section)
  |
Output: ecof_metrics.csv + optional shapefiles
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Dask `@delayed` with date batching | Handles 500+ timestep datasets without RAM overflow |
| igraph BFS double-BFS for patch length | 95% faster than MCP_Geometric; O(V+E) vs O(V log V) |
| 8-neighbour connectivity | Matches peer-reviewed methodology; prevents diagonal mis-splits |
| UTM reprojection | Ensures metre-based area and distance |
| 70% / 95% QA thresholds | Pre-fill guards against heavily clouded timesteps; post-fill validates gap-fill quality |

### Patch Length Algorithm (`compute_length_single_graph`)

```
1. Skeletonize the labeled patch -> 1-pixel-wide centreline
2. Build undirected graph from skeleton pixels:
     - Each pixel  = node
     - 8-connected neighbours = edges (weight = Euclidean distance in km)
3. Double-BFS to find longest path:
     - BFS from an arbitrary node -> find furthest node u
     - BFS from u                 -> find furthest node v
     - Path u -> v is the patch length (longest geodesic path in skeleton)
```

---

## Planned Modules (Roadmap)

### `ecofragments.tracking`  (NESP 5.6 Activity 3 -- Milestone 5, May 2026)

**Role**: Track individual patches across timesteps to identify persistent units and detect
birth / death / split / merge events.

**Algorithm (draft)**:

```
Input: da_wmask (time, y, x) + optional rcor_extent
  |
For each timestep t:
  Label connected components -> patch set P_t
  |
For each consecutive pair (t, t+1):
  Compute spatial overlap matrix between P_t and P_(t+1)
  Build directed graph:
    node = (patch_id, timestep)
    edge = spatial overlap  (weight = IoU)
  |
Classify edges:
  - Survival:  1 patch -> 1 patch (dominant overlap)
  - Split:     1 patch -> N patches
  - Merge:     N patches -> 1 patch
  - Birth:     no predecessor
  - Death:     no successor
  |
Extract patch trajectories:
  - lifespan (timesteps), centroid movement (m/timestep)
  - area and shape change over time
  |
Outputs:
  - patch_trajectories.csv  -- one row per (patch_id, timestep)
  - patch_events.csv        -- lifecycle event log
  - persistent_units.shp    -- patches with lifespan > threshold
```

**Key dependencies**: `igraph` (already in stack), `scipy.ndimage`, `geopandas`

---

### `ecofragments.resilience`  (NESP 5.6 Activity 4 -- Milestone 6, Oct 2026)

**Role**: Correlate patch dynamics (from `.tracking`) with external drivers (climate, discharge)
and predict vulnerability under future scenarios.

**Inputs**: tracking outputs + external time-series (rainfall, temperature, discharge)

**Outputs**:
- Correlation matrices (patch persistence vs. climate covariates)
- Vulnerability maps (shapefile with persistence thresholds per patch unit)
- Scenario projections (spatial datasets per climate scenario)

---

### `ecofragments.viz`  (target: Q3 2026)

**Role**: Pre-built visualisation of metric outputs, patch trajectories, and persistence rasters.

**Key plots**:
- Temporal metric time series per section
- Pixel persistence raster map (heatmap overlay)
- Patch trajectory animation (GIF / interactive)
- Section comparison heatmap (metric x section x time)

**Key dependencies**: `matplotlib`, `contextily`, optionally `folium`

---

## Data Flow

```
[Binary mask time series]        [Region boundaries]
  xr.DataArray (time, y, x)        gpd.GeoDataFrame
           |                              |
           +--------------+--------------+
                          |
              ecofragments.spatial
               (calculate_metrics)
                          |
                 ecof_metrics.csv
                per (section, date)
                          |
           ecofragments.tracking          <- PLANNED
            (patch trajectories)
                          |
           ecofragments.resilience        <- PLANNED
           (climate correlations)
                          |
               ecofragments.viz          <- PLANNED
```

---

## Dependency Stack

| Library | Role | Used in |
|---|---|---|
| `xarray`, `rioxarray` | Labelled raster arrays with CRS | All |
| `dask`, `dask-image`, `dask-regionprops` | Lazy parallel computation | `spatial` |
| `igraph` | Graph BFS patch length + future tracking | `spatial`, `tracking` |
| `scikit-image` | Connected components, skeletonization | `spatial` |
| `scipy.ndimage` | Distance transform, labelling | `spatial` |
| `geopandas`, `shapely` | Vector geometry | All |
| `rasterio` | Rasterisation, polygon extraction | `spatial` |
| `odc-geo`, `odc-stac` | CRS utilities, cloud STAC loading | Data loading |
| `pandas`, `numpy` | Data manipulation | All |
| `matplotlib`, `contextily` | Visualisation | `viz` (planned) |
