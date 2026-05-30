# irivermetrics — Scientific Paper Summaries
Generated: 2026-05-27 | Phase 0 of refactor plan
Source PDFs: `D:\RLH\5.6\references\`

---

## Paper 1 — Water Detect Accuracy (Module 1 foundation)

**Title:** Improving the accuracy of the Water Detect algorithm using Sentinel-2, Planetscope and sharpened imagery: a case study in an intermittent river
**Authors:** Tayer et al. (2023) | **Journal:** GIScience & Remote Sensing 60(1) | **DOI:** 10.1080/15481603.2023.2168676 | CC-BY

### Key findings relevant to irivermetrics
- Water Detect requires **local parameter calibration** — accuracy can vary drastically if `max_cluster` and `reg` are not tuned to local conditions
- **`max_cluster` dominates accuracy** more than `regularization`
- Best results with **Visible + NIR (VNIR) bands** — SWIR adds little unless vegetation/shadow is an issue
- Image sharpening (Planetscope-Sentinel fusion) only helps in burnt or heavily shaded areas — not worth the compute cost in most cases
- Developed automated **sensitivity analysis** to find optimal parameters by testing all combinations in a range

### Algorithm — `waterdetect_batch` parameters
- Recommended defaults for Fitzroy River (WA): `max_cluster=6`, `reg=0.07`
- These parameters feed directly into the current `waterdetect_batch()` function as `max_cluster` and `reg` arguments
- Sensitivity analysis code should ideally be separable / optional as a calibration step

### Relevance to refactor
- `waterdetect_batch` should expose `max_cluster` and `reg` prominently; document that calibration is critical
- Consider adding a `sensitivity_analysis` helper or at minimum documentation referencing this paper

---

## Paper 2 — Ecohydrological Metrics (Core irivermetrics paper)

**Title:** Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river
**Authors:** Tayer et al. (2023) | **Journal:** J. Hydrology 617, 129087 | **DOI:** 10.1016/j.jhydrol.2023.129087 | CC-BY
**Study area:** Fitzroy River, Kimberley WA | **Imagery:** Sentinel-2 NBART (10 m, 5-day return), 2017–2021 | **5 river sections** with persistence gradient (1=low to 5=high)

### Algorithm workflow (16-step attribute extraction)
1. Water detection via Water Detect → binary water mask per image
2. Fill missing values (no-data/cloud): backward fill up to 2 obs, then forward fill up to 2 obs
3. Label connected regions (8-neighbour connectivity, Scikit-image `label`)
4. Extract **pool area**, **pool perimeter**, **number of pools** (via Rasterio's `shapes`)
5. **Skeletonize** water mask → 1-pixel wide river centreline (Scikit-image `skeletonize`)
6. Label skeletonized connected segments (unique IDs per pool centreline)
7. Apply 3×3 convolution kernel to skeletonized image (OpenCV `filter2D`):
   ```
   k = [[1,1,1],[1,10,1],[1,1,1]]
   ```
   → pixel values 11 or 13 = candidate start/endpoints (1 or 3 neighbours)
8. Identify candidate start/end points (pixels with values 11 or 13 after convolution)
9. Calculate Euclidean distances between all combinations of start/endpoints
10. Select 5 most distant pairs
11. Find least-cost path using Scikit-image `MCP_Geometric` along centreline → **pool length**
12. Select most distant path as final pool length segment
13. Apply SciPy `distance_transform_edt` to water mask → distance-to-border array
14. Intersect centreline with distance array → half-width values
15. Multiply by 2 → **mean pool width** per centreline pixel
16. Average centreline pixel values per pool → final **mean pool width**

> **Note (NESP refactor):** Steps 7–12 (convolution + MCP path) are the main bottleneck. Replaced with graph theory using `igraph` BFS (breadth-first search) for finding longest path in skeletonized graph. Much faster and more reliable.

### Metrics — full definitions

**Category: Morphology (5 metrics)**

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| AWMSI | `Σ[(0.25 × pᵢ/√aᵢ) × (aᵢ/Σaᵢ)]` | dimensionless | Area-weighted Mean Shape Index. Higher = more complex/irregular perimeter |
| AWRe | `Σ[(2√(aᵢ/π)/lᵢ) × (aᵢ/Σaᵢ)]` | dimensionless | Area-weighted Elongation Ratio. 0–1; higher = more circular, lower = more elongated |
| AWMPA | `Σ(aᵢ × aᵢ)/Σaᵢ` | km² | Area-weighted Mean Pool Area |
| AWMPL | `Σ(lᵢ × aᵢ)/Σaᵢ` | km | Area-weighted Mean Pool Length |
| AWMPW | `Σ(wᵢ × aᵢ)/Σaᵢ` | km | Area-weighted Mean Pool Width |

**Category: Resilience (6 metrics)**

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| PP (Pixel Persistence) | `(WP/T) × 100` | % | % of timesteps a pixel is water. Temporal reduction — applies to full time series, not per-timestep |
| RA (Refuge Area) | `Σ(pixels where PP>90%) × pixel_area` | km² | Total area of most persistent pixels (PP threshold = 90%) |
| APSEC | `(Σaᵢ/sa) × 100` | % | Wetted Area as % of section area; → 0 as habitat contracts |
| LPSEC | `(Σlᵢ/sl) × 100` | % | Wetted Length as % of section drainage line; can exceed 100% (braided channels/floodplain) |
| PF | `N/Σaᵢ` | pools/km² | Pool Fragmentation = pools per unit wetted area; higher = more isolated pools |
| PLF | `N/Σlᵢ` | pools/km | Pool Longitudinal Fragmentation = pools per unit wetted length |

**Notation:** i = pool index; pᵢ = perimeter; aᵢ = area; lᵢ = length; wᵢ = mean width; N = total pools in section; WP = water observations for pixel; T = total observations; sa = section total area; sl = section drainage line length

> **Total: 11 metrics in original paper** (5 morphology + 6 resilience). The NESP report lists 16 metrics — later additions include: Section area, Total wetted area, Total wetted perimeter, Wetted length, Number of pools (as standalone outputs, not just inputs to weighted metrics).

### Key implementation notes
- PP and RA are **temporal reductions** — cannot be split by timestep
- PP is filtered: values < 25% are discarded when computing section mean (avoids soil bias)
- **8-neighbour connectivity** (diagonal pixels connected)
- `min_pool_size = 2 pixels` default (removes single-pixel noise)
- All spatial data reprojected to **UTM** before processing
- Composite strategy: max pixel value within ~15-day composites (resolves cloud conflict)

### Ecological interpretation
- Wet season: pool size ↑, AWMSI ↑, AWRe ↓ (more elongated), PF ↓ (connected)
- Dry season: size ↓, AWRe ↑ (more circular), PF ↑, PLF ↑ (fragmented)
- Pools become **narrower before shorter** as they dry
- Sections with high groundwater input: smoother curves, high PP, high RA

---

## Paper 3 — Hydrological Clustering (Downstream use)

**Title:** Identifying intermittent river sections with similar hydrology using remotely sensed metrics
**Authors:** Tayer et al. (2023) | **Journal:** J. Hydrology 626, 130266 | **DOI:** 10.1016/j.jhydrol.2023.130266 | CC-BY
**Study area:** 400 km reach of Fitzroy River, WA (vs 5 sections in Paper 2) | **Methodology:** irivermetrics metrics → multidimensional agglomerative clustering

### Key contributions
- Scale-up validation: metrics work at 400 km reach level with many sections
- Used all 11 irivermetrics metrics as input to **agglomerative hierarchical clustering** → 4 hydrological zone types:
  1. Highly intermittent (very low persistence, highly fragmented)
  2. Intermittent (low persistence, fragmented)
  3. Intermediate persistence (refuge pools, moderate fragmentation)
  4. Highly persistent (continuous water, low fragmentation — likely groundwater discharge zones)
- Zones match independently-derived groundwater discharge maps (environmental tracer surveys) — validates metric reliability
- Zones occur at **pool-run/riffle geomorphic unit scale** and alternate spatially along river

### Relevance to refactor
- The metric CSV output from `calculate_metrics` is the primary input — format and naming must be stable
- Section boundary sensitivity is a known issue: boundary placement can split/merge pools artificially → documented limitation
- Recommends **consistent section definition** approach aligned with geomorphic units

---

## Paper 4 — Mapping Resilience Framework (Most recent, defines new scope)

**Title:** Mapping resilience: A framework for analysing surface-water dynamics and persistent pools in non-perennial rivers using remote sensing, rainfall and river discharge data
**Authors:** Tayer et al. (2026) | **Journal:** J. Hydrology 666, 134750 | **DOI:** 10.1016/j.jhydrol.2025.134750 | CC-BY
**Study area:** 100 km Gilbert River, QLD, 1986–2023 | **Data:** WOfS Landsat Collection 3 (30 m) via STAC/odc_stac

### The 4-step framework (irivermetrics is Step 3 tool)
1. **Define scope** — research question determines analytical scale (pixel / pool-unit / section)
2. **Data collection** — remote sensing (WOfS, Sentinel, Planetscope, SAR), rainfall, discharge, optional (geology, soils, land use, evapotranspiration)
3. **Data processing** — using iRiverMetrics:
   - Spatial harmonisation → UTM, clip to AOI
   - Generate water extent metrics (irivermetrics output)
   - QA/data imputation for missing timesteps
   - **Dynamic hydrological year definition** (k-means + percentile approach — see below)
4. **Data analysis** — trend analysis, correlation with climate drivers, spatial clustering

### Dynamic hydrological year algorithm (new in this paper)
1. k-means (k=2) on monthly cumulative rainfall + zero-rain-day means → classify months as wet/dry → baseline hydrological year start = first wet month after dry month
2. Apply 20th-percentile rainfall threshold to 3-month rolling mean → identify primary wet season
3. Refine transitions using 10th-percentile raw rainfall
4. Wet season onset = start of each dynamic hydrological year; fallback for drought years

### Three analytical scales defined
- **Pixel scale:** PP raster — static persistence map over full time series
- **Section scale:** irivermetrics CSV metrics aggregated within fixed spatial units
- **Pool-unit scale:** track individual pools through time using spatial overlap between years

### Key findings (Gilbert River case study)
- 29 persistent pools identified; **none permanent** over 38 years
- Long-term trends: significant increase in rainfall + discharge → more pools, larger, less fragmented
- Strong correlation: wet-season rainfall and zero-flow duration predict pool morphology
- Off-channel pools larger and more persistent than in-channel pools (hydrogeomorphic control)

### Data inputs used
- WOfS Landsat Collection 3, 1986–2023, via `pystac_client` + `odc_stac`
- Minimum 40% valid pixels per scene
- Contiguity-filtered, pixel values: -1 = nodata, 0 = other, 1 = water
- 30 m spatial resolution

### Relevance to refactor
- irivermetrics needs to accept Dask DataArrays directly from `odc_stac` queries — confirmed requirement
- Pool-unit tracking is a **new capability not yet in current irivermetrics** — post-refactor extension
- Dynamic hydrological year tool is separate utility, not currently in irivermetrics — candidate for `utils/`
- Metric naming/outputs must be stable (this paper cites them directly)

---

## Paper 5 — Refactoring Report (Direct specification)

**Title:** Refactoring the iRiverMetrics algorithm
**Author:** Tayer, T.C. (2025) | **Publisher:** UWA / NESP Resilient Landscapes Hub | CC-BY 4.0
**Report for:** NESP Project 2.1, Task 15 (Phase 3, Milestone 6)
**Study area:** 100 km Gilbert River, QLD | **Dataset:** WOfS Landsat 1986–2024 (505 valid timesteps)

### Performance benchmarks (original → refactored)
| Config | Original | Refactored | Improvement |
|--------|---------|-----------|-------------|
| 1 section (100 km) | 65 min 7 s | 3 min 11 s | **95% faster** |
| 50 sections (~2 km each) | 12 min 1 s | 2 min 54 s | **77% faster** |
| Peak RAM (1 section) | ~31 GB | ~22 GB | **28% less** |
| Peak RAM (50 sections) | 21 GB | 40 GB | higher (parallel overhead, acceptable) |

### Changes implemented in refactor

**1. Dask integration**
- Original: loaded entire raster into RAM (memory crashes on large datasets)
- Refactored: Dask lazy evaluation + parallel processing with `delayed` functions
- Batch processing: group multiple timesteps into single tasks to reduce scheduling overhead
- Vectorised operations (xarray, NumPy, Pandas) replace loop-based approaches

**2. Pool length calculation (major bottleneck)**
- Original: convolution kernel (3×3) + OpenCV `filter2D` → Euclidean distances → `MCP_Geometric` least-cost path → select most distant of top-5 pairs
- Refactored: **`igraph` graph theory** — skeletonized pixels as nodes → BFS to find longest path in each pool segment
- Result: simpler logic, faster, more reliable, less error-prone

**3. Enhanced QA / pre-processing**
- Exclude timesteps with **< 70% valid pixels** (before fill)
- Fill no-data: forward fill → backward fill (up to 2 observations each direction)
- After filling, re-exclude timesteps with **< 95% valid pixels**
- Exclude timesteps with **100% no-data** entirely
- Rationale: high no-data proportions falsely break pool connectivity and inflate PF/PLF

**4. Expanded input flexibility**
- Original: accepted only `xarray.DataArray`
- Refactored: folder paths (`.tif` files), `xarray.DataArray`, Dask `DataArray`, Dask `Dataset`
- Enables direct integration with `odc_stac` cloud data queries

### Key challenges & solutions (Table 2)

| Challenge | Original | Refactored Solution |
|-----------|---------|-------------------|
| Scalability with large datasets | Load all to RAM → crashes | Dask lazy eval + chunked processing |
| Many small tasks overhead | Serial for loops | Batch timesteps into larger tasks; `delayed` functions |
| Pool length computation | Convolution + MCP path | `igraph` BFS on skeletonized graph |
| Limited input flexibility | DataArray only | Folder, DataArray, Dask DataArray/Dataset |

### Inputs (full parameter list)
- `da_wmask`: binary water mask time series (DataArray, Dask, or directory path)
- `rcor_extent`: shapefile or GeoDataFrame of river sections / AOI (optional; defaults to data bounds)
- `outdir`: output directory (required if rcor_extent=None)
- `section_length`: km length for automatic section definition (optional)
- `section_name_col`: column in rcor_extent with section names
- `min_pool_size`: minimum pool size in pixels (default=2)
- `img_ext`: file extension (default='.tif')
- `export_shp`: export pool shapefiles (polygon, polyline, points) (default=False)
- `export_PP`: export pixel persistence raster (default=False)
- `fill_nodata`: fill no-data values (default=True)

### All 16 metrics (Table 1 of NESP report — expanded from 11 in Paper 2)
**Morphology:** Section area, Total wetted area, Total wetted perimeter, Wetted length, AWMSI, AWRe, AWMPA, AWMPL, AWMPW, Number of pools, APSEC, LPSEC
**Resilience:** PF, PLF, PP (Pixel Persistence), RA (Refuge Area)

---

## Cross-cutting notes for refactor

### What the refactor must preserve
- All 16 metric names, formulas, and units exactly as specified in Paper 2 / NESP report
- 8-neighbour connectivity for pool labelling
- PP threshold = 90% for RA; PP filtered at <25% for section mean
- Output: CSV with time series of all metrics per section + optional pixel persistence raster + optional shapefiles (polygon/polyline/points)

### What the refactor must change (confirmed by NESP report)
- Replace convolution kernel + MCP_Geometric with `igraph` BFS for pool length
- Add Dask lazy loading throughout `calc_metrics.py`
- Add QA thresholds (70% and 95% valid pixel filters)
- Accept Dask DataArray/Dataset inputs (for WOfS/STAC workflow)
- Batch timestep processing to reduce Dask task overhead

### What is new scope (post-refactor extensions — not v1)
- Pool-unit temporal tracking (spatial overlap across years) — Paper 4
- Dynamic hydrological year algorithm — Paper 4
- Sensitivity analysis for Water Detect parameters — Paper 1

### Package structure implications
- Current `src/utils/calc_metrics.py` → rename to `irivermetrics/utils/calc_metrics.py`
- Current `src/irm_main.py` → `irivermetrics/irm_main.py`
- `igraph` must be added as a dependency (already present in current code)
- `pystac_client`, `odc_stac` needed for STAC/WOfS workflow
