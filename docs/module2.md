# calculate_metrics

**calculate_metrics** (_**da_wmask**, **rcor_extent**=None, **outdir**=None, **section_length**=None, **section_name_col**=None, **min_pool_size**=2, **img_ext**='.tif', **export_shp**=False, **export_PP**=False, **fill_nodata**=True_)

## Overview

**calculate_metrics** is the second module of the iRiverMetrics toolkit, designed to compute a range of ecohydrological metrics from binary water masks (generated or not by module 1 - [waterdetect_batch](module1.md)). These metrics representt various aspects of surface water dynamics in intermittent rivers, such as morphology, persistence, and fragmentation. For a deeper understanding of the metrics and methodologies, refer to the [original paper](https://doi.org/10.1016/j.jhydrol.2023.129087). For an application example, see [this paper](https://doi.org/10.1016/j.jhydrol.2023.130266).

## Usage Guide
### Setup
Here's an example of how to use this module to calculate surface water metrics:

1. **Parameters:**

- da_wmask : str or xarray.DataArray
    Directory path or xarray.DataArray containing binary water masks.
    **Note:** Ensure all images have consistent CRS and spatial resolutions, and names include dates in "yyyy-mm-dd" or "yyyy_mm_dd" format.

- rcor_extent : str or geopandas.GeoDataFrame, default = None
    Path to the river corridor extent (river sections) shapefile (.shp) defining the Area of Interest (AOI). Defaults to None - in this case the boundaries of the water mask will be used as the AOI.
    **Note**: If rcor_extent is None, outdir must be provided. 

- outdir : str, optional, default = None
    Destination directory for results. Defaults to a directory adjacent to the rcor_extent file if not specified.

- section_length : float, optional, default = None
    Length of river sections for metrics calculation in kilometers. If None, 

- section_name_col : str, optional, default = None
    Name of a column in `rcor_extent` to use as the section identifier in output. If None, the GeoDataFrame row index is used.

- min_pool_size: int, optional, default=2
    Minimum size of detectable water pools, specified in pixels. Defaults to 2 pixels.


- img_ext : str, optional, default = '.tif'
    File extension of the water mask images. Not required if using DataArrays.

- export_shp : bool, optional, default = False
    Whether to export detailed shapefiles of the analysed river sections. Shapefiles with wetted length, start/end and mipoints will be exported for each time step.

- export_PP : bool, optional, default = False
    Whether to export a pixel persistence GeoTIFF raster for the AOI.

- fill_nodata : bool, optional, default = True
    Whether to fill missing (no-data / cloud-obscured) observations using temporal interpolation (±2 timesteps). If False, missing pixels are treated as non-water.

2. **How it works:**
Run the module to perform the following tasks:

- Validate the input data, ensuring compatibility and consistency.
- Preprocess the data by clipping, filling nodata values, and reprojecting to UTM for consistency.
- Calculate various river metrics, including:
    - Section area
    - Total wetted area
    - Total wetted perimeter
    - Wetted length
    - Number of pools
    - Area-weighted Mean Shape Index (AWMSI)
    - Area-weighted Elongation Ratio (AWRe)
    - Area-weighted Mean Pixel Area (AWMPA)
    - Area-weighted Mean Pool Length (AWMPL)
    - Area-weighted Mean Pool Width (AWMPW)
    - Wetted Area Percentage of Section (APSEC)
    - Wetted Length Percentage of Section (LPSEC)
    - Pool fragmentation (PF)
    - Pool longitudinal fragmentation (PLF)
    - Pixel persistence (PP)
    - Refuge area (RA)
- Save the calculated metrics for each section to CSV files.
- Merges metrics from processed polygons and saves them to a CSV file.
- Export a pixel persistence raster for the entire AOI.
- Generates shapefiles for visualization and further geographic analysis if requested.

3. **Returns:**

The module generates a series of metrics for the specified river sections and section length. Metrics include section- and AOI-level values for various metrics and a pixel persistence raster. Results are stored in organised directories within the output folder. If needed, the module can export shapefiles for further analysis or visualisation.

## Usage Example
```python
from irivermetrics.irm_main import calculate_metrics

# Define input parameters

# Path to the directory containing water masks or DataArray
da_wmask = "path/to/water_masks" 
# Path to the river corridor extent (sections) shapefile (.shp)
rcor_extent = "path/to/rcor_extent.shp"
# Section length in km
section_length = 0.484 #Adjust as needed
# Define minimum pool size in pixels
min_pool_size=2 #Adjust as needed
# Output directory to store results
outdir = "path/to/output_directory"
# Input images file extension
img_ext = ".tif"
# Export shapefiles (True or False)
export_shp = True
# Whether to return water masks as a DataArray
return_da_array=False

# Calculate river metrics
calculate_metrics(da_wmask, rcor_extent, section_length, min_pool_size, outdir, img_ext, export_shp, return_da_array)
```

## Metric definitions

All metrics are computed per section per timestep unless noted otherwise.

| Metric | Formula | Unit | Category |
|--------|---------|------|----------|
| `npools` | Count of labeled connected water bodies | count | Morphology |
| `wet_area_km2` | Σ aᵢ | km² | Morphology |
| `wet_length_km` | Σ lᵢ | km | Morphology |
| `wet_perimeter_km` | Σ pᵢ | km | Morphology |
| `section_area_km2` | Section polygon area | km² | Context |
| `section_length_km` | User-supplied drainage-line length | km | Context |
| `AWMSI` | Σ[(0.25 × pᵢ/√aᵢ) × (aᵢ/Σaᵢ)] | dimensionless | Morphology |
| `AWRe` | Σ[(2√(aᵢ/π)/lᵢ) × (aᵢ/Σaᵢ)] | dimensionless | Morphology |
| `AWMPA` | Σ(aᵢ²)/Σaᵢ | km² | Morphology |
| `AWMPL` | Σ(lᵢ × aᵢ)/Σaᵢ | km | Morphology |
| `AWMPW` | Σ(wᵢ × aᵢ)/Σaᵢ | km | Morphology |
| `APSEC` | (Σaᵢ / sa) × 100 | % | Resilience |
| `LPSEC` | (Σlᵢ / sl) × 100 — requires `section_length` | % | Resilience |
| `PF` | N / Σaᵢ | pools/km² | Fragmentation |
| `PLF` | N / Σlᵢ | pools/km | Fragmentation |
| `pp_mean_%` | Mean PP of pixels where PP > 25% (temporal, section-wide) | % | Persistence |
| `ra_area_km2` | Total area of pixels where PP > 90% (temporal, section-wide) | km² | Persistence |

**Notation:** i = pool index; aᵢ = area; pᵢ = perimeter; lᵢ = pool length (igraph BFS); wᵢ = mean width (EDT along skeleton); N = total pools; sa = section polygon area; sl = section drainage-line length (`section_length` parameter).

> **PP and RA** are computed once over the full time series, not per-timestep. They appear as constant values across all rows for a given section in the output CSV.

## QA filtering

Before computing metrics, the module applies two QA passes:

1. **Pre-fill** — timesteps where fewer than 70% of section pixels are valid are excluded.
2. **Post-fill** — after temporal nodata interpolation, timesteps where fewer than 95% of section pixels are valid are excluded.

"Valid pixel" means: not NaN, not −1 (sensor no-data), not 2 (cloud/fill sentinel), and inside the section mask.

This protocol matches the NESP Technical Report (Tayer et al., NESP) and [Tayer et al. (2026)](https://doi.org/10.1016/j.jhydrol.2025.134750).

## Pool length algorithm

Pool length is the longest geodesic path through the skeletonized water pool, computed by double-BFS on the skeleton's undirected connectivity graph (using `igraph`). This replaces the original OpenCV convolution + `MCP_Geometric` approach and delivers >95% speedup on large water bodies.

See [Architecture — Pool length](./architecture.md#pool-length--igraph-double-bfs) for details.

[Back to Main README](../README.md) · [Back to docs index](./index.md)