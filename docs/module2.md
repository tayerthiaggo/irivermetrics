# calc_metrics

calc_metrics(da_wmask, rcor_extent, section_length=None, outdir=None, img_ext='.tif', export_shp=False)

## Overview

**calc_metrics** is the second module in the iRiverMetrics toolkit. This module is designed for calculating various intermittent river metrics based on binary water masks (generated or not by module 1 - [wd_batch](docs/module1.md)). It provides valuable insights into surface water morphology, persistence, resilience, fragmentation and other relevant metrics. For detailed metric and method descriptions reference the [original paper](https://doi.org/10.1016/j.jhydrol.2023.129087).

## Details

Here's an example of how to use Module Two to calculate river metrics:

1. **Input Data:**

- da_wmask (str or xarray.DataArray): Directory path or xarray.DataArray containing binary water masks.

Note:
    - Images in the directory must have a associate date in its name in the format "yyyy-mm-dd" or "yyyy_mm_dd".
    - All images in the directory must have the same coordinate reference system and spatial resolution.

- rcor_extent (str): Path to the river corridor extent (river sections) shapefile (.shp) defining the Area of Interest (AOI).

- section_length (float): Length of river sections for metrics calculation (in km).

- outdir (str, optional, default = None): Output directory for results. If None, it will be generated based on the shapefile location.

- img_ext (str, optional, default = '.tif'): The image file extension.

- export_shp (bool, optional, default = False): Flag to export shapefiles of river sections (Pool length and endpoints).

2. **How it works:**

Run the module to perform the following tasks:

- Validate the input data, ensuring compatibility and consistency.
- Preprocess the data by clipping, filling nodata values, and reprojecting to UTM for consistency.
- Calculate various river metrics, including:
    - Section area
    - Total wet area
    - Total wet perimeter
    - Wet length
    - Number of detected wet regions
    - Area-weighted Mean Shape Index (AWMSI)
    - Area-weighted Elongation Ratio (AWRe)
    - Area-weighted Mean Pixel Area (AWMPA)
    - Area-weighted Mean Pool Length (AWMPL)
    - Area-weighted Mean Pool Width (AWMPW)
    - Pixel persistence layer
- Save the calculated metrics for each section to CSV files.
- Merges metrics from processed polygons and saves them to a CSV file.
- Export a pixel persistence raster for the entire AOI, if desired.

3. **Returns:**

The module generates a series of metrics for the specified river sections and section length. Metrics include section- and AOI-level statistics for various metrics and a pixel persistence raster. Results are stored in organised directories within the output folder. If needed, the module can export shapefiles (Pool lenght and endpoints) for further analysis or visualization.

## Usage Example
```python
from src.irm_main import calc_metrics

# Define input parameters

# Path to the directory containing water masks or DataArray
da_wmask = "path/to/water_masks" 
# Path to the river corridor extent (sections) shapefile (.shp)
rcor_extent = "path/to/rcor_extent.shp"
# Section length in km
section_length = 0.484
# Output directory to store results
outdir = "path/to/output_directory"
# Input images file extension
img_ext = ".tif"
# Export shapefiles (True or False)
export_shp = True

# Calculate river metrics
calc_metrics(da_wmask, rcor_extent, section_length, outdir, img_ext, export_shp)
```

[Back to Main README](README.md)