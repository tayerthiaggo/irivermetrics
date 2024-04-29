# calculate_metrics

**calculate_metrics** (_**da_wmask**, **rcor_extent**, **section_length**=None, **min_pool_size**=2, **outdir**=None, **img_ext**='.tif', **export_shp**=False, **return_da_array**=False_)

## Overview

**calculate_metrics** is the second module of the iRiverMetrics toolkit, designed to compute a range of ecohydrological metrics from binary water masks (generated or not by module 1 - [wd_batch](module1.md)). These metrics representt various aspects of surface water dynamics in intermittent rivers, such as morphology, persistence, and fragmentation. For a deeper understanding of the metrics and methodologies, refer to the [original paper](https://doi.org/10.1016/j.jhydrol.2023.129087). For an application example, see [this paper](https://doi.org/10.1016/j.jhydrol.2023.130266).

## Usage Guide
### Setup
Here's an example of how to use this module to calculate surface water metrics:

1. **Parameters:**

- da_wmask : str or xarray.DataArray

    Directory path or xarray.DataArray containing binary water masks.

    Note: Ensure all images have consistent CRS and spatial resolutions, and names include dates in "yyyy-mm-dd" or "yyyy_mm_dd" format.

- rcor_extent : str or geopandas.GeoDataFrame

    Path to the river corridor extent (river sections) shapefile (.shp) defining the Area of Interest (AOI).

- section_length : float, optional

    Length of river sections for metrics calculation in kilometers.

- min_pool_size: int, optional, default=2

    Minimum size of detectable water pools, specified in pixels. Defaults to 2 pixels.

- outdir : str, optional, default = None

    Destination directory for results. Defaults to a directory adjacent to the rcor_extent file if not specified.

- img_ext : str, optional, default = '.tif'

    File extension of the water mask images. Not required if using DataArrays.

- export_shp : bool, optional, default = False

    Whether to export detailed shapefiles of the analysed river sections. Shapefiles with wetted length, start/end and mipoints will be exported for each time step.

- return_da_array : bool, optional, default=False
    
    Whether to return the data array of water masks along with the calculation results. Defaults to False.

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
#Add cloned directory to path
import sys
sys.path.append(r'path\to\clone\irivermetrics')

from src.irm_main import calculate_metrics

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

[Back to Main README](../README.md)