# waterdetect_batch

**waterdetect_batch** (_**input_img**, **r_lines**, **outdir**=None, **ini_file**=None, **buffer**=1000, **img_ext**='.tif', **reg**=None, **max_cluster**=None, **export_tif**=True, **return_da_array**=False_)

## Overview
 **wd_batch** is the first module in the iRiverMetrics toolkit. It's designed for efficient batch water detection on a series of multispectral satellite images over large areas using the Water Detect algorithm (Cordeiro et al., 2021) which integrates various spectral water indices and agglomerative clustering to enhance the detection and delineation of water bodies. The module is designed to operate on batches of images, providing robustness and scalability in processing while allowing fine-tuning of parameters as needed.

## Features
- Batch Processing: Process multiple images simultaneously, leveraging Dask for efficient data handling and computation.
- Flexible Configuration: Customise detection parameters via an initialization file, accommodating different environmental conditions and image characteristics.
- Detailed Output: Generate comprehensive water masks with options to export the results as GeoTIFF files for each acquisition date, or return a Dask Array.

## Usage Guide
### Setup
Here's an example of how to use waterdetect_batch module to perform batch water detection:

1. **Download .ini file:** If you need to adjust any parameters of the WaterDetect configuring file, download the default .ini file [here](WaterDetect.ini). Note, if a .ini file path is provided, a folder will be created to export the results at the same location of the .ini file. If not the results will be exported to the same path as rcor_extent shapefile. 

2. **Parameters:**
- input_img : str, xarray.DataArray or xarray.DataSet
    Provide a directory containing multispectral images (e.g., TIFF files), a DataArray (xarray.DataArray) or a DataSet (xarray.DataSet) as input. These images (or DataArray/DataSet) should contain at least 4 spectral bands (RGB+Near-infrared) for water detection.

    **Note:** Ensure all images have consistent CRS, spatial resolutions and number of bands, and names include dates in "yyyy-mm-dd" or "yyyy_mm_dd" format.
    The initial image will serve as a reference for the automated reprojection of any images with varying coordinate reference systems (CRS) or spatial resolution. If working with multiple satellites, execute the process separately for each satellite.

    The first 4 bands must be stacked as B, G, R, NIR. If there are more than 4 bands, the first 6 bands must be B, G, R, NIR, SWIR1, and SWIR2.

- r_lines : str or geopandas.GeoDataFrame
    Specify the river lines that define the rivers to be considered for water detection. These river lines help determine the area of interest (AOI) for the analysis.

- outdir : str, optional, default = None
    Output directory for results. If None, it will be generated based on the shapefile location.

- ini_file : str, optional, default = None
    The path to the WaterDetect Initialization File (.ini) file. This file contains key-value pairs for configuration settings, including spectral water indices combination, maximum clustering, and regularization. If no path is provided the WaterDetect default Initialization File (.ini) file will be used.

- buffer : int or float, optional, default = 1000 metres
    Specify a buffer distance (in metres) around the river lines. This buffer distance will be used to create polygons around the river lines, defining the extended AOI for water detection.

- img_ext : str, optional, default = .tif
    Set the file extension of the input images. This parameter helps the module recognize the image files.

- reg : float, optional, default = None
    Define the regularization of the normalized spectral indices. If None, 0.07 for four bands or 0.08 for six bands. For further information refer to [this paper](https://doi.org/10.1080/15481603.2023.2168676).

- max_cluster : int, optional, default = None
    Specify the maximum number of possible targets to be identified as clusters. If None, max_cluster = 6 for four bands or 3 for six bands. For further information refer to [this paper](https://doi.org/10.1080/15481603.2023.2168676).

- export_tif : bool, optional, default = True
    Choose whether to export resulting water masks as raster files.
    If the export_tif parameter is set to True, the module creates a folder to store raster files for each time step (e.g., .tif files).

- return_da_array : bool, optional, default = False
    Flag to return results as an xarray.DataArray. Defaults to False.

3. **How it works:**
With the correct inputs, execute the module to perform the following steps:
- Validate the data, checking for data consistency and format.
- Convert and group input rasters into a single Dask DataArray to streamline processing.
- Buffer the river corridor extent to widen the AOI.
- Clip the Dask DataArray to the new AOI.
- Execute the Water Detect algorithm on each layer, generating a binary array with water (1), non-water (0), and no data (-1) values.
- Return a DataArray with results and if the export_tif parameter is set to True, creates a folder to store raster files for each time step.

4. **Returns:**
A DataArray (xarray.DataArray) time series of water mask data. If the "export_tif" parameter is set to True, the module creates a folder to store raster files for each time step (.tif).

## Usage Example
```python
# Add cloned directory to path
import sys
sys.path.append(r'path\to\clone\irivermetrics')

from src.irm_main import waterdetect_batch

# Define input parameters

# Path to a directory containing multispectral images (e.g., TIFF files)
input_img = "path/to/images"
# Path to the river corridor extent shapefile (.shp)
r_lines = "path/to/rcor_extent.shp"
# Path to the WaterDetect configuration (.ini) file (use None for default parameters)
ini_file = "path/to/WaterDetect.ini"
# Path to the output directory
outdir= "path/to/output"
# Buffer distance (in meters) to extend the river corridor extent
buffer = 1000 #Adjust as needed
# Image file extension (e.g., '.tif')
img_ext = '.tif'
# Number of regular bands to use for processing (use None for default)
reg = None #Adjust as needed
# Maximum number of clusters for clustering (use None for default)
max_cluster = None #Adjust as needed
# Whether to export water masks as GeoTIFF files (True or False)
export_tif = True
# Whether to return a DataArray with results
return_da_array=False

# Generate a DataArray containing water masks based on the specified parameters
da_wmask = waterdetect_batch(input_img, r_lines, ini_file=None, outdir=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True, return_da_array=False)
```

[Back to Main README](../README.md)