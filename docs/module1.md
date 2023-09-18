# wd_batch

wd_batch (input_img, rcor_extent, ini_file=None, outdir=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True)

## Overview

 **wd_batch** is the first module in the iRiverMetrics toolkit. It's designed for efficient batch water detection on a series of multispectral satellite images. This module aims to identify open water bodies over large areas using the Water Detect algorithm developed by Cordeiro et al. (2021). The algorithm combines spectral water indices with various bands (e.g., Red, Blue, Green, Near-infrared, and Shortwave infrared) to highlight water features.
 
 This module provides a convenient way to perform batch water detection efficiently while allowing fine-tuning of parameters as needed.

## Details

Here's an example of how to use wd_batch module to perform batch water detection:

1. **Download .ini file:** If you need to adjust any parameters of the WaterDetect configuring file, download the default .ini file [here](WaterDetect.ini). Note, if a .ini file path is provided, a folder will be created to export the results at the same location of the .ini file. If not the results will be exported to the same path as rcor_extent shapefile. 

2. **Parameters:**

- input_img : str or xarray.DataArray

    Provide a directory containing multispectral images (e.g., TIFF files) or a DataArray (xarray.DataArray) as input. These images (or DataArray) should contain at least 4 spectral bands (RGB+Near-infrared) for water detection.

    Note:

    Images in the directory must have a associate date in its name in the format "yyyy-mm-dd" or "yyyy_mm_dd".

    All images in the directory must have the same coordinate reference system, spatial resolution, and number of bands.

    The first 4 bands must be stacked as B, G, R, NIR. If there are more than 4 bands, the first 6 bands must be B, G, R, NIR, SWIR1, and SWIR2.

- rcor_extent : str

    Specify the river lines that define the rivers to be considered for water detection. These river lines help determine the area of interest (AOI) for the analysis.

- ini_file : str

    The path to the WaterDetect default Initialization File (.ini) file. This file contains key-value pairs for configuration settings, including spectral water indices combination, maximum clustering, and regularization.

- buffer : int or float, optional, default= 1000 metres

    Specify a buffer distance (in metres) around the river lines. This buffer distance will be used to create polygons around the river lines, defining the extended AOI for water detection.

- img_ext : str, optional, default = .tif

    Set the file extension of the input images. This parameter helps the module recognize the image files.

- reg : float, optional, default = 0.07 for four bands or 0.08 for six bands

    Define the regularization parameter. For further information refer to [this paper](https://doi.org/10.1080/15481603.2023.2168676).

- max_cluster : int, optional, default = 6 for four bands or 3 for six bands
    
    Specify the maximum clustering parameter. 

- export_tif : bool, optional, default = True
    
    Choose whether to export results as raster files.

    If the "Export Results as Raster" parameter is set to True, the module creates a folder to store raster files for each time step (e.g., .tif files).

3. **How it works:**

With the correct inputs, execute the module to perform the following steps:
- Validate the data, checking for data consistency and format.
- Convert and group input rasters into a single Dask DataArray to streamline processing.
- Buffer the river corridor extent to widen the AOI.
- Clip the Dask DataArray to the new AOI.
- Execute the Water Detect algorithm on each layer, generating a binary array with water (1), non-water (0), and no data (-1) values.
- Return a DataArray with results and if the export_tif parameter is set to True, creates a folder to store raster files for each time step.

4. **Returns:**

A DataArray (xarray.DataArray) time series of water mask data. If the "Export Results as Raster" parameter is set to True, the module creates a folder to store raster files for each time step (.tif).

## Usage Example
```python
from src.irm_main import wd_batch

# Define input parameters

# Path to a directory containing multispectral images (e.g., TIFF files)
input_img = "path/to/images"
# Path to the WaterDetect configuration (.ini) file (use None for default parameters)
ini_file = "path/to/WaterDetect.ini"
# Path to the river corridor extent shapefile (.shp)
rcor_extent = "path/to/rcor_extent.shp"
# Buffer distance (in meters) to extend the river corridor extent
buffer = 1000 #Adjust as needed
# Image file extension (e.g., '.tif')
img_ext = '.tif'
# Number of regular bands to use for processing (use None for default)
reg = None
# Maximum number of clusters for clustering (use None for default)
max_cluster = None #Adjust as needed
# Whether to export water masks as GeoTIFF files (True or False)
export_tif = True

# Generate a DataArray containing water masks based on the specified parameters
da_wmask = wd_batch(input_img, ini_file, rcor_extent, buffer, img_ext, reg, max_cluster, export_tif )
```

[Back to Main README](../README.md)