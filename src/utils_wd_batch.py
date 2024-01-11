import os
import re
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import waterdetect as wd

from shapely.geometry import mapping

## shared functions

def create_new_dir(outdir, verbose=True):
    """
    Creates a new directory at the specified location.
    Parameters:
        outdir (str): The path of the directory to be created.
        verbose (bool): Whether to show the print statement or not (default is True).
    Returns:
        None
    """
    # Create output file
    os.makedirs(outdir, exist_ok=True)
    if verbose:
        print(f'{outdir} created to export results')
def get_total_memory():
    """
    Get the total available system memory (RAM) and return 95% of that value in gigabytes.

    Returns:
        float: 95% of the total available system memory in gigabytes.
    """
    total_memory_bytes = psutil.virtual_memory().total
    total_memory_gb = total_memory_bytes // (1024 ** 3)  # Convert bytes to gigabytes
    return total_memory_gb*0.95

## wd_batch utils

def validate_inputs(input_img, img_ext, rcor_extent, ini_file, buffer):
    """
    Validate input parameters for processing.

    Args:
        input_img (str or xarray.DataArray): Directory path containing image files or xarray.DataArray.
        img_ext (str): Image file extension.
        rcor_extent (str): Path to the river corridor extent shapefile (.shp).
        ini_file (str): Path to the WaterDetect Initialization (.ini) file.
        buffer (int): Buffer size for processing.

    Returns:
        tuple: A tuple containing validated input values.
            - input_img (list or xarray.DataArray): Validated input image data.
            - n_bands (int): Number of bands in the input images.
            - time_list (list): List of time values.
            - rcor_extent (str): Validated river corridor extent shapefile path.
    """
    # Check if rcor_extent has a .shp extension
    assert rcor_extent.endswith('.shp'), 'Pass river corridor extent (rcor_extent) as .shp'
    # Check if ini_file has a .ini extension
    assert ini_file.endswith('.ini'), "Use WaterDetect .ini file"
    # Validate input images or DataArray
    input_img, n_bands, time_list, rcor_extent = is_valid_input_img(input_img, img_ext, rcor_extent, buffer)        
    print('Input data validated.')
    return input_img, n_bands, time_list, rcor_extent
def is_valid_input_img(input_img, img_ext, rcor_extent, buffer):
    """
    Validate the input image data and prepare it for processing.

    Args:
        input_img (xarray.DataArray or str): Either an xarray DataArray or a folder path containing image files.
        img_ext (str): Image file extension.
        rcor_extent (str): Path to the river corridor extent shapefile (.shp).
        buffer (int): Buffer size for processing.

    Returns:
        tuple: A tuple containing validated input values.
            - input_img (xarray.DataArray): Validated input image data.
            - n_bands (int): Number of bands in the input images.
            - time_list (list): List of time values.
            - rcor_extent (str): Validated river corridor extent shapefile path.
    """
    print('Checking input data...')
    # Check if the input is a valid DataArray or folder path
    assert isinstance(input_img, xr.core.dataarray.DataArray) or (isinstance(input_img, str) and os.path.isdir(input_img)), 'Invalid Input. Pass input_img as a valid directory or DataArray'
    
    if isinstance(input_img, xr.core.dataarray.DataArray):
        # If input is a DataArray, validate and prepare it
        input_img, n_bands, time_list, rcor_extent = validate_data_array(input_img, rcor_extent)
        input_img = buffer_clip_aoi(rcor_extent, buffer, input_img)
        return input_img, n_bands, time_list, rcor_extent
    
    elif os.path.isdir(input_img):
        # If input is a directory, validate and prepare it
        input_img, n_bands, time_list, rcor_extent = validade_input_folder(input_img, rcor_extent, img_ext)
        input_img = buffer_clip_aoi(rcor_extent, buffer, input_img)
        return input_img, n_bands, time_list, rcor_extent
    # If the provided input is neither a netCDF file nor a directory, raise a ValueError
    else:
        raise ValueError('Pass input_img as a valid directory or DataArray')
def validate_data_array(input_img, rcor_extent):
    """
    Validate and preprocess an xarray DataArray.

    Args:
        input_img (xarray.DataArray): Input image data as an xarray DataArray.
        rcor_extent (str): Path to the river corridor extent shapefile (.shp).

    Returns:
        tuple: A tuple containing preprocessed input values.
            - input_img (xarray.DataArray): Preprocessed input image data.
            - n_bands (int): Number of bands in the input image.
            - time_lst (pd.DatetimeIndex): List of datetime values.
            - rcor_extent (str): Validated river corridor extent shapefile path.
    """
    # Assert that the DataArray has at least 4 bands
    assert len(input_img.band) <= 4, 'Not enough data. Dataset must have at least 4 bands (B,G,R,NIR)'
    # Get the coordinate reference system (CRS) from the input DataArray
    crs = input_img.rio.crs
    required_dimensions = ['x', 'y', 'time', 'band']
    # Check for missing dimensions in the DataArray
    missing_dimensions = [dim for dim in required_dimensions if dim not in input_img.dims]
    # Assert that all required dimensions are present
    assert not missing_dimensions, f"DataArray is missing the following required dimensions: {missing_dimensions}"
    # Get dimensions to squeeze (drop unnecessary dimensions)
    dims_to_squeeze = [dim for dim in input_img.dims if dim not in required_dimensions]
    # Squeeze and drop unnecessary dimensions
    input_img = input_img.squeeze(drop=True, dim=dims_to_squeeze)
    # Apply outlier filtering: Set values outside the range (0, 20000) to 0
    input_img = xr.where(((input_img > 0) & (input_img < 20000)), input_img, 0)
    # Restore the coordinate reference system after 'where' operation
    input_img = input_img.rio.write_crs(crs)
    # Fill NaN values with 0 and set '_FillValue' attribute to 0
    input_img = input_img.fillna(0)             
    input_img.attrs['_FillValue'] = 0
    # Transpose the dimensions to the order: 'time', 'band', 'y', 'x'
    input_img = input_img.transpose('time', 'band', 'y', 'x')
    # Check the number of bands and provide a reminder to stack them appropriately
    if len(input_img.band) == 4:
        print('Reminder: 4 bands in source must be stacked as B,G,R,NIR')
    else:
        print(f'{len(input_img.data_vars)} bands found. Reminder: First 6 bands must be stacked as B,G,R,NIR,SWIR1,SWIR2')
    # Validate the projection of the input image
    rcor_extent = validate_input_projection(crs, rcor_extent)
    # Create a list of datetime values from the input image's time dimension
    time_lst = pd.DatetimeIndex(
        [pd.to_datetime(str(img_target.time.values))
         .strftime('%Y-%m-%d') 
         for img_target in input_img
        ]
    )
    return input_img, len(input_img.band), time_lst, rcor_extent
def validade_input_folder(input_dir, rcor_extent, img_ext):
    """
    Validate and process images within a directory for input.
    Convert a directory of raster images into an xarray DataArray with time dimension.

    Args:
        input_dir (str): Directory path containing image files.
        rcor_extent (str): Path to the river corridor extent shapefile (.shp).
        img_ext (str): Image file extension.

    Returns:
        tuple: A tuple containing preprocessed input values.
            - input_img (xarray.DataArray): Preprocessed input image data.
            - n_bands (int): Number of bands in the input images.
            - time_lst (pd.DatetimeIndex): List of datetime values.
            - rcor_extent (str): Validated river corridor extent shapefile path.
    """
    # Create a list of image paths within the directory with the specified image extension (img_ext)
    img_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir) 
        if f.endswith(img_ext)
    ]
    crs_lst, res_lst, band_lst, time_values, da_images = [], [], [], [], []
    # Initialize a generator for processing images
    image_generator = process_images(img_files)
    # Process and concatenate images one at a time
    for pimg, time_value, crs, res, n_bands in image_generator:
        da_images.append(pimg)
        crs_lst.append(crs)
        res_lst.append(res)
        band_lst.append(n_bands)
        time_values.append(time_value)
    # Check consistency of CRS, resolution, and band count across images
    assert len(set(crs_lst)) == 1, 'Check projection. Images must have the same EPSG'
    assert len(set(res_lst)) == 1, 'Check spatial resolution. Images must have the same pixel size'
    assert len(set(band_lst)) == 1, 'Check spatial resolution. Images must have the same number of bands'
    # Validate the projection of the input image
    rcor_extent = validate_input_projection(crs_lst[0], rcor_extent)
    # Check the number of bands and provide a reminder to stack them appropriately
    if band_lst[0] == 4:
        print(f'Reminder: 4 bands in source must be stacked as B,G,R,NIR')
    else:
        print(f'{band_lst[0]} bands found. Reminder: First 6 bands must be stacked as B,G,R,NIR,SWIR2')
    # Create a list of datetime values from the input image's time dimension
    time_lst = pd.DatetimeIndex(
        [pd.to_datetime(str(time))
         .strftime('%Y-%m-%d') 
         for time in time_values
        ]
    )
    # Create an xarray DataArray with a 'time' dimension
    time = xr.Variable('time', time_values)
    # Concatenate image data along the 'time' dimension and sort by time
    input_img = xr.concat(da_images, dim=time).sortby('time')
    # Fill NaN values with 0 and set '_FillValue' attribute to 0
    input_img = input_img.fillna(0)             
    input_img.attrs['_FillValue'] = 0

    return input_img, band_lst[0], time_lst, rcor_extent
def process_images(img_files):
    """
    Process a list of image files, extracting relevant information.

    Args:
        img_files (list of str): List of image file paths.

    Yields:
        tuple: A tuple containing processed image data and information.
            - pimg (rioxarray.DataArray): Processed image data.
            - time_value (pd.Timestamp): Timestamp representing the image's date.
            - crs (str): Coordinate reference system (CRS) of the image.
            - res (float): Spatial resolution of the image.
            - n_bands (int): Number of bands in the image.
    """
    # Define a regular expression pattern to extract dates from image file names
    date_pattern = re.compile(r'\d{4}[-_]\d{2}[-_]\d{2}')  # Adjust the date pattern as needed
    # Loop through the provided image files
    for img in img_files:
        img_filename = os.path.basename(img)
        # Extract the date from the image file name using the defined pattern
        date_match = date_pattern.search(img_filename)
        # Check if a valid date is found in the file name; raise an error if not found
        if date_match is None:
            raise AssertionError(f'Invalid Date info in {img_filename}. File name should contain valid dates, such as yyyy-mm-dd or yyyy_mm_dd')
        # Convert the matched date string to a pandas Timestamp
        date_str = date_match.group(0).replace('_', '-')
        time_value = pd.Timestamp(date_str)
        # Open the image using rioxarray and determine its CRS, resolution, and number of bands
        pimg = rxr.open_rasterio(img).chunk(chunks='auto')
        crs = pimg.rio.crs
        res = pimg.rio.resolution()[0]
        n_bands = pimg.shape[0]
        # Yield the processed image data and associated information
        yield pimg, time_value, crs, res, n_bands
def validate_input_projection(img_crs, rcor_extent):
    """
    Validate and potentially reproject a river corridor extent shapefile to match a given CRS.

    Args:
        img_crs (str): Coordinate reference system (CRS) of the image.
        rcor_extent (str): Path to the river corridor extent shapefile (.shp).

    Returns:
        geopandas.GeoDataFrame: Validated and potentially reprojected river corridor extent.
    """
    # Read the river corridor extent shapefile using geopandas
    rcor_extent = gpd.read_file(rcor_extent) 
    # Check if the projection of the shapefile is different from the image's CRS
    if rcor_extent.crs != img_crs:
        print('rcor_extent and da_wmask projections are different! rcor_extent will be reprojected to match da_wmask')
        # Reproject the shapefile to match the CRS of the image
        rcor_extent = rcor_extent.to_crs(img_crs)
    return rcor_extent
def change_ini(ini_file, n_bands, reg, max_cluster):
    """
    Modify a WaterDetect configuration (.ini) file based on the number of bands and provided parameters.

    Args:
        ini_file (str): Path to the WaterDetect configuration (.ini) file.
        n_bands (int): Number of bands in the input images.
        reg (float): Regularization parameter.
        max_cluster (int): Maximum number of clusters.

    Returns:
        tuple: A tuple containing the modified .ini file path and the list of bands.
            - ini_file (str): Modified .ini file path.
            - bands (list of str): List of bands to be used in WaterDetect.
    """
    # Read all the lines in the .ini file into a list
    list_of_lines = open(ini_file, "r").readlines()
    # Define the default bands
    bands = ['Blue', 'Green', 'Red', 'Nir']
    # Modify specific lines in the .ini file
    list_of_lines[6] = 'reference_band = Green \n'
    list_of_lines[19] = 'save_indices = False \n'
    list_of_lines[28] = 'calc_glint = False \n'
    list_of_lines[31] = 'glint_mode = False \n'
    list_of_lines[41] = 'external_mask = True \n'
    list_of_lines[42] = 'mask_name = invalid_mask \n'
    list_of_lines[127] = "clip_band = None #['mndwi', 'Mir2', 'ndwi'] \n"
    list_of_lines[128] = "clip_inf_value = None #[-0.1, None, -0.15] \n"
    list_of_lines[129] = "clip_sup_value = None #[None, 0.075, None] \n"
    list_of_lines[144] = 'detectwatercluster = maxndwi \n'
    list_of_lines[149] = 'plot_graphs = False \n'
    # Check the number of bands and change the .ini file accordingly
    if n_bands == 4:
        # If 4 bands, and max_cluster and reg are not set yet, set them to default values
        if max_cluster == None and reg == None:
            max_cluster = 6
            reg = 0.07
        # Set the indices for specific lines in the .ini file
        list_of_lines[84] = "#\t\t    ['mndwi', 'ndwi', 'Mir2'],\n"
        list_of_lines[104] = "\t\t    ['ndwi', 'Nir' ],\n"
    else:
        # If not 4 bands, add additional bands 'Mir' and 'Mir2' to the list
        bands += ['Mir', 'Mir2']
        if max_cluster == None and reg == None:
            max_cluster = 3
            reg = 0.08

    # Set the values of max_cluster and reg in the .ini file
    list_of_lines[117] = 'regularization = ' + str(reg) + '\n'
    list_of_lines[124] = 'max_clusters = ' + str(max_cluster) + '\n'
    # Open and edit the default .ini file with the updated lines
    a_file = open(ini_file, "w")
    a_file.writelines(list_of_lines)
    a_file.close()
    # Return the name of the .ini file and the updated list of bands
    return ini_file, bands
def buffer_clip_aoi(rcor_extent, buffer, input_img):
    """
    Buffer and clip an input image to a specified river corridor extent.

    Args:
        rcor_extent (geopandas.GeoDataFrame): River corridor extent as a GeoDataFrame.
        buffer (float): Buffer distance in the same units as the CRS of the extent.
        input_img (xarray.DataArray): Input image data as an xarray DataArray.

    Returns:
        xarray.DataArray: Clipped input image data.
    """
    # Create a buffer around the extent using the specified buffer value
    rcor_extent_buffer = rcor_extent.buffer(buffer)
    # Reproject the buffer if its CRS is different from the input image's CRS
    if rcor_extent_buffer.crs.to_epsg() != input_img.rio.crs.to_epsg():
        rcor_extent_buffer = rcor_extent_buffer.to_crs(input_img.rio.crs.to_epsg())
    # Clip the input image to the buffered river corridor extent
    input_img = input_img.rio.clip(rcor_extent_buffer.geometry.apply(mapping), rcor_extent_buffer.crs)
    # Return the clipped input image data as an xarray DataArray
    return input_img
def process_image_parallel(img_target, bands, config, export_tif, date, outdir):
    """
    Process an input image to create a water mask using WaterDetect.

    Args:
        img_target (xarray.DataArray): Input image data as an xarray DataArray.
        bands (list of str): List of bands to be used in WaterDetect.
        config (WaterDetect.DWConfig): WaterDetect configuration object.
        export_tif (bool): Whether to export the water mask as a GeoTIFF file.
        date (str): Date associated with the image.
        outdir (str): Output directory for results.

    Returns:
        xarray.DataArray: Water mask generated by WaterDetect.
    """
    # Execute WaterDetect to create the water mask for the current image
    water_xarray = wd_mask(img_target, bands, config)
    # Export the water mask to a GeoTIFF file if 'export_tif' is True
    if export_tif:
        # Define the output directory for GeoTIFF files
        tif_out_dir = os.path.join(outdir, 'wmask_tif')
        # Format the date to YYYY-MM-DD for the file name
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        # Export the water mask as a GeoTIFF with LZW compression
        water_xarray.rio.to_raster(os.path.join(tif_out_dir, str(date) + '.tif'), compress='lzw')
    return water_xarray
def wd_mask(img_target, bands, config):
    """
    Apply WaterDetect (wd) to the target image and generate a water mask.

    Args:
        img_target (xarray.DataArray): Input image data as an xarray DataArray.
        bands (list of str): List of bands to be used in WaterDetect.
        config (WaterDetect.DWConfig): WaterDetect configuration object.

    Returns:
        xarray.DataArray: Water mask generated by WaterDetect.
    """
    # Call the 'create_wd_dict' function to generate a dictionary (arrays) from the target image data
    arrays, water_xarray = create_wd_dict(img_target, bands)
    # Check the number of bands to determine the WaterDetect input bands
    if len(bands) == 4:
        # Since we don't have the 'Mir' band, we cheat WaterDetect by setting certain bands to zeros
        #  (to be corrected in the WaterDetect next version)
        arrays['mndwi'] = np.zeros_like(arrays['Green'])
        arrays['mbwi'] = np.zeros_like(arrays['Green'])
        arrays['Mir2'] = np.zeros_like(arrays['Green'])
        invalid_mask = (arrays['Nir'] == 0)
        # Initialize WaterDetect with appropriate bands and invalid mask
        wmask = wd.DWImageClustering(
            bands=arrays, 
            bands_keys=['ndwi', 'Nir'], 
            invalid_mask=invalid_mask, 
            config=config, 
            glint_processor=None
            )
    else:
        # Since we don't have the 'Mir2' band, we set a specific band as the invalid mask
        invalid_mask = (arrays['Mir2'] == 0)
        # Initialize WaterDetect with appropriate bands and invalid mask
        wmask = wd.DWImageClustering(
            bands=arrays, 
            bands_keys=['mndwi', 'ndwi', 'Mir2'], 
            invalid_mask=invalid_mask, 
            config=config, 
            glint_processor=None
            )  
    # Calculate the water mask using WaterDetect
    mask = wmask.run_detect_water()
    water_xarray.values= wmask.water_mask
    # Set nodata values in the 'water_xarray' to -1
    water_xarray.rio.write_nodata(-1, inplace=True)
    # Return the water mask as an xarray DataArray
    return water_xarray
def create_wd_dict (img_target, bands):
    """
    Create a dictionary and a dummy DataArray for WaterDetect (wd) input.

    The function takes the target image data (img_target) and rescales it by dividing by a specified factor (div_factor).
    It then creates a dictionary (arrays) with each band name as the key and its corresponding rescaled values as the value.
    Additionally, it creates a dummy DataArray (water_xarray) by selecting one of the bands from the target image.

    Parameters:
        img_target (xarray.core.dataarray.DataArray): The target image data as an xarray DataArray.
        bands (list): A list containing the names of the bands in the target image.

    Returns:
        tuple: A tuple containing the dictionary (arrays) with rescaled band values and the dummy DataArray (water_xarray).
    """
    # Rescale the target image by dividing it by a specified factor (div_factor)
    div_factor = 10000
    cube = img_target/div_factor
    # Create the 'arrays' dictionary by iterating over the bands and extracting their rescaled values
    arrays = {layer: cube[i].values for i, layer in enumerate(bands)}
    # Create a dummy DataArray 'water_xarray' by selecting one of the bands from the target image (e.g., 'band=1')
    water_xarray = img_target.isel(band=1)
    # Return the dictionary with rescaled band values and the dummy DataArray
    return arrays, water_xarray
def process_results(results):
    """
    Process the water mask results obtained from processing multiple images.

    Args:
        results (list of xarray.DataArray): List of water mask results as xarray DataArrays.

    Returns:
        list of xarray.DataArray: Processed water mask results.
    """
    wd_lst = []
    # Iterate through the list of water mask results
    for water_xarray in results:
        # Append the water mask result to the list
        wd_lst.append(water_xarray)
    return wd_lst