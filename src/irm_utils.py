import os
import re
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio
import waterdetect as wd
import shapely.geometry
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, mapping
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric
from itertools import combinations
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict

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

## calc_metrics utils

def process_polygon_parallel(args_list):
    """
    Process river sections in parallel.

    Args:
        args_list (tuple): A tuple containing polygon and other parameters.
        
    Returns:
        None
    """
    print()
    try:
        # Unpack the argument list containing polygon and other parameters
        polygon, da_wmask, crs, outdir_section, export_shp, section_length = args_list
        # Call the process_polygon function to process the current polygon
        date_list, section_area, total_wet_area_list, total_wet_perimeter_list, \
        length_list, n_pools_list, AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list, \
        AWMPW_list, da_area, da_npools, da_rlines, layer, outdir_section = process_polygon(polygon, da_wmask, crs, outdir_section, export_shp)
        # Create a dictionary containing various metrics for the processed polygon
        pand_dict = {'date': date_list, 'section_area': section_area, 'wet_area_km2': [d/10**6 for d in total_wet_area_list],
                    'wet_perimeter_km2': [d/10**6 for d in total_wet_perimeter_list], 'wet_length_km': [d/1000 for d in length_list],
                    'npools': n_pools_list, 'AWMSI': AWMSI_list, 'AWRe': AWRe_list,
                    'AWMPA': [d/10**6 for d in AWMPA_list], 'AWMPL': [d/1000 for d in AWMPL_list],
                    'AWMPW': [d/1000 for d in AWMPW_list]}
        # Convert the dictionary to a pandas DataFrame
        pd_metrics = pd.DataFrame(data=pand_dict)
        # Calculate metrics using the DataFrame and section length
        pdm = calculate_metrics_df(pd_metrics, section_length)
        # Create a data array with processed data
        da_area = create_dataarray(layer, da_area, date_list, crs)
        # Calculate pixel persistence for the section
        pdm = pixel_persistence_section(da_area, pdm, interval_ranges=None)
        # Save the calculated metrics to a CSV file
        pdm.to_csv(os.path.join(outdir_section, 'pd_metrics.csv'))
    except Exception as e:
        # If an error occurs during processing, print an error message
        print(f"Error processing polygon {args_list[0]}: {e}")
def create_dataarray(layer, layer_metric, date_list, crs):
    """
    Create a DataArray with provided layer metrics and coordinates.

    Parameters:
    - layer (xarray.DataArray): Original layer data.
    - layer_metric (numpy.ndarray): Array containing layer metrics.
    - date_list (list): List of dates corresponding to the metrics.
    - crs (CRS): Coordinate Reference System.

    Returns:
    - da (xarray.DataArray): Created DataArray with metrics.
    """
    da = xr.DataArray(
        data=layer_metric,
        dims=['time', 'y', 'x'],
        coords=dict(time=date_list, y= layer.coords['y'], x= layer.coords['x']),
        attrs={'_FillValue': -1},
    )
    da.rio.write_crs(crs, inplace=True)
    da.attrs['crs'] = str(crs)
    return da
## Validation
def validate(da_wmask, rcor_extent, section_length, img_ext, expected_geometry_type):
    """
    Validates input data for processing.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray or str): Input data, either as a DataArray or a directory path.
    rcor_extent (str): Path to a file containing rcor extent (river sections) shapefile.
    section_length (int or float): Section length value.
    img_ext (str): Image extension (e.g., '.tif') for processing folder input.

    Returns:
    tuple: A tuple containing the validated DataArray and rcor extent information.
    """
    print('Checking input data...')
    # Check if da_wmask is a valid DataArray or a directory path
    assert (isinstance(da_wmask, xr.core.dataarray.DataArray) or 
           (isinstance(da_wmask, str) and os.path.isdir(da_wmask))), 'Invalid input. da_wmask must be a valid DataArray or a directory path'
    # Check the extension and type of the rcor_extent file
    rcor_extent = process_shp_input(rcor_extent, expected_geometry_type)
    # Check if section length is present
    assert section_length != None, 'Invalid input. Section length not found.'  
    # If input is a directory - process input
    if isinstance(da_wmask, str) and os.path.isdir(da_wmask):
        da_wmask = process_folder_input(da_wmask, img_ext)
    # Check if CRS information is present
    assert da_wmask.rio.crs != None, 'Invalid input. da_wmask CRS not found.'  
    # validate projection
    rcor_extent = validate_input_projection (da_wmask, rcor_extent)
    # List of dimensions that are required in the DataArray
    required_dimensions = ['x', 'y', 'time']
    # Check for missing dimensions by comparing required dimensions with DataArray dimensions
    missing_dimensions = [dim for dim in required_dimensions if dim not in da_wmask.dims]
    # Assert that there are no missing dimensions
    assert not missing_dimensions, f"Invalid input. The following dimensions are missing: {', '.join(missing_dimensions)}"
    print('Checking input data...Data validated')
    return da_wmask, rcor_extent
def process_shp_input(rcor_extent, expected_geometry_type):
    """
    Validates and loads a shapefile with specific requirements.

    Parameters:
    rcor_extent (str): Path to a shapefile (.shp) to be processed.

    Returns:
    geopandas.geodataframe.GeoDataFrame: A GeoDataFrame containing the loaded shapefile data.
    
    Raises:
    AssertionError: If the input does not meet the validation criteria.
    """
    # Check the extension of the rcor_extent file
    assert rcor_extent.endswith('.shp'), 'Invalid input. rcor_extent must be a valid shapefile(.shp)'
    # Load the shapefile using geopandas
    rcor_extent = gpd.read_file(rcor_extent) 
    # Check if there is at least one feature and if any geometry matches the expected type
    valid_geometry = False
    if expected_geometry_type == 'Polygon':
        valid_geometry = any(isinstance(geom, (Polygon, MultiPolygon)) for geom in rcor_extent.geometry)
    elif expected_geometry_type == 'LineString':
        valid_geometry = any(isinstance(geom, LineString) for geom in rcor_extent.geometry)
    else:
        raise ValueError("Invalid expected_geometry_type. Use 'Polygon' or 'LineString'.")
    
    assert not rcor_extent.empty and valid_geometry, f'Invalid input. Shapefile does not contain valid {expected_geometry_type} geometries'  
       
    return rcor_extent
def process_folder_input(input_dir, img_ext):
    """
    Convert a directory of raster images into an xarray DataArray with time dimension.

    Parameters:
        input_path (str): The path to the directory containing raster images.
        img_ext (str): The image file extension to filter files in the directory.

    Returns:
        xr.DataArray: An xarray DataArray containing the concatenated raster images
                      with a 'time' dimension, sorted by time.
    """
    # Create a list of image paths within the directory with the specified image extension (img_ext)
    img_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir) 
        if f.endswith(img_ext)
    ]
    # Compile a regular expression patterns for matching "yyyy-mm-dd" or "yyyy_mm_dd"
    date_pattern = re.compile(r'\d{4}[-_]\d{2}[-_]\d{2}')
    # Initialize lists to store metadata and image data
    crs_lst, res_lst, band_lst, time_values, da_images = [], [], [], [], []
    # Iterate through each image file in the directory
    for img in img_files:
        img_filename = os.path.basename(img)
        # Check if the filename contains valid date patterns
        date_match = date_pattern.search(img_filename)
        if date_match is None:
            raise AssertionError(f'Invalid Date info in {img_filename}. File name should contain valid dates, such as yyyy-mm-dd or yyyy_mm_dd')
        # Extract the date string from the matched pattern and convert it to a pandas Timestamp
        date_str = date_match.group(0).replace('_', '-')
        time_values.append(pd.Timestamp(date_str))
        # Open the raster image using rioxarray
        pimg = rxr.open_rasterio(img)
        # Append the image data to the list for concatenation
        da_images.append(pimg)
        # Gather metadata information for each image
        crs_lst.append(pimg.rio.crs)
        res_lst.append(pimg.rio.resolution()[0])
        band_lst.append(pimg.shape[0])  
    # Check consistency of CRS, resolution, and band count across images
    assert len(set(crs_lst)) == 1, f'Check projection. Images must have same EPSG'
    assert len(set(res_lst)) == 1, f'Check spatial resolution. Images must have same pixel size'
    assert len(set(band_lst)) == 1, f'Check spatial resolution. Images must have same number of bands'
    # Create an xarray DataArray with a 'time' dimension
    time = xr.Variable('time', time_values)
    # Concatenate image data along the 'time' dimension and sort by time
    da_wmask = xr.concat(da_images, dim=time).sortby('time')

    # # Apply chunking if chunk_size is specified
    # if chunk_size:
    #     da_wmask = da_wmask.chunk(chunks='auto')
    
    # Fill NaN values with -1 and set '_FillValue' attribute to -1
    da_wmask = da_wmask.fillna(-1)             
    da_wmask.attrs['_FillValue'] = -1
    return da_wmask
def validate_input_projection(da_wmask, gdf_shp):
    """
    Validates and potentially reprojects a shapefile to match the projection of a raster with a water mask.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray): A DataArray representing a raster with water mask.
    gdf_shp (geopandas.geodataframe.GeoDataFrame): A GeoDataFrame representing a shapefile.

    Returns:
    geopandas.geodataframe.GeoDataFrame: A GeoDataFrame, potentially reprojected to match the projection of da_wmask.

    Note:
    If the projections of da_wmask and gdf_shp are different, gdf_shp will be reprojected to match da_wmask's projection.
    """
    # Check if the projection of the shapefile is different from the raster with water mask
    if gdf_shp.crs != da_wmask.rio.crs:
        print('rcor_extent and da_wmask projections are different! rcor_extent will be reprojected to match da_wmask')
        gdf_shp = gdf_shp.to_crs(da_wmask.rio.crs)
    return gdf_shp
## Preprocessing
def preprocess(da_wmask, rcor_extent, outdir, export_shp, section_length, skip_prepare_args=False):
    """
    Perform preprocessing steps on input data and extent shapefile.

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with mask.
        rcor_extent (geopandas.GeoDataFrame): Extent shapefile rcor (river sections).

    Returns:
        xarray.DataArray: Preprocessed DataArray with nodata filled and reprojected to UTM.
        geopandas.GeoDataFrame: Reprojected extent shapefile.
    """
    print('Preprocessing...')
    # Step 1: Clip input data and extent to match dimensions
    da_wmask, rcor_extent = clip_inputs(da_wmask, rcor_extent)
    # Step 2: Fill nodata values in the DataArray
    da_wmask = fill_nodata(da_wmask)
    # Step 3: Reproject DataArray and extent to UTM CRS
    da_wmask, rcor_extent = reproject_to_utm(da_wmask, rcor_extent)
    if not skip_prepare_args:
        # Step 4: Prepare args for processing
        args_list = prepare_args(da_wmask, rcor_extent, outdir, export_shp, section_length)
    else:
        args_list = None
        
    print('Preprocessing...Done!')
    return args_list, da_wmask, rcor_extent
def clip_inputs (da_wmask, rcor_extent):
    """
    Clip input DataArray and extent shapefile to the overlapping region.

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with water mask.
        rcor_extent (geopandas.GeoDataFrame): Extent shapefile (river sections).

    Returns:
        xarray.DataArray: Clipped DataArray.
        geopandas.GeoDataFrame: Filtered extent shapefile.
    """
    # Get bounding box of DataArray
    minx, miny, maxx, maxy = da_wmask.rio.bounds()
    # Filter extent polygons that are completely within the DataArray bounding box
    rcor_extent = rcor_extent[rcor_extent.geometry.intersects(box(minx, miny, maxx, maxy))]
    # Clip the DataArray using the filtered extent
    da_wmask = da_wmask.rio.clip(rcor_extent.geometry)
    return da_wmask, rcor_extent
def fill_nodata(da_wmask):
    """
    Fill NoData values in a DataArray using adjacent valid values (backward or forward fill).

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with NoData values.

    Returns:
        xarray.DataArray: DataArray with NoData values filled using adjacent valid values.
    """
    # Convert xarray DataArray to NumPy array for efficient manipulation
    bbox_array = da_wmask.values
    # Get the number of layers in the data array.
    num_layers = bbox_array.shape[0]
    # Loop through each layer and fill NoData values using adjacent valid values
    for num in range(num_layers):
        # Check if the current layer contains any NoData values (-1).
        if np.any(bbox_array[num] == -1):
            # Find the next valid layer with data
            next_layer = num + 1
            while next_layer < num_layers and np.all(bbox_array[next_layer] == -1):
                next_layer += 1
            # If a valid next layer is found, 
            # replace NoData values in the current layer with values from the next layer.
            if next_layer < num_layers:
                # Replace NoData values with values from the next valid layer
                bbox_array[num][bbox_array[num] == -1] = bbox_array[next_layer][bbox_array[num] == -1]
            else:
                # If no valid next layer, use the previous valid layer
                prev_layer = num - 1
                while prev_layer >= 0 and np.all(bbox_array[prev_layer] == -1):
                    prev_layer -= 1
                # If a valid previous layer is found, 
                # replace NoData values in the current layer with values from the previous layer.
                if prev_layer >= 0:
                    # Replace NoData values with values from the previous valid layer
                    bbox_array[num][bbox_array[num] == -1] = bbox_array[prev_layer][bbox_array[num] == -1]
    # Create a new xarray DataArray with filled NoData values and the same coordinates and dimensions as the input.
    da_wmask = xr.DataArray(bbox_array, coords=da_wmask.coords, dims=da_wmask.dims)
    da_wmask.attrs['_FillValue'] = -1
    return da_wmask
def reproject_to_utm (da_wmask, rcor_extent):
    """
    Validate the CRS, reproject DataArray to UTM, ensure matching CRS with extent shapefile,
    and filter extent polygons that are completely within the DataArray bounding box.

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with a mask.
        rcor_extent (geopandas.GeoDataFrame): Extent shapefile for reprojection and filtering.

    Returns:
        xarray.DataArray: Reprojected DataArray with nodata fixed.
        geopandas.GeoDataFrame: Reprojected and filtered extent shapefile.
    """
    # Check if the current CRS is not UTM and needs reprojection
    if da_wmask.rio.crs != da_wmask.rio.estimate_utm_crs():
        # List of dimensions that are required in the DataArray
        required_dimensions = ['x', 'y', 'time']
        # Get the dimensions to squeeze
        dims_to_squeeze = [dim for dim in da_wmask.dims if dim not in required_dimensions]
        # Squeeze and drop unnecessary dimensions
        da_wmask = da_wmask.squeeze(drop=True, dim=dims_to_squeeze)
        # Reproject DataArray to UTM CRS
        da_wmask = da_wmask.rio.reproject(da_wmask.rio.estimate_utm_crs())
        # Fix nodata values
        da_wmask = da_wmask.where(da_wmask == 1, other = 0)
        da_wmask.attrs['_FillValue'] = 0
    
    # Check if the projection of the shapefile is different from the DataArray
    if rcor_extent.crs != da_wmask.rio.crs:
        rcor_extent = rcor_extent.to_crs(da_wmask.rio.crs)
    return da_wmask, rcor_extent
def prepare_args(da_wmask, rcor_extent, section_outdir_folder, export_shp, section_length):
    """
    Prepare a list of arguments for parallel processing.
    
    Args:
        da_wmask (xarray.DataArray): The input raster data array.
        rcor_extent (geopandas.GeoDataFrame): The GeoDataFrame containing polygons.
        outdir (str): Output directory path.
        export_shp (bool): Flag indicating whether to export shapefiles.
        section_length (float): Length of each section.

    Returns:
        list: List of arguments for parallel processing.
    """
    args_list = []
    crs = rcor_extent.crs
    # Process each polygon
    for polygon in rcor_extent.iloc:
        # Get the bounding box of the polygon
        xmin, ymin, xmax, ymax = polygon.geometry.bounds
        # Convert spatial bounds to pixel coordinates
        x_coords = da_wmask.x.values
        y_coords = da_wmask.y.values
        col_mask = (x_coords >= xmin) & (x_coords <= xmax)
        row_mask = (y_coords >= ymin) & (y_coords <= ymax)
        # Apply filtering to both x and y dimensions
        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]
        # Clip the raster using numpy indexing
        cliped_da_wmask = da_wmask[:, row_indices, col_indices]
        # Create a directory to save output metrics for the section
        outdir_section = os.path.join(section_outdir_folder, str(polygon.name))
        # Append arguments to the list
        args_list.append([polygon, cliped_da_wmask, crs, outdir_section, export_shp, section_length]) 
    return args_list
## Calculate metrics functions
def process_polygon(polygon, da_wmask, crs, outdir_section, export_shp):
    """
    Process a polygon region to calculate various metrics from raster data through time.

    Parameters:
    - polygon (GeoDataFrame): Polygon region of interest (Section).
    - da_wmask (xarray.DataArray): DataArray containing water mask.
    - crs (str): Section CRS.
    - outdir_section (str): Directory to save output metrics of Section.
    - export_shp (bool): Flag to export shapefiles.

    Returns:
    - date_list (list): List of dates for each processed time layer.
    - section_area (float): Area of the processed polygon section in square kilometers.
    - total_wet_area_list (list): List of total wet areas for each processed time layer.
    - total_wet_perimeter_list (list): List of total wet perimeters for each processed time layer.
    - length_list (list): List of total lengths of wet regions for each processed time layer.
    - n_pools_list (list): List of the number of detected wet regions for each processed time layer.
    - AWMSI_list (list): List of Area-weighted Mean Shape Index (AWMSI) values for each processed time layer.
    - AWRe_list (list): List of Area-weighted Elongation Ratio (AWRe) values for each processed time layer.
    - AWMPA_list (list): List of Area-weighted Mean Pixel Area (AWMPA) values for each processed time layer.
    - AWMPL_list (list): List of Area-weighted Mean Pool Length (AWMPL) values for each processed time layer.
    - AWMPW_list (list): List of Area-weighted Mean Pool Width (AWMPW) values for each processed time layer.
    """
    # Initialize lists to store calculated metrics
    AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list, AWMPW_list = [], [], [], [], []
    date_list, length_list, n_pools_list, total_wet_area_list, total_wet_perimeter_list = [], [], [], [], []
    da_area, da_npools, da_rlines = [], [], []
    # Calculate area of the polygon section in square kilometers
    section_area = polygon.geometry.area/10**6
    # Clip the water mask to the polygon's extent
    da_wmask = da_wmask.rio.clip([polygon.geometry])
    # Create a directory to save output metrics for the section
    create_new_dir(outdir_section, verbose=False)

    # Loop through each time layer in the water mask
    for num, layer in enumerate(da_wmask):
        # Convert time value to date format
        date = pd.to_datetime(str(da_wmask.time.values[num])).strftime('%Y-%m-%d')
        # Calculate pool area and perimeter
        dict_area_2p, area_array = calculate_pool_area_and_perimeter(layer)
        # Calculate connectivity properties - pool endpoints/length/width
        _dict_wet_prop, n_pools_index_lst, lines_index_lst, area_length_index_lst, lines_index_lst_width = calculate_connectivity_properties(layer, area_array)
        # Calculate area-weighted metrics metrics using calculated data
        total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL = calculate_metrics_AW(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width)
        # Append results to respective lists
        date_list.append(pd.to_datetime(str(da_wmask[num].time.values)).strftime('%Y-%m-%d'))
        length_list.append(sum((_dict_wet_prop[d]['length']) for d in _dict_wet_prop))
        total_wet_area_list.append(total_wet_area)
        total_wet_perimeter_list.append(total_wet_perimeter)
        n_pools_list.append(len(_dict_wet_prop))
        AWMSI_list.append(AWMSI)
        AWRe_list.append(AWRe)
        AWMPA_list.append(AWMPA)
        AWMPL_list.append(AWMPL)
        AWMPW_list.append(AWMPW)
        da_area.append(layer.values)
        da_npools.append(list_index(layer, n_pools_index_lst))
        da_rlines.append(list_index(layer, lines_index_lst))

        # Export shapefiles if export_shp is True
        if export_shp:
            try:
                save_shp (_dict_wet_prop, outdir_section, date, crs)
            except:
                pass
    return date_list, section_area, total_wet_area_list, total_wet_perimeter_list,\
           length_list, n_pools_list, AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list,\
           AWMPW_list, da_area, da_npools, da_rlines, da_wmask.isel(time=0), outdir_section
def calculate_pool_area_and_perimeter(layer):
    """
    Calculate the perimeter and area of connected regions in a given water mask layer.

    Parameters:
    - layer (xarray.DataArray): Input DataArray layer.

    Returns:
    - dict_area_2p (defaultdict): A dictionary where keys are labels of connected regions
                                  and values are lists [perimeter, area].
    - area_array (numpy.ndarray): A 2D NumPy array containing the calculated areas
                                  for each pixel's connected region.
    """
    # Initialize a dictionary to store calculated perimeter and area for each label
    dict_area_2p = defaultdict(lambda: [0.0, 0.0])
    # Group connected pixels into distinct regions
    pre_label = label(layer, connectivity=2)
    # Obtain geometric shapes for each connected region
    pre_label_shapes = list(rasterio.features.shapes((pre_label.astype('int32')), transform=layer.rio.transform()))
    # Calculate perimeter and area for each region and store in the dictionary
    for polygon, value in pre_label_shapes:
        if value == 0:
            continue  # Skip processing for group 0, which is no data
        size = pre_label[pre_label == value].size
        if size > 2: # Only process if the region contains more than 2 pixels
            shape = shapely.geometry.shape(polygon)
            perimeter = shape.length
            area = shape.area
            dict_area_2p[value][0] += perimeter
            dict_area_2p[value][1] += area
    
    # Create an area array using the calculated values and the input labels
    u, inv = np.unique(pre_label, return_inverse=True)
    areas = np.array([dict_area_2p[x][1] if x in dict_area_2p else 0.0 for x in u]) # Fill non-existent entries with 0.0
    area_array = areas[inv].reshape(pre_label.shape)

    return dict_area_2p, area_array
def calculate_connectivity_properties(layer, area_array):
    """
    Calculate connectivity properties of a layer based on provided data.
    This function calculates connectivity properties for wet regions in the given layer.
    It employs skeletonization, region labeling, path finding, and various geometric operations.
    The calculated properties are stored in several lists and dictionaries for further analysis.

    Parameters:
    - layer (xarray.DataArray): Input DataArray.
    - area_array (numpy.ndarray): Array containing calculated areas for each pixel's region.

    Returns:
    - _dict_wet_prop (dict): Dictionary containing calculated properties for wet regions.
    - endpoints_index_lst (list): List of indexes representing endpoints of paths.
    - lines_index_lst (list): List of indexes representing points along paths.
    - area_length_index_lst (list): List of tuples containing area and path length information.
    - lines_index_lst_width (list): List of tuples containing indexes and corresponding path lengths.
    """
    # Dictionary to store results
    _dict_wet_prop = {}
    endpoints_index_lst = []
    lines_index_lst = []
    lines_index_lst_width = []
    area_length_index_lst = []
    # Skeletonize the layer using Lee's method
    skeleton = skeletonize(layer.values, method='lee')
    # Label the skeletonized image to identify regions
    labeled_skeleton = label(skeleton)
    regions = regionprops(labeled_skeleton)
    # Convert the skeleton image to binary (-1 for background, 1 for foreground)
    skeleton = np.where(skeleton == 0, -1, 1)
    # Initialize the MCP_Geometric object for path finding
    m = MCP_Geometric(skeleton, fully_connected=True)
    # Set origin and lower left points
    origin = (0,0)
    lower_left = (skeleton.shape[-1], 0)
    # Iterate over regions (Pools) extracted from the labeled skeleton
    for region in regions:
        coord_list = [] # Reset the coordinate list for each region
        longest_distance = 0 # Reset the longest distance for each region
        if region.num_pixels > 2: # Process only if the region has more than 2 pixels
            # Find the furthest nodes within the region from the specified points
            closest_origin, farthest_origin = find_closest_farthest_points(origin, region)
            closest_lower_left, farthest_lower_left = find_closest_farthest_points(lower_left, region)
            points = [closest_origin, farthest_origin, closest_lower_left, farthest_lower_left]
            # Generate unique combinations of points
            unique_combinations = combine_points(points)
            # Loop through each unique point combination
            for combination in unique_combinations:
                # Find the most cost-efficient path between the combination points
                path, dist = find_most_cost_path(m, combination[0], combination[1])
                # Check if the current path's distance is greater than the longest recorded distance
                if dist > longest_distance:
                    longest_distance = dist # Update the longest distance
                    longest_path = path # Update the longest path
            # Update the lists and dictionary with calculated information
            endpoints_index_lst.extend([longest_path[0], longest_path[-1]])
            lines_index_lst.extend(longest_path)
            coord_list = np_positions_to_coord_point_list(longest_path, layer, coord_list)
            line_path = LineString(coord_list)
            _dict_wet_prop[region.label] = {
                'coord_start': Point(line_path.coords[0]),
                'coord_end': Point(line_path.coords[-1]),
                'length': line_path.length, 
                'linestring': line_path
            }
            area_length_index_lst.append(((area_array[path[0]]), line_path.length))
            lines_index_lst_width.append((longest_path, line_path.length))
    
    return _dict_wet_prop, endpoints_index_lst, lines_index_lst, area_length_index_lst, lines_index_lst_width
def find_closest_farthest_points(reference_point, region):
    """
    Finds the closest and farthest points within a given region to a reference point.

    Parameters:
    reference_point (tuple or list): The reference point's coordinates.
    region (Polygon or MultiPolygon): The region containing multiple coordinates.
    
    Returns:
    tuple: A tuple containing two coordinate points - the closest and farthest points to the reference point.
    """
    # Extract coordinates from the region
    coords = region.coords
    # Calculate distances between the reference point and all coordinates
    distances = cdist([reference_point], coords).flatten()
    # Find the index of the closest and farthest points based on distances
    closest_idx = np.argmin(distances)
    farthest_idx = np.argmax(distances)
    # Return the coordinates of the closest and farthest points
    return coords[closest_idx], coords[farthest_idx]
def combine_points(points):
    """
    Combines a list of points into unique pairs of combinations.

    Parameters:
    points (list): A list of points, where each point is represented as a tuple.

    Returns:
    list: A list of unique point combinations, where each combination is a list of point pairs.
    """
    unique_combinations = set()
    # Generate unique combinations of point indices and create sorted tuples
    for combo_indices in combinations(range(len(points)), 2):
        unique_combination = [tuple(points[i]) for i in combo_indices]
        unique_combinations.add(tuple(sorted(unique_combination)))
    # Convert the set of unique combinations back to a list of lists
    return [list(combination) for combination in unique_combinations]
def find_most_cost_path(m, start, end):
    """
    Find the most cost-efficient path using a cost matrix.

    Parameters:
    - m (CostMatrix): An instance of a cost matrix.
    - start (tuple): Starting point (row, column) in the cost matrix.
    - end (tuple): Ending point (row, column) in the cost matrix.

    Returns:
    - path (list): List of points representing the most cost-efficient path.
    - cost (float): Total cost of the found path.
    """
    costs, traceback_array = m.find_costs([start], [end])
    return m.traceback(end), costs[end]
def np_positions_to_coord_point_list(np_positions, layer, coord_list):
    """
    Convert positions in a NumPy array to a list of coordinate points.

    Parameters:
    - np_positions (numpy.ndarray): NumPy array containing positions (row, column).
    - layer (dict): Layer data containing 'x' and 'y' coordinates.
    - coord_list (list): List to store the resulting coordinate points.

    Returns:
    - coord_list (list): Updated list of coordinate points.
    """
    for np_position in np_positions:
        x = float(layer['x'][np_position[1]])
        y = float(layer['y'][np_position[0]])
        coord_list.append(Point(x, y))
    return coord_list
def calculate_metrics_AW(dict_area_2p, length_area_index_lst, layer, lines_index_lst_width):
    """
    Calculate Area-weighted metrics based on provided data and indices.

    Parameters:
    - dict_area_2p (dict): Dictionary containing area and perimeter information for regions.
    - length_area_index_lst (list): List of tuples containing length and area information.
    - layer (xarray.DataArray): Input DataArray.
    - lines_index_lst_width (list): List of tuples containing indexes and corresponding path lengths.

    Returns:
    - total_wet_area (float): Total wetted area.
    - total_wet_perimeter (float): Total wetted length.
    - AWMSI (float): Area-weighted Mean Shape Index.
    - AWMPA (float): Area-weighted Mean Pixel Area.
    - AWRe (float): Area-weighted Elongation Ratio.
    - AWMPW (float): Area-weighted Mean Pool Width.
    - AWMPL (float): Area-weighted Mean Pool Length.
    """
    # Calculate AWRe using provided length and area data
    AWRe = calculate_AWRe(length_area_index_lst)
    # Calculate AWMPW and AWMPL using provided width and length data
    AWMPW, AWMPL = calculate_AWMPW_AWMPL(layer, lines_index_lst_width)
    # Extract areas from the dictionary
    areas = [x[1] for _, x in dict_area_2p.items()]
    # Calculate total wet area
    total_wet_area= sum(areas)
    # Calculate total wet perimeter
    total_wet_perimeter = sum((x[0]) for key, x in dict_area_2p.items())
    # Calculate AWMSI
    AWMSI = (sum(((0.25*x[0]/np.sqrt(x[1]))*((x[1])/total_wet_area)) for key, x in dict_area_2p.items()))
    # Calculate AWMPA
    try:
        AWMPA = np.average(areas, weights=areas)
    except ZeroDivisionError:
        AWMPA = 0       

    return total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL
def calculate_AWRe(area_length_index_lst):
    """
    Calculate the Area-weighted Elongation Ratio (AWRe) based on area-length information.

    Parameters:
    - area_length_index_lst (list): List of tuples containing area and path length information.

    Returns:
    - AWRe (float): Area-weighted Elongation Ratio (AWRe) calculated from the provided data.

    This function computes the Area-weighted Elongation Ratio (AWRe) using area-length pairs
    representing the wet regions. It calculates Elongation Ratio (Re) for each area-length pair,
    sums them up, and then divides by the total sum of areas. The AWRe value is returned.

    If the total area sum is zero, the AWRe value is set to NaN to avoid division by zero.
    """
    # Calculate Elongation Ratio for each area-length pair 
    Re = []
    for x in area_length_index_lst:
        try:
            value = (2 * (np.sqrt(x[0]) / np.pi) / x[1]) * x[0]
            Re.append(value)
        except ZeroDivisionError:
            Re.append(np.nan)
    # Calculate the total sum of areas
    total_area_sum = np.nansum([x[0] for x in area_length_index_lst])
    if total_area_sum != 0:
        AWRe = np.nansum(Re) / total_area_sum
    else:
        AWRe = np.nan

    return AWRe
def calculate_AWMPW_AWMPL(layer, lines_index_lst_width):
    """
    Calculate the Area-weighted Mean Pool Width (AWMPW) and Area-weighted Mean Pool Length (AWMPL)
    based on width and length information of lines in a layer.

    Parameters:
    - layer (xarray.DataArray): Input DataArray.
    - lines_index_lst_width (list): List of tuples containing indexes and corresponding path lengths.

    Returns:
    - AWMPW (float): Area-weighted Mean Pool Width (AWMPW).
    - AWMPL (float): Area-weighted Mean Pool Length (AWMPL).
    """
    # Calculate euclidean distance transform array
    euc_dist_trans_array = ndimage.distance_transform_edt(layer.values)    
    width_length_list = []
    # Calculate width for each line and associate it with the corresponding length
    for idxw, length in lines_index_lst_width:
        idx, idy = zip(*idxw)
        width_seg = np.mean(euc_dist_trans_array[idx, idy])
        width_length_list.append((width_seg * 2 * layer.rio.transform()[0], length))
    # Initialize variables to store AWMPW and AWMPL           
    AWMPW = 0
    AWMPL = 0
    if width_length_list:
        # Extract widths, lengths, and areas (weights) from the width_length_list
        weights = [a for a, l in width_length_list]
        widths = [w for w, l in width_length_list]
        lengths = [l for w, l in width_length_list]
        # Calculate AWMPW by averaging widths with weights
        AWMPW = np.average(widths, weights=weights)
        # Calculate AWMPL by averaging lengths with weights
        AWMPL = np.average(lengths, weights=weights)   
    return AWMPW, AWMPL
def list_index(layer, index_lst):
    """
    Create a binary layer where specified indexes are set to 1.

    Parameters:
    - layer (xarray.DataArray): Input DataArray.
    - index_lst (list): List of tuples containing indexes.

    Returns:
    - binary_layer (numpy.ndarray): Binary layer with specified indexes set to 1.
    """
    # Create a new array filled with zeros, same shape as the input layer
    layer = np.zeros_like(layer.values)
    # If index_lst is not empty, set the specified indexes to 1 in the binary_layer
    if index_lst:
        ind_x, ind_y = zip(*index_lst)
        layer[ind_x, ind_y] = 1

    return layer
def save_shp(_dict_wet_prop, outdir, date, crs):
    """
    Save shapefiles for Pool length and endpoints based on the provided dictionary.

    Parameters:
    - _dict_wet_prop (dict): Dictionary containing pool properties information.
    - outdir (str): Directory to save shapefiles.
    - date (str): Date associated with the pool properties.
    - crs (CRS): Coordinate Reference System for the shapefiles.
    """
    # Create a directory to save shapefiles
    out_shp_dir = os.path.join(outdir, 'shp')
    create_new_dir (out_shp_dir, verbose=False)
    # Initialize lists to store data
    n_pools_data = []
    d_line_data = []
    # Loop through each property in the dictionary
    for d in _dict_wet_prop:
        start = {'date': date, 'length': _dict_wet_prop[d]['length'],  'geometry': _dict_wet_prop[d]['coord_start']}
        end = {'date': date, 'length': _dict_wet_prop[d]['length'], 'geometry': _dict_wet_prop[d]['coord_end']}
        line = {'date': date, 'length': _dict_wet_prop[d]['length'], 'geometry': _dict_wet_prop[d]['linestring']}
        
        n_pools_data.append(start)
        n_pools_data.append(end)
        d_line_data.append(line)
    # Create a GeoDataFrame for point geometries and save as shapefile
    n_pools_gdf = gpd.GeoDataFrame(n_pools_data, crs=crs)
    n_pools_gdf.to_file(os.path.join(out_shp_dir, 'dpoints_' + date + '.shp'))
    # Create a GeoDataFrame for line geometries and save as shapefile
    d_line_gdf = gpd.GeoDataFrame(d_line_data, crs=crs)
    d_line_gdf.to_file(os.path.join(out_shp_dir, 'rline_' + date + '.shp'))
def calculate_metrics_df(pd_metrics, section_length):
    """
    Calculate additional metrics and create a new DataFrame.
    APSEC: Wetted Area Percentage of Section.
    LPSEC: Wetted Length Percentage of Section.
    PF: Pool Fragmentation.
    PFL: Pool Longitudinal Fragmentation.

    Parameters:
    - pd_metrics (DataFrame): DataFrame containing calculated wetness metrics.
    - section_length (float): Length of the polygon section.

    Returns:
    - pdm (DataFrame): New DataFrame with calculated additional metrics.
    """
    # Create a copy of the input DataFrame with selected columns
    pdm = pd_metrics[['section_area', 'wet_area_km2', 'wet_perimeter_km2', 'wet_length_km', 'npools', 'AWMSI', 'AWRe', 
                            'AWMPA', 'AWMPL', 'AWMPW']].copy()
    # Add a new column for section length
    pdm['section_length'] = section_length
    # Calculate Wetted Area Percentage of Section (APSEC) and Wetted Length Percentage of Section (LPSEC)
    pdm.loc[:,'APSEC'] = ((pdm['wet_area_km2']/pdm['section_area'])*100).replace(np.inf, 0.0)
    pdm.loc[:,'LPSEC'] = ((pdm['wet_length_km']/section_length)*100).replace(np.inf, 0.0)
    
    # Calculate Pool Fragmentation (PF) and Pool Longitudinal Fragmentation (PFL)
    pdm.loc[:,'PF'] = (pdm['npools']/pdm['wet_area_km2'])
    pdm.loc[:,'PFL'] = (pdm['npools']/pdm['wet_length_km'])
    pdm[['PF', 'PFL']] = pdm[['PF', 'PFL']].replace([np.inf, -np.inf, np.nan], 0.0)

    # Convert the 'date' column to datetime format
    pdm['date'] = pd.to_datetime(pd_metrics['date'], format='%Y-%m-%d')
    # Order the columns
    col_order = ['date', 'section_area', 'section_length', 'wet_area_km2', 'wet_perimeter_km2', 'wet_length_km', 'npools', 
                    'AWMSI', 'AWRe', 'AWMPA', 'AWMPL', 'AWMPW', 'APSEC', 'LPSEC', 'PF', 'PFL']
    pdm = pdm[col_order]

    # Set specified columns to zero when 'npools' is zero
    cols_to_zero = ['wet_area_km2', 'wet_perimeter_km2', 'AWMSI', 'AWMPA', 
                    'AWMPL', 'AWMPW', 'APSEC', 'LPSEC', 'PF', 'PFL']
    pdm[cols_to_zero] = pdm[cols_to_zero].where(pdm['npools'] != 0, 0.0)
    return pdm 
## Pixel persistence metrics
def calculate_pixel_persistence(da_wmask):
    """
    Calculate pixel persistence of wet area in a DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): DataArray containing water mask over time.

    Returns:
    - p_area (xarray.DataArray): DataArray containing pixel persistence of wet area.
    """
    # Count the total number of observations in the time dimension
    total_obs = int(da_wmask['time'].count().values)
    # Calculate the number of wet observations for each pixel over time
    p_area = (da_wmask.sum(dim='time') / total_obs).astype('float32')
    # Replace pixels with zero wet area with a placeholder value
    p_area = xr.where(p_area > 0, p_area, -1)
    # Set dataarray nodata
    p_area.attrs['_FillValue'] = -1
    # Write the same CRS information as the input DataArray
    p_area = p_area.rio.write_crs(da_wmask.rio.crs)
    return p_area
def pixel_persistence_section(da_area, pdm, interval_ranges = None):
    """
    Calculate pixel persistence metrics for section.

    Parameters:
    - da_area (xarray.DataArray): DataArray containing wet area information.
    - pdm (DataFrame): DataFrame containing section metrics.
    - interval_ranges (list): List of tuples specifying lower and upper threshold ranges.

    Returns:
    - pdm (DataFrame): Updated DataFrame with pixel persistence metrics.

    This function calculates pixel persistence metrics for wet area using the provided DataArray and DataFrame.
    It calculates metrics such as mean pixel persistence (PP) and refuge area (RA).
    Additionally, if interval_ranges are provided, it calculates persistence intervals for specified ranges.
    The function returns the updated DataFrame.
    """
    # Get the pixel resolution from the DataArray
    resolution = da_area.rio.resolution()[0]
    # Calculate pixel persistence for the wet area
    p_area = calculate_pixel_persistence (da_area)
    # Calculate the mean pixel persistence of the section exceeding 0.25 (PP metric)
    pdm['PP'] = np.nanmean(xr.where(p_area >= 0.25, p_area, np.nan).values)
    # Calculate the refuge area (RA metric)
    pdm['RA'] = (((np.nansum(xr.where(p_area >= 0.9, 1, np.nan).values))*resolution*resolution)/10**6)
    # Calculate persistence interval metrics for specified ranges
    if interval_ranges:
        for lower_threshold, upper_threshold in interval_ranges:
            column_name = f'PP_{int(lower_threshold*100)}_{int(upper_threshold*100)}'
            pdm[column_name] = calculate_persistence_interval(p_area, lower_threshold, upper_threshold, resolution)
    return pdm
def calculate_persistence_interval(p_area, lower_threshold, upper_threshold, resolution):
    """
    Calculate persistence interval for specified threshold ranges.

    Parameters:
    - p_area (xarray.DataArray): DataArray containing wet area information.
    - lower_threshold (float): Lower threshold value for the persistence interval.
    - upper_threshold (float): Upper threshold value for the persistence interval.
    - resolution (float): Pixel resolution value.

    Returns:
    - interval_persistence (float): Persistence interval for the specified threshold range.

    This function calculates the persistence interval for the specified threshold range in the wet area DataArray.
    It multiplies the sum of pixels within the threshold range by the square of the resolution and converts the result to square kilometers (10^6 square meters).
    The function returns the persistence interval in square kilometers.
    """
    interval_persistence = np.nansum(xr.where((p_area >= lower_threshold) & (p_area <= upper_threshold), 1, np.nan).values)
    return interval_persistence * resolution * resolution / 10**6
