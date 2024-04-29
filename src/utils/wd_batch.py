import os
import re
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import dask.array as da
from dask import delayed
import waterdetect as wd
from shapely.geometry import mapping
from pyproj import CRS

def validate_inputs(input_img, r_lines, ini_file, outdir, buffer, img_ext, export_tif):
    """
    Validates input parameters for processing, ensuring compatibility and readiness for geospatial analysis.

    Args:
    - input_img (str or xarray.DataArray): Directory path containing image files or xarray.DataArray.
    - r_lines (str or geopandas.GeoDataFrame): Path to the river corridor extent shapefile (.shp) or GeoDataFrame.
    - ini_file (str): Path to the initialization (.ini) file.
    - outdir (str): Output directory path.
    - buffer (int): Buffer size for processing.
    - img_ext (str): Image file extension for filtering directory contents.
    - export_tif (bool): Flag indicating whether to export results as TIFF files.

    Returns:
    - tuple: Validated input values ready for processing.
    """
    # Validate river lines input
    assert isinstance(r_lines, gpd.GeoDataFrame) or (isinstance(r_lines, str) and r_lines.endswith('.shp')), \
        'River corridor extent (r_lines) must be a .shp file path or a geopandas.GeoDataFrame.'
    
    # Additional assertion: if r_lines is a GeoDataFrame and outdir is None
    assert not (isinstance(r_lines, gpd.GeoDataFrame) and outdir is None), \
        'When r_lines is a GeoDataFrame, an output directory (outdir) must be specified'
    
    # Set default outdir based on r_lines path if exporting to TIFF and outdir is not specified
    if export_tif and outdir is None:
        if isinstance(r_lines, str):  # When r_lines is a shapefile path
            outdir = os.path.join(os.path.dirname(os.path.abspath(r_lines)), 'results_iRiverMetrics')
    
    ## ADD CONDITIONAL TO USE DEFAULT INI 
    
    # Validate initialization file
    assert ini_file.endswith('.ini'), "Use WaterDetect .ini file"
    
    # Validate input images or DataArray
    input_img, n_bands, time_list = is_valid_input_img(input_img, r_lines, buffer, img_ext)        
    
    return input_img, n_bands, time_list, outdir

def is_valid_input_img(input_img, r_lines, buffer, img_ext):
    """
    Validates the input image data and prepares it for processing by checking its existence,
    compatibility with the specified river corridor extent, and applying a buffer if necessary.

    Args:
    - input_img (xarray.DataArray or str): Either an xarray DataArray or a folder path containing image files.
    - r_lines (str or geopandas.GeoDataFrame): Path to the river corridor extent shapefile (.shp)
                                                 or a GeoDataFrame containing the river lines.
    - buffer (int): Buffer size for processing.
    - img_ext (str): Expected file extension for image files when input_img is a directory path.

    Returns:
    -tuple: A tuple containing the validated and potentially modified input image data as an
            xarray DataArray, the number of bands in the input images, a list of time values
            extracted from the input data, and the validated path or GeoDataFrame of the
            river corridor extent.
    """               
    print('Validating input image data...')
    
    # Step 1: Validate the input image or directory path
    input_img, n_bands, crs = validate_input_img(input_img, img_ext)
    
    # Step 2: Validate the projection of r_lines against the input image's CRS
    r_lines = validate_input_projection(r_lines, crs)
    
    # Step 3: Apply buffer to r_lines and clip input image to this area
    input_img = buffer_clip_aoi(r_lines, buffer, input_img)
    
    # Step 4: Extract time values from the input image DataArray
    time_list = pd.DatetimeIndex(input_img['time'].values).strftime('%Y-%m-%d').tolist()

    return input_img, n_bands, time_list

def validate_input_img(input_img, img_ext):
    """
    Validates and prepares the input image data for analysis.

    Checks if the input is an xarray DataArray, an xarray Dataset, or a directory path pointing
    to image files, and processes it accordingly.

    Args:
    - input_img (xarray.DataArray, xarray.Dataset, or str): The input image data. Can be an xarray
      DataArray or Dataset, or a path to a directory containing image files.
    - img_ext (str): Image file extension to filter files in the directory. Used only if input_img is a directory.

    Returns:
    - Tuple containing the processed input data (as a DataArray), number of image bands, and the CRS.

    Raises:
    - ValueError: If input_img is not a DataArray, Dataset, or valid directory path.
    """
    # Check if the input is an xarray DataArray
    if isinstance(input_img, (xr.core.dataarray.DataArray)):
        # Validate and prepare the DataArray
        input_img, n_bands, crs = validate_data_array(input_img)
    # Check if the input is an xarray Dataset
    elif isinstance(input_img, (xr.core.dataset.Dataset)):
        # Validate and prepare the Dataset
        input_img, n_bands, crs = validate_dataset(input_img)
    # Check if the input is a directory path
    elif isinstance(input_img, str) and os.path.isdir(input_img):
        # Validate and prepare the directory path
        input_img, n_bands, crs = validate_input_folder(input_img, img_ext)
    else:
        raise ValueError('Input must be a valid xarray DataArray, Dataset, or directory path.')
    
    return input_img, n_bands, crs

def validate_data_array(input_datarray):
    """
    Validate and preprocess an xarray DataArray for geospatial analysis.

    Ensures the input DataArray has the required dimensions and bands, applies basic data cleaning,
    and restructures the DataArray for consistency in further processing.

    Args:
    - input_datarray (xarray.DataArray): Input image data as an xarray DataArray.

    Returns:
    - tuple: Processed DataArray, number of bands, and CRS.
    """
    # Determine the number of bands in the input DataArray
    n_bands = len(input_datarray.band)
    
    # Ensure the DataArray has a minimum of 4 bands (e.g., RGB and NIR)
    assert n_bands >= 4, 'Not enough data. Dataset must have at least 4 bands (B,G,R,NIR)'
    
    # Extract CRS from DataArray for geospatial operations
    crs = input_datarray.rio.crs
    
    # Define required dimensions for analysis
    required_dimensions = ['x', 'y', 'time', 'band']
    # Identify any missing dimensions in the input DataArray
    missing_dimensions = [dim for dim in required_dimensions if dim not in input_datarray.dims]
    assert not missing_dimensions, f"Missing required dimensions: {missing_dimensions}"

    # Squeeze unnecessary dimensions to simplify the DataArray
    dims_to_squeeze = [dim for dim in input_datarray.dims if dim not in required_dimensions]
    input_datarray = input_datarray.squeeze(drop=True, dim=dims_to_squeeze)
    
    # Apply basic data cleaning: filter outliers by setting values outside (0, 20000) to 0
    input_datarray = xr.where(((input_datarray > 0) & (input_datarray < 20000)), input_datarray, 0)
    
    # Restore CRS information lost during the 'where' operation
    input_datarray = input_datarray.rio.write_crs(crs)
    
    # Replace NaN values with 0 and update the '_FillValue' attribute accordingly
    input_datarray = input_datarray.fillna(0)
    input_datarray.attrs['_FillValue'] = 0
    
    # Transpose dimensions for consistent analysis, prioritizing time and band dimensions
    input_datarray = input_datarray.transpose('time', 'band', 'y', 'x').chunk(
                        {'time': -1, 'band': -1, 'x': 'auto', 'y': 'auto'})

    # Provide feedback on band structure for clarity
    if len(input_datarray.band) == 4:
        print('Reminder: Ensure bands are ordered as B,G,R,NIR for 4-band data.')
    else:
        print(f'{len(input_datarray.band)} bands found. Reminder: First 6 bands must be stacked as B,G,R,NIR,SWIR1,SWIR2')

    return input_datarray, n_bands, crs

def validate_dataset(input_dataset):
    """
    Validates and preprocesses an xarray Dataset, ensuring it has the necessary bands and dimensions.

    Converts the Dataset to a DataArray with a 'band' dimension for consistency in further processing.

    Args:
    - input_dataset (xarray.Dataset): Dataset containing multispectral satellite imagery.

    Returns:
    - A tuple of the preprocessed DataArray, number of bands, and the CRS of the input dataset.
    """
    # Determine the number of data variables, assumed to represent bands
    n_bands = len(input_dataset.data_vars)
    assert n_bands >= 4, 'Not enough data. Dataset must have at least 4 bands (B,G,R,NIR)'
    
    # Extract CRS from the dataset
    crs = input_dataset.rio.crs
    
    # Verify required spatial dimensions are present
    required_dimensions = ['x', 'y', 'time']
    missing_dimensions = [dim for dim in required_dimensions if dim not in input_dataset.dims]
    assert not missing_dimensions, f"Missing required dimensions: {missing_dimensions}"

    # Filter data to remove outliers and fill NaN values
    for var in input_dataset.data_vars:
        input_dataset[var] = xr.where(((input_dataset[var] > 0) & (input_dataset[var] < 20000)), 
                                      input_dataset[var], 0).fillna(0)
        input_dataset[var].attrs['_FillValue'] = 0

    # Ensure CRS information is retained after processing
    input_dataset.rio.write_crs(crs, inplace=True)

    # Standardize band names and stack data variables into a single DataArray
    var_names = list(input_dataset.data_vars)
    rename_bands = dict(zip(var_names, ['Blue', 'Green', 'Red', 'Nir'] + [f'Mir{i}' for i in range(1, n_bands-3)]))
    input_dataset = input_dataset.rename(rename_bands)
    final_data_array = input_dataset.to_array(dim='band').transpose('time', 'band', 'y', 'x')
    final_data_array.attrs['_FillValue'] = 0
    
    # Provide feedback on band structure
    if len(final_data_array.band) == 4:
        print('Reminder: Ensure bands are ordered as B,G,R,NIR for 4-band data.')
    else:
        print(f'{len(final_data_array.band)} bands found. Reminder: First 6 bands must be stacked as B,G,R,NIR,SWIR1,SWIR2')

    return final_data_array, n_bands, crs

def validate_input_folder(input_dir, img_ext):   
    """
    Processes a directory of raster images into an xarray DataArray with a time dimension.

    Validates the existence of image files, their CRS, spatial resolution, and band count. 
    Converts images to a unified CRS if necessary.

    Args:
    - input_dir (str): Directory containing image files.
    - img_ext (str): Image file extension (e.g., '.tif').

    Returns:
    - A tuple of the processed DataArray, number of bands, and CRS.
    """
    # List all image files in directory with specified extension
    img_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir) 
        if f.endswith(img_ext)
    ]
    
    # Ensure at least one image file is found
    if len(img_files) == 0:
        print(img_files)
        raise FileNotFoundError("No image files found in the input directory.")

    # Open first image to extract reference CRS and resolution
    with rxr.open_rasterio(img_files[0]) as first_raster:
        reference_crs = first_raster.rio.crs
        crs_obj = CRS(reference_crs)
        if crs_obj.is_geographic:
            reference_crs = first_raster.rio.estimate_utm_crs()
            first_raster = first_raster.rio.reproject(reference_crs)
            reference_resolution = first_raster.rio.resolution()[0] 
        else:
            reference_crs = reference_crs
            reference_resolution = first_raster.rio.resolution()[0]
    
    # Process and load each image, aligning to the reference CRS and resolution
    crs_lst, res_lst, band_lst, time_values, da_images = [], [], [], [], []
    # Initialize a generator for processing images
    image_generator = process_images(img_files, reference_crs, reference_resolution)

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

    # Check the number of bands and provide a reminder to stack them appropriately
    if band_lst[0] == 4:
        na_value = 0
        print(f'Reminder: 4 bands in source must be stacked as B,G,R,NIR')
    elif band_lst[0] == 1:
        na_value = -1
        print(f'Single band raster found as water mask')
    else:
        na_value = 0
        print(f'{band_lst[0]} bands found. Reminder: First 6 bands must be stacked as B,G,R,NIR,SWIR2')
    
    # Create an xarray DataArray with a 'time' dimension
    time = xr.Variable('time', time_values)
    # Concatenate image data along the 'time' dimension and sort by time
    input_img = xr.concat(da_images, dim=time
                          ).sortby('time'
                            ).chunk({'auto'})
    # Fill NaN values with na_value and set '_FillValue' attribute to na_value
    input_img = replace_nodata(input_img, na_value)         
    input_img.attrs['_FillValue'] = na_value

    return input_img, band_lst[0], crs

def process_images(img_files, reference_crs, reference_resolution):   
    """
    Processes a list of image files into a consistent format and extracts relevant information.

    Iterates over image files, reprojects them to a common CRS, matches spatial resolution,
    and extracts time information from file names.

    Args:
    - img_files (list of str): Paths to image files to be processed.
    - reference_crs (CRS object or str): The target CRS for reprojection.
    - reference_resolution (float): The target spatial resolution.

    Yields:
    - Each iteration yields processed image data and metadata as a tuple.
    """
    # Regular expression for extracting date from file names
    date_pattern = re.compile(r'\d{4}[-_]\d{2}[-_]\d{2}')  # Adjust the date pattern as needed
    
    # Loop through the provided image files
    for img in img_files:
        # Extract file name
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
        pimg = rxr.open_rasterio(img).chunk({'band': -1, 'x': 'auto', 'y': 'auto'})  # Chunking for Dask

        if pimg.rio.crs != reference_crs or pimg.rio.resolution()[0] != reference_resolution:
            pimg = pimg.rio.reproject(reference_crs, resolution=reference_resolution)

        crs = pimg.rio.crs
        res = pimg.rio.resolution()[0]
        n_bands = pimg.shape[0]
        
        # Yield the processed image data and associated information
        yield pimg, time_value, crs, res, n_bands

def replace_nodata(input_img, new_nodata_val):
    """
    Replace NoData values in a DataArray with a specified new value.

    Args:
    - input_img (xarray.DataArray): Input DataArray with potential NoData values.
    - new_nodata_val: New value to replace NoData values with.

    Returns:
    - xarray.DataArray: Updated DataArray with NoData values replaced.
    """
    # Check for existing NoData value in the attributes
    nodata_val = input_img.attrs.get('_FillValue', None)
    if nodata_val is None:
        nodata_val = input_img.attrs.get('nodatavals', None)
    
    # Replace existing NoData values with the new specified value
    if nodata_val is not None:
        # Replace the NoData value with the new value where it occurs
        input_img = input_img.where(input_img != nodata_val, new_nodata_val)
    else:
        # If NoData values are represented as NaNs, use fillna to replace them
        input_img = input_img.fillna(new_nodata_val)
    
    return input_img

def validate_input_projection(r_lines, img_crs):
    """
    Ensures the CRS of the river corridor extent matches the image CRS, reprojecting if necessary.

    Args:
    - img_crs (str or dict): CRS of the input imagery.
    - r_lines (str or geopandas.GeoDataFrame): Path to river corridor extent shapefile or GeoDataFrame.

    Returns:
    - geopandas.GeoDataFrame: River corridor extent with matching CRS.
    """
    # Load shapefile as GeoDataFrame if a path is provided, else use the provided GeoDataFrame
    r_lines = gpd.read_file(r_lines) if isinstance(r_lines, str) else r_lines
    
    # Validate CRS presence in river lines GeoDataFrame
    assert r_lines.crs is not None, "r_lines must have a defined CRS."
    
    # Reproject r_lines to match the image CRS if they differ
    if r_lines.crs != img_crs:
        print('Reprojecting river corridor extent to match image CRS.')
        # Reproject the shapefile to match the CRS of the image
        r_lines = r_lines.to_crs(img_crs)
    
    return r_lines

def buffer_clip_aoi(r_lines, buffer, input_img):
    """
    Applies a buffer to river lines and clips the input image to this buffered area.

    Args:
    - r_lines (geopandas.GeoDataFrame): GeoDataFrame representing river corridors.
    - buffer (float): Buffer size to apply around river lines, in units of r_lines' CRS.
    - input_img (xarray.DataArray): Satellite imagery or similar raster data to be clipped.

    Returns:
    - xarray.DataArray: The input image data clipped to the buffered river corridor extent.
    """
    # Apply buffer to the river lines
    r_lines_buffer = r_lines.buffer(buffer)
    
    # Ensure CRS compatibility between buffered extent and input image, reproject if necessary
    if r_lines_buffer.crs.to_epsg() != input_img.rio.crs.to_epsg():
        r_lines_buffer = r_lines_buffer.to_crs(input_img.rio.crs.to_epsg())
        
    # Clip the input image using the buffered extent
    input_img = input_img.rio.clip(r_lines_buffer.geometry.astype(object).apply(mapping), r_lines_buffer.crs)
    
    return input_img

def change_ini(ini_file, n_bands, reg, max_cluster):   
    """
    Modifies WaterDetect configuration file based on input parameters and number of bands.

    Args:
    - ini_file (str): Path to the .ini configuration file.
    - n_bands (int): Number of image bands to be considered.
    - reg (float): Regularization parameter for WaterDetect algorithm.
    - max_cluster (int): Maximum number of water clusters.

    Returns:
    - A tuple of the modified .ini file path and a list of band names.
    """
    # Read the existing .ini file content
    list_of_lines = open(ini_file, "r").readlines()
    
    # Default bands for WaterDetect processing
    bands = ['Blue', 'Green', 'Red', 'Nir']
    
    # Modify .ini file content based on the number of bands and given parameters
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
    
    # Adjust configurations based on the provided number of bands
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

    # Update max_cluster and regularization parameters in the .ini file
    list_of_lines[117] = 'regularization = ' + str(reg) + '\n'
    list_of_lines[124] = 'max_clusters = ' + str(max_cluster) + '\n'
    
    # Write the modified content back to the .ini file
    a_file = open(ini_file, "w")
    a_file.writelines(list_of_lines)
    a_file.close()

    return ini_file, bands

def setup_directories(ini_file, outdir, export_tif):
    """
    Configures directories for output files and ensures the initialization file exists.

    Args:
    - ini_file (str): Path to the initialization (.ini) file. If None, defaults to a standard path.
    - outdir (str): Base directory for saving output files. A subdirectory for TIFF files is created if export_tif is True.
    - export_tif (bool): Flag indicating whether TIFF files will be exported, affecting directory setup.

    Returns:
    - Tuple containing paths to the output directory and the .ini file.
    """
    # Default initialization file setup
    if ini_file is None:
        # Set default .ini file path if not provided
        ini_file = os.path.join(os.getcwd(), 'docs', 'WaterDetect.ini')
    # Output directory setup for TIFF export
    if export_tif:
        create_new_dir(outdir, verbose=False)
        outdir = os.path.join(outdir, 'wd_batch')
        create_new_dir(outdir)

    return outdir, ini_file

def create_new_dir(outdir, verbose=True):
    """
    Creates a new directory at the specified location.
    
    Args:
    - outdir (str): The path of the directory to be created.
    - verbose (bool): Whether to show the print statement or not (default is True).
    
    Returns:
    - None
    """
    # Create output file
    os.makedirs(outdir, exist_ok=True)
    if verbose:
        print(f'{outdir} created to export results.\n')

def get_total_memory():
    """
    Calculates 95% of the total system memory to determine an optimal memory limit for processing.

    Useful for setting memory constraints in applications that can be configured to use a maximum amount of RAM.

    Returns:
    - float: Available system memory to use (95% of total) in gigabytes.
    """
    # Obtain total system memory in bytes
    total_memory_bytes = psutil.virtual_memory().total
    # Convert total memory from bytes to gigabytes and calculate 95% of it
    total_memory_gb = total_memory_bytes // (1024 ** 3)  # Convert bytes to gigabytes
    
    return total_memory_gb*0.95

def process_image_parallel(img_target, bands, config, export_tif, date, outdir):
    """
    Applies WaterDetect algorithm on a given image to identify water bodies, optionally exporting results.

    Utilizes specified spectral bands and algorithm configuration to process satellite imagery,
    generating a binary water mask. Optionally exports the result as a GeoTIFF.

    Args:
    - img_target (xarray.DataArray): Satellite image data.
    - bands (list of str): Spectral bands to use in the analysis.
    - config (WaterDetect.DWConfig): Configuration parameters for WaterDetect.
    - export_tif (bool): Flag to export water mask as GeoTIFF.
    - date (str): Acquisition date of the satellite image.
    - outdir (str): Directory to save the GeoTIFF, if exporting.

    Returns:
    - xarray.DataArray: Binary water mask generated by WaterDetect.
    """
    # Generate water mask using WaterDetect
    water_xarray = wd_mask(img_target, bands, config)
    
    # Export the water mask to a GeoTIFF file if 'export_tif' is True
    if export_tif:
        # Convert date string to a standardized format
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        # Export water mask to GeoTIFF with compression
        water_xarray.rio.to_raster(os.path.join(outdir, str(date) + '.tif'), compress='lzw')
    
    return water_xarray

def wd_mask(img_target, bands, config):
    """
    Generates a water mask using the WaterDetect algorithm on multispectral imagery.

    Utilizes specified bands and configuration to identify water bodies within the image.

    Args:
    - img_target (xarray.DataArray): The input satellite image.
    - bands (list of str): Bands selected for water detection.
    - config (WaterDetect.DWConfig): Configuration for the WaterDetect algorithm.

    Returns:
    - xarray.DataArray: The generated water mask.
    """
    # Prepare data for WaterDetect
    arrays, water_xarray = create_wd_dict(img_target, bands)
    
    # Adjust arrays based on available bands, using zeros for missing bands as a temporary workaround
    if len(bands) == 4:
        # Simulate missing bands with zeros for compatibility with WaterDetect expectations
        arrays['mndwi'] = np.zeros_like(arrays['Green'])
        arrays['mbwi'] = np.zeros_like(arrays['Green'])
        arrays['Mir2'] = np.zeros_like(arrays['Green'])
        invalid_mask = (arrays['Nir'] == 0)
    else:
        # Use available bands, assuming all necessary bands are present
        invalid_mask = (arrays['Mir2'] == 0)
    
    # Initialize WaterDetect with the prepared data
    wmask = wd.DWImageClustering(
        bands=arrays, 
        bands_keys=['mndwi', 'ndwi', 'Mir2'], 
        invalid_mask=invalid_mask, 
        config=config, 
        glint_processor=None
        )  
    
    # Execute water detection
    wmask.run_detect_water()
    
    # Convert WaterDetect's NumPy water mask to a Dask array for integration with xarray
    dask_water_mask = da.from_array(wmask.water_mask, chunks='auto')
    # Create a new xarray DataArray for the water mask, preserving original metadata
    water_xarray = water_xarray.copy(data=dask_water_mask)
    water_xarray.rio.write_nodata(-1, inplace=True)  

    return water_xarray

def create_wd_dict (img_target, bands):
    """
    Prepares input data for WaterDetect by rescaling and organizing into a dictionary.

    Args:
    - img_target (xarray.DataArray): Target image data for water detection.
    - bands (list of str): Names of the spectral bands to be used.

    Returns:
    - tuple: Contains a dictionary with band data and a dummy xarray.DataArray.
    """
    # Specify the division factor for rescaling, if necessary     
    div_factor = 1
    # Rescale the target image data
    cube = (img_target/div_factor)
    # Create a dictionary mapping band names to their respective rescaled data arrays
    arrays = {layer: cube[i].values for i, layer in enumerate(bands)}
    # Select a single band from the target image to create a dummy DataArray
    # This DataArray will be used as a template for metadata and spatial references
    water_xarray = img_target.isel(band=1)  

    return arrays, water_xarray

@delayed
def concatenate(results_with_dates):
    """
    Concatenates a list of xarray DataArrays along a new time dimension.

    Args:
    - results_with_dates (list of tuples): Each tuple contains an xarray DataArray and its associated date.

    Returns:
    - xarray.DataArray: Concatenated DataArray with a new 'time' dimension.
    """
    # Unpack the results and dates from the input list of tuples
    final_results, final_dates = zip(*results_with_dates)
    # Create a Pandas Index for the dates, which will be used as the 'time' dimension
    # Concatenate the DataArrays along the new 'time' dimension
    return xr.concat(final_results, dim=pd.Index(final_dates, name='time')).chunk(chunks='auto')