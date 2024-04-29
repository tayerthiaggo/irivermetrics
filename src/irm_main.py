import os
import waterdetect as wd
from dask import delayed
from dask.distributed import Client, as_completed
from tqdm import tqdm
import pandas as pd

from .utils import wd_batch
from .utils import calc_metrics
from .utils import gen_sections

## Module 1
def waterdetect_batch(input_img, r_lines, ini_file=None, outdir=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True, return_da_array=False):
    """
    Detects water bodies in a batch of images using the WaterDetect algorithm and generates a water mask time series.

    Args:
    - input_img (str, xarray.DataArray or xarray.DataSet): Path to the directory containing image files, an xarray.DataArray or an xarray.DataSet of images.
    - r_lines (str): Path to the file defining river lines for water detection.
    - ini_file (str, optional): Path to the WaterDetect initialization (.ini) file. Defaults to None.
    - outdir (str, optional): Output directory for results. Defaults to None.
    - buffer (int, optional): Buffer size in pixels for processing. Defaults to 1000.
    - img_ext (str, optional): Extension of image files to process. Defaults to '.tif'.
    - reg (float, optional): Registration method for image alignment. Defaults to None.
    - max_cluster (int, optional): Maximum number of clusters for the algorithm. Defaults to None.
    - export_tif (bool, optional): Flag to export results as GeoTIFF files. Defaults to True.
    - return_da_array (bool, optional): Flag to return results as an xarray.DataArray. Defaults to False.

    Returns:
    - xarray.DataArray or None: Returns a DataArray containing the water mask time series if `return_da_array` is True; otherwise returns None.
    """
    # Initial validation and preprocessing of input parameters
    input_img, n_bands, time_lst, outdir = wd_batch.validate_inputs(input_img, r_lines, ini_file, outdir, buffer, img_ext, export_tif)
    
    # Adjust the initialization file based on input image properties.
    ini_file, bands = wd_batch.change_ini(ini_file, n_bands, reg, max_cluster)
    print('Input data validated.\n')

    # Set up WaterDetect configuration for water detection
    config = wd.DWConfig(config_file=ini_file)
    config.detect_water_cluster
    
    # Ensure output directories are properly set up for storing results
    outdir, ini_file = wd_batch.setup_directories(ini_file, outdir, export_tif)
    print('\nExecuting...')

    # Define retry logic for robust image processing
    max_retries = 3 # Allows up to 3 retry attempts for processing images
    with Client(memory_limit=f'{wd_batch.get_total_memory()}GB') as client:
        retries = 0 # Track the number of retry attempts
        # Prepare tuples of images and their corresponding dates
        to_retry = list(zip(input_img, time_lst))
        delayed_results = [] # Collect results for final concatenation
        
        # Distribute shared configuration and bands to workers
        scattered_config = client.scatter(config, broadcast=True)
        scattered_bands = client.scatter(bands, broadcast=True)

        # Process images with retry logic for handling failures
        while to_retry and retries < max_retries:
            future_to_date = {} # Track processing futures to their corresponding dates.
            for img, date in to_retry:
                scattered_img = client.scatter(img)
                # Submit image for processing
                future = client.submit(wd_batch.process_image_parallel, scattered_img, scattered_bands, scattered_config, export_tif, date, outdir)
                future_to_date[future] = (img, date) # Map future to its date

            failed_tasks = []  # Track tasks that fail for retry
            total_images = len(future_to_date) # Total number of images to process
            processed_images = 0 # Count of successfully processed images

            # Monitor and collect results from processing tasks
            for future in as_completed(future_to_date):
                try:
                    result = future.result() # Retrieve the result from the future
                    _, date = future_to_date[future] # Extract the corresponding date
                    # Append the result and its date to delayed_results using Dask's delayed function
                    delayed_result = delayed(lambda x, y: (x, y))(result, pd.to_datetime(date))
                    delayed_results.append(delayed_result)

                    processed_images += 1
                    # Update the progress of processed images
                    print(f'Processed {processed_images}/{total_images} images', end='\r')
                except Exception as e:
                    # Handle exceptions and append failed tasks for retrying
                    failed_img, failed_date = future_to_date[future]  # Correctly unpack the tuple
                    failed_tasks.append((failed_img, failed_date))  # Re-append the failed task correctly
                    print(f'Error processing image for date {failed_date}: {e} (we will try again)')

            # Prepare for the next iteration with failed tasks
            to_retry = failed_tasks
            retries += 1
        
        # Check if all tasks are completed
        if not to_retry:
            print('All tasks completed successfully.')
        else:
            print(f'Failed to process {len(to_retry)} image(s) after {max_retries} retries.')

    if return_da_array:
        print('Working on results...')
        # Compute the concatenated results and sort by time
        da_wmask = wd_batch.concatenate(delayed_results).compute().sortby('time')
        print('Done!')
        return da_wmask
    else:
        return None

## Module 2
def estimate_section_for_pool(da_wmask, lower_thresh, higher_thresh, min_num_pixel=4, outdir=None, export_shp=False, img_ext='tif'):
    """
    Estimates sections for pools within a given water mask based on persistence and size thresholds.

    This function processes a time series of water mask data to identify stable water presence,
    applies defined persistence thresholds to distinguish potential pool areas from other water bodies,
    and filters these areas by size to ensure they meet a minimum pixel requirement. The identified pool
    areas are then converted into spatial polygons, with overlapping polygons merged to form concise pool
    sections.

    Args:
    - da_wmask (str, xarray.DataArray or xarray.DataSet): Path to the directory, xarray.DataArray or xarray.DataSet containing the water mask time series data.
    - lower_thresh (float): Lower threshold for pixel persistence to identify water presence.
    - higher_thresh (float): Higher threshold for persistent water areas to refine pool candidates.
    - min_num_pixel (int): The minimum number of pixels required for a cluster to be considered a pool.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the estimated pool sections, their locations, and geometries.
    """
    rcor_extent=None
    section_length=None
    
    if export_shp:
        assert outdir is not None, 'if export_shp is True. must provide an output directory.'
    
    # Validate the input water mask to ensure compatibility
    da_wmask, rcor_extent, _ = calc_metrics.validate(da_wmask, rcor_extent, section_length, img_ext, module='pool')
    
    
    da_wmask = binary_dilation_ts(da_wmask)
    # Calculate pixel persistence to identify stable water presence across the time series
    args_list = calc_metrics.preprocess(da_wmask, rcor_extent, outdir)
    
    PP = args_list[0][2]
    
    print('Estimating pool maximum extent...')
    # Generate a pool mask by applying persistence thresholds and filtering by size
    # This step distinguishes potential pool areas from the surrounding water body
    pool_mask = gen_sections.process_masks(PP, lower_thresh, higher_thresh, min_num_pixel, radius=3)

    # Convert the identified pool areas from the mask into spatial polygons
    # This step translates pixel clusters into geometries within a GeoDataFrame
    pools_aoi = gen_sections.process_pool_mask_to_gdf(pool_mask, PP)
    
    # Merge any overlapping polygons to consolidate closely situated pools into single sections
    # This helps reduce redundancy and simplifies the representation of pool areas
    pools_aoi = gen_sections.merge_overlapping_polygons(pools_aoi)
    
    if export_shp:
        pools_aoi.to_file(os.path.join(outdir, 'pools_aoi.shp'))
    
    print('Done!')
    
    return pools_aoi

# Module 3
def calculate_metrics(da_wmask, rcor_extent, section_length=None, min_pool_size=2, outdir=None, img_ext='.tif', export_shp=False, return_da_array=False):   
    """
    Calculates ecohydological metrics for intermittent river sections using water mask data and predefined river corridor extents.

    Args:
    - da_wmask (str or xarray.DataArray): The path to the directory containing water mask images or an xarray.DataArray of water masks.
    - rcor_extent (str): The file path to the river corridor extent shapefile, defining the sections for metrics calculation.
    - section_length (float, optional): The length of each river section for the metrics calculation in kilometers. Defaults to None.
    - outdir (str, optional): The output directory where results will be saved. If not specified, a directory will be generated near the shapefile location.
    - img_ext (str, optional): The file extension for input water mask images. Defaults to '.tif'.
    - export_shp (bool, optional): A flag indicating whether to export the metrics calculation results as shapefiles. Defaults to False.
    - return_da_array (bool, optional): A flag indicating whether to return the data array along with the calculation results. Defaults to False.

    Returns:
        pandas.DataFrame or (pandas.DataFrame, xarray.DataArray, str): Returns a DataFrame containing the calculated metrics. If `return_da_array` is True, it also returns the data array of water masks and the river corridor extent path.
    """
    # Initial validation and preprocessing of inputs to ensure compatibility and correctness
    da_wmask, rcor_extent, crs = calc_metrics.validate(da_wmask, rcor_extent, section_length, img_ext, module='calc_metrics')
    
    da_wmask = binary_dilation_ts(da_wmask).astype('int8')
    # Prepare the output directory for storing calculation results
    outdir = calc_metrics.setup_directories_cm(rcor_extent, outdir)  
    
    # return da_wmask, rcor_extent, crs
    
    # Initialize a Dask client to manage distributed computing resources
    with Client(memory_limit=f"{wd_batch.get_total_memory()}GB") as client:
        print(f"Dask Dashboard available at: {client.dashboard_link}")
        
        # Preprocess data and generate a list of arguments for parallel processing of each river section
        args_list = calc_metrics.preprocess(da_wmask, rcor_extent, outdir)
        
        # return args_list
        print('Calculating metrics...')

        # Submit each task to the Dask cluster for parallel execution
        futures = [client.submit(calc_metrics.process_polygon_parallel, 
                                 (feature, cliped_da_wmask, clipped_PP, section_length, export_shp, outdir, min_pool_size)) 
                   for feature, cliped_da_wmask, clipped_PP in args_list]
        
        # Collect results from completed tasks
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Polygons'):
            result = future.result()  # Retrieve task result
            if result is not None:
                results.append(result)
        
    if export_shp:
        # Attempt to export results as shapefiles, if enabled
        try:
            print('Exporting shapefiles...')
            calc_metrics.save_shp(results, outdir, crs)
            print('Shapefiles exported!')
           
        except Exception as e:
            print(f"Failed to export shapefiles due to: {e}")
    
    all_metrics = pd.concat([res[0] for res in results if res is not None], ignore_index=True)
    # Save merged metrics to a CSV file
    all_metrics.to_csv(os.path.join(outdir, 'Calculated_metrics.csv'))
    
    print('Metric Calculation Successfull.')
    
    # Conditional return based on the `return_da_array` flag
    if return_da_array:
        # Return the DataFrame alongside the original data array and corridor extent path
        return all_metrics, da_wmask, rcor_extent 
    else:
        return all_metrics
    
from dask_image.ndmorph import binary_dilation
import numpy as np
import xarray as xr
def binary_dilation_ts(da_wmask):
    
    # Define a structuring element for dilation with the given radius; this is a square matrix of ones
    structure = np.ones((1, 2, 2), dtype=bool)  

    # Step 1: Create a mask for valid data (where data is not -1)
    valid_data_mask = da_wmask != -1

    # Step 2: Apply binary dilation only on valid data (convert -1 to 0 temporarily for dilation)
    # Temporarily set no data values to 0 to apply dilation
    temp_data = da_wmask.where(valid_data_mask, 0)

    # Define the structuring element for dilation
    structure = np.ones((1, 3, 3), dtype=bool)

    # Apply binary dilation on the valid data
    # Note: Ensure to keep the original shape [time, y, x] in the structuring element if necessary
    dilated_data = binary_dilation(temp_data.data, structure=structure).astype(int)

    # Step 3: Create the final DataArray, restoring the -1 values
    da_wmask = xr.where(valid_data_mask, dilated_data, -1)
    
    return da_wmask