import os
# import xarray as xr
import waterdetect as wd
import dask
from dask import delayed
from dask.distributed import Client, as_completed
import pandas as pd

from src.utils_wd_batch import *
from src.utils_calc_metrics import *
from src.utils_gen_sections import *


## Module 1
def wd_batch(input_img, r_lines, ini_file=None, outdir=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True, return_da_array=True):
    """
    Process a batch of input images to detect water using WaterDetect.

    Args:
        input_img (str or xarray.DataArray): Directory path containing image files or xarray.DataArray.
        r_lines (str): River lines that define the rivers to be considered for water detection.
        ini_file (str): Path to the WaterDetect Initialization (.ini) file (default is None).
        buffer (int, optional): Buffer size for processing (default is 1000).
        img_ext (str, optional): Image file extension (default is '.tif').
        reg (float, optional): Registration method for image alignment (default is None).
        max_cluster (int, optional): Maximum number of clusters (default is None).
        export_tif (bool, optional): Whether to export GeoTIFF files (default is True).

    Returns:
        xarray.DataArray: A DataArray containing the water mask time series.
    """
    # Validate and preprocess the input images, bands, time list, and river corridor extent
    input_img, n_bands, time_lst, outdir = validate_inputs(input_img, r_lines, ini_file, outdir, buffer, img_ext, export_tif)
    # Update the WaterDetect configuration file based on the number of bands
    ini_file, bands = change_ini(ini_file, n_bands, reg, max_cluster)
    print('Input data validated.\n')

    # Configure WaterDetect using the updated .ini file
    config = wd.DWConfig(config_file=ini_file)
    config.detect_water_cluster
    
     # Set up and verify the output directories based on the given river lines and .ini file
    outdir, ini_file = setup_directories(ini_file, outdir, export_tif)
    print('\nExecuting...')

    max_retries = 3 # Maximum number of retries for processing images
    # Start parallel processing using Dask
    with Client(memory_limit=f'{get_total_memory()}GB') as client:
        # Initialize the retry counter
        retries = 0 
        # Prepare tuples of images and their corresponding dates
        to_retry = list(zip(input_img, time_lst))
        # Prepare a list to store delayed results for concatenation
        delayed_results = []
        # Scatter configuration and bands data across Dask workers
        scattered_config = client.scatter(config, broadcast=True)
        scattered_bands = client.scatter(bands, broadcast=True)

        # Retry loop for processing images
        while to_retry and retries < max_retries:
            # Map each image processing future to its corresponding date for tracking
            future_to_date = {}
            for img, date in to_retry:
                scattered_img = client.scatter(img)
                # Submit the image processing task to the Dask cluster
                future = client.submit(process_image_parallel, scattered_img, scattered_bands, scattered_config, export_tif, date, outdir)
                future_to_date[future] = (img, date) # Map future to its date

            failed_tasks = []  # List to store failed tasks for retrying
            total_images = len(future_to_date) # Total number of images to process
            processed_images = 0 # Counter for processed images

            # Iterate over the completed tasks
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
        da_wmask = concatenate(delayed_results).compute().sortby('time')
        print('Done!')
        return da_wmask
    else:
        return None

## Module 2 

# to do 

# CREATE FOLDER TO EXPORT RESULTS AND EXPORT RESULTS
# Issues

# -	Rivers of the same order, if running in parallel, will be merged – fix it
# -	Small polygons at the end vertices were clipped by higher river order – merge to adjacent polygon?
# -	Gaps inside a few polygons – why?

def estimate_section_for_cachtment(da_wmask, r_lines, str_order_col=None, initial_buffer=1000):
    """
    Estimates the size of sections for a given catchment area based on water mask data and river lines.

    This function orchestrates the process of validating and preprocessing the input data, 
    processing the line features, and then postprocessing the results to estimate the section size.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray or str): Input data, either as a DataArray or a directory path.
    r_lines (str): Path to a file containing river lines or similar linear features shapefile.
    str_order_col (str): The name of the column in `r_lines` representing stream order.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the estimated section sizes along with additional
                  processed geospatial data.
    """
    # Preprocess the input datasets and obtain necessary parameters for further processing
    initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list = validate_and_preprocess(da_wmask, r_lines, str_order_col, initial_buffer)
    # Process the line features based on the preprocessed data and obtain buffer list,
    # section lengths, widths, and original indices
    buff_list, section_lengths, section_widths, original_indices = process_lines(initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list)
    # Postprocess the results from line processing to generate a final GeoDataFrame
    # containing the estimated section sizes and other relevant data
    result_gdf = postprocess_results(r_lines_dissolved, buff_list, section_lengths, section_widths, original_indices, str_order_col, crs)
    # Return the final GeoDataFrame with the processed results
    return result_gdf

## Module 3
# to do 

## docstrings, comments


def estimate_section_for_pool(da_wmask, lower_thresh, higher_thresh, min_num_pixel):
    
    
    ## not retrieve other variables
    
    da_wmask, _, _ = validate_input_img(da_wmask, img_ext='.tif')

    PP = calculate_pixel_persistence(da_wmask)

    ## Processes PP based on lower and higher thresholds.
    pool_mask = process_masks(PP, lower_thresh, higher_thresh, min_num_pixel, radius=3)

    pools_aoi = process_pool_mask_to_gdf(pool_mask, PP)
    
    return pools_aoi


## Module 4
def calc_metrics(da_wmask, rcor_extent, section_length=None, outdir=None, img_ext='.tif', export_shp=False):
    """
    Calculate intermittent river metrics based on water mask data and river corridor extent.

    Args:
        da_wmask (str or xarray.DataArray): Directory path or xarray.DataArray containing Water masks.
        rcor_extent (str): Path to the river corridor extent (river sections) shapefile (.shp).
        section_length (float): Length of river sections for metrics calculation (km).
        outdir (str, optional): Output directory for results. If None, it will be generated based on the shapefile location (default is None).
        img_ext (str, optional): Image file extension (default is '.tif').
        export_shp (bool, optional): Whether to export results as shapefiles.

    Returns:
        None
    """
    # Validate and preprocess input data
    da_wmask, rcor_extent, crs = validate(da_wmask, rcor_extent, section_length, img_ext, module='calc_metrics')
    # Set up output directories and verify ini_file path
    outdir, section_outdir_folder = setup_directories_cm(rcor_extent, outdir)
    # Preprocess data and get arguments list for parallel processing
    args_list, da_wmask, rcor_extent = preprocess(da_wmask, rcor_extent, crs, section_outdir_folder, export_shp, section_length)
    # Create a Dask client with memory limit
    print('Calculating metrics...')
    
    # Create a Dask client with memory limit
    with Client(memory_limit=f"{get_total_memory()}GB") as client:
        # Scatter the input data to workers
        args_list = [client.scatter(args) for args in args_list]
        # Process polygons in parallel using Dask's delayed function
        delayed_results = [dask.delayed(process_polygon_parallel)(args) for args in args_list]
        results = dask.compute(*delayed_results)
    
    if export_shp:
        try:
            print('Exporting shapefiles...')
            save_shp(results, outdir, crs)
            print('Shapefiles exported!')
        except:
            pass
    
    # Calculate pixel persistence layer
    persistence_layer = calculate_pixel_persistence(da_wmask)
    # Create output directory for persistence raster
    outdir_persistence = os.path.join(outdir, '02.Persistence_raster')
    create_new_dir(outdir_persistence, verbose=False)
    print('Persistence raster will be exported to ', outdir_persistence)
    # Export pixel persistence raster
    persistence_layer.rio.to_raster(os.path.join(outdir_persistence, 'pixel_persistence.tif'), compress='lzw', lock=True)
    # Loop through processed polygons and combine metrics
    result = pd.DataFrame()
    for polygon in rcor_extent.iloc:
        pd_dir = os.path.join(section_outdir_folder, f'{polygon.name}', 'pd_metrics.csv')
        df = pd.read_csv(pd_dir, index_col=0)
        df['Section'] = polygon.name
        result = pd.concat([result, df], axis=0, ignore_index=True)
    # Save merged metrics to a CSV file
    result.to_csv(os.path.join(outdir, 'Calculated_metrics.csv'))

    print('Calculating metrics...All done!')

    return da_wmask, rcor_extent
