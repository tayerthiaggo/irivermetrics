import os
import gc
import pandas as pd
import waterdetect as wd
from dask import delayed, compute
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar

from .utils import wd_batch
from .utils import calc_metrics

## Module 1
def waterdetect_batch(input_img, r_lines, outdir=None, ini_file=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True, return_da_array=False):
    """
    Detects water bodies in a batch of images using the WaterDetect algorithm and generates a water mask time series.

    Args:
    - input_img (str, xarray.DataArray or xarray.DataSet): Path to the directory containing image files, an xarray.DataArray or an xarray.DataSet.
    - r_lines (str or geopandas.GeoDataFrame): Path to the file defining river lines for water detection or GeoDataFrame.
    - ini_file (str, optional): Path to the WaterDetect initialization (.ini) file. Defaults to None.
    - outdir (str, optional): Output directory for results. Defaults to None.
    - buffer (int, optional): Buffer size in pixels for processing. Defaults to 1000m.
    - img_ext (str, optional): Extension of image files to process. Defaults to '.tif'.
    - reg (float, optional): Regularization of the normalized spectral indices. Defaults to None.
    - max_cluster (int, optional): Maximum number of possible targets to be identified as clusters. Defaults to None.
    - export_tif (bool, optional): Flag to export results as GeoTIFF files. Defaults to True.
    - return_da_array (bool, optional): Flag to return results as an xarray.DataArray. Defaults to False.

    Returns:
    - xarray.DataArray or None: If return_da_array is True, returns a DataArray containing the water mask time series; otherwise, returns None.
    """
    print('\nExecuting waterdetect_batch.\n')
    # Validate input data and preprocess parameters
    input_img, n_bands, time_lst, outdir, ini_file, _ = wd_batch.validate_inputs(input_img, r_lines, ini_file, outdir, buffer, img_ext, export_tif)
    
    # Adjust the initialization file based on input image properties
    ini_file, bands = wd_batch.change_ini(ini_file, n_bands, reg, max_cluster)
    print('Input data validated.\n')

    # Initialize WaterDetect configuration for the water detection process
    config = wd.DWConfig(config_file=ini_file)
    config.detect_water_cluster
    
    # Setup output directories to store results
    outdir = wd_batch.setup_directories(outdir, export_tif)
    print('Executing...')

    # Define retry mechanism to handle image processing failures
    max_retries = 3 # Allow up to 3 retries
    with Client(memory_limit=f'{wd_batch.get_total_memory()}GB') as client:
        print(f"Dask Dashboard available at: {client.dashboard_link}")
        retries = 0 # Track the number of retry attempts
        # Prepare tuples of images and their corresponding dates
        to_retry = list(zip(input_img, time_lst))
        delayed_results = [] # Collect results for final concatenation
        
        # Distribute shared configuration and bands to workers
        scattered_config = client.scatter(config, broadcast=True)
        scattered_bands = client.scatter(bands, broadcast=True)

        # Process images with retry logic for handling failures
        while to_retry and retries < max_retries:
            future_to_date = {} # Track processing futures to their corresponding dates
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

# Module 2
def calculate_metrics(da_wmask, rcor_extent=None, outdir=None, section_length=None, min_pool_size=2, img_ext='.tif', export_shp=False, export_PP = False):   
    """
    Calculates ecohydrological metrics for defined sections of intermittent rivers using water mask data.
    These metrics assist in understanding the water availability and ecological conditions of the river sections.

    Args:
    - da_wmask (str or xarray.DataArray): Path to a directory containing water mask images or an xarray.DataArray of water masks.
    - rcor_extent (str): Path to the river corridor extent shapefile, which defines the sections for metrics calculation. If None, the entire extent of the water mask images will be used.
    - section_length (float, optional): The desired length of each river section for metrics calculation in kilometers. Defaults to None.
    - min_pool_size (int, optional): Minimum size of water pools to consider in the analysis in pixels. Defaults to 2 pixels.
    - outdir (str, optional): Directory to save the output results. Defaults to the same location as the input shapefile if not specified.
    - img_ext (str, optional): File extension for the input water mask images. Defaults to '.tif'.
    - export_shp (bool, optional): Whether to export part of the results as shapefiles (pool wetted area/line, start/end midpoints). Defaults to False.
    - export_PP (bool, optional): Whether to export pixel persistence data as a raster. Defaults to False.

    Returns:
    - pandas.DataFrame: DataFrame containing the calculated metrics. 
    """   
    # Validate and preprocess inputs
    da_wmask, rcor_extent, crs, pixel_size, outdir = calc_metrics.validate(da_wmask, rcor_extent, outdir, img_ext)
    da_wmask, rcor_extent = calc_metrics.preprocess(da_wmask, rcor_extent)
                  
    print('Calculating metrics...')
    
    # Calculate Pixel Persistence
    PP = calc_metrics.calculate_pixel_persistence(da_wmask).chunk('auto')
    if export_PP:
        PP.rio.to_raster(os.path.join(outdir, 'full_scene_persistence.tif'), compress='lzw')
    
    # Process Pixel Persistence metrics
    PP_metric_tasks = [calc_metrics.process_PP_metrics (PP, feature, pixel_size)
                        for _, feature in rcor_extent.iterrows()]       
    
    PP_metrics = compute(*PP_metric_tasks)
    PP_df = pd.DataFrame(PP_metrics, columns=['section', 'PP_mean', 'RA_area_km2'])
    
    # Free memory after PP metrics computation
    del PP_metric_tasks, PP_metrics, PP
    gc.collect()
    
    # Prepare metric tasks using Dask
    metric_tasks = [calc_metrics.process_feature_time(feature, da_wmask.sel(time=time_value), min_pool_size, 
                                                      section_length, pixel_size, export_shp, crs)
                    for _, feature in rcor_extent.iterrows() for time_value in da_wmask.time.data]

    with ProgressBar():
        # Compute all metric tasks using Dask
        computed_tasks = compute(*metric_tasks)
       
    del metric_tasks
    gc.collect()
   
    # Extracting the metric results from computed tasks
    attributes_results = pd.concat([task[0] for task in computed_tasks if task[0] is not None], ignore_index=True)
    # Group by date and section and apply the calculation function
    metrics_df = attributes_results.groupby(
        ['date', 'section']).apply(
        calc_metrics.process_metrics, include_groups=False).reset_index()
    
    metrics_results = pd.merge(metrics_df, PP_df, on='section', how='left')
    metrics_results.to_csv(os.path.join(outdir, 'Calculated_metrics.csv'))
    
    if export_shp:
        geo_dfs = [pd.concat([task[i] for task in computed_tasks], ignore_index=True) for i in range(1, 4)]
        for df, name in zip(geo_dfs, ['Points', 'Lines', 'Polygons']):
            df.to_file(os.path.join(outdir, f'result_{name}.shp'))

    return metrics_results