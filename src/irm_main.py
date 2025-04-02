import os
import gc
import pandas as pd
import waterdetect as wd
from dask import delayed, compute
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar, Profiler


from .utils import wd_batch
from .utils import calc_metrics

import dask.dataframe as dd

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

## Module 2
def calculate_metrics(da_wmask, 
                    rcor_extent=None, 
                    outdir=None, 
                    section_length=None,
                    section_name_col=None, 
                    min_pool_size=2, 
                    img_ext='.tif', 
                    export_shp=False, 
                    export_PP=False, 
                    fill_nodata=True):   
    """
    [Your existing docstring]
    """
    # Validate and preprocess inputs
    da_wmask, rcor_extent, section_length, crs, pixel_size, outdir = calc_metrics.validate(da_wmask, 
                                                                           rcor_extent, 
                                                                           outdir, 
                                                                           section_length,
                                                                           img_ext,
                                                                           section_name_col)
    da_wmask, rcor_extent = calc_metrics.preprocess(da_wmask, 
                                                    rcor_extent, 
                                                    fill_nodata)
    
    # return da_wmask
    date_list = pd.to_datetime(da_wmask.time.data).strftime('%Y-%m-%d').to_list()       
    features = list(rcor_extent.iterrows())
    
    if len(features) < 3:
        batch_size = 6
    else:
        batch_size = 36

    print(f'Using {batch_size} date batches for processing.')
    
    summary_tasks = []
    export_tasks = []
    
    for _, feature in features:
        # Preprocess the feature
        pre_task = calc_metrics.preprocess_feature(
            da_wmask, feature, section_name_col, pixel_size, min_pool_size
        )
        
        # Generate batches of dates
        time_step_batches = calc_metrics.batch_date_list(date_list, batch_size=batch_size)
        
        # Create batch tasks
        batch_tasks = [
            calc_metrics.process_feature_batch(
                preprocessed=pre_task,
                batch_dates=batch_dates,
                pixel_size=pixel_size,
                section_length=section_length,
            )
            for batch_dates in time_step_batches
        ]
        
        summary_task = delayed(pd.concat)(batch_tasks, ignore_index=True)
        summary_tasks.append(summary_task)
        
        if export_shp:
            # Create export_shapefiles task
            export_task = calc_metrics.export_shapefiles(
                preprocessed=pre_task,
                outdir=outdir,
                pixel_size=pixel_size,
                summary_ddf=summary_task,
                crs=crs,
                min_pool_size=min_pool_size
            )
            export_tasks.append(export_task)
        
    print('Computing metrics... (this may take a while)')
    with ProgressBar():
        # Compute all tasks in parallel
        tasks_results = compute(*summary_tasks)
        # Concatenate all results into a single DataFrame
        attributes_results = pd.concat(tasks_results, ignore_index=True)
    
    metrics_df = attributes_results.groupby(['date', 'section'], observed=False
        ).apply(calc_metrics.process_metrics, include_groups=False
        ).sort_values(by=['section', 'date']
        ).reset_index()
    
    metrics_df['date'] = pd.to_datetime(metrics_df['date'])
    # metrics_df['section'] = metrics_df['section'].astype('int32')
    metrics_df['npools'] = metrics_df['npools'].astype('int32')
    
    metrics_df.to_csv(os.path.join(outdir, 'irm_metrics.csv'))
        
    if export_shp:
        print('Exporting shapefiles...')
        with ProgressBar():
            # Compute all tasks in parallel
            export_results = compute(*export_tasks)
        
        # export_results is a list of tuples: (polygons_gdf, lines_gdf, points_gdf)
        polygons_list, lines_list, points_list = zip(*export_results)
        
        # Concatenate all GeoDataFrames
        concatenated_polygons = pd.concat(polygons_list, ignore_index=True)
        concatenated_lines = pd.concat(lines_list, ignore_index=True)
        concatenated_points = pd.concat(points_list, ignore_index=True)
        
        # Export concatenated GeoDataFrames as single shapefiles
        concatenated_polygons.to_file(f"{outdir}/irm_Polygons.shp")
        concatenated_lines.to_file(f"{outdir}/irm_Lines.shp")
        concatenated_points.to_file(f"{outdir}/irm_Points.shp")
    
    if export_PP:
        print('Exporting pixel persistence raster...')
        PP = calc_metrics.calculate_pixel_persistence(da_wmask)
        #### ADD MASK BEFORE EXPORTING -- CREATE FUNCTION -- 
        # SEE update_nodata_in_rcor_extent AND fill_nodata_darray
        PP.rio.to_raster(os.path.join(outdir, 'Pixel_Persistence.tif'), compress='lzw')
    
    print('\nAll Done!')
    
    return metrics_df