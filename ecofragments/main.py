import os
import pandas as pd
from dask import delayed, compute
from dask.diagnostics import ProgressBar

from .utils import calc_metrics

import warnings

def calculate_metrics(da_wmask, 
                    rcor_extent=None, 
                    outdir=None, 
                    section_length=None,
                    section_name_col=None, 
                    min_patch_size=2, 
                    img_ext='.tif', 
                    export_shp=False, 
                    export_PP=False, 
                    fill_nodata=True):   

    warnings.filterwarnings("ignore", category=UserWarning)

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
            da_wmask, feature, section_name_col, pixel_size, min_patch_size
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
                min_patch_size=min_patch_size
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
    metrics_df['n_patches'] = metrics_df['n_patches'].astype('int32')
    
    metrics_df.to_csv(os.path.join(outdir, 'ecof_metrics.csv'))
        
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
        concatenated_polygons.to_file(f"{outdir}/ecof_Polygons.shp")
        concatenated_lines.to_file(f"{outdir}/ecof_Lines.shp")
        concatenated_points.to_file(f"{outdir}/ecof_Points.shp")
    
    if export_PP:
        print('Exporting pixel persistence raster...')
        PP = calc_metrics.calculate_pixel_persistence(da_wmask)
        #### ADD MASK BEFORE EXPORTING -- CREATE FUNCTION -- 
        # SEE update_nodata_in_rcor_extent AND fill_nodata_darray
        PP.rio.to_raster(os.path.join(outdir, 'Pixel_Persistence.tif'), compress='lzw')
    
    print('\nAll Done!')
    
    return metrics_df