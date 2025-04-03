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



class irm:
    def __init__(self, 
                    min_pool_size=2, 
                    img_ext='.tif', 
                    export_tif=True,                        
                    export_shp=False, 
                    export_PP=False, 
                    fill_nodata=True):
        
        """
        Initializes the irm instance with default parameters.

        Args:
        - buffer (int): Buffer size in pixels for processing. Defaults to 1000.
        - img_ext (str): Extension of image files to process. Defaults to '.tif'.
        - export_tif (bool): Flag to export results as GeoTIFF files. Defaults to True.
        """
        self.min_pool_size = min_pool_size
        self.img_ext = img_ext
        self.export_tif = export_tif                       
        self.export_shp = export_shp 
        self.export_PP = export_PP 
        self.fill_nodata = fill_nodata
        
    def calculate_metrics(self,
                        da_wmask, 
                        rcor_extent=None, 
                        outdir=None, 
                        section_length=None,
                        section_name_col=None):

        """
        [Your existing docstring]
        """    
        
        # Validate and preprocess inputs
        da_wmask, rcor_extent, section_length, crs, pixel_size, outdir = calc_metrics.validate(da_wmask, 
                                                                            rcor_extent, 
                                                                            outdir, 
                                                                            section_length,
                                                                            self.img_ext,
                                                                            section_name_col)
        da_wmask, rcor_extent = calc_metrics.preprocess(da_wmask, 
                                                        rcor_extent, 
                                                        self.fill_nodata)
    
        date_list = pd.to_datetime(da_wmask.time.data).strftime('%Y-%m-%d').to_list()       
        batch_size = 36
        features = list(rcor_extent.iterrows())
                
        summary_tasks = []
        export_tasks = []
        
        for _, feature in features:
            # Preprocess the feature
            pre_task = calc_metrics.preprocess_feature(
                da_wmask, feature, section_name_col, pixel_size, self.min_pool_size
            )
            
            # Generate batches of dates
            time_step_batches = calc_metrics.batch_date_list(date_list, batch_size=batch_size)
            
            # Create batch tasks
            batch_tasks = [
                calc_metrics.process_feature_batch(
                    preprocessed=pre_task,
                    batch_dates=batch_dates,
                    pixel_size=pixel_size,
                    outdir=outdir,
                    export_PP=self.export_PP,
                    export_shp=self.export_shp,
                    section_length=section_length,
                    crs=crs
                )
                for batch_dates in time_step_batches
            ]
            
            summary_task = delayed(pd.concat)(batch_tasks, ignore_index=True)
            summary_tasks.append(summary_task)
            
            if self.export_shp:
                # Create export_shapefiles task
                export_task = calc_metrics.export_shapefiles(
                    preprocessed=pre_task,
                    outdir=outdir,
                    pixel_size=pixel_size,
                    summary_ddf=summary_task,
                    crs=crs,
                    min_pool_size=self.min_pool_size
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
            
        if self.export_shp:
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
        
        if self.export_PP:
            print('Exporting pixel persistence raster...')
            PP = calc_metrics.calculate_pixel_persistence(da_wmask)
            PP.rio.to_raster(os.path.join(outdir, 'Pixel_Persistence.tif'), compress='lzw')
        
        print('\nAll Done!')
        
        return metrics_df