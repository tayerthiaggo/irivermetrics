import os
from dask import delayed
from dask.distributed import Client, as_completed
from tqdm import tqdm
import pandas as pd

from .utils import wd_batch
from .utils import calc_metrics
from .utils import gen_sections

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
    
    
    da_wmask = calc_metrics.binary_dilation_ts(da_wmask)
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