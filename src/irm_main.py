import os
import xarray as xr
import waterdetect as wd
import dask
from dask import delayed
import dask.distributed
from dask.distributed import Client

from src.irm_utils import *

## Module 1
def wd_batch(input_img, rcor_extent, ini_file=None, outdir=None, buffer=1000, img_ext='.tif', reg=None, max_cluster=None, export_tif=True):
    """
    Process a batch of input images to detect water using WaterDetect.

    Args:
        input_img (str or xarray.DataArray): Directory path containing image files or xarray.DataArray.
        rcor_extent (str): River lines or polygon shapefile that define the rivers to be considered for water detection.
        ini_file (str): Path to the WaterDetect Initialization (.ini) file (default is None).
        buffer (int, optional): Buffer size for processing (default is 1000).
        img_ext (str, optional): Image file extension (default is '.tif').
        reg (float, optional): Registration method for image alignment (default is None).
        max_cluster (int, optional): Maximum number of clusters (default is None).
        export_tif (bool, optional): Whether to export GeoTIFF files (default is True).

    Returns:
        xarray.DataArray: A DataArray containing the water mask time series.
    """
    # Check if outdir is not specified
    if outdir == None:
        # If not, set the outdir to a default directory named 'results_iRiverMetrics' within the directory of rcor_extent
        outdir = os.path.join(os.path.dirname(os.path.abspath(rcor_extent)), 'results_iRiverMetrics')
    # Check if ini_file is not specified
    if ini_file == None:
        # If not, set the ini_file to a default WaterDetect.ini file located in the 'docs' directory of the current working directory
        ini_file = os.path.join(os.getcwd(), 'docs', 'WaterDetect.ini')
    create_new_dir(outdir)
    outdir = os.path.join(outdir, 'wd_batch')
    create_new_dir(outdir)
    # Create a directory to export GeoTIFF files if 'export_tif' is True
    if export_tif:
        tif_out_dir = os.path.join(outdir, 'wmask_tif')
        create_new_dir(tif_out_dir)
    # Validate input images
    input_img, n_bands, time_lst, rcor_extent = validate_inputs(input_img, img_ext, rcor_extent, ini_file, buffer)
    # Edit the WaterDetect configuration (.ini) file based on the number of bands
    ini_file, bands = change_ini(ini_file, n_bands, reg, max_cluster)
    # Configure WaterDetect using the edited .ini file
    config = wd.DWConfig(config_file=ini_file)
    config.detect_water_cluster
    print('Executing...')
    with Client(memory_limit=f"{get_total_memory()}GB") as client:
        # Create a list of delayed tasks for processing images in parallel
        wd_delayed_results = [delayed(process_image_parallel)(img_target, bands, config, export_tif, date, outdir) for img_target, date in zip(input_img, time_lst)]
        wd_results = dask.compute(*wd_delayed_results)
        # Process collected results
        wd_lst = process_results(wd_results)
    print('Working on results...')
    # Create a time DataArray using the dates from the time list
    time_layers = xr.Variable('time', time_lst)
    # Concatenate the water mask results to create a water mask time series
    da_wmask = xr.concat(wd_lst, dim=time_layers).chunk(chunks='auto')
    return da_wmask
## Module 2
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
    da_wmask, rcor_extent = validate(da_wmask, rcor_extent, section_length, img_ext, expected_geometry_type='Polygon')
    # Determine the output directory
    if outdir == None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(rcor_extent)), 'results_iRiverMetrics')
    # Create the main output directory  
    create_new_dir(outdir)
    # Create output directories for metrics and section results
    outdir = os.path.join(outdir, 'metrics')
    create_new_dir(outdir, verbose=False)
    print('Results from Metrics module will be exported to ', outdir)

    section_outdir_folder = os.path.join(outdir, '02.Section_results')
    create_new_dir(section_outdir_folder, verbose=False)
    print('Section results will be exported to ', section_outdir_folder)

    # Preprocess data and get arguments list for parallel processing
    args_list, da_wmask, rcor_extent = preprocess(da_wmask, rcor_extent, section_outdir_folder, export_shp, section_length)

    # Create a Dask client with memory limit
    print('Calculating metrics...')
    # Create a Dask client with memory limit
    with Client(memory_limit=f"{get_total_memory()}GB") as client:
        # Scatter the input data to workers
        args_list = [client.scatter(args) for args in args_list]
        # Process polygons in parallel using Dask's delayed function
        delayed_results = [dask.delayed(process_polygon_parallel)(args) for args in args_list]
        dask.compute(*delayed_results)

    # Calculate pixel persistence layer
    persistence_layer = calculate_pixel_persistence(da_wmask)
    # Create output directory for persistence raster
    outdir_persistence = os.path.join(outdir, '01.Persistence_raster')
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
    result.to_csv(os.path.join(outdir, 'Result_merged_sections.csv'))

    print('Calculating metrics...All done!')

    return da_wmask, rcor_extent
