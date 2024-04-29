import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
import dask.array as da
from dask import delayed
from dask_image.ndmorph import binary_dilation
import rasterio
from rasterio import features
import shapely.geometry
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, box
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric
from itertools import combinations
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict
from odc.geo.xr import xr_reproject
from pyproj import CRS

from src.utils import wd_batch

def validate(da_wmask, rcor_extent, section_length, img_ext, module):
    """
    This function performs a series of checks and transformations on the input data to ensure they are in the correct format and structure for further processing.

    Parameters:
    - da_wmask (xarray.core.dataarray.DataArray or str): Water mask data as a DataArray or path to water mask images.
    - rcor_extent (geopandas.geodataframe.GeoDataFrame or str): geodatafrmae or path to river corridor extent shapefile.
    - section_length (int or float): Length of river section for analysis, required for 'calc_metrics' module.
    - img_ext (str): File extension for image processing, e.g., '.tif'.
    - module (str): Processing module name, e.g., 'calc_metrics'.

    Returns:
    - tuple: Validated water mask DataArray, processed river corridor extent data, and CRS.
    
    Raises:
    - AssertionError: If input validations fail, indicating incorrect data types, missing files, or unsupported formats.
    
    Notes:
    - Replaces NaN values in the DataArray with -1 and sets '_FillValue' attribute.
    - Validates and adjusts the CRS of the DataArray and shapefile to ensure compatibility.
    """   
    print('Checking input data...')
    
    # Validate da_wmask as DataArray or directory path
    assert (isinstance(da_wmask, xr.core.dataarray.DataArray) or 
           (isinstance(da_wmask, str) and os.path.isdir(da_wmask))), 'Invalid input. da_wmask must be a valid DataArray or a directory path'

    # Process directory input to create DataArray
    if isinstance(da_wmask, str) and os.path.isdir(da_wmask):
        # Create DataArray and validade rasters within the provided folder
        da_wmask, n_bands, crs = wd_batch.validate_input_folder(da_wmask, img_ext)
    
    # Ensure da_wmask is a DataArray with a CRS attribute
    if isinstance(da_wmask, xr.core.dataarray.DataArray):
        crs = da_wmask.rio.crs

    # Replace NaN values and validate CRS
    da_wmask = wd_batch.replace_nodata(da_wmask, -1)
    da_wmask.attrs['_FillValue'] = -1
    da_wmask, crs = validate_data_array_cm(da_wmask, crs)
    
    if rcor_extent is None:
        # Extract bounding box coordinates
        bounds = da_wmask.rio.bounds()
        # Create a polygon from the bounding box
        bbox_polygon = box(*bounds)
        # Create a GeoDataFrame from the polygon
        rcor_extent = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=crs)
    
    if rcor_extent is not None:
        # Validate rcor_extent shapefile for correct file extension and CRS
        rcor_extent = validate_shp_input(rcor_extent, crs, module)
    
    # Check for section length if required by the module
    if module == 'calc_metrics':
        # Check if section length is present
        assert section_length != None, 'Invalid input. Section length not found.' 

    print('Input data validated.')
    return da_wmask, rcor_extent, crs

def validate_data_array_cm(da_wmask, crs):
    """
    Adjusts a DataArray with water mask data for proper dimensions and CRS.

    Ensures the DataArray has required spatial dimensions ('x', 'y') and optionally 'time',
    and adjusts the CRS to a uniform projection if necessary, favoring UTM for geographic coordinates.

    Parameters:
    - da_wmask (xarray.DataArray): DataArray containing water mask data.
    - crs (str, dict, or pyproj.CRS): CRS of the DataArray. Can be a PROJ string, a dictionary, or a pyproj.CRS object.

    Returns:
    - tuple: The adjusted DataArray and its corresponding CRS.
    
    Raises:
    - AssertionError: For missing required dimensions or absence of CRS information.

    Note:
    - Reprojects geographic (lat/lon) CRS to UTM for consistent spatial analysis.
    """   
    # Validate required dimensions are present
    required_dimensions = ['x', 'y', 'time']
    # Check for missing dimensions by comparing required dimensions with DataArray dimensions
    missing_dimensions = [dim for dim in required_dimensions if dim not in da_wmask.dims]
    assert not missing_dimensions, f"Missing dimensions: {', '.join(missing_dimensions)}"
    
    # Drop unnecessary dimensions
    dims_to_squeeze = [dim for dim in da_wmask.dims if dim not in required_dimensions]
    # Squeeze and drop unnecessary dimensions
    da_wmask = da_wmask.squeeze(drop=True, dim=dims_to_squeeze)
    
     # Ensure CRS is provided
    assert crs != None, 'Invalid input. da_wmask CRS not found.'
    
    # Process CRS and reproject if necessary
    crs_obj = CRS(crs)
    # Reproject DataArray to UTM CRS if original CRS is geographic (lat/lon)
    if crs_obj.is_geographic:
        # Estimate and reproject to UTM for geographic CRS
        crs = da_wmask.rio.estimate_utm_crs()
        da_wmask = xr_reproject(da_wmask, crs)
        
    da_wmask = da_wmask.chunk('auto')
    da_wmask.attrs['_FillValue'] = -1
    
    return da_wmask, crs

def validate_shp_input(rcor_extent, img_crs, module):
    """
    Validates and loads a shapefile for river corridor analysis, ensuring compatibility with the specified module.

    Parameters:
    - rcor_extent (str or geopandas.GeoDataFrame): Path to the shapefile or a GeoDataFrame to be validated.
    - img_crs (str or dict): The coordinate reference system of the image data to ensure CRS compatibility.
    - module (str): The processing module name, affecting validation criteria ('generate_sections' or 'calc_metrics').

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing the validated and potentially reprojected shapefile data.

    Raises:
    - AssertionError: If the shapefile does not meet required criteria based on the specified module.
    """   
    # Validate input type and extension for rcor_extent
    assert isinstance(rcor_extent, gpd.GeoDataFrame) or (isinstance(rcor_extent, str) and rcor_extent.endswith('.shp')), \
        'rcor_extent input must be a .shp file path or a GeoDataFrame.'

    # Load and validate shapefile projection
    rcor_extent = wd_batch.validate_input_projection(rcor_extent, img_crs)

    # Validation based on the specified module
    if module == 'generate_sections':
        # Validate for LineString geometries
        valid_geometry = any(isinstance(geom, (LineString, MultiLineString)) for geom in rcor_extent.geometry)
        assert not rcor_extent.empty and valid_geometry, \
            'Shapefile must contain valid LineString geometries for generate_sections module.' 

    elif module == 'calc_metrics':
        # Validate for Polygon geometries
        valid_geometry = any(isinstance(geom, (Polygon, MultiPolygon)) for geom in rcor_extent.geometry)
        assert not rcor_extent.empty and valid_geometry, \
            'Shapefile must contain valid Polygon geometries for calc_metrics module.'

    return rcor_extent

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

def setup_directories_cm(rcor_extent, outdir):
    """
    Prepares and returns the directory path for exporting results from the Calculate Metrics module.

    This function creates a structured directory hierarchy under the specified output directory. If no output
    directory is provided, it defaults to a 'results_iRiverMetrics' folder in the same directory as the rcor_extent file.

    Parameters:
    - rcor_extent (str): Path to the river corridor extent shapefile, used to derive the default output directory location.
    - outdir (str, optional): Base path for the output directory. If None, defaults to a subdirectory next to the rcor_extent file.

    Returns:
    - str: The path to the created directory where metric results will be stored.

    Notes:
    - The function ensures the creation of a 'metrics' subdirectory within the specified or derived output directory.
    """
    # Validate input parameters
    if outdir == None:
        if not os.path.isfile(rcor_extent):
            raise FileNotFoundError(f"The specified rcor_extent path does not exist: {rcor_extent}")
        outdir = os.path.join(os.path.dirname(os.path.abspath(rcor_extent)), 'results_iRiverMetrics')
        
    # Determine the base output directory
    outdir = os.path.join(outdir, 'metrics')
    # Create output folder
    wd_batch.create_new_dir(outdir, verbose=False)
    print('Results from Calculate Metrics module will be exported to ', outdir)
    
    return outdir

def preprocess(da_wmask, rcor_extent, outdir):   
    """
    Performs preprocessing steps on input water mask and river corridor extent shapefile to prepare for analysis.

    The preprocessing includes clipping the water mask and extent shapefile to matching dimensions, filling no-data
    values in the water mask, calculating a pixel persistence layer, and preparing arguments for subsequent processing.

    Parameters:
    - da_wmask (xarray.DataArray): Input DataArray containing the water mask.
    - rcor_extent (geopandas.GeoDataFrame): GeoDataFrame representing the river corridor extent.
    - outdir (str): Output directory path for saving intermediate files, such as the pixel persistence layer.

    Returns:
    - list: A list of arguments prepared for further processing, including preprocessed DataArray and GeoDataFrame.

    Notes:
    - This function saves the pixel persistence layer as a TIFF file in the specified output directory.
    """
    print('Preprocessing...')
    # Step 1: Clip input data and extent to match dimensions
    da_wmask, rcor_extent = match_input_extent(da_wmask, rcor_extent)
    
    # Step 2: Fill nodata values in the DataArray
    da_wmask = fill_nodata(da_wmask, rcor_extent)    

    # Step 3: Calculate and save pixel persistence layer
    persistence_layer = calculate_pixel_persistence(da_wmask)
    persistence_layer.rio.to_raster(os.path.join(outdir, 'pixel_persistence.tif'), compress='lzw')
    
    # Step 4: Prepare arguments for further processing
    args_list = prepare_args(da_wmask, persistence_layer, rcor_extent)
    
    return args_list

def match_input_extent(da_wmask, rcor_extent):
    """
    Clips an input DataArray and a GeoDataFrame to their overlapping region to ensure spatial consistency.

    This function identifies the overlapping region between the spatial extent of a water mask DataArray
    and a river corridor extent shapefile. Both the DataArray and the GeoDataFrame are then clipped to this
    overlapping region to facilitate consistent, focused analysis.

    Parameters:
    - da_wmask (xarray.DataArray): Input DataArray containing the water mask.
    - rcor_extent (geopandas.GeoDataFrame): GeoDataFrame representing the river corridor extent.

    Returns:
    - xarray.DataArray: The DataArray clipped to the overlapping region.
    - geopandas.GeoDataFrame: The GeoDataFrame filtered to the overlapping region.

    Raises:
    - ValueError: If no overlapping region is found or if the clipping results in empty datasets.
    """
    # Define bounding box of the DataArray
    minx, miny, maxx, maxy = da_wmask.rio.bounds()
    # Filter GeoDataFrame to features intersecting with the bounding box
    rcor_extent = rcor_extent[rcor_extent.geometry.intersects(box(minx, miny, maxx, maxy))]
    
    if rcor_extent.empty:
        raise ValueError("No overlapping region found between da_wmask and rcor_extent.")

    # # Clip DataArray to the bounding box of filtered extent
    # da_wmask = da_wmask.rio.clip(rcor_extent.geometry, all_touched=True)
        
    return da_wmask, rcor_extent

def fill_nodata(da_wmask, rcor_extent):
    """
    Fills NoData values in a DataArray, specifically targeting NoData within a specified river corridor extent,
    and then using parallel processing with Dask to fill remaining NoData values across the entire DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): The DataArray to process, assumed to have a 'time' dimension.
    - rcor_extent (geopandas.GeoDataFrame): The geographical extent within which NoData values are specifically updated.

    Returns:
    - xarray.DataArray: The DataArray with NoData values filled.
    """
    if rcor_extent is not None:
        # Update 'no data' values within the specified river corridor extent
        da_wmask = update_nodata_in_rcor_extent(da_wmask, rcor_extent)
    crs = da_wmask.rio.crs
    
    # Ensure data is 8-bit before applying fill_nodata_layer function
    da_wmask = da_wmask.astype('int8')
    
    # Apply fill_nodata_layer function with Dask's map_overlap to the updated DataArray
    filled_data = da.map_overlap(
        fill_nodata_layer,
        da_wmask.data,
        depth={'time': 2},  # Buffer size for two layers ahead/behind
        boundary={'time': 'reflect'}, # Reflect boundary for edge chunks
        dtype=da_wmask.dtype
    )
       
    # Convert the filled Dask array back to an xarray DataArray
    da_wmask = xr.DataArray(filled_data, coords=da_wmask.coords, dims=da_wmask.dims)
    da_wmask.attrs['_FillValue'] = -1
    
    # Replace remaining NoData values (2) with -1 
    da_wmask = xr.where(da_wmask != 2, da_wmask, -1)
    # Update CRS information
    da_wmask.rio.write_crs(crs, inplace=True)
    
    return da_wmask

def update_nodata_in_rcor_extent(da_wmask, rcor_extent):
    """
    Update 'no data' values within a specified river corridor extent in an xarray DataArray. 
    The river corridor extent is defined by a shapefile, which is dissolved to a single geometry,
    rasterized, and then used to identify and update 'no data' areas within the xarray DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): The xarray DataArray to be updated.
    - rcor_extent (geopandas.GeoDataFrame): The GeoDataFrame representing the river corridor extent.

    Returns:
    - xarray.DataArray: Updated xarray DataArray where 'no data' areas within the river corridor extent are updated.
    """   
    # Dissolve the shapefile to create a single, unified geometry
    gdf = rcor_extent.dissolve() 
    # Ensure the polygon is in the same CRS as the xarray
    gdf = gdf.to_crs(da_wmask.rio.crs.to_string())
    
    # Define rasterization transformation and shape
    transform = rasterio.transform.from_bounds(*da_wmask.rio.bounds(), len(da_wmask.coords['x']), len(da_wmask.coords['y']))
    raster_shape = (len(da_wmask.coords['y']), len(da_wmask.coords['x']))
    
    # Rasterize the dissolved and transformed GeoDataFrame
    rasterized_polygon = features.rasterize(shapes=gdf.geometry, out_shape=raster_shape, transform=transform, fill=0, default_value=1) #, dtype='int8')
    # Convert the rasterized polygon to a Dask array
    rasterized_polygon_dask = da.from_array(rasterized_polygon, chunks='auto')
    
    # Apply the rasterized mask to identify and update NoData values within the river corridor
    river_corridor_raster = da_wmask.isel(time=0).copy()
    river_corridor_raster = river_corridor_raster.copy(data=rasterized_polygon_dask)
    # Identify areas with -1 in the water mask within the river corridor
    no_data_areas = (da_wmask == -1) & (river_corridor_raster == 1)
    # Ensure the updated DataArray retains the original CRS
    da_wmask = xr.where(no_data_areas, 2, da_wmask).rio.write_crs(da_wmask.rio.crs, inplace=True)
    
    return da_wmask

def fill_nodata_layer(chunk):   
    """
    Fill NoData values in a chunk of a DataArray.

    For each layer in the chunk, NoData values (marked as 2) are filled by checking
    values from adjacent layers. The function first attempts to fill from the
    immediate next and previous layers, and if unsuccessful, checks two layers ahead and behind.

    Parameters:
    - chunk (numpy.ndarray): A chunk of the DataArray being processed.

    Returns:
    - numpy.ndarray: The chunk with NoData values filled.
    """
    # Total number of layers in the chunk
    num_layers = chunk.shape[0] 

    # Iterate through each layer, focusing on filling NoData values from adjacent layers
    for num in range(1, num_layers - 2):
        # Process only layers that contain NoData values (marked as 2)
        if np.any(chunk[num] == 2):
            # Check and fill NoData values from adjacent layers
            for offset in [1, 2, -1, -2]:
                adj_layer = num + offset # Index of the adjacent layer
                # Ensure the adjacent layer index is within the chunk's range
                if 0 <= adj_layer < num_layers:
                    # Create a mask for valid filling positions
                    valid_mask = (chunk[num] == 2) & (chunk[adj_layer] != 2)
                    # Fill NoData values from the adjacent layer where valid
                    chunk[num][valid_mask] = chunk[adj_layer][valid_mask]

    # Handle edge layers separately to avoid index errors
    for num in [0, num_layers - 1]:
        if np.any(chunk[num] == 2):
            # Determine the offset range based on whether it's the first or last layer
            for offset in [1, 2] if num == 0 else [-1, -2]:  # Check +1, +2 for first and -1, -2 for last layer
                adj_layer = num + offset
                # Check if the adjacent layer is within the valid range
                if 0 <= adj_layer < num_layers:
                    valid_mask = (chunk[num] == 2) & (chunk[adj_layer] != 2)
                    chunk[num][valid_mask] = chunk[adj_layer][valid_mask]
    return chunk

def prepare_args(da_wmask, persistence_layer, rcor_extent):
    """
    Prepares a list of arguments for parallel processing of geospatial data, focusing on clipping the data
    to the extents defined by river corridor geometries.

    Parameters:
    - da_wmask (xarray.DataArray): Input raster data array representing a water mask.
    - persistence_layer (xarray.DataArray): DataArray representing the pixel persistence layer.
    - rcor_extent (geopandas.GeoDataFrame): GeoDataFrame containing river corridor extent polygons.

    Returns:
        list: A list of arguments, each a tuple with (feature geometry, clipped water mask DataArray, clipped pixel persistence DataArray).
    """   
    args_list = []
    for index, feature in rcor_extent.iterrows():
        # Extract bounds for each polygon
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        # Use the bounds to clip the da_wmask and persistence_layer
        col_mask = (da_wmask.x >= xmin) & (da_wmask.x <= xmax)
        row_mask = (da_wmask.y >= ymin) & (da_wmask.y <= ymax)
        
        # Clip the data arrays to the bounds of the current feature
        cliped_da_wmask = da_wmask.sel(x=col_mask, y=row_mask)
        clipped_PP = persistence_layer.sel(x=col_mask, y=row_mask)
        
        # Append the polygon and the clipped DataArrays as a tuple to the args list   
        args_list.append((feature, cliped_da_wmask, clipped_PP))
    
    return args_list

@delayed
def process_single_layer(layer, polygon, section_length, export_shp, min_pool_size):
    
    dict_area_2p, area_array = calculate_area_perimeter(layer, min_pool_size)
    
    _dict_wet_prop, area_length_index_lst, lines_index_lst_width = calculate_connectivity(layer, area_array, min_pool_size)
           
    total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL = calculate_metrics_AW(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width)
    
    date = pd.to_datetime(str(layer.time.values)).strftime('%Y-%m-%d')
    res_metrics_data, res_points_data, res_lines_data = compile_results(date, polygon, section_length, export_shp, 
                                                                        total_wet_area, total_wet_perimeter, AWMSI, 
                                                                        AWMPA, AWRe, AWMPW, AWMPL, _dict_wet_prop)
    
    return res_metrics_data, res_points_data, res_lines_data

def calculate_area_perimeter(layer, min_pool_size):
    dict_area_2p, area_array = calculate_pool_area_and_perimeter(layer, min_pool_size)
    return dict_area_2p, area_array

def calculate_connectivity(layer, area_array, min_pool_size):
    _dict_wet_prop, area_length_index_lst, lines_index_lst_width = calculate_connectivity_properties(layer, area_array, min_pool_size)
    return _dict_wet_prop, area_length_index_lst, lines_index_lst_width

def calculate_additional_metrics(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width):
    total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL = calculate_metrics_AW(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width)
    return total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL

def compile_results(date, polygon, section_length, 
                    export_shp, total_wet_area, total_wet_perimeter, 
                    AWMSI, AWMPA, AWRe, AWMPW, AWMPL, _dict_wet_prop):
    
    res_metrics_data, res_points_data, res_lines_data = [], [], []

    # Collect the calculated metrics for this layer
    res_metrics_data.append({
        'date': date,
        'section': polygon.name,
        'npools': len(_dict_wet_prop),
        'section_area_km2': polygon.geometry.area / 1e6,  # Convert to square kilometers
        'section_length': section_length,
        'wet_area_km2': total_wet_area/ 1e6, # Convert area to square kilometers
        'wet_perimeter_km': total_wet_perimeter / 1e3, # Convert to kilometers
        'wet_length_km': sum(_dict_wet_prop[d]['length'] for d in _dict_wet_prop) / 1e3,
        'AWMSI': AWMSI,
        'AWRe': AWRe,
        'AWMPA': AWMPA / 1e6, # Convert to square kilometers
        'AWMPL': AWMPL / 1e3, # Convert to kilometers
        'AWMPW': AWMPW / 1e3 # Convert to kilometers
    })

    # Optional: Export analysis results as shapefiles
    if export_shp:
        # Process points and lines data for shapefile export
        for region_label, props in _dict_wet_prop.items():
            point_entry = {
                'Date': date, 
                'Section': polygon.name, 
                'Region': region_label
            }
            res_points_data.extend([
                {**point_entry, 'Type': 'Coord_start', 'geometry': props['coord_start']},
                {**point_entry, 'Type': 'Coord_end', 'geometry': props['coord_end']},
                {**point_entry, 'Type': 'Midpoint', 'geometry': props['midpoint']}
            ])
            res_lines_data.append({
                'Date': date, 
                'Section': polygon.name, 
                'Region': region_label, 
                'Length': props['length'], 
                'geometry': props['linestring']
            })
    
    return res_metrics_data, res_points_data, res_lines_data

def process_polygon_parallel(args_list):  
    """
    Process river sections in parallel to compute various ecohydrological metrics.

    This function performs spatial analysis on a given river section defined by a polygon.
    It calculates metrics such as area, perimeter, connectivity properties, and more, based
    on the input water mask and persistence layer. Optionally, it can export the results as
    shapefiles.

    Parameters:
    - args_list (tuple): A tuple containing parameters for processing a single river section.
                        Expected order: (polygon, da_wmask, PP, section_length, export_shp, outdir).

    Returns:
    - tuple: Returns a tuple of DataFrames (metrics_data, points_data, lines_data) containing the computed metrics.
                If `export_shp` is False, only metrics_data DataFrame is returned.
    """
    try:
        # Unpack the arguments for clarity
        polygon, da_wmask, PP, section_length, export_shp, outdir, min_pool_size = args_list

        # Initialize lists to collect data for each metric
        metrics_data = []
        points_data = []
        lines_data = []
        
        # Clip the input water mask to the polygon's extent for focused analysis
        da_wmask = da_wmask.rio.clip([polygon.geometry]).chunk('auto').persist()
        # Convert da_wmask into a list of delayed tasks, one per layer
        tasks = [process_single_layer(da_wmask[num], polygon, section_length, export_shp, min_pool_size) for num in range(da_wmask.shape[0])]

        # Use dask.compute to execute all tasks in parallel
        polygon_results = dask.compute(*tasks)
                
        # Aggregate results
        metrics_data, points_data, lines_data = [], [], []

        for res_metrics_data, res_points_data, res_lines_data in polygon_results:
            metrics_data.extend(res_metrics_data)
            points_data.extend(res_points_data)
            lines_data.extend(res_lines_data)

        # Convert collected metrics data into a DataFrame for further analysis and reporting
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics['date'] = pd.to_datetime(df_metrics['date'], format='%Y-%m-%d')

        # Perform vectorized calculations for additional metrics directly on the DataFrame
        df_metrics['APSEC'] = (df_metrics['wet_area_km2'] / df_metrics['section_area_km2']) * 100
        df_metrics['LPSEC'] = (df_metrics['wet_length_km'] / df_metrics['section_length']) * 100
        df_metrics['PF'] = df_metrics['npools'] / df_metrics['wet_area_km2']
        df_metrics['PFL'] = df_metrics['npools'] / df_metrics['wet_length_km']
        # Replace infinite values and NaNs
        df_metrics.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        # Set specified columns to zero when 'npools' is zero using DataFrame.where
        cols_to_zero = ['wet_area_km2', 'wet_perimeter_km', 'AWMSI', 'AWMPA', 'AWMPL', 'AWMPW', 'APSEC', 'LPSEC', 'PF', 'PFL']
        df_metrics[cols_to_zero] = df_metrics[cols_to_zero].where(df_metrics['npools'] != 0, 0.0)
        
        # Calculate pixel persistence for the section and update metrics DataFrame
        PP = PP.rio.clip([polygon.geometry]).compute()
        df_metrics = pixel_persistence_section(PP, df_metrics, interval_ranges = None)

        # Conditionally return data based on `export_shp` flag
        if export_shp:
            # Convert points_data and lines_data to DataFrames as before
            df_points = pd.DataFrame(points_data)
            df_lines = pd.DataFrame(lines_data)

            return df_metrics, df_points, df_lines
        else:
            return df_metrics

    except Exception as e:       
        print(f"Error processing polygon: {e}")
        # Return an empty DataFrame or a meaningful error indicator for this task
        return pd.DataFrame()

def calculate_pool_area_and_perimeter(layer, min_pool_size):   
    """
    Calculate the perimeter and area of connected regions in a given water mask layer.

    Parameters:
    - layer (numpy.ndarray): Input 2D array representing a single layer of the water mask.

    Returns:
    - dict_area_2p (defaultdict): A dictionary mapping labels of connected regions to [perimeter, area].
    - area_array (numpy.ndarray): Array of the same shape as `layer`, where each pixel's value is the area of its connected region.
    """
    # Initialize a dictionary to store calculated perimeter and area for each label
    dict_area_2p = defaultdict(lambda: [0.0, 0.0])
      
    # Label connected pixels in the layer
    pre_label = label(layer, connectivity=2)
    # Calculate geometric properties for each labeled region
    pre_label_shapes = list(rasterio.features.shapes((pre_label.astype('int32')), transform=layer.rio.transform()))
    
    # Calculate perimeter and area for each region and store in the dictionary
    for polygon, value in pre_label_shapes:
        if value == 0:
            continue  # Skip processing for group 0, which is no data
        size = pre_label[pre_label == value].size
        if size >= min_pool_size: # Only process if the region contains more than X pixels
            shape = shapely.geometry.shape(polygon)
            perimeter = shape.length
            area = shape.area
            dict_area_2p[value][0] += perimeter
            dict_area_2p[value][1] += area

    # Generate an area array based on labeled regions and their calculated areas
    u, inv = np.unique(pre_label, return_inverse=True)
    areas = np.array([dict_area_2p[x][1] if x in dict_area_2p else 0.0 for x in u]) # Fill non-existent entries with 0.0
    area_array = areas[inv].reshape(pre_label.shape)

    return dict_area_2p, area_array

def calculate_connectivity_properties(layer, area_array, min_pool_size):   
    """
    Calculates connectivity properties for wet regions in a layer, employing image processing and geometric analyses.
    
    Skeletonization is used to reduce regions to their simplest form to facilitate pathfinding. The function identifies
    the longest paths within these skeletonized regions, calculating properties such as length, endpoints, and centroids.
    
    Parameters:
    - layer (numpy.ndarray): Input 2D array representing a single layer of the water mask.
    - area_array (numpy.ndarray): Array with calculated areas for each pixel's region, used for additional metrics.
    
    Returns:
    - _dict_wet_prop (dict): Properties of wet regions, including start and end coordinates, length, linestring, and centroid.
    - area_length_index_lst (list): List of tuples with area and path length for each region.
    - lines_index_lst_width (list): List of tuples with path indices and corresponding lengths.
    """
    # Initializations for results storage
    _dict_wet_prop = {}
    endpoints_index_lst = []
    lines_index_lst = []
    lines_index_lst_width = []
    area_length_index_lst = []
    
    # Skeletonization to simplify the connectivity analysis
    skeleton = skeletonize(layer, method='lee')
    
    # Labeling the skeletonized image to identify distinct regions
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
        if region.num_pixels >= min_pool_size: # Process only if the region has more than X pixels
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
            # Extract the midpoint of the line
            midpoint = line_path.interpolate(0.5, normalized=True)
            
            _dict_wet_prop[region.label] = {
                # 'date': layer['time'].values[0].strftime('%Y-%m-%d'),
                'coord_start': Point(line_path.coords[0]),
                'coord_end': Point(line_path.coords[-1]),
                'length': line_path.length, 
                'linestring': line_path,
                'midpoint': midpoint
            }
            area_length_index_lst.append(((area_array[path[0]]), line_path.length))
            lines_index_lst_width.append((longest_path, line_path.length))

    return _dict_wet_prop, area_length_index_lst, lines_index_lst_width

def find_closest_farthest_points(reference_point, region):
    """
    Finds the closest and farthest points within a given region to a reference point.
    
    Parameters:
    - reference_point (tuple or list): The reference point's coordinates (x, y).
    - region (shapely.geometry.Polygon): The region as a Polygon object.
    
    Returns:
    - tuple: Contains the coordinates of the closest and farthest points to the reference point.
    """
    # Ensure access to the coordinates of the polygon's exterior
    coords = region.coords
    # Calculate distances from the reference point to all points in the region
    distances = cdist([reference_point], coords).flatten()
    # Find indices of the closest and farthest points
    closest_idx = np.argmin(distances)
    farthest_idx = np.argmax(distances)
    
    # Return the coordinates of the closest and farthest points
    return coords[closest_idx], coords[farthest_idx]

def combine_points(points):
    """
    Generates unique pairs of combinations from a list of points.

    This function is useful for creating potential pairings of points when analyzing
    distances or connections between them in geospatial analyses.

    Parameters:
    - points (list): A list of points, where each point is represented as a tuple (x, y).

    Returns:
    - list: A list of unique point combinations, where each combination is represented
            as a list containing two point tuples.
    """
    unique_combinations = set()
    # Iterate over all possible 2-point combinations
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
    # Iterate over each position in the input NumPy array
    for np_position in np_positions:
        # Retrieve the corresponding 'x' and 'y' coordinates for the current position
        x = float(layer['x'][np_position[1]])
        y = float(layer['y'][np_position[0]])
        # Append a new Point object with these coordinates to the coord_list
        coord_list.append(Point(x, y))
    return coord_list

def calculate_metrics_AW(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width):   
    """
    Computes Area-weighted metrics based on provided data and indices.

    Parameters:
    - dict_area_2p (dict): Dictionary with keys as region labels and values as [perimeter, area] lists.
    - area_length_index_lst (list): List of tuples, each containing (area, path length) for regions.
    - layer (xarray.DataArray): The data array from which these regions are derived.
    - lines_index_lst_width (list): List of tuples, each containing (path indices, path length) for region paths.

    Returns:
    - A tuple containing calculated area-weighted metrics, including: 
        - total_wet_area (float): Total wetted area.
        - total_wet_perimeter (float): Total wetted length.
        - AWMSI (float): Area-weighted Mean Shape Index.
        - AWMPA (float): Area-weighted Mean Pixel Area.
        - AWRe (float): Area-weighted Elongation Ratio.
        - AWMPW (float): Area-weighted Mean Pool Width.
        - AWMPL (float): Area-weighted Mean Pool Length.
    """
    # Ensure areas and perimeters are numpy arrays for efficient computation
    areas = np.array([info[1] for info in dict_area_2p.values()])
    perimeters = np.array([info[0] for info in dict_area_2p.values()])
    
    # Total area and perimeter of all regions
    total_wet_area = areas.sum()
    total_wet_perimeter = perimeters.sum()
    
    # Area-Weighted Mean Shape Index (AWMSI)
    AWMSI = np.sum((0.25 * perimeters / np.sqrt(areas)) * (areas / total_wet_area))
    
    # Area-Weighted Mean Pixel Area (AWMPA)
    AWMPA = np.average(areas, weights=areas) if areas.size > 0 else 0
    
    # Calculate AWRe, AWMPW, and AWMPL using their respective functions
    AWRe = calculate_AWRe(area_length_index_lst)
    AWMPW, AWMPL = calculate_AWMPW_AWMPL(layer, lines_index_lst_width)

    return total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL

def calculate_AWRe(area_length_index_lst):
    """
    Computes the Area-weighted Elongation Ratio (AWRe) based on area-length information.

    Parameters:
    - area_length_index_lst (list): A list of tuples, each containing the area and length
                                    of a region.

    Returns:
    - AWRe (float): The computed Area-weighted Elongation Ratio. If the total area sum is zero,
                    returns NaN to indicate the calculation is undefined.
    """   
    # Initialize arrays for areas and lengths for vectorized computation 
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
    # Euclidean distance transform to find the distance to the nearest background pixel
    euc_dist_trans_array = ndimage.distance_transform_edt(layer.values)
    
    # Calculate pixel scale based on the layer's spatial resolution to convert distances to real-world measurements
    pixel_scale = 2 * layer.rio.transform()[0] # Assuming uniform pixel size
    
    # Initialize accumulators for weighted width, length, and total weight calculations
    total_width = total_length = total_weight = 0
    
    for idxw, length in lines_index_lst_width:
        # Extract row and column indices
        idx, idy = zip(*idxw)
        # Calculate mean width of the segment, scaled to real-world distances
        width_seg = np.mean(euc_dist_trans_array[idx, idy]) * pixel_scale
        # Calculate weight as the product of segment width and length
        weight = width_seg * length
        
        # Accumulate weighted width and length, along with the total weight
        total_width += width_seg * weight
        total_length += length * weight
        total_weight += weight
    
    # Calculate area-weighted averages, ensuring no division by zero
    AWMPW = total_width / total_weight if total_weight > 0 else 0
    AWMPL = total_length / total_weight if total_weight > 0 else 0
    
    return AWMPW, AWMPL

def save_shp(results, outdir, crs):   
    """
    Saves points and lines data from analysis results as shapefiles, including setting the appropriate CRS.

    Parameters:
    - results (tuple): A tuple containing DataFrames of metrics, points, and lines data.
    - outdir (str): The output directory path where the shapefiles will be saved.
    - crs (str or dict): The coordinate reference system to set for the GeoDataFrames before saving.
    """
    # Initialize lists to aggregate points and lines data
    all_points_data = []
    all_lines_data = []
    # Extract and aggregate points and lines data from each result in the tuple
    for result in results:
        _, points_df, lines_df = result
        all_points_data.append(points_df)
        all_lines_data.append(lines_df)
        
    # Concatenate all DataFrame parts into single DataFrames for points and lines
    points_gdf = gpd.GeoDataFrame(pd.concat(all_points_data, ignore_index=True), geometry='geometry')
    lines_gdf = gpd.GeoDataFrame(pd.concat(all_lines_data, ignore_index=True), geometry='geometry')
    
    # Set the CRS for both GeoDataFrames
    points_gdf.crs = crs
    lines_gdf.crs = crs
    
    # Save the GeoDataFrames as shapefiles
    points_gdf.to_file(os.path.join(outdir, 'result_points.shp'))
    lines_gdf.to_file(os.path.join(outdir, 'result_lines.shp'))

def calculate_metrics_df(pd_metrics, section_length):
    """
    Calculate additional metrics and create a new DataFrame.
    APSEC: Wetted Area Percentage of Section.
    LPSEC: Wetted Length Percentage of Section.
    PF: Pool Fragmentation.
    PFL: Pool Longitudinal Fragmentation.

    Parameters:
    - pd_metrics (DataFrame): DataFrame containing initial calculated wetness metrics.
    - section_length (float): Length of the section being analyzed, necessary for some metric calculations.
    
    Returns:
    - pdm (DataFrame): Enhanced DataFrame with additional metrics included.
    """
    # Ensure all necessary columns are present
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

def pixel_persistence_section(PP, df_metrics, interval_ranges=None):
    """
    Calculate pixel persistence metrics for a section, integrating with Dask and Xarray for efficiency.

    Parameters:
    - PP (xarray.DataArray): Pixel Persistence array, indicating the persistence of conditions across time.
    - df_metrics (pandas.DataFrame): DataFrame to which persistence metrics will be added.
    - interval_ranges (list of tuples, optional): List of (lower_bound, upper_bound) tuples for additional
      persistence interval calculations.

    Returns:
    - pandas.DataFrame: The input DataFrame enhanced with pixel persistence metrics.
    """
    # Calculate pixel area in km^2 based on raster resolution
    pixel_area_km2 = abs(PP.rio.resolution()[0] * PP.rio.resolution()[1]) / 10**6
    
    # Mean pixel persistence (%)
    pp_threshold = 0.1
    df_metrics['PP_%'] = (PP >= pp_threshold).mean(dim=['x', 'y']).data * 100
    
    # Calculate refuge area (RA) in km^2
    ra_threshold = 0.9
    df_metrics['RA_km2'] = ((PP >= ra_threshold).sum(dim=['x', 'y']) * pixel_area_km2).data
    
    # Calculate persistence intervals if specified
    if interval_ranges:
        for lower_threshold, upper_threshold in interval_ranges:
            column_name = f'PP_{int(lower_threshold*100)}_{int(upper_threshold*100)}'
            df_metrics[column_name] = (((PP >= lower_threshold) & (PP < upper_threshold)).sum(dim=['x', 'y']) * pixel_area_km2).data

    return df_metrics

def calculate_pixel_persistence(da_wmask):   
    """
    Calculates the pixel persistence of wet areas within a series of water mask observations.

    This function determines the frequency of wet conditions for each pixel across the temporal
    dimension of the input DataArray, providing a measure of how persistently each location remains
    wet over time.

    Parameters:
    - da_wmask (xarray.DataArray): An xarray DataArray containing a water mask over time, with dimensions
                                   including 'time', 'x', and 'y', and values indicating wet (1) or dry (0) conditions.

    Returns:
    - p_area (xarray.DataArray): An xarray DataArray representing the pixel persistence of wet areas, with values
                                 ranging from 0 to 1, where 1 indicates 100% persistence (always wet) and -1 for areas
                                 with no wet observations.
    """
    # Total number of observations in the time dimension
    total_obs = da_wmask.sizes['time']
    
    # Mask out no data values (-1) by setting them to 0 temporarily for the calculation
    da_wmask = da_wmask.where(da_wmask == 1, 0)
    
    # Sum wet observations over time and normalize by total observations to get persistence ratio
    p_area = (da_wmask.sum(dim='time') / total_obs).astype('float32')
    
    # Apply a placeholder value for pixels with no wet observations
    p_area = xr.where(p_area > 0, p_area, -1)
    
    # Set DataArray attributes for NoData value
    p_area.attrs['_FillValue'] = -1
    
    # Write the same CRS information as the input DataArray
    p_area = p_area.rio.write_crs(da_wmask.rio.crs)
    
    return p_area

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
