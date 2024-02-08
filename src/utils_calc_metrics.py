import os
import re
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import dask.array as da
import rasterio
from rasterio import features
import waterdetect as wd
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

from src import utils_wd_batch as utils_wdb

def validate(da_wmask, rcor_extent, section_length, img_ext, module):
    """
    Validates input data for processing in a GIS context, especially for water mask and river corridor data.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray or str): Input data, either as a DataArray representing a water mask or a directory path containing water mask images.
    rcor_extent (str): Path to a file containing river corridor extent (rcor) shapefile.
    section_length (int or float): Length of the river section for metric calculations.
    img_ext (str): Image file extension (e.g., '.tif') used to process images in the input folder.
    module (str): Name of the processing module to be used (e.g., 'calc_metrics').

    Returns:
    tuple: A tuple containing the validated DataArray for the water mask and the processed river corridor extent data.
    
    Notes:
    - This function performs a series of checks and transformations on the input data to ensure they are in the correct format and structure for further processing.
    - It also handles the replacement of NaN values in the DataArray and verifies the file type and CRS compatibility of the shapefile.
    """
    print('Checking input data...')
    
    # Validate if da_wmask is either a DataArray or a valid directory path
    assert (isinstance(da_wmask, xr.core.dataarray.DataArray) or 
           (isinstance(da_wmask, str) and os.path.isdir(da_wmask))), 'Invalid input. da_wmask must be a valid DataArray or a directory path'

    # Process the input if it is a directory path
    if isinstance(da_wmask, str) and os.path.isdir(da_wmask):
        # Create DataArray and validade rasters within the provided folder
        da_wmask, n_bands, crs = utils_wdb.validate_input_folder(da_wmask, img_ext)
    
    if isinstance(da_wmask, xr.core.dataarray.DataArray):
        crs = da_wmask.rio.crs

    # Replace NaN values in the DataArray with -1 and set the '_FillValue' attribute accordingly
    da_wmask = utils_wdb.replace_nodata(da_wmask, -1)
    da_wmask.attrs['_FillValue'] = -1
    # Validate and potentially transform the CRS of the DataArray
    da_wmask, crs = validate_data_array_cm(da_wmask, crs)
    # Validate the rcor_extent shapefile input, ensuring correct file extension and CRS compatibility
    rcor_extent = validate_shp_input(rcor_extent, crs, module)
    # If the module is 'calc_metrics', ensure that the section length is provided
    if module == 'calc_metrics':
        # Check if section length is present
        assert section_length != None, 'Invalid input. Section length not found.' 

    print('Checking input data...Data validated')
    return da_wmask, rcor_extent, crs

def validate_data_array_cm(da_wmask, crs):
    """
    Validates and adjusts a given DataArray containing a water mask to ensure it has the required dimensions
    and a proper Coordinate Reference System (CRS).

    Parameters:
    da_wmask (xarray.DataArray): The DataArray containing the water mask data.
    crs (str or dict): The coordinate reference system of the DataArray. It can be a string or a dictionary.

    Returns:
    tuple: A tuple containing the processed DataArray and its corresponding CRS.
    
    Raises:
    AssertionError: If the required dimensions are missing or if the CRS information is not provided.

    Note:
    - The function ensures that the DataArray has the dimensions 'x', 'y', and 'time'.
    - If the CRS is geographic (lat/lon), the function reprojects the DataArray to an estimated UTM CRS.
    """
    # List of dimensions that are required in the DataArray
    required_dimensions = ['x', 'y', 'time']
    # Check for missing dimensions by comparing required dimensions with DataArray dimensions
    missing_dimensions = [dim for dim in required_dimensions if dim not in da_wmask.dims]
    # Assert that there are no missing dimensions
    assert not missing_dimensions, f"Invalid input. The following dimensions are missing: {', '.join(missing_dimensions)}"
    
    # Get dimensions to squeeze (drop unnecessary dimensions)
    dims_to_squeeze = [dim for dim in da_wmask.dims if dim not in required_dimensions]
    # Squeeze and drop unnecessary dimensions
    da_wmask = da_wmask.squeeze(drop=True, dim=dims_to_squeeze)
    
    # Check if CRS information is present
    assert crs != None, 'Invalid input. da_wmask CRS not found.'
    # Create CRS object from input
    crs_obj = CRS(crs)
    # Reproject DataArray to UTM CRS if original CRS is geographic (lat/lon)
    if crs_obj.is_geographic:
        # Estimate UTM CRS based on geographic coordinates
        crs = da_wmask.rio.estimate_utm_crs()
        # Reproject DataArray to the estimated UTM CRS
        da_wmask = xr_reproject(da_wmask, crs)
    return da_wmask, crs

def validate_shp_input(rcor_extent, img_crs, module):
    """
    Validates and loads a shapefile with specific requirements.

    Parameters:
    rcor_extent (str): Path to a shapefile (.shp) to be processed.

    Returns:
    geopandas.geodataframe.GeoDataFrame: A GeoDataFrame containing the loaded shapefile data.
    
    Raises:
    AssertionError: If the input does not meet the validation criteria.
    """
    # Check if r_lines is a string with .shp extension or a GeoDataFrame
    assert isinstance(rcor_extent, gpd.GeoDataFrame) or (isinstance(rcor_extent, str) and rcor_extent.endswith('.shp')), \
        'Pass river corridor extent (r_lines) as a .shp file path or a geopandas.GeoDataFrame'

    # validate projection
    rcor_extent = utils_wdb.validate_input_projection(rcor_extent, img_crs)

    if module == 'generate_sections':
        # Check if there is at least one feature and if any geometry matches the expected type
        valid_geometry = any(isinstance(geom, (LineString, MultiLineString)) for geom in rcor_extent.geometry)
        assert not rcor_extent.empty and valid_geometry, f'Invalid input. Shapefile does not contain valid LineString geometries'  

    elif module == 'calc_metrics':
        # Check if there is at least one feature and if any geometry matches the expected type
        valid_geometry = any(isinstance(geom, (Polygon, MultiPolygon)) for geom in rcor_extent.geometry)
        assert not rcor_extent.empty and valid_geometry, f'Invalid input. Shapefile does not contain valid Polygon geometries' 

    return rcor_extent

def setup_directories_cm(rcor_extent, outdir):
    """
    Set up and return the output and initialization file directories.
    """
    # Determine the output directory
    if outdir == None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(rcor_extent)), 'results_iRiverMetrics')
    # Create the main output directory  
    create_new_dir(outdir, verbose=False)
    # Create output directories for metrics and section results
    outdir = os.path.join(outdir, 'metrics')
    create_new_dir(outdir, verbose=False)
    print('Results from Calculate Metrics module will be exported to ', outdir)

    section_outdir_folder = os.path.join(outdir, '01.Section_results')
    create_new_dir(section_outdir_folder, verbose=False)
    print('Section results will be exported to ', section_outdir_folder)

    return outdir, section_outdir_folder

def create_new_dir(outdir, verbose=True):
    """
    Creates a new directory at the specified location.
    Parameters:
        outdir (str): The path of the directory to be created.
        verbose (bool): Whether to show the print statement or not (default is True).
    Returns:
        None
    """
    # Create output file
    os.makedirs(outdir, exist_ok=True)
    if verbose:
        print(f'{outdir} created to export results')

def preprocess(da_wmask, rcor_extent, crs, outdir, export_shp, section_length):
    """
    Perform preprocessing steps on input data and extent shapefile.

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with mask.
        rcor_extent (geopandas.GeoDataFrame): Extent shapefile rcor (river sections).

    Returns:
        xarray.DataArray: Preprocessed DataArray with nodata filled and reprojected to UTM.
        geopandas.GeoDataFrame: Reprojected extent shapefile.
    """
    print('Preprocessing...')
    # Step 1: Clip input data and extent to match dimensions
    da_wmask, rcor_extent = match_input_extent(da_wmask, rcor_extent)
    print('filling no data')
    # Step 2: Fill nodata values in the DataArray
    da_wmask = fill_nodata(da_wmask, rcor_extent)    
    # Step 3: Prepare args for processing
    args_list = prepare_args(da_wmask, rcor_extent, crs, outdir, export_shp, section_length)
    print('Preprocessing...Done!')
    return args_list, da_wmask, rcor_extent

def match_input_extent(da_wmask, rcor_extent):
    """
    Clip input DataArray and extent shapefile to the overlapping region.

    Parameters:
        da_wmask (xarray.DataArray): Input DataArray with water mask.
        rcor_extent (geopandas.GeoDataFrame): Extent shapefile (river sections).

    Returns:
        xarray.DataArray: Clipped DataArray.
        geopandas.GeoDataFrame: Filtered extent shapefile.
    """
    # Get bounding box of DataArray
    minx, miny, maxx, maxy = da_wmask.rio.bounds()
    # Filter extent polygons that are completely within the DataArray bounding box
    rcor_extent = rcor_extent[rcor_extent.geometry.intersects(box(minx, miny, maxx, maxy))]
    # Clip the DataArray using the filtered extent
    da_wmask = da_wmask.rio.clip(rcor_extent.geometry)
        
    return da_wmask, rcor_extent

def fill_nodata(da_wmask, rcor_extent):
    """
    Apply the fill_nodata_layer function to an entire DataArray using Dask for parallel processing.

    This function uses Dask's map_overlap to apply the filling logic to each chunk
    of the DataArray. It handles NoData values (marked as 2) by attempting to fill them
    with values from adjacent time layers.

    Parameters:
        da_wmask (xarray.DataArray): The DataArray to process.

    Returns:
        xarray.DataArray: The DataArray with NoData values filled.
    """
    # Update 'no data' values within a specified river corridor extent
    da_wmask = update_nodata_in_rcor_extent(da_wmask, rcor_extent)
    crs = da_wmask.rio.crs
    
    # Apply the fill_nodata_layer function with Dask's map_overlap
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
    
    # Convert any remaining NoData values (2) back to -1   
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
    da_wmask (xarray.DataArray): The xarray DataArray to be updated.
    rcor_extent (geopandas.GeoDataFrame): The GeoDataFrame representing the river corridor extent.

    Returns:
    xarray.DataArray: The updated xarray DataArray where 'no data' areas within the river 
                      corridor extent are updated.
    """
    # Dissolve the shapefile to create a single geometry
    gdf = rcor_extent.dissolve() 
    # Ensure the polygon is in the same CRS as the xarray
    gdf = gdf.to_crs(da_wmask.rio.crs.to_string())
    # Create a transformation and raster shape from xarray
    transform = rasterio.transform.from_bounds(*da_wmask.rio.bounds(), len(da_wmask.coords['x']), len(da_wmask.coords['y']))
    raster_shape = (len(da_wmask.coords['y']), len(da_wmask.coords['x']))
    # Rasterize the dissolved shapefile
    rasterized_polygon = features.rasterize(shapes=gdf.geometry, out_shape=raster_shape, transform=transform, fill=0, default_value=1) #, dtype='int8')
    # Convert the rasterized polygon to a Dask array for efficient computation
    rasterized_polygon_dask = da.from_array(rasterized_polygon, chunks='auto')
    # Create a copy of the first timestep of the xarray DataArray and apply the rasterized polygon as a mask
    river_corridor_raster = da_wmask.isel(time=0).copy()
    river_corridor_raster = river_corridor_raster.copy(data=rasterized_polygon_dask)
    # Identify areas with -1 in the water mask within the river corridor
    no_data_areas = (da_wmask == -1) & (river_corridor_raster == 1)
    # Update these areas to 2 and retain the original CRS
    da_wmask = xr.where(no_data_areas, 2, da_wmask).rio.write_crs(da_wmask.rio.crs, inplace=True)
    
    return da_wmask

def fill_nodata_layer(chunk):
    """
    Fill NoData values in a chunk of a DataArray.

    For each layer in the chunk, NoData values (marked as 2) are filled by checking
    values from adjacent layers. The function first attempts to fill from the
    immediate next and previous layers, and if unsuccessful, checks two layers ahead and behind.

    Parameters:
        chunk (numpy.ndarray): A chunk of the DataArray being processed.

    Returns:
        numpy.ndarray: The chunk with NoData values filled.
    """
    # Total number of layers in the chunk
    num_layers = chunk.shape[0] 

    # Iterate through each layer, except the first and last
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

    # Special handling for the first and last layers in the chunk
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

def prepare_args(da_wmask, rcor_extent, crs, section_outdir_folder, export_shp, section_length):
    """
    Prepares a list of arguments for parallel processing of geospatial data.

    Args:
        da_wmask (xarray.DataArray): The input raster data array representing a water mask.
        rcor_extent (geopandas.GeoDataFrame): The GeoDataFrame containing river corridor extent polygons.
        crs (str or dict): Coordinate reference system of the input data.
        section_outdir_folder (str): Base output directory path for sectioned outputs.
        export_shp (bool): Flag indicating whether to export the results as shapefiles.
        section_length (float): Length of each section for processing.

    Returns:
        list: A list of arguments, where each entry is a list containing parameters for processing a specific section.
    """
    args_list = []

    # Iterate over each feature (polygon) in the river corridor extent GeoDataFrame
    for feature in rcor_extent.iterrows():
        # Extract the GeoDataFrame row representing the feature
        feature = feature[1]
        # Calculate the bounding box coordinates of the current polygon
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        # Define masks for selecting data within the bounding box
        col_mask = (da_wmask.x >= xmin) & (da_wmask.x <= xmax)
        row_mask = (da_wmask.y >= ymin) & (da_wmask.y <= ymax)
        # Use Dask's lazy indexing to select the portion of da_wmask within the polygon's bounding box
        cliped_da_wmask = da_wmask.sel(x=col_mask, y=row_mask, method='nearest')
        # Create a directory to save output metrics for the section
        outdir_section = os.path.join(section_outdir_folder, str(feature.name))
        # Add the set of parameters as a list to the args_list for parallel processing
        args_list.append([feature, cliped_da_wmask, crs, outdir_section, export_shp, section_length])
    return args_list

def process_polygon_parallel(args_list):
    """
    Process river sections in parallel.

    Args:
        args_list (tuple): A tuple containing polygon and other parameters.
        
    Returns:
        None
    """
    try:
        # Unpack the argument list containing polygon and other parameters
        polygon, da_wmask, crs, outdir_section, export_shp, section_length = args_list
        # Call the process_polygon function to process the current polygon
        date_list, section_area, total_wet_area_list, total_wet_perimeter_list, \
        length_list, n_pools_list, AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list, \
        AWMPW_list, da_area, da_npools, da_rlines, layer, outdir_section, points_data, lines_data = process_polygon(polygon, da_wmask, crs, outdir_section, export_shp)
        # Create a dictionary containing various metrics for the processed polygon
        pand_dict = {'date': date_list, 'section_area': section_area, 'wet_area_km2': [d/10**6 for d in total_wet_area_list],
                    'wet_perimeter_km2': [d/10**6 for d in total_wet_perimeter_list], 'wet_length_km': [d/1000 for d in length_list],
                    'npools': n_pools_list, 'AWMSI': AWMSI_list, 'AWRe': AWRe_list,
                    'AWMPA': [d/10**6 for d in AWMPA_list], 'AWMPL': [d/1000 for d in AWMPL_list],
                    'AWMPW': [d/1000 for d in AWMPW_list]}
        # Convert the dictionary to a pandas DataFrame
        pd_metrics = pd.DataFrame(data=pand_dict)
        # Calculate metrics using the DataFrame and section length
        pdm = calculate_metrics_df(pd_metrics, section_length)
        # Create a data array with processed data
        da_area = create_dataarray(layer, da_area, date_list, crs)
        # Calculate pixel persistence for the section
        pdm = pixel_persistence_section(da_area, pdm, interval_ranges=None)
        # Save the calculated metrics to a CSV file
        pdm.to_csv(os.path.join(outdir_section, 'pd_metrics.csv')) 
        
        return points_data, lines_data
      
    except Exception as e:
        # If an error occurs during processing, print an error message
        print(f"Error processing polygon {args_list[0]}: {e}")

def process_polygon(polygon, da_wmask, crs, outdir_section, export_shp):
    """
    Process a polygon region to calculate various metrics from raster data through time.

    Parameters:
    - polygon (GeoDataFrame): Polygon region of interest (Section).
    - da_wmask (xarray.DataArray): DataArray containing water mask.
    - crs (str): Section CRS.
    - outdir_section (str): Directory to save output metrics of Section.
    - export_shp (bool): Flag to export shapefiles.

    Returns:
    - date_list (list): List of dates for each processed time layer.
    - section_area (float): Area of the processed polygon section in square kilometers.
    - total_wet_area_list (list): List of total wet areas for each processed time layer.
    - total_wet_perimeter_list (list): List of total wet perimeters for each processed time layer.
    - length_list (list): List of total lengths of wet regions for each processed time layer.
    - n_pools_list (list): List of the number of detected wet regions for each processed time layer.
    - AWMSI_list (list): List of Area-weighted Mean Shape Index (AWMSI) values for each processed time layer.
    - AWRe_list (list): List of Area-weighted Elongation Ratio (AWRe) values for each processed time layer.
    - AWMPA_list (list): List of Area-weighted Mean Pixel Area (AWMPA) values for each processed time layer.
    - AWMPL_list (list): List of Area-weighted Mean Pool Length (AWMPL) values for each processed time layer.
    - AWMPW_list (list): List of Area-weighted Mean Pool Width (AWMPW) values for each processed time layer.
    """
    # Initialize lists to store calculated metrics
    AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list, AWMPW_list = [], [], [], [], []
    date_list, length_list, n_pools_list, total_wet_area_list, total_wet_perimeter_list = [], [], [], [], []
    da_area, da_npools, da_rlines = [], [], []
    # Calculate area of the polygon section in square kilometers
    section_area = polygon.geometry.area/10**6
    # Clip the water mask to the polygon's extent
    da_wmask = da_wmask.rio.clip([polygon.geometry])
    # Create a directory to save output metrics for the section
    create_new_dir(outdir_section, verbose=False)


    points_data = []
    lines_data = []
    # Loop through each time layer in the water mask
    for num, layer in enumerate(da_wmask):
        # Convert time value to date format
        date = pd.to_datetime(str(da_wmask.time.values[num])).strftime('%Y-%m-%d')
        # Calculate pool area and perimeter
        dict_area_2p, area_array = calculate_pool_area_and_perimeter(layer)
        # Calculate connectivity properties - pool endpoints/length/width
        _dict_wet_prop, n_pools_index_lst, lines_index_lst, area_length_index_lst, lines_index_lst_width = calculate_connectivity_properties(layer, area_array)
        # Calculate area-weighted metrics metrics using calculated data
        total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL = calculate_metrics_AW(dict_area_2p, area_length_index_lst, layer, lines_index_lst_width)
        # Append results to respective lists
        date_list.append(pd.to_datetime(str(da_wmask[num].time.values)).strftime('%Y-%m-%d'))
        length_list.append(sum((_dict_wet_prop[d]['length']) for d in _dict_wet_prop))
        total_wet_area_list.append(total_wet_area)
        total_wet_perimeter_list.append(total_wet_perimeter)
        n_pools_list.append(len(_dict_wet_prop))
        AWMSI_list.append(AWMSI)
        AWRe_list.append(AWRe)
        AWMPA_list.append(AWMPA)
        AWMPL_list.append(AWMPL)
        AWMPW_list.append(AWMPW)
        da_area.append(layer.values)
        da_npools.append(list_index(layer, n_pools_index_lst))
        da_rlines.append(list_index(layer, lines_index_lst))
        
        for region_label, props in _dict_wet_prop.items():
            # Add point data
            points_data.append({'Date': date, 'Polygon_id': polygon.name, 'Region_label': region_label, 'Type': 'Coord_start', 'geometry': props['coord_start']})
            points_data.append({'Date': date, 'Polygon_id': polygon.name, 'Region_label': region_label, 'Type': 'Coord_end', 'geometry': props['coord_end']})
            points_data.append({'Date': date, 'Polygon_id': polygon.name, 'Region_label': region_label, 'Type': 'Centroid', 'geometry': props['centroid']})
            
            # Add line data
            lines_data.append({'Date': date, 'Polygon_id': polygon.name, 'Region_label': region_label, 'Length': props['length'], 'geometry': props['linestring']})


        # # Export shapefiles if export_shp is True
        # if export_shp:
        #     try:
        #         save_shp (_dict_wet_prop, outdir_section, date, crs)
        #     except:
        #         pass
        
    return date_list, section_area, total_wet_area_list, total_wet_perimeter_list,\
           length_list, n_pools_list, AWMSI_list, AWRe_list, AWMPA_list, AWMPL_list,\
           AWMPW_list, da_area, da_npools, da_rlines, da_wmask.isel(time=0), outdir_section, points_data, lines_data

def calculate_pool_area_and_perimeter(layer):
    """
    Calculate the perimeter and area of connected regions in a given water mask layer.

    Parameters:
    - layer (xarray.DataArray): Input DataArray layer.

    Returns:
    - dict_area_2p (defaultdict): A dictionary where keys are labels of connected regions
                                  and values are lists [perimeter, area].
    - area_array (numpy.ndarray): A 2D NumPy array containing the calculated areas
                                  for each pixel's connected region.
    """
    # Initialize a dictionary to store calculated perimeter and area for each label
    dict_area_2p = defaultdict(lambda: [0.0, 0.0])

    # Group connected pixels into distinct regions
    pre_label = label(layer, connectivity=2)
    # Obtain geometric shapes for each connected region
    pre_label_shapes = list(rasterio.features.shapes((pre_label.astype('int32')), transform=layer.rio.transform()))
    # Calculate perimeter and area for each region and store in the dictionary
    for polygon, value in pre_label_shapes:
        if value == 0:
            continue  # Skip processing for group 0, which is no data
        size = pre_label[pre_label == value].size
        if size > 2: # Only process if the region contains more than 2 pixels
            shape = shapely.geometry.shape(polygon)
            perimeter = shape.length
            area = shape.area
            dict_area_2p[value][0] += perimeter
            dict_area_2p[value][1] += area
    
    # Create an area array using the calculated values and the input labels
    u, inv = np.unique(pre_label, return_inverse=True)
    areas = np.array([dict_area_2p[x][1] if x in dict_area_2p else 0.0 for x in u]) # Fill non-existent entries with 0.0
    area_array = areas[inv].reshape(pre_label.shape)

    return dict_area_2p, area_array

def calculate_connectivity_properties(layer, area_array):
    """
    Calculate connectivity properties of a layer based on provided data.
    This function calculates connectivity properties for wet regions in the given layer.
    It employs skeletonization, region labeling, path finding, and various geometric operations.
    The calculated properties are stored in several lists and dictionaries for further analysis.

    Parameters:
    - layer (xarray.DataArray): Input DataArray.
    - area_array (numpy.ndarray): Array containing calculated areas for each pixel's region.

    Returns:
    - _dict_wet_prop (dict): Dictionary containing calculated properties for wet regions.
    - endpoints_index_lst (list): List of indexes representing endpoints of paths.
    - lines_index_lst (list): List of indexes representing points along paths.
    - area_length_index_lst (list): List of tuples containing area and path length information.
    - lines_index_lst_width (list): List of tuples containing indexes and corresponding path lengths.
    """
    # Dictionary to store results
    _dict_wet_prop = {}
    endpoints_index_lst = []
    lines_index_lst = []
    lines_index_lst_width = []
    area_length_index_lst = []
    # Skeletonize the layer using Lee's method
    skeleton = skeletonize(layer.values, method='lee')
    # Label the skeletonized image to identify regions
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
        if region.num_pixels > 2: # Process only if the region has more than 2 pixels
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
            # Extract the centroid of the region
            centroid = [tuple(int(x) for x in region.centroid)] # Returns (row, col) format
            centroid = np_positions_to_coord_point_list(centroid, layer, [])    
            
            _dict_wet_prop[region.label] = {
                # 'date': layer['time'].values[0].strftime('%Y-%m-%d'),
                'coord_start': Point(line_path.coords[0]),
                'coord_end': Point(line_path.coords[-1]),
                'length': line_path.length, 
                'linestring': line_path,
                'centroid': centroid[0]
            }
            area_length_index_lst.append(((area_array[path[0]]), line_path.length))
            lines_index_lst_width.append((longest_path, line_path.length))
    
    return _dict_wet_prop, endpoints_index_lst, lines_index_lst, area_length_index_lst, lines_index_lst_width

def find_closest_farthest_points(reference_point, region):
    """
    Finds the closest and farthest points within a given region to a reference point.

    Parameters:
    reference_point (tuple or list): The reference point's coordinates.
    region (Polygon or MultiPolygon): The region containing multiple coordinates.
    
    Returns:
    tuple: A tuple containing two coordinate points - the closest and farthest points to the reference point.
    """
    # Extract coordinates from the region
    coords = region.coords
    # Calculate distances between the reference point and all coordinates
    distances = cdist([reference_point], coords).flatten()
    # Find the index of the closest and farthest points based on distances
    closest_idx = np.argmin(distances)
    farthest_idx = np.argmax(distances)
    # Return the coordinates of the closest and farthest points
    return coords[closest_idx], coords[farthest_idx]

def combine_points(points):
    """
    Combines a list of points into unique pairs of combinations.

    Parameters:
    points (list): A list of points, where each point is represented as a tuple.

    Returns:
    list: A list of unique point combinations, where each combination is a list of point pairs.
    """
    unique_combinations = set()
    # Generate unique combinations of point indices and create sorted tuples
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
    for np_position in np_positions:
        x = float(layer['x'][np_position[1]])
        y = float(layer['y'][np_position[0]])
        coord_list.append(Point(x, y))
    return coord_list

def calculate_metrics_AW(dict_area_2p, length_area_index_lst, layer, lines_index_lst_width):
    """
    Calculate Area-weighted metrics based on provided data and indices.

    Parameters:
    - dict_area_2p (dict): Dictionary containing area and perimeter information for regions.
    - length_area_index_lst (list): List of tuples containing length and area information.
    - layer (xarray.DataArray): Input DataArray.
    - lines_index_lst_width (list): List of tuples containing indexes and corresponding path lengths.

    Returns:
    - total_wet_area (float): Total wetted area.
    - total_wet_perimeter (float): Total wetted length.
    - AWMSI (float): Area-weighted Mean Shape Index.
    - AWMPA (float): Area-weighted Mean Pixel Area.
    - AWRe (float): Area-weighted Elongation Ratio.
    - AWMPW (float): Area-weighted Mean Pool Width.
    - AWMPL (float): Area-weighted Mean Pool Length.
    """
    # Calculate AWRe using provided length and area data
    AWRe = calculate_AWRe(length_area_index_lst)
    # Calculate AWMPW and AWMPL using provided width and length data
    AWMPW, AWMPL = calculate_AWMPW_AWMPL(layer, lines_index_lst_width)
    # Extract areas from the dictionary
    areas = [x[1] for _, x in dict_area_2p.items()]
    # Calculate total wet area
    total_wet_area= sum(areas)
    # Calculate total wet perimeter
    total_wet_perimeter = sum((x[0]) for key, x in dict_area_2p.items())
    # Calculate AWMSI
    AWMSI = (sum(((0.25*x[0]/np.sqrt(x[1]))*((x[1])/total_wet_area)) for key, x in dict_area_2p.items()))
    # Calculate AWMPA
    try:
        AWMPA = np.average(areas, weights=areas)
    except ZeroDivisionError:
        AWMPA = 0       

    return total_wet_area, total_wet_perimeter, AWMSI, AWMPA, AWRe, AWMPW, AWMPL

def calculate_AWRe(area_length_index_lst):
    """
    Calculate the Area-weighted Elongation Ratio (AWRe) based on area-length information.

    Parameters:
    - area_length_index_lst (list): List of tuples containing area and path length information.

    Returns:
    - AWRe (float): Area-weighted Elongation Ratio (AWRe) calculated from the provided data.

    This function computes the Area-weighted Elongation Ratio (AWRe) using area-length pairs
    representing the wet regions. It calculates Elongation Ratio (Re) for each area-length pair,
    sums them up, and then divides by the total sum of areas. The AWRe value is returned.

    If the total area sum is zero, the AWRe value is set to NaN to avoid division by zero.
    """
    # Calculate Elongation Ratio for each area-length pair 
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
    # Calculate euclidean distance transform array
    euc_dist_trans_array = ndimage.distance_transform_edt(layer.values)    
    width_length_list = []
    # Calculate width for each line and associate it with the corresponding length
    for idxw, length in lines_index_lst_width:
        idx, idy = zip(*idxw)
        width_seg = np.mean(euc_dist_trans_array[idx, idy])
        width_length_list.append((width_seg * 2 * layer.rio.transform()[0], length))
    # Initialize variables to store AWMPW and AWMPL           
    AWMPW = 0
    AWMPL = 0
    if width_length_list:
        # Extract widths, lengths, and areas (weights) from the width_length_list
        weights = [a for a, l in width_length_list]
        widths = [w for w, l in width_length_list]
        lengths = [l for w, l in width_length_list]
        # Calculate AWMPW by averaging widths with weights
        AWMPW = np.average(widths, weights=weights)
        # Calculate AWMPL by averaging lengths with weights
        AWMPL = np.average(lengths, weights=weights)   
    return AWMPW, AWMPL

def list_index(layer, index_lst):
    """
    Create a binary layer where specified indexes are set to 1.

    Parameters:
    - layer (xarray.DataArray): Input DataArray.
    - index_lst (list): List of tuples containing indexes.

    Returns:
    - binary_layer (numpy.ndarray): Binary layer with specified indexes set to 1.
    """
    # Create a new array filled with zeros, same shape as the input layer
    layer = np.zeros_like(layer.values)
    # If index_lst is not empty, set the specified indexes to 1 in the binary_layer
    if index_lst:
        ind_x, ind_y = zip(*index_lst)
        layer[ind_x, ind_y] = 1

    return layer

def save_shp(results, outdir, crs):
    
    # Create a directory to save shapefiles
    out_shp_dir = os.path.join(outdir, '03.shp')
    create_new_dir (out_shp_dir, verbose=False)
    
    # After processing all polygons and aggregating results
    all_points_data = []
    all_lines_data = []
    for points, lines in results:
        all_points_data.extend(points)
        all_lines_data.extend(lines)

    # Convert to GeoDataFrames
    points_gdf = gpd.GeoDataFrame(all_points_data)
    lines_gdf = gpd.GeoDataFrame(all_lines_data)

    # Ensure geometry columns are correctly recognized
    points_gdf['geometry'] = points_gdf['geometry'].apply(Point)
    lines_gdf['geometry'] = lines_gdf['geometry'].apply(LineString)

    # Set the CRS
    points_gdf.set_crs(crs, inplace=True)
    lines_gdf.set_crs(crs, inplace=True)

    # Save as shapefiles
    points_gdf.to_file(os.path.join(out_shp_dir, 'result_points.shp'))
    lines_gdf.to_file(os.path.join(out_shp_dir, 'result_lines.shp'))

def calculate_metrics_df(pd_metrics, section_length):
    """
    Calculate additional metrics and create a new DataFrame.
    APSEC: Wetted Area Percentage of Section.
    LPSEC: Wetted Length Percentage of Section.
    PF: Pool Fragmentation.
    PFL: Pool Longitudinal Fragmentation.

    Parameters:
    - pd_metrics (DataFrame): DataFrame containing calculated wetness metrics.
    - section_length (float): Length of the polygon section.

    Returns:
    - pdm (DataFrame): New DataFrame with calculated additional metrics.
    """
    # Create a copy of the input DataFrame with selected columns
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

def create_dataarray(layer, layer_metric, date_list, crs):
    """
    Create a DataArray with provided layer metrics and coordinates.

    Parameters:
    - layer (xarray.DataArray): Original layer data.
    - layer_metric (numpy.ndarray): Array containing layer metrics.
    - date_list (list): List of dates corresponding to the metrics.
    - crs (CRS): Coordinate Reference System.

    Returns:
    - da (xarray.DataArray): Created DataArray with metrics.
    """
    da = xr.DataArray(
        data=layer_metric,
        dims=['time', 'y', 'x'],
        coords=dict(time=date_list, y= layer.coords['y'], x= layer.coords['x']),
        attrs={'_FillValue': -1},
    )
    da.rio.write_crs(crs, inplace=True)
    da.attrs['crs'] = str(crs)
    return da

def pixel_persistence_section(da_area, pdm, interval_ranges = None):
    """
    Calculate pixel persistence metrics for section.

    Parameters:
    - da_area (xarray.DataArray): DataArray containing wet area information.
    - pdm (DataFrame): DataFrame containing section metrics.
    - interval_ranges (list): List of tuples specifying lower and upper threshold ranges.

    Returns:
    - pdm (DataFrame): Updated DataFrame with pixel persistence metrics.

    This function calculates pixel persistence metrics for wet area using the provided DataArray and DataFrame.
    It calculates metrics such as mean pixel persistence (PP) and refuge area (RA).
    Additionally, if interval_ranges are provided, it calculates persistence intervals for specified ranges.
    The function returns the updated DataFrame.
    """
    # Get the pixel resolution from the DataArray
    resolution = da_area.rio.resolution()[0]
    # Calculate pixel persistence for the wet area
    p_area = calculate_pixel_persistence (da_area)
    # Calculate the mean pixel persistence of the section exceeding 0.25 (PP metric)
    pdm['PP_%'] = xr.where(p_area >= 0.25, p_area, np.nan).mean().data * 100    
    # Calculate the refuge area (RA metric)
    ra_area = xr.where(p_area >= 0.9, 1, np.nan)
    ra_sum = ra_area.sum(dim=['x', 'y']).data  # assuming 'x' and 'y' are your spatial dimensions
    pdm['RA_km2'] = (ra_sum * resolution * resolution) / 10**6
    
    # Calculate persistence interval metrics for specified ranges
    if interval_ranges:
        for lower_threshold, upper_threshold in interval_ranges:
            column_name = f'PP_{int(lower_threshold*100)}_{int(upper_threshold*100)}'
            pdm[column_name] = calculate_persistence_interval(p_area, lower_threshold, upper_threshold, resolution)
    return pdm

def calculate_pixel_persistence(da_wmask):
    """
    Calculate pixel persistence of wet area in a DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): DataArray containing water mask over time.

    Returns:
    - p_area (xarray.DataArray): DataArray containing pixel persistence of wet area.
    """
    # Count the total number of observations in the time dimension using Dask
    total_obs = da_wmask['time'].size
    # Calculate the number of wet observations for each pixel over time
    p_area = (da_wmask.sum(dim='time') / total_obs).astype('float32')
    # Replace pixels with zero wet area with a placeholder value
    p_area = xr.where(p_area > 0, p_area, -1)
    # Set dataarray nodata
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
