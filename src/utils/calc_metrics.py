# Standard library imports
import os
from collections import deque

# Third-party imports
import dask.array as da
import dask_image.ndmeasure as dask_ndmeasure
from dask import delayed, compute
from dask_regionprops import regionprops
import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.xr import xr_reproject
from pyproj import CRS
from rasterio.features import rasterize, shapes
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, shape
from scipy.ndimage import label, distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import regionprops_table
import xarray as xr
from igraph import Graph
from scipy.ndimage import generate_binary_structure
import dask.dataframe as dd

# Local imports
from src.utils import wd_batch

# Public functions
def validate(da_wmask, rcor_extent, outdir, section_length, img_ext, section_name_col):
    """
    Validate and preprocess input data for the Calculate Metrics module.

    Parameters:
    - da_wmask (xarray.DataArray or str): Water mask data as a DataArray or path to water mask images.
    - rcor_extent (geopandas.GeoDataFrame or str): Path to river corridor extent shapefile or GeoDataFrame.
    - outdir (str): Output directory path.
    - img_ext (str): Image file extension.

    Returns:
    - tuple: Validated water mask DataArray, processed river corridor extent data, CRS, pixel size, and output directory.
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
    
    if section_length is None:
        section_length = np.nan
            
    outdir = setup_directories_cm(rcor_extent, outdir)
    
    if rcor_extent is None:
        assert outdir != None, 'Invalid input. rcor_extent or outdir must be provided.'
        # Extract bounding box coordinates
        bounds = da_wmask.rio.bounds()
        # Create a polygon from the bounding box
        bbox_polygon = box(*bounds)
        # Create a GeoDataFrame from the polygon
        rcor_extent = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=crs)
    
    if rcor_extent is not None:
        # Validate rcor_extent shapefile for correct file extension and CRS
        rcor_extent = validate_shp_input(rcor_extent, crs, section_name_col)
    
    da_wmask, crs, pixel_size = validate_data_array_cm(da_wmask, crs)

    print('Input data validated.\n')
    return da_wmask, rcor_extent, section_length, crs, pixel_size, outdir

def preprocess(da_wmask, rcor_extent, fill_nodata):
    print('Preprocessing...')
    da_wmask, rcor_extent = match_input_extent(da_wmask, rcor_extent)
    
    original_time_count = da_wmask.sizes['time']
    # Exclude time steps with only nodata values
    valid_time_mask = da_wmask.isnull().all(dim=['y', 'x'])
    da_wmask = da_wmask.sel(time=~valid_time_mask)
    valid_time_count = da_wmask.sizes['time'] 
    
    if valid_time_count < original_time_count:
        print(f'    Excluding {original_time_count - valid_time_count} time steps with only nodata values.')
            
    if rcor_extent is not None:
        # Update 'no data' values within the specified river corridor extent
        da_wmask, mask_array = update_nodata_in_rcor_extent(da_wmask, rcor_extent)
    
    if fill_nodata:
        print('    Filling nodata values...')
        da_wmask = fill_nodata_darray(da_wmask, mask_array)
    else:
        da_wmask_data = da.where(da_wmask == 1, 1, 0)
        da_wmask = xr.DataArray(da_wmask_data.astype(np.int8),
                            coords=da_wmask.coords, 
                            dims=da_wmask.dims,
                            attrs=da_wmask.attrs
                            )
    
    assert da_wmask.sizes['time'] > 1, 'No time steps remaining after preprocessing. - Not enough data to calculate metrics.'
    
    print(f"    {da_wmask.sizes['time']} valid time steps will be used to calculate metrics.")
    
    print('Preprocessing complete.\n')
    return da_wmask, rcor_extent

@delayed
def preprocess_feature(da_wmask, feature, section_name_col, pixel_size, min_pool_size):
    """
    Preprocess a single feature.
    """
    preprocess_result = preprocess_feature_operations(da_wmask, feature, section_name_col)
       
    da_wmask_feature = preprocess_result['da_wmask_feature']
    section = preprocess_result['section']
    section_area = preprocess_result['section_area']
        
    pp_mean, ra_area = calculate_pixel_persistence_metrics(
        da_wmask_feature, 
        pixel_size
    )
    
    labeled_layer, labeled_skel, distance_transform_layer = pre_process_layer(
        da_wmask_feature, min_pool_size
    )
    
    return {
        'da_wmask_feature': da_wmask_feature,
        'section': section,
        'section_area': section_area,
        'pp_mean': pp_mean,
        'ra_area': ra_area,
        'labeled_layer': labeled_layer,
        'labeled_skel': labeled_skel,
        'distance_transform_layer': distance_transform_layer
    }

def batch_date_list(date_list, batch_size=15):
    """Batch the date list into smaller groups."""
    return [date_list[i:i + batch_size] for i in range(0, len(date_list), batch_size)]

@delayed
def process_feature_batch(preprocessed, 
                          batch_dates, 
                          pixel_size,
                          section_length):
    """
    Process a batch of dates for a preprocessed feature.
    """
    # Extract preprocessed data
    da_wmask_feature = preprocessed['da_wmask_feature']
    section = preprocessed['section']
    section_area = preprocessed['section_area']
    pp_mean = preprocessed['pp_mean']
    ra_area = preprocessed['ra_area']
    labeled_layer = preprocessed['labeled_layer']
    labeled_skel = preprocessed['labeled_skel']
    distance_transform_layer = preprocessed['distance_transform_layer']
    
    # Find indices for the batch dates
    # Convert batch_dates to a set for faster lookup
    batch_dates_set = set(batch_dates)
    time_indices = [
        i for i, date in enumerate(da_wmask_feature.time.data)
        if pd.to_datetime(date).strftime('%Y-%m-%d') in batch_dates_set
    ]
    
    summary_df_list = []
    for i in time_indices:
        time_step = pd.to_datetime(da_wmask_feature.time.data[i]).strftime('%Y-%m-%d')
        summary_df = summarize_block(           
            labeled_layer.isel(time=i).data,
            labeled_skel.isel(time=i).data,
            distance_transform_layer.isel(time=i).data,
            time_step,
            pixel_size,
            section,
            section_area,
            section_length,
            pp_mean,
            ra_area
        )
        summary_df_list.append(summary_df)
    
    # Concatenate all summaries for this batch
    if summary_df_list:
        final_df = pd.concat(summary_df_list, ignore_index=True)
    else:
        # Handle case with no matching dates
        final_df = pd.DataFrame()
    
    return final_df

@delayed
def export_shapefiles(preprocessed, 
                              outdir, 
                              pixel_size, 
                              summary_ddf, 
                              crs, 
                              min_pool_size):
    """
    Export shapefiles for a feature based on preprocessed data and summary DataFrame.
    """
    da_wmask_feature = preprocessed['da_wmask_feature']
    section = preprocessed['section']

    masked_da_coords = da_wmask_feature.coords
    polygont_gdf = export_polygons(da_wmask_feature, outdir, section, min_pool_size, pixel_size)
    line_gdf, point_gdf = export_points_and_lines(summary_ddf, section, masked_da_coords, outdir, crs)
    
    return polygont_gdf, line_gdf, point_gdf

def process_metrics(group):
    areas = group['area_km2']
    lengths = group['length_km']
    widths = group['width_km']
    perimeters = group['perimeter_km']
    section_area_km2 = group['section_area_km2'].iloc[0]
    section_length = group['section_length_km'].iloc[0]
    npools = areas.size
    pp_mean = group['pp_mean_%'].iloc[0]
    ra_area = group['ra_area_km2'].iloc[0]
    
    if areas.sum() == 0:            
        LPSEC = np.nan if np.isnan(section_length) or section_length == 0 else 0
        return pd.Series({
        'section_area_km2': section_area_km2,
        'section_length_km': section_length,
        'npools': int(0),  
        'wet_area_km2': 0,
        'wet_length_km': 0, 
        'wet_perimeter_km': 0, 
        'AWMSI': 0, 
        'AWRe': np.nan,
        'AWMPA': 0, 
        'AWMPL': 0, 
        'AWMPW': 0, 
        'PF': 0, 
        'PFL': 0, 
        'APSEC': 0, 
        'LPSEC': LPSEC,
        'pp_mean_%': pp_mean,
        'ra_area_km2': ra_area
        })
    
    else:
        total_wetted_area = np.sum(areas)
        total_wetted_perimeter = np.sum(perimeters)
        total_wetted_length = np.sum(lengths)

        AWMSI = np.sum((0.25 * perimeters / np.sqrt(areas)) * (areas / total_wetted_area))
        AWMPA = np.average(areas, weights=areas)
        radii = 2 * (np.sqrt(areas) / np.pi)
        AWRe = np.nansum((radii / lengths) * areas) / total_wetted_area
        AWMPL = np.average(lengths, weights=areas)
        AWMPW = np.average(widths, weights=areas)
        PF = npools / total_wetted_area
        PFL = npools / total_wetted_length
        APSEC = (total_wetted_area / section_area_km2) * 100
        
        LPSEC = (total_wetted_length / section_length) * 100 if not np.isnan(section_length) or section_length != 0 else np.nan

        return pd.Series({
            'section_area_km2': section_area_km2,
            'section_length_km': section_length,
            'npools': int(npools), 
            'wet_area_km2': total_wetted_area,
            'wet_length_km': total_wetted_length, 
            'wet_perimeter_km': total_wetted_perimeter,
            'AWMSI': AWMSI, 
            'AWRe': AWRe,
            'AWMPA': AWMPA, 
            'AWMPL': AWMPL, 
            'AWMPW': AWMPW, 
            'PF': PF, 
            'PFL': PFL,
            'APSEC': APSEC, 
            'LPSEC': LPSEC,
            'pp_mean_%': pp_mean,
            'ra_area_km2': ra_area,
        })

def calculate_pixel_persistence(
    da_wmask_feature: xr.DataArray, 
):
    # Total number of observations in the time dimension
    total_obs = da_wmask_feature.sizes['time'] 
    # Sum wet observations over time
    wet_sum = da_wmask_feature.sum(dim='time')
    # Normalize by total observations to get persistence ratio
    p_area = wet_sum / total_obs
    return p_area

# Helper functions
### Validate ###
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
    pixel_size = da_wmask.rio.resolution()[0]
    
    return da_wmask, crs, pixel_size

def validate_shp_input(rcor_extent, img_crs, section_name_col):
    """
    Validates and loads a shapefile for river corridor analysis, ensuring compatibility with the specified module.

    Parameters:
    - rcor_extent (str or geopandas.GeoDataFrame): Path to the shapefile or a GeoDataFrame to be validated.
    - img_crs (str or dict): The coordinate reference system of the image data to ensure CRS compatibility.

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

    # Validate for Polygon geometries
    valid_geometry = any(isinstance(geom, (Polygon, MultiPolygon)) for geom in rcor_extent.geometry)
    assert not rcor_extent.empty and valid_geometry, \
        'Shapefile must contain valid Polygon geometries. For processing the entire water mask extent, set rcor_extent to None'
    rcor_extent.sindex
    
    if section_name_col != None:
        assert section_name_col in rcor_extent.columns, f"Invalid section_name_col: {section_name_col}. Column not found in rcor_extent.\n Available columns: {list(rcor_extent.columns)}"
    
    return rcor_extent

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
    else:
        outdir = os.path.join(outdir, 'results_iRiverMetrics')
    # Determine the base output directory
    outdir = os.path.join(outdir, 'metrics')
    # Create output folder
    wd_batch.create_new_dir(outdir, verbose=False)
    print('Results from Calculate Metrics module will be exported to ', outdir)
    
    return outdir

### Preprocess ###
def clip_data(data, xmin, xmax, ymin, ymax):
    """Clips the data array based on given bounds."""
    col_mask = (data.x >= xmin) & (data.x <= xmax)
    row_mask = (data.y >= ymin) & (data.y <= ymax)
    return data.sel(x=col_mask, y=row_mask)

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

    xmin, ymin, xmax, ymax = rcor_extent.dissolve().geometry.total_bounds
    da_wmask = clip_data(da_wmask, xmin, xmax, ymin, ymax)
    
    return da_wmask, rcor_extent

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
    if gdf.crs != da_wmask.rio.crs:
        gdf = gdf.to_crs(da_wmask.rio.crs.to_string())

    # Extract relevant information
    shapes = [(geom, 1) for geom in gdf.geometry if not geom.is_empty]
    transform = da_wmask.rio.transform()
    out_shape = da_wmask.isel(time=0).shape

    # Rasterize the geometry into a numpy array
    mask_array = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    da_wmask_array = da.where(((da_wmask.isnull()) | (da_wmask == -1)) & (mask_array == 1), 2, da_wmask)
    
    da_wmask = xr.DataArray(da_wmask_array.astype('int8'),
                        coords=da_wmask.coords, 
                        dims=da_wmask.dims,
                        attrs=da_wmask.attrs
                        )
       
    valid_pixel_mask = (
    da_wmask.notnull() &  # Check for not NaN
    (da_wmask != -1) &    # Exclude NoData (-1)
    (da_wmask != 2) &     # Exclude 2
    (mask_array == 1)     # Include only where mask_array == 1
    )
    # Count the valid pixels where the mask is True
    valid_pixel_counts = valid_pixel_mask.sum(dim=["y", "x"])
    total_pixel_counts = np.sum(mask_array == 1)
    valid_data_proportion = valid_pixel_counts / total_pixel_counts
    # Create a mask to select time steps where at least 70% of the valid pixels are not -1 or 2
    valid_time_mask = valid_data_proportion >= 0.7
    
    num_invalid_time_steps = (~valid_time_mask).sum().values
    if num_invalid_time_steps > 0:
        # Print the message with the number of invalid time steps
        print(f"    {num_invalid_time_steps} layer(s) excluded - below 70% valid data threshold.")
    # Filter only the valid time steps
    da_wmask = da_wmask.sel(time=valid_time_mask)
    da_wmask = da_wmask.where(mask_array == 1, other= -1)
    
    return da_wmask, mask_array

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

def fill_nodata_darray(da_wmask, mask_array):
    """
    Fills NoData values in a DataArray, specifically targeting NoData within a specified river corridor extent,
    and then using parallel processing with Dask to fill remaining NoData values across the entire DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): The DataArray to process, assumed to have a 'time' dimension.
    - rcor_extent (geopandas.GeoDataFrame): The geographical extent within which NoData values are specifically updated.

    Returns:
    - xarray.DataArray: The DataArray with NoData values filled.
    """        
    # Apply fill_nodata_layer function with Dask's map_overlap to the updated DataArray
    filled_data = da.map_overlap(
        fill_nodata_layer,
        da_wmask.data,
        depth={'time': 2},  # Buffer size for two layers ahead/behind
        boundary={'time': 'reflect'}, # Reflect boundary for edge chunks
        dtype=da_wmask.dtype
    )
    da_wmask_data = da.where(filled_data == 1, filled_data, 0)
    da_wmask = xr.DataArray(da_wmask_data.astype(np.int8),
                    coords=da_wmask.coords,
                    dims=da_wmask.dims,
                    attrs=da_wmask.attrs
                    )
    
    ## Check valid pixels
    print('    Checking filled NoData values quality...')
    valid_pixel_mask = (
    da_wmask.notnull() &  # Check for not NaN
    (da_wmask != -1) &    # Exclude NoData (-1)
    (mask_array == 1)     # Include only where mask_array == 1
    )
    # Count the valid pixels where the mask is True
    valid_pixel_counts = valid_pixel_mask.sum(dim=["y", "x"])
    total_pixel_counts = np.sum(mask_array == 1)
    valid_data_proportion = valid_pixel_counts / total_pixel_counts
    # Create a mask to select time steps where at least 95% of the valid pixels are not -1
    valid_time_mask = valid_data_proportion >= 0.95
    
    num_invalid_time_steps = (~valid_time_mask).sum().values
    if num_invalid_time_steps > 0:
        # Print the message with the number of invalid time steps
        print(f"    {num_invalid_time_steps} layer(s) excluded - below 95% valid data threshold after filling nodata.")
    # Filter only the valid time steps
    da_wmask = da_wmask.sel(time=valid_time_mask)
    
    return da_wmask

### Pre_process_feature ###
def preprocess_feature_operations(da_wmask, feature, section_name_col):
    
    if section_name_col is None:
        section = str(feature.name)
    else:
        section = str(feature[section_name_col])

    section_area = feature.geometry.area / 1e6
    xmin, ymin, xmax, ymax = feature.geometry.bounds
    da_wmask_feature = clip_data(da_wmask, xmin, xmax, ymin, ymax)
    transform = da_wmask_feature.rio.transform()    
    mask = create_mask(da_wmask_feature, feature, transform)
    
    da_wmask_feature = da_wmask_feature.where(
        mask == 1, other=0)

    return {
        'da_wmask_feature': da_wmask_feature,
        'section': section,
        'section_area': section_area
    }

def create_mask(da_wmask_feature, feature, transform):
    shapes = [(feature.geometry, 1)]
    mask_array = rasterize(
        shapes=shapes,
        out_shape=(da_wmask_feature.sizes['y'], da_wmask_feature.sizes['x']),
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=True
    )
    mask_da = xr.DataArray(
        mask_array,
        coords={'y': da_wmask_feature['y'], 'x': da_wmask_feature['x']},
        dims=('y', 'x')
    )
    return mask_da

def calculate_pixel_persistence_metrics(
    da_wmask_feature: xr.DataArray, 
    pixel_size: float,
):
    p_area = calculate_pixel_persistence(da_wmask_feature)
    # Convert pixel size from meters to kilometers and calculate area in km²
    pixel_area_km2 = pixel_size**2 / 10**6
    
    # Calculate mean persistence for pixels with p_area > 0.1
    pp_mean = p_area.where(p_area > 0.1).mean(skipna=True).values.item()
    ra_area = p_area.where(p_area > 0.9, other=0).sum(skipna=True) * pixel_area_km2

    return pp_mean, ra_area.values.item()

def find_connected_components(block, min_pool_size):
    structure = np.ones((3, 3), dtype=int) 
    labeled_array, _ = label(block, structure=structure)
    # Remove small objects
    labeled_array = remove_small_objects(labeled_array, min_size=min_pool_size)
    return labeled_array.astype(np.int16)

def skeletonize_label(labeled_layer):
    skel = skeletonize(labeled_layer).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int) 
    labeled_skel, _ = label(skel, structure=structure)
    return labeled_skel

def distance_transform(labeled_layer_block):
    binary_layer = (labeled_layer_block >= 1).astype(np.uint8)
    distance_transform_layer_block = distance_transform_edt(binary_layer)
    return distance_transform_layer_block.astype(np.float32)

def pre_process_layer(da_wmask_feature, min_pool_size):
    # Apply find_connected_components_2d
    labeled_layer = xr.apply_ufunc(
        find_connected_components,
        da_wmask_feature,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int16],
        kwargs={'min_pool_size': min_pool_size}
    )

    # Apply adaptive_skeletonization_2d
    labeled_skel = xr.apply_ufunc(
        skeletonize_label,
        labeled_layer,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int16]
    )

    # Apply distance_transform_2d
    distance_transform_layer = xr.apply_ufunc(
        distance_transform,
        labeled_layer,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    return labeled_layer, labeled_skel, distance_transform_layer

### process_feature_batch ###
def summarize_block(labeled_block, labeled_skel_block, distance_transform_block, time_step, pixel_size,
                        section,
                        section_area,
                        section_length,
                        pp_mean,
                        ra_area):
    """
    Summarizes information from each block and computes width metrics.

    Parameters:
    - labeled_block (numpy.ndarray): 2D array with labeled data.
    - distance_transform_block (numpy.ndarray): 2D array with distance transform data.
    - time_step (int): The current time step index.
    - pixel_size (float): Scalar value to convert length units.

    Returns:
    - pandas.DataFrame: DataFrame with columns ['label', 'length', 'width_km', 'total_area_km2', 'total_perimeter_km', 'num_features', 'date'].
    """
    # Ensure blocks are NumPy arrays
    labeled_block = np.asarray(labeled_block)
    distance_transform_block = np.asarray(distance_transform_block)
    labeled_skel_block = np.asarray(labeled_skel_block)
    
    area_perimeter_df = compute_area_and_perimeter_df(labeled_block, pixel_size)

    if area_perimeter_df.empty:
        # If no features found, populate feature-wise columns with NaN
        final_df = pd.DataFrame({
            'label': 0,
            'length_km': 0,
            'width_km': 0,
            'area_km2': 0,
            'perimeter_km': 0,
            'date': time_step,
            'path': [None],
            'section': section,
            'section_area_km2': section_area,  
            'section_length_km': section_length,
            'pp_mean_%': pp_mean,
            'ra_area_km2': ra_area
        })
        return final_df

    # # Compute lengths using igraph
    # length_df = compute_length_igraph(labeled_skel_block, pixel_size)
    
    length_df = compute_length_single_graph(labeled_skel_block, pixel_size)
    
    # Process width using the paths
    paths = length_df['path'].tolist()
    width_df = process_edt_width(paths, distance_transform_block, pixel_size)

    if width_df.empty:
        # If no widths were calculated, set width_km to NaN
        length_df['width_km'] = np.nan
    else:
        # Assign the calculated widths to the length DataFrame
        length_df = length_df.reset_index(drop=True)
        width_df = width_df.reset_index(drop=True)
        length_df['width_km'] = width_df['width_km']

    # Add time_step information
    length_df['date'] = time_step
    length_df['section'] = section
    length_df['section_area_km2'] = section_area
    length_df['section_length_km'] = section_length
    length_df['pp_mean_%'] = pp_mean
    length_df['ra_area_km2'] = ra_area

    # Merge per-feature and per-block summaries
    # Assuming area_perimeter_df has one row per block
    # We need to merge it with each row of length_df
    final_df = length_df.merge(area_perimeter_df, on='label', how='left')

    # Select only the required columns
    final_df = final_df[['label', 'length_km', 'width_km', 
                         'area_km2', 'perimeter_km', 
                         'date', 'path', 'section', 
                         'section_area_km2', 'section_length_km',
                         'pp_mean_%', 'ra_area_km2']]

    return final_df

def compute_length_single_graph(labeled_skel_block, pixel_size):
    # Get the coordinates of skeleton pixels and their labels
    skeleton_pixels = np.argwhere(labeled_skel_block > 0)
    labels = labeled_skel_block[labeled_skel_block > 0]

    # Create a DataFrame for easy manipulation
    pixels_df = pd.DataFrame({
        'y': skeleton_pixels[:, 0],
        'x': skeleton_pixels[:, 1],
        'label': labels
    })
    pixels_df.index = pixels_df.index.astype(int)

    # Map coordinates to indices
    coord_to_idx = {(row.y, row.x): idx for idx, row in pixels_df.iterrows()}

    # Generate edges based on 8-connectivity
    structure = generate_binary_structure(2, 2)
    neighbors_offsets = np.argwhere(structure) - 1  # Offsets for neighbors

    edges = []
    for idx, row in pixels_df.iterrows():
        y, x = row.y, row.x
        label = row.label
        for dy, dx in neighbors_offsets:
            ny, nx = y + dy, x + dx
            neighbor_coord = (ny, nx)
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                neighbor_label = pixels_df.loc[neighbor_idx, 'label']
                if neighbor_label == label and neighbor_idx > idx:
                    edges.append((idx, neighbor_idx))

    # Create a single graph for all labels
    g = Graph()
    g.add_vertices(len(pixels_df))

    # Assign the 'name' attribute to vertices
    for idx, vertex in enumerate(g.vs):
        vertex['name'] = pixels_df.index[idx]

    if edges:
        g.add_edges(edges)

    # Initialize results list
    results = []
    labels_set = pixels_df['label'].unique()

    for label_val in labels_set:
        # Get indices of nodes belonging to the current label
        label_indices = pixels_df[pixels_df['label'] == label_val].index.tolist()
        if len(label_indices) == 0:
            continue
        
        # Extract the subgraph for the current label using induced_subgraph
        subgraph = g.induced_subgraph(label_indices)

        # Find the longest path in the subgraph
        path_length, path_coords = find_longest_path_subgraph(subgraph, pixels_df, pixel_size=30)

        results.append({
            'label': label_val,
            'length_km': path_length / 1e3,  # Convert to km
            'path': path_coords
        })


    df = pd.DataFrame(results)

    return df

def find_longest_path_subgraph(subgraph, pixels_df, pixel_size):
    if subgraph.vcount() == 0:
        return np.nan, np.array([])

    degrees = subgraph.degree()
    if not degrees:
        return np.nan, np.array([])
    max_deg_vertex = int(np.argmax(degrees))

    # First BFS to find one endpoint
    far_node = bfs_farthest_node(subgraph, max_deg_vertex)
    # Second BFS to find the actual farthest node from far_node
    farthest_node = bfs_farthest_node(subgraph, far_node)

    # Get the shortest path between far_node and farthest_node
    path = subgraph.get_shortest_paths(far_node, to=farthest_node, output="vpath")[0]
    if not path:
        return np.nan, np.array([])

    # Convert vertex indices back to global indices
    global_indices = [subgraph.vs[idx]['name'] for idx in path]
    # Get the coordinates from pixels_df
    path_coords = pixels_df.loc[global_indices, ['y', 'x']].values
    path_length = calculate_total_distance(path_coords, pixel_size)

    return path_length, path_coords

def bfs_farthest_node(graph, start_vertex):
    """
    Perform BFS to find the farthest node from the start_vertex.

    Parameters:
    - graph (igraph.Graph): The graph.
    - start_vertex (int): The starting vertex index.

    Returns:
    - int: The index of the farthest node.
    """
    # Compute shortest paths from the start_vertex to all other vertices
    distances = graph.shortest_paths_dijkstra(source=start_vertex, target=None, mode='ALL')[0]

    # Find the maximum distance and the corresponding vertex
    farthest_distance = max(distances)
    farthest_vertex = distances.index(farthest_distance)

    return farthest_vertex

def calculate_total_distance(points, pixel_size):
    """
    Calculate the total Euclidean distance of a path.

    Parameters:
    - points (numpy.ndarray): Array of (x, y) coordinates.
    - pixel_size (float): Size of a pixel to convert distances.

    Returns:
    - float: Total distance.
    """
    if len(points) < 2:
        return pixel_size  # Minimal length
    differences = np.diff(points, axis=0)
    distances = np.linalg.norm(differences * pixel_size, axis=1)
    total_distance = np.sum(distances)
    return total_distance

def process_edt_width(paths, distance_transform_block, pixel_size):
    """
    Process width using the provided paths and distance transform data.

    Parameters:
    - paths (list of numpy.ndarray): List of path coordinates.
    - distance_transform_block (numpy.ndarray): 2D array with distance transform data.
    - pixel_size (float): Size of a pixel to convert distances.

    Returns:
    - pandas.DataFrame: DataFrame with columns ['width_km'].
    """
    # Filter out empty paths
    non_empty_paths = [path for path in paths if len(path) > 0]  # Corrected condition

    if not non_empty_paths:
        return pd.DataFrame(columns=['width_km'])  # Return empty DataFrame if no valid paths

    # Flatten the list of paths
    flat_points = np.vstack(non_empty_paths)  # Shape: (total_points, 2)

    # Separate row and column indices
    flat_idx, flat_idy = flat_points[:, 0], flat_points[:, 1]

    # Ensure indices are within the bounds of the distance_transform_block
    flat_idx = np.clip(flat_idx, 0, distance_transform_block.shape[0] - 1)
    flat_idy = np.clip(flat_idy, 0, distance_transform_block.shape[1] - 1)

    # Get the corresponding distance transform values
    widths = distance_transform_block[flat_idx, flat_idy]

    # Determine the split indices for each path
    lengths = [len(path) for path in non_empty_paths]
    split_indices = np.cumsum(lengths)[:-1]

    # Split the widths back into individual paths
    width_segments = np.split(widths, split_indices)

    # Calculate the mean width for each path
    mean_widths = np.array([segment.mean() if len(segment) > 0 else np.nan for segment in width_segments])

    # Convert to km (assuming pixel_size is in meters and width is diameter)
    width_km = (mean_widths * pixel_size * 2) / 1e3  # diameter to radius if necessary

    # Create the DataFrame
    df = pd.DataFrame({'width_km': width_km})

    return df

def compute_area_and_perimeter_df(labeled_block, pixel_size):
    
    labeled_block = labeled_block.squeeze()
    labeled_block = np.asarray(labeled_block)
    
    assert labeled_block.ndim in [2, 3], f"labeled_block has invalid number of dimensions: {labeled_block.ndim}"
    
    # Compute region properties
    props = regionprops_table(labeled_block, properties=('label', 'area', 'perimeter_crofton'))
    
    # Calculate total area in km²
    area_km2 = props['area'] * (pixel_size ** 2) / 1e6
    
    # Calculate total perimeter in km
    perimeter_km = props['perimeter_crofton'] * pixel_size / 1e3
    
    # props['label'] = np.arange(1, len(props['label']) + 1)
    
    # Create a DataFrame with the results
    df = pd.DataFrame({
        'area_km2': area_km2,
        'perimeter_km': perimeter_km,
        'label': np.arange(1, len(props['label']) + 1),
    })
    
    return df


### Export Poligons ###

def export_polygons(da_wmask_feature, outdir, section, min_pool_size, pixel_size):
    
    time_data = da_wmask_feature.time.data
    date_list = pd.to_datetime(time_data).strftime('%Y-%m-%d').to_list()
    
    transform = da_wmask_feature.rio.transform()
    crs = da_wmask_feature.rio.crs

    polygon_gdf = extract_polygons_map_blocks(da_wmask_feature, date_list, transform, crs, section)
    
    filtered_polygon_gdf = filter_polygons(polygon_gdf, min_pool_size, pixel_size)
    
    return filtered_polygon_gdf
    
    filtered_polygon_gdf.to_file(os.path.join(outdir, f'Polygons_section_{section}.shp'))
def process_polygons_gdf(block, date, transform, crs, section):
    """
    Process a single block (time step) to extract polygons.

    Parameters:
    - block (numpy.ndarray): 2D array representing the raster for one time step.
    - date (str or datetime): Date corresponding to the current time step.
    - transform (Affine): Affine transformation for the raster.
    - crs (str or dict): Coordinate reference system.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing polygons for the block.
    """
    data_uint8 = block.astype('uint8')
    mask = data_uint8 == 1
    polygon_list = []
    for geom, value in shapes(data_uint8, mask=mask, connectivity=8, transform=transform):
        if value == 1:
            polygon = {
                'Date': date,
                'Type': 'polygon',
                'Section': section,
                'geometry': shape(geom)
            }
            polygon_list.append(polygon)
            
    # Define the expected columns
    columns = ['Date', 'Section', 'Type', 'geometry']
            
    polygon_gdf = gpd.GeoDataFrame(polygon_list, columns=columns, crs=crs)
    return polygon_gdf

def extract_polygons_map_blocks(da_wmask_feature, date_list, transform, crs, section):
    """
    Apply shp_process_polygons to each time step in the Dask array using map_blocks.

    Parameters:
    - dask_array (dask.array.Array): 3D Dask array with dimensions (time, y, x).
    - time_coords (list or array-like): List of dates corresponding to each time step.
    - transform (Affine): Affine transformation for the raster.
    - crs (str or dict): Coordinate reference system.

    Returns:
    - list of dask.delayed GeoDataFrame: List of GeoDataFrames for each time step.
    """
    polygon_tasks = []
    for i in range(da_wmask_feature.shape[0]):
        date = date_list[i]
        block = da_wmask_feature[i]
        # Use map_blocks to apply the function to each block
        task = da.map_blocks(
            process_polygons_gdf,
            block,
            date,
            transform,
            crs,
            section,
            dtype=object,
            chunks=(),
            name=f'polygon_{i}',
        )
        polygon_tasks.append(task)
    polygon_gdfs = compute(*polygon_tasks)
    # Filter out any None results just in case
    polygon_gdfs = [gdf for gdf in polygon_gdfs if gdf is not None]
    
    if not polygon_gdfs:
        # Return an empty GeoDataFrame with the correct columns and CRS
        combined_gdf = gpd.GeoDataFrame(columns=['Date', 'Section', 'Type', 'geometry'], crs=crs)
    else:
        # Concatenate all GeoDataFrames
        combined_gdf = gpd.GeoDataFrame(pd.concat(polygon_gdfs, ignore_index=True), crs=crs)
    
    return combined_gdf

def filter_polygons(polygon_gdf, min_pool_size, pixel_size):
    
    # Calculate minimum area based on min_pool_size and pixel_size
    min_area = min_pool_size * (pixel_size ** 2)  # e.g., 2 * 30^2 = 1800 m²

    # Ensure the CRS is projected for accurate area calculations
    if not polygon_gdf.crs.is_projected:
        raise ValueError("CRS must be projected to calculate area in meters.")
    
    # Calculate area in square meters
    polygon_gdf['area_m2'] = polygon_gdf.geometry.area
    
    # Calculate area in square kilometers
    polygon_gdf['area_km2'] = polygon_gdf['area_m2'] / 1e6  # 1 km² = 1,000,000 m²
    
    # Filter polygons based on minimum area
    filtered_gdf = polygon_gdf[polygon_gdf['area_m2'] >= min_area].copy()
    
    return filtered_gdf

### Export points and Lines ###

def retrieve_coordinates(indices, masked_da_coords):
    indices = np.array(indices)
    # Directly use numpy for coordinate extraction and shapely creation
    coords = np.column_stack((
        np.array(masked_da_coords['x'])[indices[:, 1]],
        np.array(masked_da_coords['y'])[indices[:, 0]]
    ))
    return LineString(coords)

def create_points(row):
    return [
        {'Date': row['date'], 'section': row['section'], 'line': int(row.name), 'Type': 'coord_start', 'geometry': Point(row['geometry'].coords[0])},
        {'Date': row['date'], 'section': row['section'], 'line': int(row.name), 'Type': 'coord_end', 'geometry': Point(row['geometry'].coords[-1])},
        {'Date': row['date'], 'section': row['section'], 'line': int(row.name), 'Type': 'midpoint', 'geometry': row['geometry'].interpolate(0.5, normalized=True)}
    ]

def export_points_and_lines(summary_ddf, section, masked_da_coords, outdir, crs):
    """
    Process and create GeoDataFrames for points and lines from attributes.

    Parameters:
    - df_attrs (pandas.DataFrame): DataFrame containing attributes.
    - masked_da_coords (xarray.DataArray): Masked DataArray coordinates.
    - crs (pyproj.CRS): Coordinate reference system.

    Returns:
    - tuple: GeoDataFrames for lines and points.
    """
    summary_ddf = summary_ddf[summary_ddf['path'
        ].apply(lambda x: x is not None and len(x) > 1)
        ]
    
    summary_ddf['geometry'] = summary_ddf['path'].apply(
        lambda x: retrieve_coordinates(x, masked_da_coords)
        )

    line_gdf = summary_ddf[['date', 'section', 'label', 'length_km', 'geometry']]
    line_gdf = gpd.GeoDataFrame(line_gdf, crs=crs)
    
    
    
    point_gdf = gpd.GeoDataFrame(
        [item for sublist in list(line_gdf.apply(create_points, axis=1)) for item in sublist],
        crs=crs
    )
    
    return line_gdf, point_gdf
    
    line_gdf.to_file(os.path.join(outdir, f'LineStrings_section_{section}.shp'))
    point_gdf.to_file(os.path.join(outdir, f'Points_section_{section}.shp'))
