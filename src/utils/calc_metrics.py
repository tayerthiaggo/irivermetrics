# Standard library imports
import os
from collections import deque

# Third-party imports
import dask.array as da
import dask_image.ndmeasure as dask_ndmeasure
from dask import delayed
from dask_regionprops import regionprops
import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.xr import xr_reproject
from pyproj import CRS
from rasterio.features import rasterize, shapes
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, shape
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
import xarray as xr
import networkx as nx

# Local imports
from src.utils import wd_batch

# Public functions
def validate(da_wmask, rcor_extent, outdir, img_ext):
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
    
        # Prepare the output directory to store results
    # if outdir is None:
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
        rcor_extent = validate_shp_input(rcor_extent, crs)

    # Replace NaN values and validate CRS
    da_wmask = wd_batch.replace_nodata(da_wmask, -1)
    da_wmask.attrs['_FillValue'] = -1
    da_wmask, crs, pixel_size = validate_data_array_cm(da_wmask, crs)

    print('Input data validated.')
    return da_wmask, rcor_extent, crs, pixel_size, outdir

def preprocess(da_wmask, rcor_extent):
    print('Preprocessing...')
    da_wmask, rcor_extent = match_input_extent(da_wmask, rcor_extent)
    da_wmask = fill_nodata(da_wmask, rcor_extent)
    return da_wmask, rcor_extent

def calculate_pixel_persistence(da_wmask):   
    """
    Calculate pixel persistence for the entire DataArray.

    Parameters:
    - da_wmask (xarray.DataArray): DataArray containing water mask data.

    Returns:
    - xarray.DataArray: DataArray containing pixel persistence values.
    """
    # Total number of observations in the time dimension
    total_obs = da_wmask.sizes['time'] 
    # Sum wet observations over time
    wet_sum = da_wmask.sum(dim='time')
    # Normalize by total observations to get persistence ratio
    p_area = wet_sum / total_obs
    # Set pixels with no wet observations to -1
    PP = p_area.where(p_area > 0, -1).chunk('auto')
    PP.attrs['_FillValue'] = -1
    return PP

@delayed
def process_PP_metrics (PP, feature, pixel_size):
    """
    Process pixel persistence metrics for a specific feature.

    Parameters:
    - PP (xarray.DataArray): DataArray containing pixel persistence values.
    - feature (geopandas.GeoDataFrame): Feature to process.
    - pixel_size (float): Size of each pixel in the DataArray.

    Returns:
    - dict: Dictionary containing pixel persistence metrics for the feature.
    """
    section = feature.name
    PP_feature = clip_da_to_boundaries(PP, feature)
    transform = PP_feature.rio.transform()
    mask = mask_da(PP_feature, feature, transform)
    masked_PP = da.where(mask == 1, PP_feature, 0)

    pixel_area_km2 = pixel_size**2 / 10**6
    
    pp_mean = np.array(masked_PP[masked_PP > 0.1]).mean()
    ra_area = np.array(da.where(masked_PP > 0.9, pixel_area_km2, 0)).sum()

    PP_row = {
        'section': section,
        'PP_mean': pp_mean,
        'RA_area_km2': ra_area
    }
    return PP_row

@delayed
def process_feature_time(feature, da_wmask_time, min_pool_size, section_length, pixel_size, export_shp, crs):
    """
    Process metrics for a specific feature (Polygon) over time.

    Parameters:
    - feature (geopandas.GeoDataFrame): Feature to process.
    - da_wmask_time (xarray.DataArray): DataArray for the specific time step.
    - min_pool_size (int): Minimum size of water pools to consider in the analysis in pixels.
    - section_length (float): Length of the section in kilometers.
    - pixel_size (float): Size of each pixel in the DataArray.
    - export_shp (bool): Whether to export shapefiles.
    - crs (pyproj.CRS): Coordinate reference system.

    Returns:
    - tuple: DataFrame with metrics and GeoDataFrames for points, lines, and polygons if export_shp is True.
    """
    section = feature.name
    section_area = feature.geometry.area / 1e6
    date = pd.to_datetime(da_wmask_time.time.data).strftime('%Y-%m-%d')
    
    da_wmask_feature = clip_da_to_boundaries(da_wmask_time, feature)
    transform = da_wmask_feature.rio.transform()
    mask = mask_da(da_wmask_feature, feature, transform)
    masked_da = da_wmask_feature.where(mask == 1, other=0)  
    masked_da_coords = masked_da.coords
    layer = masked_da.data.rechunk('auto')
    
    if da.logical_not(layer).all():
        df_attrs = pd.DataFrame([{
        'label': 0, 'path': [], 'length': 0, 'width': 0,
        'area_km2': 0, 'length_km': 0, 'width_km': 0, 'perimeter_km': 0,
        'date': date, 'section': section, 'section_area_km2': section_area,
        'section_length': section_length}])

        point_gdf, line_gdf, polygon_gdf = None, None, None
    else:
        df_attrs = process_layer(layer, min_pool_size, pixel_size, date, section, section_area, section_length)

    if export_shp:
        point_gdf, line_gdf, polygon_gdf = process_export_shp (layer, df_attrs, masked_da_coords, transform, crs)
    else:
        point_gdf, line_gdf, polygon_gdf = None, None, None
   
    return df_attrs, point_gdf, line_gdf, polygon_gdf

def process_metrics(group):
    areas = group['area_km2']
    lengths = group['length_km']
    widths = group['width_km']
    perimeters = group['perimeter_km']
    section_area_km2 = group['section_area_km2'].iloc[0]
    section_length = group['section_length'].iloc[0]

    npools = int(areas.size)
    
    if npools == 0:
        return pd.Series({
        'section_area_km2': section_area_km2,
        'section_length': section_length,
        'npools': npools,  'wet_area_km2': 0,
        'wet_perimeter_km': 0, 'wet_length_km': 0, 'AWMSI': 0, 'AWRe': np.nan,
        'AWMPA': 0, 'AWMPL': 0, 'AWMPW': 0, 'PF': 0, 'PFL': 0, 'APSEC': 0, 'LPSEC': 0
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
        LPSEC = (total_wetted_length / section_length) * 100

        return pd.Series({
            'section_area_km2': section_area_km2,
            'section_length': section_length,
            'npools': npools, 
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
            'LPSEC': LPSEC
        })


# Helper functions
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
    
    pixel_size = da_wmask.rio.resolution()[0]
    
    return da_wmask, crs, pixel_size

def validate_shp_input(rcor_extent, img_crs):
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
    gdf = gdf.to_crs(da_wmask.rio.crs.to_string())
        
    transform = da_wmask.rio.transform()
    mask_dummy = da_wmask.isel(time=0).data.copy()
    shapes = gdf.geometry
    # Map the function across blocks
    mask = da.map_blocks(
        map_rasterize,
        mask_dummy,
        chunks=da_wmask.chunks[1:3],
        meta=np.array((), dtype=da_wmask.dtype),
        shapes=shapes,
        transform=transform
    )
    da_wmask_array = da.where((da_wmask == -1) & (mask == 1), 2, da_wmask)
    da_wmask = xr.DataArray(da_wmask_array,
                        coords=da_wmask.coords, 
                        dims=da_wmask.dims,
                        attrs=da_wmask.attrs
                        )
        
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
    
    # Apply fill_nodata_layer function with Dask's map_overlap to the updated DataArray
    filled_data = da.map_overlap(
        fill_nodata_layer,
        da_wmask.data,
        depth={'time': 2},  # Buffer size for two layers ahead/behind
        boundary={'time': 'reflect'}, # Reflect boundary for edge chunks
        dtype=da_wmask.dtype
    )
    da_wmask_data = da.where(filled_data == 1, filled_data, 0)
    da_wmask = xr.DataArray(da_wmask_data,
                    coords=da_wmask.coords, 
                    dims=da_wmask.dims,
                    attrs=da_wmask.attrs
                    )
    return da_wmask

def label_and_distance_layer(layer, min_pool_size):
    """
    Labels connected components in a binary layer and computes distance transform.

    Parameters:
    - layer (dask.array.Array): The binary layer to process.
    - min_pool_size (int): Minimum size of water pools to consider in the analysis in pixels.

    Returns:
    - tuple: Labeled layer, number of features, properties, and distance transform layer.
    """
    distance_transform_layer = da.map_overlap(
        lambda block: distance_transform_edt(block),
        layer,
        depth=1,
        dtype=float,
    ).rechunk('auto')
    
    structure = np.ones((3, 3))  
    labeled_layer, _ = dask_ndmeasure.label(layer, structure=structure)
    
    filtered_labels = da.map_overlap(
        lambda block: remove_small_objects(block, min_size=min_pool_size, connectivity=2),
        labeled_layer,
        depth=1,
        dtype='int16',
    )
    
    labeled_layer, num_features = dask_ndmeasure.label(filtered_labels, structure=structure)
    props = regionprops(labeled_layer, properties=('label', 'area', 'perimeter_crofton'))
    
    return labeled_layer.rechunk('auto'), num_features, props, distance_transform_layer

def adaptive_skeletonization(labeled_layer, num_features, max_depth=10):
    """
    Perform skeletonization with adaptive boundary conditions to ensure connectivity.

    Parameters:
    - labeled_layer (dask.array.Array): The labeled layer to skeletonize.
    - num_features (int): Number of features in the labeled layer.
    - max_depth (int): Maximum depth for boundary conditions. Defaults to 10.

    Returns:
    - tuple: Coordinates and labels of skeletonized features.
    """
    depth = 1
    num_features_skel = 0
    structure = np.ones((3, 3))

    while num_features != num_features_skel and depth <= max_depth:
        # Apply map_overlap with the adaptive boundary parameter
        skeleton = da.map_overlap(
            lambda block: skeletonize(block),
            labeled_layer,
            depth=(depth, depth),
            boundary='reflect',
            dtype='uint8'
        )
        
        # Label connected components
        labeled_skel, num_features_skel = dask_ndmeasure.label(skeleton, structure=structure)
        
        if num_features == num_features_skel:
            break
        
        depth += 2  # Increase depth for the next iteration if needed
   
    pixels = np.array(da.argwhere(labeled_skel > 0))
    labels = np.array(labeled_skel[labeled_skel > 0])

    return pixels, labels

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
    
def shp_process_points_and_lines(df_attrs, masked_da_coords, crs):
    """
    Process and create GeoDataFrames for points and lines from attributes.

    Parameters:
    - df_attrs (pandas.DataFrame): DataFrame containing attributes.
    - masked_da_coords (xarray.DataArray): Masked DataArray coordinates.
    - crs (pyproj.CRS): Coordinate reference system.

    Returns:
    - tuple: GeoDataFrames for lines and points.
    """
    df_attrs = df_attrs[df_attrs['path'].apply(lambda x: len(x) > 1)]
    
    df_attrs['geometry'] = df_attrs['path'].apply(
        lambda x: retrieve_coordinates(x, masked_da_coords)
        )

    line_gdf = df_attrs[['date', 'section', 'label', 'length', 'geometry']]

    line_gdf = gpd.GeoDataFrame(line_gdf, crs=crs)

    # Apply the function to each row and explode the list of dictionaries into a flat DataFrame
    point_gdf = gpd.GeoDataFrame(
        [item for sublist in line_gdf.apply(create_points, axis=1).tolist() for item in sublist],
        crs=crs
    )
    return line_gdf, point_gdf

def shp_process_polygons(layer, df_attrs, transform, crs):
    """
    Process and create GeoDataFrame for polygons from a layer.

    Parameters:
    - layer (dask.array.Array): Binary layer containing polygons.
    - df_attrs (pandas.DataFrame): DataFrame containing attributes.
    - transform (Affine): Affine transformation for the layer.
    - crs (pyproj.CRS): Coordinate reference system.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing polygons.
    """
    data_uint8 = layer.astype('uint8')
    mask = data_uint8 == 1
    # Generate polygons from shapes where mask is True
    polygon_list = [{'Date': df_attrs['date'].iloc[0], 'section': df_attrs['section'].iloc[0], 'Type': 'polygon', 'geometry': shape(polygon[0])}
                     for polygon in shapes(data_uint8, mask=mask, connectivity = 8, transform=transform)]
    polygon_gdf = gpd.GeoDataFrame([gdf for gdf in polygon_list if gdf is not None], crs=crs)
    return polygon_gdf

def clip_da_to_boundaries(da_wmask, feature):
    xmin, ymin, xmax, ymax = feature.geometry.bounds
    col_mask = (da_wmask.x >= xmin) & (da_wmask.x <= xmax)
    row_mask = (da_wmask.y >= ymin) & (da_wmask.y <= ymax)
    da_wmask_feature = da_wmask.sel(x=col_mask, y=row_mask)
    return da_wmask_feature

def map_rasterize(block, shapes, transform):
    # Get block metadata
    block_shape = block.shape[-2:]
    # Rasterize the geometry for the current block
    rasterized_polygon = rasterize(
        shapes=shapes,
        out_shape=block_shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype='int8',
    )
    return rasterized_polygon

def mask_da(da_wmask_feature, feature, transform):
    """
    Mask a DataArray using a feature's geometry.

    Parameters:
    - da_wmask_feature (xarray.DataArray): DataArray to be masked.
    - feature (geopandas.GeoDataFrame): Feature used for masking.
    - transform (Affine): Affine transformation for the DataArray.

    Returns:
    - dask.array.Array: Masked DataArray.
    """
    shapes = [(feature.geometry, 1)]
    # Map the function across blocks
    mask = da.map_blocks(
        map_rasterize,
        da_wmask_feature.data,
        chunks=da_wmask_feature.chunks,
        meta=np.array((), dtype=da_wmask_feature.dtype),
        shapes=shapes,
        transform=transform
    )
    return mask

def process_length(pixels, labels):
    """
    Process pool length from skeletonized coordinates.

    Parameters:
    - pixels (numpy.ndarray): Coordinates of the skeletonized pixels.
    - labels (numpy.ndarray): Labels of the skeletonized pixels.

    Returns:
    - pandas.DataFrame: DataFrame containing length and width metrics.
    """
    def bfs_farthest_node(G, start_node):
        visited = {start_node: 0}
        queue = deque([start_node])
        last_node = start_node

        while queue:
            node = queue.popleft()
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)
                    last_node = neighbor

        return last_node, visited

    def calculate_total_distance(points):
        if len(points) < 2:
            return 1
        differences = np.diff(points, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        total_distance = np.sum(distances)
        return total_distance

    def find_longest_path_in_subgraph(G, label):
        subgraph = G.subgraph([node for node in G if G.nodes[node]['label'] == label])
        if subgraph.number_of_nodes() == 0:
            return label, [], 0, 0

        start_node = next(iter(subgraph.nodes))
        far_node, _ = bfs_farthest_node(subgraph, start_node)
        farthest_node, distances = bfs_farthest_node(subgraph, far_node)

        path = []
        current_node = farthest_node
        while distances[current_node] > 0:
            path.append(current_node)
            current_node = next(n for n in subgraph.neighbors(current_node) if distances[n] == distances[current_node] - 1)
        path.append(far_node)

        path_length = calculate_total_distance(np.array(path))
        
        return label, path, path_length

    def calculate_length(G):
        labels = set(nx.get_node_attributes(G, 'label').values())
        results = [find_longest_path_in_subgraph(G, label) for label in labels]
        df = pd.DataFrame(results, columns=['label', 'path', 'length'])      
        return df
    
    # Create a graph
    G = nx.Graph()

    # Define 8-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                
    for (x, y), label in zip(pixels, labels):
        G.add_node((x, y), label=label)
        for dx, dy in neighbors:
            neighbor = (x + dx, y + dy)
            if neighbor in G.nodes and G.nodes[neighbor]['label'] == label:
                G.add_edge((x, y), neighbor)
                
    df = calculate_length(G)
    
    return df

def process_edt_width(paths, distance_transform_layer):
    """
    Process width metrics using distance transform values.

    Parameters:
    - paths (list): List of paths to process.
    - distance_transform_layer (dask.array.Array): Distance transform layer.

    Returns:
    - list: List of mean widths for each path.
    """
    # Flatten the list of paths and convert to numpy arrays
    flat_idx, flat_idy = zip(*[point for path in paths for point in path])
    flat_idx = np.array(flat_idx)
    flat_idy = np.array(flat_idy)
    
    # Get the corresponding distance transform values
    widths = distance_transform_layer.vindex[flat_idx, flat_idy]
    
    # Split the widths back into lists corresponding to each path
    split_indices = np.cumsum([len(path) for path in paths])[:-1]
    width_segments = np.split(widths, split_indices)
    
    # Calculate the mean width for each path
    mean_widths = [np.mean(segment) for segment in width_segments]
    
    return mean_widths

def process_layer(layer, min_pool_size, pixel_size, date, section, section_area, section_length):
    """
    Process a binary layer to calculate various metrics.

    Parameters:
    - layer (dask.array.Array): Binary layer to process.
    - min_pool_size (int): Minimum size of water pools to consider in the analysis in pixels.
    - pixel_size (float): Size of each pixel in the DataArray.
    - date (str): Date of the layer.
    - section (int): Section number.
    - section_area (float): Area of the section in square kilometers.
    - section_length (float): Length of the section in kilometers.

    Returns:
    - pandas.DataFrame: DataFrame containing calculated metrics.
    """
    labeled_layer, num_features, props, distance_transform_layer = label_and_distance_layer(layer, min_pool_size)
            
    pixels, labels = adaptive_skeletonization(labeled_layer, num_features, max_depth=10)
    
    df_attrs = process_length(pixels, labels)
    
    df_attrs['width'] = process_edt_width(df_attrs['path'], distance_transform_layer)
        
    df_attrs['area_km2'] = (props['area'].to_dask_array(lengths=True) * (pixel_size**2)) / 1e6
    df_attrs['perimeter_km'] = (props['perimeter_crofton'].to_dask_array(lengths=True) * pixel_size) / 1e3
    df_attrs['length_km'] = (df_attrs['length'] * pixel_size) / 1e3
    df_attrs['width_km'] = (df_attrs['width'] * pixel_size * 2) / 1e3
    df_attrs['date'] = date
    df_attrs['section'] = section
    df_attrs['section_area_km2'] = section_area
    df_attrs['section_length'] = section_length
   
    return df_attrs

def process_export_shp(layer, df_attrs, masked_da_coords, transform, crs):
    """
    Process and export shapefiles for points, lines, and polygons.

    Parameters:
    - layer (dask.array.Array): Binary layer containing polygons.
    - df_attrs (pandas.DataFrame): DataFrame containing attributes.
    - masked_da_coords (xarray.DataArray): Masked DataArray coordinates.
    - transform (Affine): Affine transformation for the layer.
    - crs (pyproj.CRS): Coordinate reference system.

    Returns:
    - tuple: GeoDataFrames for points, lines, and polygons.
    """
    line_gdf, point_gdf = shp_process_points_and_lines(df_attrs, masked_da_coords, crs)
    polygon_gdf = shp_process_polygons(layer, df_attrs, transform, crs)
    return point_gdf, line_gdf, polygon_gdf