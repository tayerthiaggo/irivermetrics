import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
from skimage.measure import label
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from dask_image.ndmorph import binary_dilation

## Pool Scale

def process_masks(PP, lower_thresh, higher_thresh, min_num_pixel, radius=3):    
    """
    Processes water masks from an input data array based on specified threshold values, 
    filters out small regions, and retains regions present in both lower and higher 
    threshold masks after dilation.

    Args:
    - PP (np.ndarray): Input pixel persistence 2D data array for which masks will be generated.
    - lower_thresh (float): Lower threshold value for generating the initial mask.
    - higher_thresh (float): Higher threshold value for generating a secondary mask used 
                              for filtering.
    - min_num_pixel (int): Minimum number of pixels a region must have to be retained in 
                           the higher threshold mask.
    - radius (int, optional): Radius for the dilation structure used to dilate the lower 
                              threshold mask. Defaults to 3.

    Returns:
    - np.ndarray: A 2D integer array where each non-zero pixel is part of a region that 
                  exceeds the lower threshold, overlaps with regions exceeding the higher 
                  threshold, and the region size meets the minimum pixel criteria.
    """   
    # Generate binary masks for both thresholds: one for the lower and one for the higher
    binmask_lt = PP >= lower_thresh
    binmask_ht = PP >= higher_thresh
    # Define a structuring element for dilation with the given radius; this is a square matrix of ones
    structure = np.ones((radius, radius), dtype=bool)

    # Dilate the lower threshold mask to increase the size of regions; 
    # this helps in bridging small gaps between close-by regions
    dilated_mask_lt = binary_dilation(binmask_lt.data, structure=structure).astype(int)
    labeled_lt = label(dilated_mask_lt, connectivity=2)

    # Label connected components in both the dilated lower threshold mask and the higher threshold mask
    # Connectivity=2 implies 8-connectivity in 2D, considering diagonals as connections
    labeled_ht = label(binmask_ht.data, connectivity=2)

    # Filter out small regions in the higher threshold mask based on the minimum number of pixels
    label_sizes = np.bincount(labeled_ht.ravel())[1:]  # Ignoring background (label 0)
    filtered_labels = np.nonzero(label_sizes >= min_num_pixel)[0] + 1 
    mask = np.isin(labeled_ht, filtered_labels)
    labeled_ht[~mask] = 0
    labeled_ht[labeled_ht != 0] = 1

    # Retain only the regions in the dilated lower threshold mask that overlap with filtered higher threshold mask regions
    overlay_mask = labeled_lt * labeled_ht
    
    # Identify unique, non-zero regions in the overlay mask to ensure they are from both initial masks
    unique_values = np.unique(overlay_mask)[1:]  # Exclude zero
    retained_mask = np.isin(labeled_lt, unique_values)

    # The final mask is the lower threshold mask adjusted for overlap and filtering criteria
    pool_mask = labeled_lt * retained_mask

    return pool_mask.astype('int32')

def process_pool_mask_to_gdf(pool_mask, PP):
    """
    Converts a binary raster (pool_mask) into a GeoDataFrame by polygonizing the raster data,
    filtering out polygons with a value of 0, filling holes within the polygons, applying a buffer 
    to each polygon, and simplifying the geometry.

    Args:
    - pool_mask (np.ndarray): A 2D array where each element represents a pixel in the raster. 
                              Non-zero values are considered as features to be polygonized.
    - PP (xarray.DataArray): The input pixel persistence data array with spatial metadata, including the 
                             coordinate reference system (CRS) and affine transform for 
                             georeferencing.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing polygons derived from the input raster, 
                        with holes filled, and geometries buffered and simplified. Each 
                        polygon is assigned a unique 'id'.
    """
    # Polygonize the raster data to generate polygons for each contiguous non-zero region
    polygons = list(rasterio.features.shapes(pool_mask, transform=PP.rio.transform()))

    # Filter out polygons corresponding to a value of 0, which are not of interest
    filtered_polygons = [
        {"type": "Feature", "geometry": geom, "properties": {"id": value}} 
        for geom, value in polygons if value != 0
    ]
    # Convert the filtered polygons into a GeoDataFrame, preserving the spatial reference
    pools_aoi = gpd.GeoDataFrame.from_features(filtered_polygons, crs=PP.rio.crs)
    
    # Fill holes in each polygon to ensure solid geometries
    pools_aoi['geometry'] = pools_aoi['geometry'].apply(fill_polygon_holes)

    # Buffer each polygon to increase their area slightly; the buffer distance is three times the pixel size.
    pools_aoi['geometry'] = pools_aoi['geometry'].buffer(PP.rio.resolution()[0]*3)
    
    # Simplify the geometries to reduce complexity while maintaining the general shape.
    pools_aoi['geometry'] = pools_aoi['geometry'].simplify(PP.rio.resolution()[0])
    
    # Assign a unique ID to each polygon for easier identification and analysis
    pools_aoi['id'] = range(0, len(pools_aoi))

    return pools_aoi

def fill_polygon_holes(geometry):
    """
    Removes holes in a given Polygon or MultiPolygon geometry by reconstructing
    the geometry using only the exterior coordinates.
    
    Args:
    - geometry (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The input
      geometry from which holes will be removed.

    Returns:
    - shapely.geometry.Polygon or shapely.geometry.MultiPolygon: A new geometry of the
      same type as the input, but with all holes removed. If the input geometry type
      is neither Polygon nor MultiPolygon, the original geometry is returned.
    """
    if geometry.geom_type == 'Polygon':
        # For a Polygon, create a new Polygon using only the exterior coordinates.
        return Polygon(list(geometry.exterior.coords))
    elif geometry.geom_type == 'MultiPolygon':
        # For a MultiPolygon, iterate over each Polygon, removing holes,
        # and create a new MultiPolygon from the results.
        return MultiPolygon([Polygon(list(part.exterior.coords)) for part in geometry])
    return geometry

def merge_overlapping_polygons(pools_aoi):
    """
    Merges overlapping polygons within a GeoDataFrame.

    This function iterates over a GeoDataFrame (gdf) containing polygon geometries,
    identifies and merges overlapping polygons into a single polygon, and repeats
    this process until no further overlaps exist. The function ensures the geometry
    column is valid before proceeding with the merge operations.

    Parameters:
    - gdf (geopandas.GeoDataFrame): A GeoDataFrame with a column of polygon geometries.

    Returns:
    - geopandas.GeoDataFrame: A new GeoDataFrame with overlapping polygons merged.

    Note:
    - The function maintains the original CRS (Coordinate Reference System) of the input GeoDataFrame.
    - It uses a spatial index to efficiently find potential overlaps among polygons.
    """
    # Ensure the geometry column is valid by filtering out invalid geometries
    pools_aoi = pools_aoi[pools_aoi.is_valid]
    
    merged = False # Initialize merged flag to False to enter the while loop
    while not merged:
        # Create a spatial index for the GeoDataFrame to optimize overlap detection
        sindex = pools_aoi.sindex
        
        new_geoms = [] # Initialize a list to store the geometries of merged or non-overlapping polygons
        merged_indices = set() # Keep track of indices of polygons that have been merged
        
        for idx, polygon in pools_aoi.iterrows():
            # Skip the polygon if it has already been merged
            if idx in merged_indices:
                continue
            # Find indices of potentially overlapping polygons using the spatial index
            possible_matches_index = list(sindex.intersection(polygon.geometry.bounds))
            # Retrieve the potentially overlapping polygons using their indices
            possible_matches = pools_aoi.iloc[possible_matches_index]
            # Filter to get truly overlapping polygons
            overlaps = possible_matches[possible_matches.geometry.overlaps(polygon.geometry)]
            
            # Include the current polygon in the set of overlaps to ensure it is merged
            overlaps = pd.concat([overlaps, pools_aoi.loc[[idx]]])
            
            if len(overlaps) > 1:
                # Merge all overlapping polygons into a single geometry
                new_geom = unary_union(overlaps.geometry)
                new_geoms.append(new_geom)
                # Update the set of merged indices with the indices of merged polygons
                merged_indices.update(overlaps.index)
            else:
                # If there are no overlaps, add the original polygon geometry to the list
                new_geoms.append(polygon.geometry)
                
        # Create a new GeoDataFrame with the merged and original non-overlapping geometries
        new_gdf = gpd.GeoDataFrame(geometry=new_geoms, crs=pools_aoi.crs)
        
        # Check if no further merges occurred by comparing lengths of the old and new GeoDataFrames
        if len(new_gdf) == len(pools_aoi):
            merged = True # If true, exit the loop
        else:
            # If merges occurred, set the new GeoDataFrame as the one to iterate over in the next loop
            pools_aoi = new_gdf
    
    return new_gdf