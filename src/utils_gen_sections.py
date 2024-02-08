import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, LineString, MultiLineString
from rtree import index
from shapely.ops import snap, split, linemerge

from dask_image.ndmorph import binary_dilation
from skimage.measure import label

from src.utils_calc_metrics import validate, match_input_extent, fill_nodata, calculate_pixel_persistence

from src import utils_calc_metrics as utils_cm

### Catchment scale
def validate_and_preprocess(da_wmask, r_lines, str_order_col, initial_buffer):
    """
    Validates and preprocesses the input data for further processing in GIS operations.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray or str): The input dataset containing water masks, either as a DataArray or a directory path.
    r_lines (str): Path to a file containing river lines or similar linear features shapefile.
    str_order_col (str): The name of the column in `r_lines` representing stream order.

    Returns:
    tuple: A tuple containing processed data including initial buffer, filtered PP, CRS,
           dissolved river lines, and a list of stream orders. Specifically:
           - initial_buffer (int): Buffer size derived from preprocessing.
           - filtered_PP (xarray.core.dataarray.DataArray): Processed and filtered Pixel Persistence array.
           - crs (Coordinate Reference System): The CRS of the processed geospatial data.
           - r_lines_dissolved (GeoDataFrame): Dissolved river lines after processing.
           - str_order_list (list): List of stream orders extracted from `r_lines`.
    """
    # Call validate function to check and prepare the dataset
    da_wmask, r_lines = utils_cm.validate(da_wmask, r_lines, section_length=None, img_ext='.tif', module='generate_sections')

    # Preprocess the inputs `da_wmask` and `r_lines` to make them ready for further processing
    da_wmask, r_lines = preprocess_inputs(da_wmask, r_lines, initial_buffer)

    # Execute line pre-processing and store the results in multiple variables
    initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list = line_pre_processing (da_wmask, r_lines, str_order_col)
    
    return initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list

def preprocess_inputs(da_wmask, r_lines, initial_buffer):
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
    # Step 1: Buffer r_lines with initial buffer value
    initial_rcor_extent = r_lines.dissolve().buffer(initial_buffer, cap_style=2)
    initial_rcor_extent = gpd.GeoDataFrame(geometry=initial_rcor_extent, crs=r_lines.crs)
    
    # Step 2: Clip input data and extent to match dimensions
    da_wmask, r_lines = utils_cm.match_input_extent(da_wmask, initial_rcor_extent)

    # Step 3: Fill nodata values in the DataArray
    da_wmask = utils_cm.fill_nodata(da_wmask, initial_rcor_extent)
    
    return da_wmask, r_lines

def line_pre_processing (da_wmask, r_lines, str_order_col):
    """
    Pre-process the lines for section size estimation.
    """
    
    initial_buffer, filtered_PP = estimate_initial_buffer(da_wmask)  # Assuming estimate_initial_buffer is defined elsewhere
    crs = r_lines.crs

    # Check if str_order_col is provided, if not, create and populate a default column
    if str_order_col is None:
        # Create a new column 'default_col' and populate with a default value (e.g., 0)
        r_lines['str_order'] = 0
        str_order_col = 'str_order'
    
    r_lines_dissolved = r_lines.dissolve(by=str_order_col).sort_values(by=str_order_col, ascending=False)

    r_lines_dissolved['str_order'] = r_lines_dissolved.index
    str_order_list = list(r_lines_dissolved['str_order'])
    
    return initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list 

def estimate_initial_buffer (da_wmask):
    """
    Estimate the initial buffer size based on pixel persistence.

    Parameters:
    - da_wmask (xarray.DataArray): Input DataArray with water mask.

    Returns:
    - initial_buffer: The calculated initial buffer size.
    - filtered_PP: The filtered pixel persistence based on a threshold of 50%.
    """
    # Calculate pixel persistence using the input water mask
    PP = utils_cm.calculate_pixel_persistence(da_wmask)
    # Apply a threshold to retain values above 50%, converting to binary format
    filtered_PP = xr.where(PP > 0.5, 1, 0).astype('int8')
    # Calculate the Euclidean distance transform of the binary image
    euc_dist_trans_array = ndimage.distance_transform_edt(filtered_PP.values)   
    # Find the maximum distance, then scale and round it, ensuring a minimum size
    initial_buffer = round(euc_dist_trans_array.max() * filtered_PP.rio.transform()[0] * 1.1)
    # Round to next tens for standardization
    initial_buffer = ((initial_buffer // 10) + 1) * 10
    # Return the calculated initial buffer and the filtered pixel persistence
    return initial_buffer, filtered_PP

def process_lines (initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list):
    """
    Processes line features by buffering, splitting, and cleaning to prepare for GIS analysis.

    Parameters:
    initial_buffer (int): Buffer size derived from preprocessing.
    filtered_PP (xarray.core.dataarray.DataArray): Processed and filtered Pixel Persistence array.
    crs (Coordinate Reference System): The CRS of the processed geospatial data.
    r_lines_dissolved (GeoDataFrame): Dissolved river lines after processing.
    str_order_list (list): List of stream orders extracted from `r_lines`.

    Returns:
    tuple: A tuple containing the results of the line processing. Specifically:
           - buff_list (list): List of buffer geometries for each processed line section.
           - section_lengths (list): List of lengths for each line section.
           - section_widths (list): List of widths for each line section.
           - original_indices (list): List of original indices from the input data.
    """
    # Initialize empty lists for storing section lengths, widths, and original indices
    section_lengths, section_widths, original_indices = [], [], []
    # Create an empty DataFrame to store sections
    sections = pd.DataFrame()
    # Initialize an empty list to store buffer geometries
    buff_list = []
    # Initialize a variable for GeoDataFrame sections as None
    gdf_sections = None
    # Setup a context manager to ignore specific warnings during runtime
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Loop through each stream order in `str_order_list`
        for str_ord in str_order_list:
            print(str_ord)
            # Filter `r_lines_dissolved` for the current stream order and select the first row if not empty
            feature = r_lines_dissolved[r_lines_dissolved['str_order'] == str_ord]
            feature = feature.iloc[0] if not feature.empty else None      
            # Check if the feature is not None
            if feature is not None:
                # Create buffer geometries for the feature
                polygon_list = buffer_features(feature, initial_buffer)                
                # Estimate section length and width for the buffered polygons
                section_length, section_width = estimate_section_length_and_width(filtered_PP, polygon_list)
                # Create a buffer around the line geometry with specific width and styles
                line_buffer = gpd.GeoDataFrame(geometry = [feature.geometry.buffer(section_width, cap_style=2, join_style=2)], crs=crs)
                # Split lines and buffer them based on section length and width
                buffered_rivers, buffered_simplified_rivers = split_and_buffer_lines(feature, section_length, section_width, crs)
                # Clean the sections by erasing overlapping areas and merging polygons
                final_cleaned_sections = erase_overlapping_and_clean(buffered_rivers, buffered_simplified_rivers, line_buffer, crs)
                # Erase sections of higher order rivers if `gdf_sections` is not None
                if gdf_sections is not None:
                    final_cleaned_sections = gpd.GeoDataFrame(geometry = final_cleaned_sections.geometry, crs=crs)
                    final_cleaned_sections = gpd.overlay(final_cleaned_sections, gdf_sections, how='difference'
                                            ).buffer(-0.1, cap_style=2, join_style=2).buffer(0.1, cap_style=2, join_style=2
                                            ).explode(ignore_index=True).reset_index(drop=True)
                # Extend the lists with the calculated section lengths, widths, and original indices 
                section_lengths.extend([section_length] * len(final_cleaned_sections))
                section_widths.extend([section_width] * len(final_cleaned_sections))
                original_indices.extend([feature.name] * len(final_cleaned_sections))
                # Concatenate the current sections to the existing DataFrame
                sections = pd.concat([sections, final_cleaned_sections.geometry])
                # Convert the sections DataFrame into a GeoDataFrame
                gdf_sections = gpd.GeoDataFrame(geometry=sections[0])
                # Append the geometry of final cleaned sections to the buffer list
                buff_list.append(final_cleaned_sections.geometry)
    return buff_list, section_lengths, section_widths, original_indices

def buffer_features(feature, buffer_size):
    """
    Buffers a feature and splits it if it's a MultiPolygon.

    Parameters:
    feature (GeoSeries): The feature to process.
    buffer_size (float): The size of the buffer.

    Returns:
    List[Polygon]: A list of buffered polygons.
    """
    buffered_feature = feature.geometry.buffer(buffer_size, cap_style=2)
    if buffered_feature.geom_type == 'MultiPolygon':
        return list(buffered_feature.geoms)
    return [buffered_feature]

def estimate_section_length_and_width (filtered_PP, polygon_list):
    """
    Estimate the length and width of a section based on provided polygon geometries and raster data.

    This function clips the raster data array to the buffered geometries, then calculates 
    the Euclidean distance to determine the width, and uses skeletonization and region properties 
    to estimate the section length.

    Args:
    filtered_PP: A rasterio dataset object representing pre-processed data.
    polygon_list: A list of polygon geometries for clipping the raster data.

    Returns:
    A tuple containing the estimated section length and width.
    """ 
    # Clip the raster data array to the buffered geometries
    PP_clipped = filtered_PP.rio.clip(polygon_list)
    # Calculate Euclidean distance
    euc_dist_trans_array = ndimage.distance_transform_edt(PP_clipped.values)   
    # Determine section width based on the maximum Euclidean distance, ensuring it's at least 5 units
    section_width = round(max(euc_dist_trans_array.max() * 2, 5) * PP_clipped.rio.transform()[0])
    # Round width to the next tens for standardization
    section_width = ((section_width // 10) + 1) * 10
    # Skeletonize the layer using Lee's method to find the longest path (section length)
    skeleton = skeletonize(PP_clipped.values, method='lee')
    # Label the skeletonized image to identify connected regions
    labeled_skeleton = label(skeleton)
    regions = regionprops(labeled_skeleton)
    # Calculate average region length considering the number of regions
    total_num_pixels = sum(region.area for region in regions)
    average_pixels_per_region  = round((total_num_pixels / len(regions) if len(regions) > 0 else 1))
    # section_length = (section_length if section_length > 5 else 5) * PP_clipped.rio.transform()[0]
    # Convert average pixel count to actual length, ensuring it's at least 5 units
    section_length = max(average_pixels_per_region, 5) * PP_clipped.rio.transform()[0]
    # Round length to the next tens for standardization
    section_length = ((section_length // 10) + 1) * 10   
    return int(section_length), int(section_width)

def split_and_buffer_lines(feature, section_length, buffer_size, crs):
    """
    Buffers and splits a line feature based on the section length.

    Parameters:
    feature (GeoSeries): The line feature to process.
    section_length (float): The length of each section to split the line into.
    buffer_size (float): The size of the buffer.

    Returns:
    List[Polygon]: A list of buffered and split line sections.
    """
    splitted_lines = []
    feature = feature.copy()
    try:
        # Attempt to merge lines if they are multilines
        feature.geometry = linemerge(feature.geometry)
    except:
        pass

    # Process each line or multiline
    if feature.geometry.geom_type == 'MultiLineString':
        for line in feature.geometry.geoms:
            splitted_line = split_line_section_length(line, section_length)
            splitted_lines.append(splitted_line)
    else:
        splitted_line = split_line_section_length(feature.geometry, section_length)
        splitted_lines.append(splitted_line)

    # Flatten the list of lines and buffer them
    flat_lines = [item for sublist in splitted_lines for item in sublist]
    buffered_rivers_lst = [
        line.buffer(buffer_size, cap_style=2, join_style=1) 
                               for line in flat_lines]
    
    # Create a GeoDataFrame from the buffer list
    buffered_rivers = gpd.GeoDataFrame(geometry=buffered_rivers_lst, crs=crs)
    
    buffered_simplified_rivers_lst  = [
        LineString([line.coords[0], line.coords[-1]]).buffer(buffer_size, cap_style=2, join_style=1)
                                for line in flat_lines]

    buffered_simplified_rivers = gpd.GeoDataFrame(geometry=buffered_simplified_rivers_lst, crs=crs)
    
    return buffered_rivers, buffered_simplified_rivers

def split_line_section_length(line, section_length):
    """
    Split a line into segments of specified section length.

    This function redistributes vertices on the line and then splits it into
    segments. The section length is adjusted to ensure a whole number of segments
    are created, staying within 10% of the original section length.

    Args:
    - line (shapely.geometry.LineString): The line to be split.
    - section_length (float): The desired length of each section.

    Returns:
    - List[shapely.geometry.LineString]: Segments of the line.
    """
    # Redistribute vertices on the line
    line = redistribute_vertices(line, 10)
    # Calculate the total length of the line
    total_length = line.length
    # Determine the number of segments required
    num_segments = max(round(total_length / section_length), 1)
    # Adjust the section length to ensure an even distribution of segments.
    adjusted_min_length = total_length / num_segments
    
    if 0.9 * section_length <= adjusted_min_length <= 1.1 * section_length:
        section_length = adjusted_min_length
        
    # Generate points along the line at regular intervals
    mpoints = MultiPoint([line.interpolate(i * section_length) for i in range(num_segments)])
    # Split the line at the generated points
    splitted_line = split(snap(line, mpoints, 0.001), mpoints)
    
    # Filter out geometries to retain only LineString objects
    splitted_line = [geom for geom in splitted_line.geoms if isinstance(geom, LineString)]
    
    # # Filter out geometries to retain only LineString objects
    # splitted_line = [geom for geom in splitted_line.geoms if geom.geom_type == 'LineString']
    return splitted_line

def redistribute_vertices(geom, distance):
    """
    Redistribute vertices of a LineString or MultiLineString geometry.

    This function ensures that the vertices of the geometry are spaced apart
    by a specified distance. It handles LineString and MultiLineString geometries.

    Args:
    - geom (shapely.geometry.LineString/MultiLineString): The input geometry.
    - distance (float): The distance between each vertex.

    Returns:
    - shapely.geometry.LineString/MultiLineString: The geometry with redistributed vertices.

    Raises:
    - ValueError: If the input geometry type is neither LineString nor MultiLineString.
    """
    if geom.geom_type == 'LineString':
        # Calculate the number of vertices needed
        num_vert = max(int(round(geom.length / distance)), 1)
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        # Recursively apply the function to each part of the MultiLineString
        parts = [redistribute_vertices(part, distance)
                 for part in geom.geoms]
        # Filter out empty parts and return a MultiLineString
        return MultiLineString([p for p in parts if not p.is_empty])
    else:
        # Raise an error for unhandled geometry types
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

def erase_overlapping_and_clean (buffered_rivers, buffered_simplified_rivers, line_buffer, crs):
    """
    Performs overlay operations and cleaning on the provided GeoDataFrames.

    Parameters:
    buffered_rivers (GeoDataFrame): Original buffered GeoDataFrame.
    gdf_buff_simplified (GeoDataFrame): Simplified buffered GeoDataFrame.
    gdf_bend_buffer (GeoDataFrame): GeoDataFrame for bend buffering.
    crs (CRS): Coordinate Reference System for the resulting GeoDataFrames.

    Returns:
    GeoDataFrame: The final cleaned and processed GeoDataFrame.
    """
    # Erase overlaps in the original buffered GeoDataFrame
    buffered_rivers_erased = erase_overlaps(buffered_rivers, crs)
    # Erase overlaps in the simplified buffered GeoDataFrame
    buffered_simplified_rivers_erased = erase_overlaps(buffered_simplified_rivers, crs)
    # Perform overlay to find the difference between the original and simplified GeoDataFrames
    # Buffer the result to clean edges
    diff_polygons = buffered_rivers_erased.overlay(buffered_simplified_rivers_erased, how='difference', keep_geom_type=True
                                        ).buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2
                                        ).explode(ignore_index=True).reset_index(drop=True)  
    
    diff_polygons = diff_polygons[~diff_polygons.geometry.is_empty]                     
    diff_polygons = gpd.GeoDataFrame(geometry=diff_polygons, crs=crs)
        
    
    # Clean sections of the diff_polygons
    difference_geometry = merge_polygons_on_shared_boundary (buffered_simplified_rivers_erased, diff_polygons)

    # Dissolve the original buffered GeoDataFrame and clean the edges
    buffered_rivers_dissolved = buffered_rivers.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    buffered_rivers_dissolved = gpd.GeoDataFrame(geometry=buffered_rivers_dissolved, crs=crs)
    
    # Dissolve the cleaned diff GeoDataFrame and clean the edges
    difference_geometry_dissolved = difference_geometry.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    difference_geometry_dissolved = gpd.GeoDataFrame(geometry=difference_geometry_dissolved, crs=crs)
       
    # Perform an overlay to find the difference between the dissolved original and diff GeoDataFrames
    # Explode the result to separate multipart geometries into singlepart
    difference_overlay = buffered_rivers_dissolved.overlay(difference_geometry_dissolved, how='difference', keep_geom_type=True
                                        ).buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2
                                        ).explode(ignore_index=True).reset_index(drop=True)
    
    difference_overlay = difference_overlay[~difference_overlay.geometry.is_empty]
    difference_overlay = gpd.GeoDataFrame(geometry=difference_overlay, crs=crs)
    
    
    # Clean sections of the dissolved difference GeoDataFrame
    cleaned_sections = merge_polygons_on_shared_boundary (difference_geometry, difference_overlay)
    
    final_cleaned_sections = merge_small_polygons_with_nearest_neighbors(cleaned_sections)

    #Check for gaps
    
    # Dissolve the original buffered GeoDataFrame and clean the edges
    final_cleaned_sections_dissolved = final_cleaned_sections.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    final_cleaned_sections_dissolved = gpd.GeoDataFrame(geometry=final_cleaned_sections_dissolved, crs=crs)
    
    difference_overlay_gaps = line_buffer.overlay(final_cleaned_sections_dissolved, how='difference', keep_geom_type=True
                                    ).buffer(-1, cap_style=2, join_style=2).buffer(1, cap_style=2, join_style=2
                                    ).explode(ignore_index=True).reset_index(drop=True)
                                    
    difference_overlay_gaps = difference_overlay_gaps[~difference_overlay_gaps.geometry.is_empty]
    # difference_overlay_gaps = difference_overlay_gaps[difference_overlay_gaps.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    difference_overlay_gaps = gpd.GeoDataFrame(geometry=difference_overlay_gaps, crs=line_buffer.crs)
    
    if not difference_overlay_gaps.empty:
        # Clean sections of the dissolved difference GeoDataFrame
        cleaned_sections_gaps = merge_polygons_on_shared_boundary (final_cleaned_sections, difference_overlay_gaps)

        final_cleaned_sections = merge_small_polygons_with_nearest_neighbors(cleaned_sections_gaps)

        final_cleaned_sections = final_cleaned_sections.buffer(-0.1, cap_style=2, join_style=2).buffer(0.1, cap_style=2, join_style=2
                                            ).explode(ignore_index=True).reset_index(drop=True)
    
    return final_cleaned_sections

def erase_overlaps(buffered_rivers, crs):
    """
    Removes overlapping areas from geometries in a GeoDataFrame.
    
    Parameters:
    gdf_buff_list (GeoDataFrame): A GeoDataFrame containing geometries.
    crs (CRS): Coordinate Reference System to be used for the output GeoDataFrame.
    
    Returns:
    GeoDataFrame: A GeoDataFrame with overlapping areas removed.
    """
    diff_lst = []
    # Iterate through each buffer geometry to remove overlaps
    for geom_i in buffered_rivers.geometry:
        # Find geometries that overlap with the current geometry
        overlaps = buffered_rivers[buffered_rivers.geometry.overlaps(geom_i)]
        
        for geom_j in overlaps.geometry:
            try:
                # Calculate the difference between the geometries               
                intersection = geom_i.intersection(geom_j)
                if isinstance(geom_i, (Polygon, MultiPolygon)):
                    geom_i = geom_i.difference(intersection)
            # except Exception as e:
            except:
                # print(f"Skipping problematic geometry due to error: {e}")
                continue  # Skip to the next geometry

        # Buffer the geometry slightly inwards and outwards to clean the edges
        geom_i = geom_i.buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
        # Handle MultiPolygon by keeping the largest polygon
        if isinstance(geom_i, MultiPolygon):
            geom_i = max(geom_i.geoms, key=lambda polygon: polygon.area)
            # diff_lst.append(geom_i)
        # Add Polygon if it's not empty
        if isinstance(geom_i, Polygon) and not geom_i.is_empty:
            diff_lst.append(geom_i)
    return gpd.GeoDataFrame(geometry=diff_lst, crs=crs)

def merge_polygons_on_shared_boundary(target_geometries, polygons_to_merge, buffer_distance=0.02):
    """
    Merges each polygon in 'polygons_to_merge' with its nearest neighbour in 'target_geometries' 
    based on the longest shared boundary.

    Parameters:
    target_geometries (GeoDataFrame): A GeoDataFrame containing the main geometries.
    polygons_to_merge (GeoDataFrame): A GeoDataFrame containing polygons that need to be merged.
    buffer_distance (float): The buffer distance for boundary calculation.

    Returns:
    GeoDataFrame: The updated 'target_geometries' GeoDataFrame after merging.
    """   
    # Buffer the geometries in polygons_to_merge for intersection checks
    polygons_to_merge['buffered_geometry'] = polygons_to_merge.geometry.buffer(buffer_distance)
    # Create a spatial index for target_geometries for efficient querying
    target_geometries_sindex = target_geometries.sindex
    # Iterate through each polygon in polygons_to_merge
    for idx, row in polygons_to_merge.iterrows():
        # The current polygon to be merged
        current_polygon = row['buffered_geometry']
        
        # if current_polygon.is_valid:
        # Find indices of target geometries that potentially intersect with the current polygon
        neighbour_indices = list(target_geometries_sindex.intersection(current_polygon.bounds))
        # Extract potential neighboring geometries
        intersecting_candidates = target_geometries.iloc[neighbour_indices]
        # Filter to find actual overlapping neighbors
        overlapping_neighbours = intersecting_candidates[intersecting_candidates.geometry.overlaps(current_polygon)]
        # If there are overlapping neighbors, proceed with merging
        if not overlapping_neighbours.empty:
            # Find the index of the neighbor with the longest shared boundary
            longest_neighbour_idx = get_longest_shared_boundary(current_polygon, overlapping_neighbours, buffer_distance)
            # If a neighbor with the longest shared boundary is found
            if longest_neighbour_idx is not None:
                # Merge the current polygon with this neighbor
                merged_polygon = current_polygon.union(target_geometries.at[longest_neighbour_idx, 'geometry'])
                # Update the geometry in target_geometries
                target_geometries.at[longest_neighbour_idx, 'geometry'] = merged_polygon.buffer(buffer_distance).buffer(-buffer_distance)
    return target_geometries

def get_longest_shared_boundary(polygon, neighbours, buffer_distance=0.01):
    """
    Finds the neighbour polygon that shares the longest boundary with the given polygon.
    
    Parameters:
    polygon (Polygon): The reference polygon.
    neighbours (GeoDataFrame): A GeoDataFrame containing neighbour polygons.
    buffer_distance (float): The buffer distance for boundary calculation.
    
    Returns:
    int: The index of the neighbor with the longest shared boundary. Returns None if no shared boundaries are found.
    """
    max_length = 0
    longest_neighbor_idx = None
    # Buffer the boundary of the polygon
    polygon_buffered_boundary = polygon.boundary.buffer(buffer_distance)
    # Iterate through each neighbour to find the longest shared boundary
    for idx, neighbour in neighbours.iterrows():
        # Buffer the boundary of the neighbour
        neighbour_buffered_boundary = neighbour.geometry.boundary.buffer(buffer_distance)
        # Calculate the shared boundary
        shared_boundary = polygon_buffered_boundary.intersection(neighbour_buffered_boundary)
        # Skip if there is no shared boundary
        if shared_boundary.is_empty:
            continue
        # Calculate the length of the overlapping boundary
        shared_boundary_length = shared_boundary.length
        # Update max length and index if current boundary is longer
        if shared_boundary_length > max_length:
            max_length = shared_boundary_length
            longest_neighbour_idx = idx
    return longest_neighbour_idx

def merge_small_polygons_with_nearest_neighbors(cleaned_sections, std_dev_trheshold=2):
    """
    Merges smaller polygons in 'cleaned_sections' with their nearest neighbor based on area.

    Parameters:
    cleaned_sections (GeoDataFrame): A GeoDataFrame containing geometries to be processed.
    small_polygons (GeoDataFrame): A subset of 'cleaned_sections' containing smaller polygons.

    Returns:
    GeoDataFrame: The updated 'cleaned_sections' after merging smaller polygons.
    """
    # Calculate the area for each polygon
    cleaned_sections['area'] = cleaned_sections['geometry'].area
    # Calculate mean and standard deviation
    mean_area = cleaned_sections['area'].mean()
    std_dev_area = cleaned_sections['area'].std()
    # Threshold for filtering
    threshold = mean_area - std_dev_trheshold * std_dev_area
    # Filter polygons that are smaller than the threshold
    small_polygons = cleaned_sections[cleaned_sections['area'] < threshold]
    # Create spatial index for cleaned_sections
    cleaned_sections_sindex = cleaned_sections.sindex
    # Store the indices of small polygons to drop later
    indices_to_drop = []
    for idx, small_polygon in small_polygons.iterrows():
        # Get the current small polygon's geometry
        polygon_i = small_polygon.geometry
        # Find indices of geometries in cleaned_sections that potentially intersect the small polygon
        possible_matches_index = list(cleaned_sections_sindex.intersection(polygon_i.bounds))
        # Exclude the current small polygon from its potential matches
        possible_matches_index = [i for i in possible_matches_index if i != idx]
        # Proceed only if there are potential matches
        if possible_matches_index:
            # Retrieve the potential matching polygons from cleaned_sections
            possible_matches = cleaned_sections.iloc[possible_matches_index]
            # Filter to get only those polygons that actually intersect the small polygon
            intersected_cleaned_sections = possible_matches[possible_matches.geometry.intersects(polygon_i.buffer(0.02))]
            # If there are intersecting polygons
            if not intersected_cleaned_sections.empty:
                # Find the polygon with the smallest area among the intersecting ones
                intersected_index = intersected_cleaned_sections['area'].idxmin()
                # Merge the small polygon with the selected intersecting polygon
                merged_polygon = intersected_cleaned_sections.at[intersected_index, 'geometry'].union(polygon_i)
                # Update the geometry in cleaned_sections
                cleaned_sections.at[intersected_index, 'geometry'] = merged_polygon             
                # Add index to the list for later removal
                indices_to_drop.append(idx)      
    # Remove the original small polygons from cleaned_sections
    cleaned_sections = cleaned_sections.drop(indices_to_drop).reset_index(drop=True)
    # Update cleaned_sections to only keep valid geometries with non-zero area
    cleaned_sections = cleaned_sections[cleaned_sections['geometry'].notnull() & cleaned_sections['geometry'].area > 0]
    # Ensure all geometries are either Polygons or MultiPolygons
    cleaned_sections['geometry'] = [geom if isinstance(geom, (Polygon, MultiPolygon)) else None for geom in cleaned_sections.geometry]
    
    # Drop any rows with null geometries
    cleaned_sections = cleaned_sections.dropna(subset=['geometry'])
    
    return cleaned_sections

def postprocess_results(r_lines_dissolved, buff_list, section_lengths, section_widths, original_indices, str_order_col, crs):
    """
    Postprocesses the results by creating a GeoDataFrame with buffered geometries and
    performing spatial joins and merges for final GIS analysis.

    Parameters:
    r_lines_dissolved (GeoDataFrame): Dissolved river lines used in the processing.
    buff_list (list): List of buffer geometries from the line processing.
    section_lengths (list): List of lengths for each line section.
    section_widths (list): List of widths for each line section.
    original_indices (list): List of original indices from the input data.
    str_order_col (str): The name of the stream order column in the input data.
    crs (Coordinate Reference System): The CRS of the processed geospatial data.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the postprocessed results with buffered geometries,
                  section lengths, widths, and additional information from spatial joins.
    """
    # Combine buffered geometries into a GeoDataFrame with specified CRS
    buffered_gdf = gpd.GeoDataFrame(geometry=[item for sublist in buff_list for item in sublist], crs=crs)
    # Add columns for section lengths, widths, and stream order to the GeoDataFrame
    buffered_gdf['section_length'] = section_lengths
    buffered_gdf['section_width'] = section_widths
    buffered_gdf['stream_order'] = original_indices  # Add the original indices
    # Perform a spatial join between buffered_gdf and r_lines_dissolved to find intersecting features
    intersection_df = gpd.sjoin(buffered_gdf, r_lines_dissolved, predicate='intersects', how='inner', lsuffix='original', rsuffix=str_order_col)
    # Filter rows where the stream order matches the index of stream order column
    intersection_df = intersection_df[intersection_df['stream_order'] != intersection_df[f'index_{str_order_col}']]
    # Select only the index of stream order column
    intersection_df = intersection_df[f'index_{str_order_col}']
    # Merge the intersection_df with the buffered_gdf based on indices
    result_gdf = buffered_gdf.merge(intersection_df, 
                                    left_index=True, right_index=True, how='left')
    # Replace NaN values in the index of stream order column with -1
    result_gdf[f'index_{str_order_col}'].fillna(-1, inplace=True)
    # Rename the column to 'nodes' for clarity
    result_gdf.rename(columns={f'index_{str_order_col}': f'nodes'}, inplace=True)
    
    # Return the final processed GeoDataFrame
    return result_gdf

## Pool Scale


def process_masks(PP, lower_thresh, higher_thresh, min_num_pixel, radius=3):
    """
    Processes masks based on lower and higher thresholds.

    Parameters:
    - PP: Input data array.
    - lower_thresh: Lower threshold value.
    - higher_thresh: Higher threshold value.
    - radius: Radius for dilation structure.
    - min_num_pixel: Minimum number of pixels for region to be retained.

    Returns:
    - filtered_lower_mask: The final filtered lower mask.
    """
    # Create binary masks based on thresholds
    binmask_lt = PP > lower_thresh
    binmask_ht = PP > higher_thresh
    # Create structure for dilation
    structure = np.ones((radius, radius), dtype=bool)

    # Process lower threshold mask
    # Apply binary dilation and label the connected components
    dilated_mask_lt = binary_dilation(binmask_lt.data, structure=structure).astype(int)
    labeled_lt = label(dilated_mask_lt, connectivity=2)

    # Process higher threshold mask
    # Label connected components in the higher threshold mask
    labeled_ht = label(binmask_ht.data, connectivity=2)

    # Filter out small regions in higher threshold mask
    label_sizes = np.bincount(labeled_ht.ravel())[1:]  # Ignoring background (label 0)
    filtered_labels = np.nonzero(label_sizes >= min_num_pixel)[0] + 1 
    mask = np.isin(labeled_ht, filtered_labels)
    labeled_ht[~mask] = 0
    labeled_ht[labeled_ht != 0] = 1

    # Overlay masks to retain overlapping regions
    overlay_mask = labeled_lt * labeled_ht
    unique_values = np.unique(overlay_mask)[1:]  # Exclude zero
    retained_mask = np.isin(labeled_lt, unique_values)

    # Final filtered lower mask with only overlapping regions
    pool_mask = labeled_lt * retained_mask

    return pool_mask.astype('int32')

def process_pool_mask_to_gdf(pool_mask, PP, pixel_size=30):
    """
    Processes a data array to a GeoDataFrame by polygonizing the raster,
    filtering out polygons with value 0, filling holes, applying convex hull,
    and buffering by the specified pixel size.

    Parameters:
    - data_array: xarray.DataArray, the input data array.
    - pixel_size: int, the pixel size for buffering the polygons.

    Returns:
    - GeoDataFrame with processed polygons.
    """
    # Polygonize the raster
    polygons = list(rasterio.features.shapes(pool_mask, transform=PP.rio.transform()))

    # Filter out polygons with a value of 0
    filtered_polygons = [
        {"type": "Feature", "geometry": geom, "properties": {"id": value}} 
        for geom, value in polygons if value != 0
    ]
    # Create GeoDataFrame
    pools_aoi = gpd.GeoDataFrame.from_features(filtered_polygons, crs=PP.rio.crs)
    pools_aoi['geometry'] = pools_aoi['geometry'].apply(fill_polygon_holes)

    # Apply convex hull and buffer
    pools_aoi['geometry'] = pools_aoi['geometry'].convex_hull
    pools_aoi['geometry'] = pools_aoi['geometry'].buffer(PP.rio.resolution()[0])
    
    pools_aoi['id'] = range(1, len(pools_aoi) + 1)

    return pools_aoi

# Fill polygon holes
def fill_polygon_holes(geometry):
    if geometry.geom_type == 'Polygon':
        return Polygon(list(geometry.exterior.coords))
    elif geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(list(part.exterior.coords)) for part in geometry])
    return geometry