import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, LineString, MultiLineString
from shapely.ops import snap, split, linemerge
from dask_image.ndmorph import binary_dilation

### Catchment scale

def validate_and_preprocess(da_wmask, r_lines, str_order_col, initial_buffer):
    """
    Validates and preprocesses inputs for estimating sections at catchment scale.

    Ensures data compatibility, processes river lines, and prepares water mask data for analysis.

    Args:
    - da_wmask (xarray.DataArray or str): Water mask dataset or path to image directory.
    - r_lines (str): Path to river lines shapefile.
    - str_order_col (str): Column in river lines shapefile representing stream order.
    - initial_buffer (int): Initial buffer size for processing.

    Returns:
    - Tuple of processed data elements for catchment scale analysis.
    """
    # Call validate function to check and prepare the dataset
    da_wmask, r_lines, crs = calc_metrics.validate(da_wmask, r_lines, section_length=None, img_ext='.tif', module='generate_sections')

    # Preprocess the inputs `da_wmask` and `r_lines` to make them ready for further processing
    da_wmask, r_lines = preprocess_inputs(da_wmask, r_lines, initial_buffer)

    # Execute line pre-processing and store the results in multiple variables
    initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list = line_pre_processing (da_wmask, r_lines, str_order_col)
    
    return initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list

def preprocess_inputs(da_wmask, r_lines, initial_buffer):
    """
    Applies initial buffering to river lines and preprocesses water mask and river line inputs.

    Args:
    - da_wmask (xarray.DataArray): Water mask data as an xarray DataArray.
    - r_lines (geopandas.GeoDataFrame): River corridor extent as a GeoDataFrame.
    - initial_buffer (int or float): Buffer distance to apply to the river lines.

    Returns:
    - xarray.DataArray: The preprocessed water mask DataArray.
    - geopandas.GeoDataFrame: The buffered and potentially reprojected river lines GeoDataFrame.
    """
    print('Preprocessing data...')
    # Dissolve and buffer the river lines using the specified buffer distance
    initial_rcor_extent = r_lines.dissolve().buffer(initial_buffer, cap_style=2)
    initial_rcor_extent = gpd.GeoDataFrame(geometry=initial_rcor_extent, crs=r_lines.crs)
    
    # Clip the water mask DataArray to the extent of the buffered river lines
    da_wmask, r_lines = calc_metrics.match_input_extent(da_wmask, initial_rcor_extent)

    # Fill NoData values in the clipped water mask DataArray
    da_wmask = calc_metrics.fill_nodata(da_wmask, initial_rcor_extent)
    
    return da_wmask, r_lines

def line_pre_processing (da_wmask, r_lines, str_order_col):
    """
    Processes river lines for catchment analysis, including buffer estimation and dissolving by stream order.

    Args:
    - da_wmask (xarray.DataArray): Water mask data.
    - r_lines (geopandas.GeoDataFrame): GeoDataFrame containing river lines.
    - str_order_col (str): Column name indicating stream order in r_lines. If None, a default order (zero) is used.

    Returns:
    - tuple: Contains processed elements including buffer size, pixel persistence, CRS, dissolved river lines, and stream orders.
    """
    # Estimate initial buffer size and filter pixel persistence from water mask
    initial_buffer, filtered_PP = estimate_initial_buffer(da_wmask)
    # Ensure CRS consistency for spatial operations
    crs = r_lines.crs

    # Handle cases where stream order column is not specified
    if str_order_col is None:
        # Assign a default stream order if none is specified
        r_lines['str_order'] = 0
        str_order_col = 'str_order'
    
    # Dissolve river lines based on stream order, creating a single geometry per order
    r_lines_dissolved = r_lines.dissolve(by=str_order_col).sort_values(by=str_order_col, ascending=False)

    # Extract the list of unique stream orders for further analysis
    r_lines_dissolved['str_order'] = r_lines_dissolved.index
    str_order_list = list(r_lines_dissolved['str_order'])
    
    return initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list 

def estimate_initial_buffer (da_wmask):
    """
    Determines an initial buffer for spatial analysis based on the persistence of water presence.

    Args:
    - da_wmask (xarray.DataArray): Water mask data.

    Returns:
    - int: Suggested initial buffer size for further processing.
    - xarray.DataArray: Filtered Pixel Persistence indicating areas consistently identified as water.
    """
    # Calculate pixel persistence using the input water mask
    PP = calc_metrics.calculate_pixel_persistence(da_wmask)
    # Apply a threshold to retain values above 50%, converting to binary format
    filtered_PP = xr.where(PP > 0.5, 1, 0).astype('int8')
    # Perform Euclidean distance transform to identify the furthest point from water
    euc_dist_trans_array = ndimage.distance_transform_edt(filtered_PP.values)   
    # Find the maximum distance, then scale and round it, ensuring a minimum size
    initial_buffer = round(euc_dist_trans_array.max() * filtered_PP.rio.transform()[0] * 1.1)
    # Round to next tens for standardization
    initial_buffer = ((initial_buffer // 10) + 1) * 10
    
    return initial_buffer, filtered_PP

def process_lines (initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list):    
    """
    Applies geometric operations on river lines for spatial analysis preparation.

    Args:
    - initial_buffer (int): Buffer size for spatial operations.
    - filtered_PP (xarray.DataArray): Pixel Persistence indicating water presence.
    - crs (CRS): Coordinate Reference System of the spatial data.
    - r_lines_dissolved (GeoDataFrame): Dissolved river lines by stream order.
    - str_order_list (list): Ordered list of stream orders for processing.

    Returns:
    - A tuple of lists: Buffers, section lengths, widths, and original indices of processed line features.
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
    Applies a buffer around a geometric feature, handling both Polygon and MultiPolygon cases.

    This function buffers the input feature by the specified size. If the result is a MultiPolygon,
    it splits the MultiPolygon into individual Polygon geometries.

    Args:
    - feature (GeoSeries or GeoDataFrame): The geographic feature(s) to buffer.
    - buffer_size (float): The distance to buffer around the feature.

    Returns:
    - A list of buffered Polygon geometries.
    """
    # Apply buffer with specified size and styles
    buffered_feature = feature.geometry.buffer(buffer_size, cap_style=2)
    # Check if the buffered geometry is a MultiPolygon and split if necessary
    if buffered_feature.geom_type == 'MultiPolygon':
        return list(buffered_feature.geoms)
    
    return [buffered_feature]

def estimate_section_length_and_width (filtered_PP, polygon_list):   
    """
    Estimates the length and width of river sections using raster and vector data.

    This approach combines raster analysis for width estimation with morphological operations
    for length estimation, providing a basis for river section characterization.

    Args:
    - filtered_PP (xarray.DataArray): Pixel Persistence data indicating water presence.
    - polygon_list (list of Polygon): Buffered geometries for clipping the Pixel Persistence data.

    Returns:
    - tuple: Estimated section length and width in the units of the data's CRS.
    """
    # Clip the raster data array to the buffered geometries
    PP_clipped = filtered_PP.rio.clip(polygon_list)
    
    # Calculate Euclidean distance to determine the width of the section
    euc_dist_trans_array = ndimage.distance_transform_edt(PP_clipped.values)   
    # Determine section width based on the maximum Euclidean distance, ensuring it's at least 5 units
    section_width = round(max(euc_dist_trans_array.max() * 2, 5) * PP_clipped.rio.transform()[0])
    # Round width to the next tens for standardization
    section_width = ((section_width // 10) + 1) * 10
    
    # Skeletonize clipped Pixel Persistence data to estimate section length
    skeleton = skeletonize(PP_clipped.values, method='lee')
    # Label the skeletonized image to identify connected regions
    labeled_skeleton = label(skeleton)
    regions = regionprops(labeled_skeleton)
    # Calculate average region length considering the number of regions
    total_num_pixels = sum(region.area for region in regions)
    average_pixels_per_region  = round((total_num_pixels / len(regions) if len(regions) > 0 else 1))
    # Convert average pixel count to actual length, ensuring it's at least 5 units
    section_length = max(average_pixels_per_region, 5) * PP_clipped.rio.transform()[0]
    # Round length to the next tens for standardization
    section_length = ((section_length // 10) + 1) * 10   
    
    return int(section_length), int(section_width)

def split_and_buffer_lines(feature, section_length, buffer_size, crs):
    """
    Splits and buffers a line feature for spatial analysis.

    Args:
    - feature (GeoSeries): Line feature to be processed.
    - section_length (float): Length to split the line feature into sections.
    - buffer_size (float): Buffer size to apply to each section.
    - crs (CRS): Coordinate Reference System of the feature.

    Returns:
    - Tuple: Containing two GeoDataFrames of buffered line sections (regular and simplified).
    """
    # Initialize lists to store buffered geometries
    splitted_lines = []
    # Copy the input feature to avoid modifying the original
    feature = feature.copy()
    
    # Attempt to merge lines if they are multi-lines, useful for continuous processing
    try:
        # Attempt to merge lines if they are multilines
        feature.geometry = linemerge(feature.geometry)
    except:
        pass

    # Ensure feature is a single geometry for processing
    if feature.geometry.geom_type == 'MultiLineString':
        for line in feature.geometry.geoms:
            splitted_line = split_line_section_length(line, section_length)
            splitted_lines.append(splitted_line)
    else:
        splitted_line = split_line_section_length(feature.geometry, section_length)
        splitted_lines.append(splitted_line)

    # Flatten the list of lists to a single list of line geometries
    flat_lines = [item for sublist in splitted_lines for item in sublist]
    
    # Buffer the flattened list of line geometries
    buffered_rivers_lst = [
        line.buffer(buffer_size, cap_style=2, join_style=1) 
                               for line in flat_lines]
    
    # Convert the list of buffered geometries to a GeoDataFrame
    buffered_rivers = gpd.GeoDataFrame(geometry=buffered_rivers_lst, crs=crs)
    # Simplify the buffered geometries by creating straight lines between start and end points, then buffering
    buffered_simplified_rivers_lst  = [
        LineString([line.coords[0], line.coords[-1]]).buffer(buffer_size, cap_style=2, join_style=1)
                                for line in flat_lines]

    # Convert the simplified buffered geometries to a GeoDataFrame
    buffered_simplified_rivers = gpd.GeoDataFrame(geometry=buffered_simplified_rivers_lst, crs=crs)
    
    return buffered_rivers, buffered_simplified_rivers

def split_line_section_length(line, section_length):
    """
    Splits a given line into multiple segments of approximately equal length.

    The function aims to divide the line into segments that are as close as possible to the specified section length.
    It first redistributes vertices along the line to have a uniform distribution before splitting.

    Args:
    - line (shapely.geometry.LineString): The line geometry to split.
    - section_length (float): Target length for each line segment.

    Returns:
    - List[shapely.geometry.LineString]: A list of LineString objects representing the divided segments of the original line.
    """
    # Ensure vertices are evenly distributed along the line
    line = redistribute_vertices(line, 10)
    # Calculate the total number of segments needed, ensuring at least one segment
    total_length = line.length
    num_segments = max(round(total_length / section_length), 1)
    
    # Adjust section length to fit an integer number of segments within the total length
    adjusted_min_length = total_length / num_segments
    if 0.9 * section_length <= adjusted_min_length <= 1.1 * section_length:
        section_length = adjusted_min_length
        
    # Generate split points at adjusted intervals along the line
    mpoints = MultiPoint([line.interpolate(i * section_length) for i in range(num_segments)])
    
    # Use 'snap' to ensure split points are exactly on the line, then split the line at these points
    splitted_line = split(snap(line, mpoints, 0.001), mpoints)
    
    # Filter the resulting geometries to retain only LineString types
    splitted_line = [geom for geom in splitted_line.geoms if isinstance(geom, LineString)]
    
    return splitted_line

def redistribute_vertices(geom, distance):
    """
    Adjusts the vertices of a line geometry to be evenly spaced.

    The function aims to create a uniform distribution of vertices along a line or set of lines,
    ensuring that the spacing between consecutive vertices approximates a specified distance.

    Args:
    - geom (LineString or MultiLineString): The geometry to redistribute vertices on.
    - distance (float): Desired distance between vertices.

    Returns:
    - LineString or MultiLineString: Geometry with vertices redistributed.

    Raises:
    - ValueError: If geom is not a LineString or MultiLineString.
    """
    # Process for a single LineString
    if geom.geom_type == 'LineString':
        # Calculate the number of vertices needed
        num_vert = max(int(round(geom.length / distance)), 1)
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    
    # Process each LineString within a MultiLineString separately
    elif geom.geom_type == 'MultiLineString':
        # Recursively apply the function to each part of the MultiLineString
        parts = [redistribute_vertices(part, distance)
                 for part in geom.geoms]
        # Filter out empty parts and return a MultiLineString
        return MultiLineString([p for p in parts if not p.is_empty])
    
    # else:
    #     # Raise an error for unhandled geometry types
    #     raise ValueError('unhandled geometry %s', (geom.geom_type,))

def erase_overlapping_and_clean (buffered_rivers, buffered_simplified_rivers, line_buffer, crs):
    """
    Performs overlay operations to erase overlapping areas and clean the resulting geometries.

    This function handles the refinement of buffered river geometries by erasing overlaps between
    the original and simplified buffered geometries, cleaning up the differences, and finally merging
    any small polygons with their nearest neighbors to create a more coherent set of river geometries.

    Parameters:
    - buffered_rivers (GeoDataFrame): GeoDataFrame containing the original buffered river geometries.
    - buffered_simplified_rivers (GeoDataFrame): GeoDataFrame containing simplified buffered river geometries.
    - line_buffer (GeoDataFrame): GeoDataFrame containing buffers around the river lines, used for gap filling.
    - crs (CRS): Coordinate Reference System of the input GeoDataFrames.

    Returns:
    - GeoDataFrame: The final cleaned and processed set of river geometries.
    """
    # Erase overlaps between the original and simplified buffered rivers to clean geometries
    buffered_rivers_erased = erase_overlaps(buffered_rivers, crs)
    buffered_simplified_rivers_erased = erase_overlaps(buffered_simplified_rivers, crs)
    
    # Calculate the geometric difference between the erased buffered rivers and the simplified versions
    # Then clean the edges of the result to smooth and standardize the geometries
    diff_polygons = buffered_rivers_erased.overlay(buffered_simplified_rivers_erased, how='difference', keep_geom_type=True
                                        ).buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2
                                        ).explode(ignore_index=True).reset_index(drop=True)  
    
    diff_polygons = diff_polygons[~diff_polygons.geometry.is_empty]                     
    diff_polygons = gpd.GeoDataFrame(geometry=diff_polygons, crs=crs)
        
    # Merge polygons based on shared boundaries for coherence
    difference_geometry = merge_polygons_on_shared_boundary (buffered_simplified_rivers_erased, diff_polygons)

    # Dissolve the buffered rivers for a unified geometry and clean the edges
    buffered_rivers_dissolved = buffered_rivers.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    buffered_rivers_dissolved = gpd.GeoDataFrame(geometry=buffered_rivers_dissolved, crs=crs)
    
    # Clean the dissolved difference geometry in the same way
    difference_geometry_dissolved = difference_geometry.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    difference_geometry_dissolved = gpd.GeoDataFrame(geometry=difference_geometry_dissolved, crs=crs)
       
    # Find the difference between the dissolved original buffered rivers and the cleaned difference geometry
    # This helps to identify and remove any remaining overlaps or inconsistencies
    difference_overlay = buffered_rivers_dissolved.overlay(difference_geometry_dissolved, how='difference', keep_geom_type=True
                                        ).buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2
                                        ).explode(ignore_index=True).reset_index(drop=True)
    difference_overlay = difference_overlay[~difference_overlay.geometry.is_empty]
    difference_overlay = gpd.GeoDataFrame(geometry=difference_overlay, crs=crs)

    # Further clean the sections by merging small polygons with their nearest neighbors
    cleaned_sections = merge_polygons_on_shared_boundary (difference_geometry, difference_overlay)
    final_cleaned_sections = merge_small_polygons_with_nearest_neighbors(cleaned_sections)
    final_cleaned_sections_dissolved = final_cleaned_sections.dissolve().buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
    final_cleaned_sections_dissolved = gpd.GeoDataFrame(geometry=final_cleaned_sections_dissolved, crs=crs)
    
    # Fill any remaining gaps by overlaying the line buffer with the cleaned sections and processing the differences
    difference_overlay_gaps = line_buffer.overlay(final_cleaned_sections_dissolved, how='difference', keep_geom_type=True
                                    ).buffer(-1, cap_style=2, join_style=2).buffer(1, cap_style=2, join_style=2
                                    ).explode(ignore_index=True).reset_index(drop=True)                       
    difference_overlay_gaps = difference_overlay_gaps[~difference_overlay_gaps.geometry.is_empty]
    # difference_overlay_gaps = difference_overlay_gaps[difference_overlay_gaps.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    difference_overlay_gaps = gpd.GeoDataFrame(geometry=difference_overlay_gaps, crs=line_buffer.crs)
    
    # If there are gaps, merge the cleaned sections with the gap polygons and clean the result
    if not difference_overlay_gaps.empty:
        # Clean sections of the dissolved difference GeoDataFrame
        cleaned_sections_gaps = merge_polygons_on_shared_boundary (final_cleaned_sections, difference_overlay_gaps)
        final_cleaned_sections = merge_small_polygons_with_nearest_neighbors(cleaned_sections_gaps)
        final_cleaned_sections = final_cleaned_sections.buffer(-0.1, cap_style=2, join_style=2).buffer(0.1, cap_style=2, join_style=2
                                            ).explode(ignore_index=True).reset_index(drop=True)
    
    return final_cleaned_sections

def erase_overlaps(buffered_rivers, crs):
    """
    Removes overlaps between polygon geometries within a GeoDataFrame to ensure that
    each geometry is distinct, simplifying the spatial representation of buffered rivers.

    Overlapping geometries are subtracted from each other, and the resulting geometries
    are cleaned up by applying a slight inward and outward buffer. This process also
    ensures that only the largest polygon is kept in case of MultiPolygon geometries,
    focusing on the primary spatial feature.

    Parameters:
    - buffered_rivers (GeoDataFrame): Contains buffered river geometries that may overlap.
    - crs (CRS): The coordinate reference system to apply to the output GeoDataFrame.

    Returns:
    - GeoDataFrame: Contains the cleaned geometries with overlaps removed, preserving the CRS.
    """
    diff_lst = []
    # Iterate through each buffer geometry to remove overlaps
    for geom_i in buffered_rivers.geometry:
        # Find geometries that overlap with the current geometry
        overlaps = buffered_rivers[buffered_rivers.geometry.overlaps(geom_i)]
        
        # Process each overlapping geometry
        for geom_j in overlaps.geometry:
            try:
                # Calculate the difference to remove the overlapping area              
                intersection = geom_i.intersection(geom_j)
                if isinstance(geom_i, (Polygon, MultiPolygon)):
                    geom_i = geom_i.difference(intersection)
            except:
                continue  # Skip to the next geometry

        # Apply a slight buffer to clean up edges of the geometry
        geom_i = geom_i.buffer(-0.02, cap_style=2).buffer(0.02, cap_style=2)
        
        # In case of MultiPolygon, keep only the largest polygon to simplify the representation
        if isinstance(geom_i, MultiPolygon):
            geom_i = max(geom_i.geoms, key=lambda polygon: polygon.area)
        
        # Add the cleaned geometry to the list if it's not empty
        if isinstance(geom_i, Polygon) and not geom_i.is_empty:
            diff_lst.append(geom_i)
    
    return gpd.GeoDataFrame(geometry=diff_lst, crs=crs)

def merge_polygons_on_shared_boundary(target_geometries, polygons_to_merge, buffer_distance=0.02):
    """
    Merges polygons based on shared boundaries to refine and consolidate spatial representations.

    This function iterates through a set of polygons intended for merging and identifies the best
    candidate for merging within a target geometry collection based on the length of the shared boundary.
    The merging process is aimed at spatial consolidation of closely related or overlapping geometries.

    Parameters:
    - target_geometries (GeoDataFrame): Target geometries for potential merging.
    - polygons_to_merge (GeoDataFrame): Polygons designated for merging into the target geometries.
    - buffer_distance (float): A small buffer distance to facilitate the identification of shared boundaries.

    Returns:
    - GeoDataFrame: Updated target geometries after merging operations.
    """
    # Apply a slight buffer to polygons_to_merge to ensure shared boundaries are identified
    polygons_to_merge['buffered_geometry'] = polygons_to_merge.geometry.buffer(buffer_distance)
    # Prepare a spatial index for efficient querying of target_geometries
    target_geometries_sindex = target_geometries.sindex
    
    # Iterate through polygons intended for merging
    for idx, row in polygons_to_merge.iterrows():
        current_polygon = row['buffered_geometry']
        
        # Potential neighbors based on spatial index query
        neighbour_indices = list(target_geometries_sindex.intersection(current_polygon.bounds))
        intersecting_candidates = target_geometries.iloc[neighbour_indices]
        
        # Actual neighbors based on overlap calculation
        overlapping_neighbours = intersecting_candidates[intersecting_candidates.geometry.overlaps(current_polygon)]
        
        # Proceed if there are actual neighboring geometries
        if not overlapping_neighbours.empty:
            # Identify the neighbor with the longest shared boundary
            longest_neighbour_idx = get_longest_shared_boundary(current_polygon, overlapping_neighbours, buffer_distance)
            
            # Merge current polygon with the identified neighbor
            if longest_neighbour_idx is not None:
                # Merge the current polygon with this neighbor
                merged_polygon = current_polygon.union(target_geometries.at[longest_neighbour_idx, 'geometry'])
                # Apply inward and outward buffer to clean the merged geometry
                target_geometries.at[longest_neighbour_idx, 'geometry'] = merged_polygon.buffer(buffer_distance).buffer(-buffer_distance)
    
    return target_geometries

def get_longest_shared_boundary(polygon, neighbours, buffer_distance=0.01):
    """
    Finds the neighbour polygon that shares the longest boundary with the given polygon.
    
    Parameters:
    - polygon (Polygon): The reference polygon to compare with its neighbours.
    - neighbours (GeoDataFrame): A GeoDataFrame containing neighbour polygons that could share a boundary with the reference polygon.
    - buffer_distance (float): The buffer distance for expanding the boundaries of the polygons to calculate shared boundaries effectively.
    
    Returns:
    - int: The index of the neighbor with the longest shared boundary. Returns None if no shared boundaries are found.
    """
    max_length = 0
    longest_neighbor_idx = None
    
    # Buffer the boundary of the reference polygon to ensure slight overlaps are considered
    polygon_buffered_boundary = polygon.boundary.buffer(buffer_distance)
    
    # Iterate through each neighbour to find the longest shared boundary
    for idx, neighbour in neighbours.iterrows():
        # Buffer the boundary of the neighbour
        neighbour_buffered_boundary = neighbour.geometry.boundary.buffer(buffer_distance)
        # Calculate the intersection (shared boundary) between 
        # the buffered boundaries of the reference polygon and the current neighbor
        shared_boundary = polygon_buffered_boundary.intersection(neighbour_buffered_boundary)
        
        # If there's no intersection, continue to the next neighbor
        if shared_boundary.is_empty:
            continue
        
        # Calculate the length of the shared boundary
        shared_boundary_length = shared_boundary.length
        
        # If this shared boundary is longer than any previously found, 
        # update max_length and longest_neighbor_idx
        if shared_boundary_length > max_length:
            max_length = shared_boundary_length
            longest_neighbour_idx = idx
    
    return longest_neighbour_idx

def merge_small_polygons_with_nearest_neighbors(cleaned_sections, std_dev_trheshold=2):
    """
    Merges smaller polygons within a GeoDataFrame to their nearest neighboring polygon, based on a threshold determined 
    by the mean area and standard deviation of all polygons' areas. This operation is intended to consolidate minor geometries 
    into larger, more significant shapes for analysis.

    Parameters:
    - cleaned_sections (GeoDataFrame): GeoDataFrame containing geometries to be evaluated and potentially merged.
    - std_dev_trheshold (int): Multiplier for the standard deviation to define the size threshold for considering a polygon 'small'.

    Returns:
    - GeoDataFrame: The GeoDataFrame after merging smaller polygons with their nearest neighbors.
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
    - r_lines_dissolved (GeoDataFrame): Dissolved river lines used in the processing.
    - buff_list (list): List of buffer geometries from the line processing.
    - section_lengths (list): List of lengths for each line section.
    - section_widths (list): List of widths for each line section.
    - original_indices (list): List of original indices from the input data.
    - str_order_col (str): The name of the stream order column in the input data.
    - crs (Coordinate Reference System): The CRS of the processed geospatial data.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the postprocessed results with buffered geometries,
                  section lengths, widths, and additional information from spatial joins.
    """
    # Combine buffered geometries into a GeoDataFrame with specified CRS
    buffered_gdf = gpd.GeoDataFrame(geometry=[item for sublist in buff_list for item in sublist], crs=crs)
    
    # Add columns for section lengths, widths, and stream order to the GeoDataFrame
    buffered_gdf['section_length'] = section_lengths
    buffered_gdf['section_width'] = section_widths
    buffered_gdf['stream_order'] = original_indices  # Add the original indices
    
    # Perform a spatial join between buffered_gdf and r_lines_dissolved to find intersecting features
    # This join uses the 'intersects' predicate to identify geometries in buffered_gdf that intersect with r_lines_dissolved
    intersection_df = gpd.sjoin(buffered_gdf, r_lines_dissolved, predicate='intersects', how='inner', lsuffix='original', rsuffix=str_order_col)
    
    # Filter rows where the stream order matches the index of stream order column
    # This step removes self-intersections or redundancies in the spatial join results
    intersection_df = intersection_df[intersection_df['stream_order'] != intersection_df[f'index_{str_order_col}']]
    
    # Select only the index of stream order column, which identifies intersecting features
    intersection_df = intersection_df[f'index_{str_order_col}']
    
    # Merge the intersection_df with the buffered_gdf based on indices to combine 
    # spatial join results with original buffered geometries
    result_gdf = buffered_gdf.merge(intersection_df, 
                                    left_index=True, right_index=True, how='left')
    
    # Replace NaN values in the index of stream order column with -1
    # This step is crucial for handling non-intersecting geometries, assigning them a default value
    result_gdf[f'index_{str_order_col}'].fillna(-1, inplace=True)
    
    # Rename the column to 'nodes' for clarity
    result_gdf.rename(columns={f'index_{str_order_col}': f'nodes'}, inplace=True)
    
    return result_gdf
