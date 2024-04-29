 
    
## Module 2 
# to do 

# CREATE FOLDER TO EXPORT RESULTS AND EXPORT RESULTS
# Issues

# -	Rivers of the same order, if running in parallel, will be merged – fix it
# -	Small polygons at the end vertices were clipped by higher river order – merge to adjacent polygon?
# -	Gaps inside a few polygons – why?

def estimate_section_for_cachtment(da_wmask, r_lines, str_order_col=None, initial_buffer=1000):
    """
    Estimates the size of sections for a given catchment area based on water mask data and river lines.

    This function orchestrates the process of validating and preprocessing the input data, 
    processing the line features, and then postprocessing the results to estimate the section size.

    Parameters:
    da_wmask (xarray.core.dataarray.DataArray or str): Input data, either as a DataArray or a directory path.
    r_lines (str): Path to a file containing river lines or similar linear features shapefile.
    str_order_col (str): The name of the column in `r_lines` representing stream order.
    initial_buffer (int): Initial buffer for analysis. 

    Returns:
    GeoDataFrame: A GeoDataFrame containing the estimated section sizes along with additional
                  processed geospatial data.
    """
    # Preprocess the input datasets and obtain necessary parameters for further processing
    initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list = gen_sections.validate_and_preprocess(da_wmask, r_lines, str_order_col, initial_buffer)
    # Process the line features based on the preprocessed data and obtain buffer list,
    # section lengths, widths, and original indices
    buff_list, section_lengths, section_widths, original_indices = gen_sections.process_lines(initial_buffer, filtered_PP, crs, r_lines_dissolved, str_order_list)
    # Postprocess the results from line processing to generate a final GeoDataFrame
    # containing the estimated section sizes and other relevant data
    result_gdf = gen_sections.postprocess_results(r_lines_dissolved, buff_list, section_lengths, section_widths, original_indices, str_order_col, crs)
    # Return the final GeoDataFrame with the processed results
    return result_gdf