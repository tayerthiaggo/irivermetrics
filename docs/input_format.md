# HydroFragments Input Formats

HydroFragments accepts 3D raster time series representing surface water observations across time, y, and x dimensions.

## Input Kinds

HydroFragments supports three input kinds configured via `HydroConfig`:

1. **`generic_binary`:**
   - Boolean or 0/1 integer arrays.
   - `1` (or `True`) = wet pixel, `0` (or `False`) = dry pixel.

2. **`watermask_tsfill`:**
   - Standard WaterMask-TSFill monthly composite arrays.
   - Values: `0` = dry, `1` = wet, `254` / `255` = invalid/unobserved.

3. **`dea_wofs`:**
   - Digital Earth Australia WOfS bitflag arrays.
   - Bit flags parsed automatically to derive water and valid observation masks.

## Spatial and Temporal Requirements

- **Dimensions:** DataArrays must have dimensions `("time", "y", "x")`.
- **Coordinate Alignment:** `water` and `valid_obs` masks must share identical spatial coordinates and time stamps.
- **CRS:** Projected coordinate reference systems with linear meter units (e.g., `EPSG:3577` Albers Equal Area) are required for accurate area and length calculations.
- **Cadence:** Monthly composite series are standard. Irregular sub-monthly observations can be composited using `hydroseason` prior to metric analysis.
