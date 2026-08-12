# Test Fixture Provenance and Tiers

HydroFragments test fixtures are organized into three active tiers:

## Tier A — Synthetic Analytic Fixtures

**Location:** `tests/fixtures/analytic_masks.py`

Hand-built boolean masks with documented, hand-calculable ground truth (component count, pixel area, connectivity behavior).

| Fixture function | Ground truth locked in |
|---|---|
| `empty_mask` | 0 connected components, 0 wet pixels |
| `full_mask` | 1 connected component covering every pixel |
| `diagonal_pair_mask` | 1 component under 8-connectivity (2 under 4-connectivity) |
| `one_pixel_noise_mask` | 2 components (1 px + 4 px) before any small-object filter |
| `mask_with_hole` | 1 component, hole does not split it, area = footprint - hole |
| `long_bar_mask` | 1 component, longest path = length - 1 |
| `padded_square_mask` | 1 component with real dry background |
| `chunk_crossing_mask` | 1 component spanning chunk boundaries |

## Tier B — Bundled Test Fixtures

| Fixture | Path | Description |
|---|---|---|
| NetCDF Water Mask | `tests/wmask_ts.nc` | 63-timestep NetCDF raster cube for integration tests |
| River Corridor AOI | `tests/rcor_extent.shp` | 7-section polygon shapefile for spatial section tests |

## Tier C — Real Validation-Catchment Fixtures

| Fixture | Path | Description |
|---|---|---|
| Real monthly water-mask cube (Fitzroy) | `data/wofs_monthly_masks_1986_2026.zarr` | 480-month Fitzroy River Basin Zarr dataset |
| Real drainage centreline (Fitzroy Kimberley) | `data/fitzroy_kimberley_drainage.gpkg` | 291-reach channel centreline GeoPackage |
| AOI polygon (Fitzroy Kimberley) | `data/fitzroy_kimberley_aoi.geojson` | AOI boundary GeoJSON |
