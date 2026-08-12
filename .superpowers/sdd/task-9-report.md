# Task 9 Report: Verified GeoTIFF and opt-in NetCDF writers

**Date:** 2026-08-12  
**HEAD:** d65e0d7  
**Status:** Complete

## Summary

Implemented verified, windowed GeoTIFF writers and optional NetCDF export for all Section 2.4 raster products. Writers use same-directory temporary files, reopen validation, and `Path.replace()` for atomic publication. Preflight rejects pre-existing final artifacts.

## Changes

### `hydrofragments/output/rasters.py`

- Added `RasterProductContract` registry for all Section 2.4 products (dtype, nodata, units, filenames).
- Implemented windowed GeoTIFF writes (256×256 tiles clamped to GDAL 16-pixel minimum, DEFLATE, dtype predictors, BigTIFF `IF_SAFER`, band descriptions, metadata tags).
- Added `write_verified_geotiff`, `validate_geotiff`, `write_geotiff_from_dataarray`, `preflight_raster_artifacts`.
- Added `build_refuge_overlap_from_checkpoint` for per-HY-pair overlap categories.
- Added `export_rasters_from_checkpoint` orchestrating persistence, temporal, refuge-stability, zones, and optional NetCDF (`spatial.nc`).
- Added `write_verified_netcdf` with `h5netcdf` opt-in and actionable error when the extra is missing.
- Checkpoint export aligns non-georeferenced accumulator arrays to the grid contract before write.

### `pyproject.toml`

- Added optional dependency: `netcdf = ["h5netcdf>=1.4"]`.

### `tests/output/test_rasters.py`

- Round-trip tests for every Section 2.4 product (GeoTIFF).
- Preflight, shifted-grid, missing-CRS, truncated-file, checkpoint export, and NetCDF tests.

## Verification

```powershell
python -m pytest tests/output/test_rasters.py -q
```

Result: **26 passed** (with environment PROJ warnings on Windows; CRS validation uses identifier intersection for robust comparison).

## Commit

```
feat: write verified georeferenced raster products
```

## Notes

- NetCDF requires `pip install 'hydrofragments[netcdf]'`.
- GeoTIFF CRS validation tolerates equivalent projected CRS identifiers when PROJ EPSG resolution is unavailable (common on mixed PROJ installs).
