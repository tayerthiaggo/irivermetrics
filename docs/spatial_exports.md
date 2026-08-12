# Spatial exports

Optional GIS-ready vector and raster products complement the canonical tidy metric tables. Spatial exports are **off by default** so the default tabular workflow pays no polygonization, checkpoint, or serialization cost.

## Quick decision guide

| Goal | Use |
|------|-----|
| Statistical modelling, dashboards, database loads | Metric tables (`metrics/` Parquet or CSV) |
| Cartography in QGIS/ArcGIS | GeoTIFF rasters and GeoPackage vectors |
| Multidimensional scientific exchange | Opt-in NetCDF (`pip install hydrofragments[netcdf]`) |

## Configuration

Use configuration schema **`1.1.0`**. Spatial product selection belongs in `execution_config()` and does not change the scientific configuration hash.

```python
from hydrofragments import HydroConfig

config = HydroConfig.from_mapping(
    {
        "config_schema_version": "1.1.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
        "output": {
            "output_dir": "runs/demo_01",  # required when spatial_products is non-empty
            "formats": ["parquet"],        # validated: parquet, csv
            "spatial_products": [],        # default: exports off
            "raster_formats": ["geotiff"], # geotiff (default) and/or netcdf
        },
    }
)
```

### Opt-in examples

Persistence rasters only:

```python
"output": {
    "output_dir": "runs/persistence",
    "spatial_products": ["persistence_rasters"],
}
```

Monthly pool polygons:

```python
"output": {
    "output_dir": "runs/pools",
    "spatial_products": ["monthly_pools"],
}
```

All products appropriate for a georeferenced cube with hydrological-year inputs:

```python
"output": {
    "output_dir": "runs/full",
    "spatial_products": [
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
    ],
}
```

The deprecated `include_vectors: true` alias maps to `monthly_pools` for one schema cycle. Do not set both to conflicting values.

## Output layout

`output_dir` names the **final run directory**, not a shared parent folder. The directory must be absent or empty before analysis starts. HydroFragments stages artifacts, validates them, then commits the bundle with one directory rename. `run_manifest.json` is written **last**.

```text
<output_dir>/
  config.json
  metrics/
  metric_coverage.csv
  vectors/spatial.gpkg
    monthly_pools
    zones
    reaches
    reach_wet_monthly
  rasters/occurrence.tif
  rasters/valid_observation_count.tif
  rasters/refuge_mask.tif
  rasters/zones.tif
  rasters/recurrence.tif
  rasters/recurrence_valid_year_count.tif
  rasters/hydroperiod_by_year.tif
  rasters/hydroperiod_valid_month_count_by_year.tif
  rasters/refuge_overlap_by_hy.tif
  rasters/refuge_stability_frequency.tif
  rasters/refuge_stability_union_pair_count.tif
  rasters/spatial.nc              # when raster_formats includes netcdf
  run_manifest.json
```

Only requested and applicable paths are created. Unavailable requested products fail **preflight** with `SpatialProductUnavailable` rather than silently omitting files.

## CRS and grid

Exports preserve the source cube grid and CRS. HydroFragments does **not** reproject during export. A cube without resolvable CRS/transform fails early when spatial output is requested; tabular analysis remains allowed.

Every raster and vector product is validated against a frozen `SpatialGrid` contract (CRS, affine transform, coordinate order, shape). Equal-shaped arrays with a shifted transform are rejected.

## Product reference

### `persistence_rasters`

| Artifact | dtype | nodata | units |
|----------|------:|--------|-------|
| `rasters/occurrence.tif` | float32 | NaN | percent, 0–100 |
| `rasters/valid_observation_count.tif` | uint32 | 4294967295 | months |
| `rasters/refuge_mask.tif` | uint8 | 255 | 0=false, 1=true |

**Prerequisites:** Georeferenced water cube with valid CRS/transform.

**Workflow:** `analyze()` and `analyze_from_dea()` when cube grid is valid.

**Cost:** Moderate I/O; counters are accumulated in the monthly pass (same pass as scalar persistence metrics). Enabling exports reuses completed checkpoints rather than re-reading the cube.

### `temporal_rasters`

| Artifact | dtype | nodata | units |
|----------|------:|--------|-------|
| `rasters/recurrence.tif` | float32 | NaN | percent, 0–100 |
| `rasters/recurrence_valid_year_count.tif` | uint16 | 65535 | calendar years |
| `rasters/hydroperiod_by_year.tif` | float32 | NaN | fraction, 0–1 (multi-band by calendar year) |
| `rasters/hydroperiod_valid_month_count_by_year.tif` | uint8 | 255 | months, 0–12 |

**Prerequisites:** Georeferenced cube; sufficient temporal record for the estimator.

### `refuge_stability_rasters`

| Artifact | dtype | nodata | units / codes |
|----------|------:|--------|---------------|
| `rasters/refuge_overlap_by_hy.tif` | uint8 | 255 | 0=dry, 1=lost, 2=new, 3=stable (multi-band by HY pair) |
| `rasters/refuge_stability_frequency.tif` | float32 | NaN | percent, 0–100 per-pixel stability frequency |
| `rasters/refuge_stability_union_pair_count.tif` | uint16 | 65535 | valid HY pairs wet in either year |

**Prerequisites:** Hydrological-year anchors from `hydroyear_extent` and at least two valid end-dry states.

**Nodata semantics:** Pixels with zero eligible HY pairs are nodata. The per-pixel frequency raster is **not** the scalar Jaccard `refuge_spatial_stability` metric.

### `monthly_pools`

**Path:** `vectors/spatial.gpkg` layer `monthly_pools`

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64[ns] | Month timestamp |
| `pool_id` | string | `YYYY-MM-DD:<window_id>:<label_id>` |
| `label_id` | int32 | Connected-component label |
| `n_pixels` | int32 | Pixel count |
| `area_m2` | float64 | Polygon area |
| `perimeter_m` | float64 | Perimeter |
| `major_axis_length_m` | float64 | Major axis |
| `width_m` | float64 | Nullable width |
| `elongation_ratio` | float64 | Nullable |
| `shape_index` | float64 | Nullable |
| `geometry` | Polygon/MultiPolygon | Source CRS |

Aggregate monthly metrics (AWRE, AWMSI) are **not** duplicated on every feature. Polygon area and count match the measured label properties within raster/vector tolerance.

**Checkpoint-only design:** Pool polygons are polygonized during the monthly pass into durable checkpoint partitions, then streamed into the GeoPackage. `HydroResult.write()` and `write_output_tables()` reject an in-memory run-wide `GeoDataFrame`.

### `zones`

**Paths:** `vectors/spatial.gpkg` layer `zones`, `rasters/zones.tif`

Dissolved zone polygons with `zone_id`, `zone_name`, `area_km2`, `source`, and geometry. Raster uses uint8 zone codes (0 outside/no zone).

**Prerequisites:** Explicit zone input (`AnalysisInputs.zones` or DEA workflow zone result). Unavailable from a cube-only `analyze()` call.

### `reach_profiles`

**Paths:** `vectors/spatial.gpkg` layers `reaches` (geometry) and `reach_wet_monthly` (non-spatial table keyed by `reach_id`, `date` with `is_wet`, `length_m`, `lpsec_contribution_pct`).

**Prerequisites:** Real channel context (`SpatialContext` with drainage geometry). Unavailable from a cube-only `analyze()` call.

## Performance and storage

Controlled benchmark evidence (synthetic fixtures, export-off median regression ≤10%, all-products peak RSS ≤125% of core) is recorded in `benchmarks/results/dynamics_spatial_exports.md`.

Rules of thumb:

- **Export off:** No polygonization, vector checkpoint writes, or raster serialization.
- **Export on:** Reuses the same monthly materialization and label pass as metrics; no second full-cube read.
- **GeoTIFF default:** Tiled 256×256, DEFLATE compression — not advertised as Cloud Optimized GeoTIFF (COG).
- **NetCDF:** Opt-in extra; single write pass, slower than GeoTIFF for large grids.

## Opening outputs

### Python

```python
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

from hydrofragments.output.manifest import validate_result_bundle

bundle = Path("runs/demo_01")
manifest = validate_result_bundle(bundle)
print(manifest["manifest_schema_version"])  # 1.1.0

occurrence = xr.open_dataarray(bundle / "rasters" / "occurrence.tif")
pools = gpd.read_file(bundle / "vectors" / "spatial.gpkg", layer="monthly_pools")
```

### QGIS / GDAL

```bash
gdalinfo runs/demo_01/rasters/occurrence.tif
ogrinfo -al -so runs/demo_01/vectors/spatial.gpkg monthly_pools
```

Load rasters with **Render type → Singleband pseudocolor** for percent products. Check band descriptions for hydroperiod calendar years and refuge-overlap HY pairs.

## Version boundaries

| Version | Scope |
|---------|-------|
| Package | `0.1.0` (`hydrofragments.__version__`) |
| Config schema | `1.0.0` (accepted), `1.1.0` (spatial products) |
| Metric row schema | `1.1.0` (new `EdgeFlag` values for dynamics) |
| Run manifest schema | `1.1.0` (artifact inventory with digests) |

Readers accept legacy metric/manifest `1.0.0` datasets under their original contracts.

## See also

- [Dynamics metrics](metrics/dynamics.md) — reconnection timing and refuge stability scalars
- [Offline example](../examples/spatial_exports.py) — synthetic cube, bundle write, manifest validation
- [Final metrics covered](final_metrics_covered.md) — metric availability matrix
