# HydroFragments

HydroFragments `0.1.0` quantifies river surface-water patch dynamics from aligned water and valid-observation time series.

> iRiverMetrics is now HydroFragments.

## What it measures

HydroFragments derives surface-water landscape metrics across seven metric families:
- **Extent:** Water surface area, percentage of section covered (APSEC), and valid observation bounds.
- **Persistence:** Season-stratified water occurrence frequency and refuge area.
- **Fragmentation:** Pool count (number of pools), largest patch index (LPI), and patch size distributions.
- **Morphology:** Area-weighted relative edge (AWRe), area-weighted mean shape index (AWMSI), and pool width statistics.
- **Dynamics:** Extent contraction, reconnection timing after end-dry, and refuge spatial stability.
- **Channel:** Channel-aligned wet reach profile and pool spacing along drainage centrelines.
- **Connectivity:** Structural river connectivity indices (RC, TCF, DCI).

## Installation

CI-tested on Python 3.10–3.13.

```bash
git clone https://github.com/tayerthiaggo/HydroFragments.git
cd HydroFragments
python -m pip install -e ".[test]"
```

Optional NetCDF spatial export:

```bash
python -m pip install -e ".[netcdf]"
```

## Quickstart

```python
import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube

times = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
water = xr.DataArray(
    np.array(
        [
            [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=bool,
    ),
    dims=("time", "y", "x"),
    coords={"time": times},
)

cube = open_water_cube(water, input_kind="generic_binary")
config = HydroConfig.from_mapping(
    {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
    }
)
result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
print(result.metrics_table[["date", "metric", "value"]].head())
```

Without `output.output_dir`, `analyze()` performs no filesystem writes and returns `result.output_dir is None` with an in-memory manifest dictionary.

## Outputs

When `output.output_dir` is set, `analyze()` (or `analyze_from_dea()`) writes a complete result bundle to that directory. The path names one final run directory; it must be absent or empty before the run starts.

Default layout (only selected products are created):

```text
<output_dir>/
  config.json
  metrics/                    partitioned canonical Parquet dataset
  metrics.csv                 only when CSV is selected
  metric_coverage.csv
  vectors/spatial.gpkg        optional GeoPackage layers
  rasters/*.tif               optional GeoTIFF products
  rasters/spatial.nc          optional consolidated NetCDF (requires [netcdf])
  run_manifest.json           written last; full artifact inventory
```

- **`metrics/`** — Hive-partitioned Parquet dataset (`metric_family`, `value_type`); there is no single-file `metrics.parquet` at the bundle root.
- **`metric_coverage.csv`** — Per-metric runtime status and skip reasons (CSV only; no coverage Parquet).
- **`run_manifest.json`** — Provenance, config hashes, and SHA-256 digests for every artifact. Not `manifest.json`.
- **Spatial products** — Off by default. Enable with `output.spatial_products` in config schema `1.1.0`. See [Spatial exports](docs/spatial_exports.md).

`HydroResult.write(path, formats=("parquet",))` writes metric tables and coverage only. It does not export spatial products; request those in `OutputConfig` before calling `analyze()`.

## Scientific scope and limitations

HydroFragments quantifies surface-water patch geometry and landscape structure in intermittent river corridors from satellite-derived surface water masks. It does not model subsurface hydrology, water depth, flow velocity, or ecological condition.

## Documentation

- [Docs Index](docs/index.md)
- [Project Overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Input Format](docs/input_format.md)
- [Metric Specification](docs/metric_specification.md)
- [Dynamics metrics](docs/metrics/dynamics.md)
- [Spatial exports](docs/spatial_exports.md)
- [Scientific Foundation](docs/scientific-foundation.md)
- [Validation Status](docs/validation_status.md)
- [Testing Guide](docs/testing.md)
- [Changelog](CHANGELOG.md)

## Citation

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river, *Journal of Hydrology*, DOI [10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087).

## License

[MIT License](LICENSE)
