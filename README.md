# HydroFragments

HydroFragments `0.1.0` quantifies river surface-water patch dynamics from aligned water and valid-observation time series.

> iRiverMetrics is now HydroFragments.

## What it measures

HydroFragments derives surface-water landscape metrics across seven metric families:
- **Extent:** Water surface area, percentage of section covered (APSEC), and valid observation bounds.
- **Persistence:** Season-stratified water occurrence frequency and refuge area.
- **Fragmentation:** Pool count (number of pools), largest patch index (LPI), and patch size distributions.
- **Morphology:** Area-weighted relative edge (AWRe), area-weighted mean shape index (AWMSI), and pool width statistics.
- **Dynamics:** Wetting and drying transition rates and temporal persistence classes.
- **Channel:** Channel-aligned wet reach profile and pool spacing along drainage centrelines.
- **Connectivity:** Structural river connectivity indices (RC, TCF, DCI).

## Installation

CI-tested on Python 3.10 and 3.11.

```bash
git clone https://github.com/tayerthiaggo/HydroFragments.git
cd HydroFragments
python -m pip install -e ".[test]"
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
        "output": {"output_dir": "hydrofragments_out"},
    }
)
result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
print(result.metrics_table[["date", "metric", "value"]].head())
```

## Outputs

`analyze()` writes reproducible tidy tables and JSON manifests to `config.output.output_dir`:
- `metrics.parquet` / `metrics.csv`: Tidy metric values with timestamps, AOI IDs, and unit attributes.
- `metric_coverage.parquet` / `metric_coverage.csv`: Observation validity and coverage fraction records.
- `manifest.json`: Full run provenance, hash digest, software environment, and input configuration snapshot.

## Scientific scope and limitations

HydroFragments quantifies surface-water patch geometry and landscape structure in intermittent river corridors from satellite-derived surface water masks. It does not model subsurface hydrology, water depth, flow velocity, or ecological condition.

## Documentation

- [Docs Index](docs/index.md)
- [Project Overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Input Format](docs/input_format.md)
- [Metric Specification](docs/metric_specification.md)
- [Scientific Foundation](docs/scientific-foundation.md)
- [Validation Status](docs/validation_status.md)
- [Testing Guide](docs/testing.md)
- [Changelog](CHANGELOG.md)

## Citation

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river, *Journal of Hydrology*, DOI [10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087).

## License

[MIT License](LICENSE)
