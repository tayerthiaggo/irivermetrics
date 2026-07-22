# HydroFragments

> **Status (v1.2.0rc1):** Public package namespace is `hydrofragments`. The legacy
> `ecofragments.calculate_metrics` facade remains for one deprecation cycle and
> returns a **non-canonical** wide pivot of retained v1.2 metrics only. Canonical
> output is tidy Parquet via `hydrofragments.analyze()`. See
> [docs/migration_v1_2.md](docs/migration_v1_2.md).

HydroFragments quantifies **surface-water patch dynamics** in intermittent rivers
from binary or WaterMask-TSFill monthly water time series. Scope is river
surface water only — not generic terrestrial/urban patch metrics, and not flow,
depth, or ecological condition.

## Shipped in v1.2.0 core (`contracts_core`)

- Occurrence (season-stratified, valid-observation denominator)
- Refuge area
- APSEC (fixed AOI denominator)
- Number of pools, LPI, AWRe, AWMSI

Explicitly deferred: LPSEC, HY/dry-down, connectivity (RC/TCF/DCI runtime),
MESH/width distributions, and CUDA acceleration.

## Install

```bash
git clone https://github.com/tayerthiaggo/HydroFragments.git
cd HydroFragments
python -m pip install -e ".[test]"
```

CPU-only install: no CuPy/CUDA packages are required. Optional GPU extras are not
enabled in this release candidate.

## Quickstart

**New to HydroFragments?** [`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb)
walks through the same pipeline below in plain language, with a plot, runnable
end-to-end on a bundled tiny fixture in under two minutes -- no external data
required. From there, [`examples/02_dea_via_tsfill.ipynb`](examples/02_dea_via_tsfill.ipynb)
covers the real Digital Earth Australia / WaterMask-TSFill workflow, and
[`examples/03_metrics_walkthrough.ipynb`](examples/03_metrics_walkthrough.ipynb)
tours every metric family and the 4-zone spatial stratification.

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

### Command line

Once you have a real config file and input, the same pipeline is available
without opening a notebook or writing a script:

```bash
hydrofragments analyze --config cfg.yaml --input data.zarr --aoi my_reach --out results/
```

`--config` accepts YAML or JSON (see `HydroConfig.from_mapping` for the exact
schema); `--input` is passed straight to `open_water_cube`. Run
`hydrofragments analyze --help` for the full argument list.

## Legacy import (deprecated)

```python
from ecofragments import calculate_metrics  # emits DeprecationWarning
```

Dropped legacy metrics (`PF`, `PLF`, `AWMPA`, `AWMPL`, `AWMPW`) raise
`LegacyMetricMigrationError` when explicitly requested. They are never restored
in compatibility output.

## Documentation

- [Docs index](docs/index.md)
- [Architecture](docs/architecture.md)
- [Input format](docs/input_format.md)
- [Migration v1.2](docs/migration_v1_2.md)
- [Historical `calculate_metrics` reference](docs/module2.md) (legacy, quarantined)
- [Quickstart notebook](examples/01_quickstart.ipynb)
- [DEA / WaterMask-TSFill notebook](examples/02_dea_via_tsfill.ipynb)
- [Metrics walkthrough notebook](examples/03_metrics_walkthrough.ipynb)

## Citation

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D.
(2023) Ecohydrological metrics derived from multispectral images to characterize
surface water in an intermittent river, *Journal of Hydrology*, DOI
[10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087).

Lineage note: this repository is a clean HydroFragments v1.2 rewrite; predecessor
public history is not preserved (Decision Q10 option C).
