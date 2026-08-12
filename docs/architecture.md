# HydroFragments Architecture

HydroFragments provides a modular architecture for computing surface-water landscape dynamics from satellite time series.

## Package Layout

```text
hydrofragments/
├── api.py               # Main public entrypoint (analyze, open_water_cube)
├── config.py            # HydroConfig schema and validation
├── models.py            # AnalysisInputs, WaterCube, MetricRecord data classes
├── section_analysis.py  # Monthly section analysis engine
├── workflow.py          # High-level DEA/WOfS workflow orchestrator
├── metrics/             # Vectorized metric implementations
│   ├── extent.py        # Surface area, APSEC, coverage fraction
│   ├── persistence.py   # Occurrence frequency, refuge area
│   ├── patch_bundle.py  # Patch labeling and connected components
│   ├── morphology.py   # AWRe, AWMSI, pool width statistics
│   ├── dynamics.py     # Transition rates and temporal classes
│   └── connectivity.py # River connectivity indices (RC, TCF, DCI)
├── spatial/             # Spatial operations and channel centrelines
├── io/                  # Raster and Zarr IO adapters
└── output/              # Tidy Parquet/CSV and JSON manifest generation
```

## Core Execution Flow

1. **Input Intake:** `open_water_cube()` wraps input DataArrays into a validated `WaterCube`.
2. **Configuration:** `HydroConfig` configures spatial, temporal, compute, and output settings.
3. **Monthly Analysis Engine:** `analyze_section_rows()` in `section_analysis.py` processes data month-by-month, bounding memory usage.
4. **Metric Resolution:** Selected metric families are calculated via optimized NumPy/SciPy/xarray operations.
5. **Output Assembly:** `write_output_tables()` exports tidy Parquet/CSV metrics tables and JSON provenance manifests.
