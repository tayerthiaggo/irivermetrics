# HydroFragments v1.2 migration guide

## Package identity

| Before | After |
|--------|-------|
| `pip install` name `ecofragments` | `hydrofragments` |
| `from ecofragments import calculate_metrics` | `from hydrofragments import analyze, open_water_cube` |
| Wide `ecof_metrics.csv` (16 legacy columns) | Tidy Parquet + `run_manifest.json` |

`ecofragments` remains importable for **one deprecation cycle** with warnings.

## Canonical workflow

```python
from hydrofragments import HydroConfig, analyze, open_water_cube

cube = open_water_cube(dataset_or_array, input_kind="watermask_tsfill")
config = HydroConfig.from_mapping({...})
result = analyze(cube, aoi_id="reach-01", config=config)
result.write("run_output", formats=("parquet",))
```

## Compatibility facade

`ecofragments.calculate_metrics(...)` still runs for section shapefiles and legacy
call sites, but:

- Output includes **retained v1.2 metrics only** in a wide pivot.
- Output is labelled non-canonical; do not use it as a schema authority.
- `export_shp` / `export_PP` raise migration errors (use v1.2 export paths).

### Retained wide columns (compat pivot)

| Legacy column | v1.2 metric | Notes |
|---------------|-------------|-------|
| `n_patches` | `number_of_pools` | same intent |
| `APSEC` | `apsec` | fixed AOI denominator |
| `AWMSI` | `awmsi` | |
| `AWRe` | `awre` | major-axis path in core |
| `LPI` | `lpi` | new canonical fragmentation axis |
| `pp_mean_%` | `occurrence` | **semantics changed** — season-stratified valid-obs denominator |
| `ra_area_km2` | `refuge_area` | threshold from `persistence.refuge_threshold` |

### Dropped metrics (explicit migration errors)

| Metric | Status | Replacement / guidance |
|--------|--------|------------------------|
| `PF` | removed | use `lpi` + `number_of_pools` with fixed AOI context |
| `PLF` | removed | use `lpi` when channel `L_ref` is available |
| `AWMPA` | removed | not in v1.2 register |
| `AWMPL` | removed | not in v1.2 register |
| `AWMPW` | removed | width distribution deferred (resolution-floor guard) |
| `LPSEC` | excluded from core | requires real drainage `L_ref` contract |
| `NNI` | cut | not scientifically defensible in v1.2 |

Requesting dropped metrics via `legacy_metrics=[...]` raises
`LegacyMetricMigrationError` with per-metric guidance.

## Input changes

- Canonical input is a `WaterCube` with aligned `water` and `valid_obs`.
- WaterMask-TSFill sentinels `254`/`255` must decode before any signed cast.
- See [input_format.md](./input_format.md).

## Not in v1.2.0

- `waterdetect_batch` (never existed in this repo)
- HY detection, extent contraction, connectivity runtime metrics
- CUDA acceleration (optional later tranche)
