# HydroFragments architecture (v1.2.0rc1)

HydroFragments is a **river surface-water** metrics toolkit. It is not a generic
terrestrial/urban patch engine. v1.2 focuses on scientifically defensible extent,
persistence, and morphology metrics with tidy output and explicit guards.

## Package layout

```
hydrofragments/
  api.py              # open_water_cube, validate_inputs, analyze, compare_results
  config.py           # HydroConfig, config_hash
  schema.py           # tidy output contract
  models.py           # WaterCube, HydroResult, MetricRecord
  compat.py           # ecofragments migration helpers (no duplicate kernels)
  io/                 # adapters, validity, alignment
  temporal/           # cadence, compositing
  compute/            # Dask policy, chunk budgets
  patches/            # CPU reference labeling and morphology
  metrics/            # registry + metric implementations
  output/             # Parquet, manifest, rasters
  guards/             # scientific + comparison guards
  pipeline.py         # monthly checkpoint orchestration

ecofragments/
  main.py             # deprecated calculate_metrics facade only
  utils/calc_metrics.py  # legacy kernels — characterisation tests only
```

## Public API

| Entry point | Role |
|-------------|------|
| `open_water_cube()` | Canonical `WaterCube` boundary |
| `validate_inputs()` | Contract checks without metric compute |
| `analyze()` | Core execution; tidy metrics + manifest |
| `compare_results()` | Refuses incompatible runs by default |

## Compatibility facade

`ecofragments.calculate_metrics` routes to `hydrofragments.compat.calculate_metrics_compat`.
It emits `DeprecationWarning`, returns a **non-canonical** wide pivot of retained
metrics, and raises `LegacyMetricMigrationError` for dropped legacy metrics.

Dropped metrics are never recomputed: `PF`, `PLF`, `AWMPA`, `AWMPL`, `AWMPW`, `LPSEC`
(channel core), NNI, graph centralities.

## Compute model

- Temporal reductions and occurrence stay lazy until the monthly checkpoint.
- Patch morphology runs on bounded CPU component crops (exact reference).
- CUDA is optional and not part of the v1.2.0rc1 release candidate.

## Deferred tranches

Pixel-temporal (recurrence/hydroperiod), HY/dry-down, channel/zones, connectivity,
and CUDA each have independent decision gates documented in
`docs/audit/implementation_plan.md`.
