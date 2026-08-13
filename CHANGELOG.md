# Changelog

All notable changes to HydroFragments are documented here.

## [Unreleased]

### Added

- **Dynamics profile wiring:** `reconnection_timing` and `refuge_spatial_stability`
  emit from `analyze()` when hydrological-year prerequisites exist, alongside
  `extent_contraction`.
- **Spatial export system (config schema 1.1.0):** Optional GeoTIFF rasters,
  GeoPackage vectors, and opt-in NetCDF (`[netcdf]` extra). Products are off by
  default; enabling them does not change metric table values.
- **Validated result bundles:** Atomic staging, artifact inventory with SHA-256
  digests, and `run_manifest.json` written last (manifest schema 1.1.0).
- **Documentation:** [Spatial exports guide](docs/spatial_exports.md), updated
  [dynamics metrics](docs/metrics/dynamics.md), and offline
  [examples/spatial_exports.py](examples/spatial_exports.py).

### Changed

- **README outputs:** Documents partitioned `metrics/` Parquet, `metric_coverage.csv`,
  and `run_manifest.json`; clarifies side-effect-free `analyze()` without
  `output_dir` and table-only `HydroResult.write()`.
- **Metric row schema 1.1.0:** Machine-readable `EdgeFlag` values for dynamics
  edge cases (no new columns).
- **Reconnection thresholds:** `DynamicsConfig` percentage thresholds on 0–100
  scale, included in scientific hash.

### Performance

- Controlled benchmark evidence: export-off median regression ≤10% vs true
  baseline `12a6dbd`; all-products peak RSS ≤125% of core on synthetic fixtures.
  Export-off peak RSS documents a 32 MiB constant overhead (not O(time)
  retention). See `benchmarks/results/dynamics_spatial_exports.md`.

## [0.1.0] - 2026-08-12

### Added

- First public HydroFragments release.
- Surface-water extent, persistence, fragmentation, morphology, dynamics,
  channel, and connectivity metric workflows.
- DEA/Water Observations integration, reproducible manifests, validation
  guards, examples, benchmarks, and scientific documentation.
