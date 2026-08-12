# Final metrics covered by HydroFragments

Source checked: `README.md`, `hydrofragments/metrics/registry.py`, public API wiring in
`hydrofragments/api.py`, metric modules under `hydrofragments/metrics/`, and
`docs/validation_status.md`.

## Current release core

These metrics are in the default `contracts_core` profile and are the v1.2.0rc1
headline output.

| Metric ID | Name | Unit | Brief explanation |
|---|---|---:|---|
| `occurrence` | Occurrence frequency | percent | Long-term wetness of each pixel using valid observations only. Missing imagery is not counted as dry; the shipped estimator equal-weights supported calendar months. |
| `refuge_area` | Refuge area | km2 | Area of pixels whose occurrence meets the configured refuge threshold, default `0.90`, and clears the valid-observation support floor. |
| `apsec` | Area percent of section | percent | Monthly wetted area divided by fixed AOI or section area. Measures visible surface-water extent against a stable denominator. |
| `number_of_pools` | Number of pools | count | Count of connected wet patches after the configured connectivity rule and minimum patch-size filter. |
| `lpi` | Largest patch index | percent | Area of the largest wet patch divided by fixed AOI or landscape area. Shows dominance of one large refuge versus many smaller ones. |
| `awre` | Area-weighted elongation ratio | dimensionless | Area-weighted pool compactness or elongation signal. Uses a locked length method so skeleton and major-axis lengths are not silently mixed. |
| `awmsi` | Area-weighted mean shape index | dimensionless | Area-weighted boundary complexity of wet patches. Complements AWRe by describing edge complexity rather than elongation alone. |

## Optional or gated profiles

These metrics are covered by the repo and registry, but need selected profiles and
input dependencies. Some are implemented as kernels but not part of the default
core release path.

| Metric ID | Profile / status | Unit | Brief explanation |
|---|---|---:|---|
| `recurrence` | `pixel_temporal` | percent | Inter-annual reliability of wetness, using valid years/month support instead of assuming missing months are dry. |
| `hydroperiod` | `pixel_temporal` | fraction | Within-year fraction of valid observed months where a pixel is wet. Summarised per year. |
| `extent_contraction` | `dynamics`, HY + dual-composite gated | percent_per_month | Monthly APSEC slope over the drying limb of a hydrological year. This is surface-water extent contraction, not streamflow or discharge recession. |
| `reconnection_timing` | `dynamics`, HY + connectivity support gated | month | Lag after end-dry until LPSEC or LPI crosses the configured percent threshold. LPSEC preferred when complete; proxy use is flagged. Wired in `analyze()` when prerequisites exist. |
| `refuge_spatial_stability` | `dynamics`, HY + end-dry masks gated | dimensionless | Jaccard overlap of consecutive end-dry refuge footprints on common-valid support. First year and empty union are non-reportable. Wired in `analyze()` when prerequisites exist. |
| `lpsec` | `channel`, real-channel gated | percent | Wetted channel length divided by fixed real channel reference length. Requires validated drainage/channel context. |
| `inter_pool_gap` | `channel`, real-channel gated | km | Along-channel dry-gap distances between bounded wet segments; repo emits summary statistics such as mean, median, max, and CV. |
| `pool_width` | `secondary`, width-floor gated | m | Planform pool-width distribution from distance transform. Width is surface width only, not depth or water volume. |
| `mesh` | `secondary`, validation-gated; currently disabled | m2 | Effective mesh size: fixed-denominator patch-size distribution metric. Fitzroy validation found high redundancy with LPI, so it is gated off unless validation supports use. |
| `realised_connectivity` | `connectivity`; runtime deferred in README | dimensionless | Snapshot connectivity of a fixed drainage graph using active edges or reachable node pairs. Positioned against DCI/PC/IIC literature. |
| `tcf` | `connectivity`; runtime deferred in README | percent | Temporal connectivity frequency for fixed graph nodes: active valid months divided by valid months. |

## Spatial export products

Optional GIS products (config schema `1.1.0`, off by default) are documented in
[spatial_exports.md](spatial_exports.md). Metric tables remain identical whether
spatial export is enabled or disabled.

| Product key | Typical artifacts | Prerequisites |
|---|---|---|
| `persistence_rasters` | occurrence, valid_observation_count, refuge_mask GeoTIFFs | Georeferenced cube |
| `temporal_rasters` | recurrence, hydroperiod GeoTIFFs | Georeferenced cube, temporal record |
| `refuge_stability_rasters` | refuge overlap, stability frequency GeoTIFFs | HY anchors, ≥2 end-dry states |
| `monthly_pools` | `vectors/spatial.gpkg` layer `monthly_pools` | Georeferenced cube, patch filtering |
| `zones` | zones GPKG layer + `zones.tif` | Explicit zone input or DEA workflow |
| `reach_profiles` | `reaches` + `reach_wet_monthly` layers | Channel `SpatialContext` |

## Not final metrics

The legacy metrics `PF`, `PLF`, `AWMPA`, `AWMPL`, `AWMPW`, `PCF`, `NNI`,
`degree_centrality`, and `betweenness_centrality` are not final HydroFragments
metrics. The schema forbids or retires them because they were circular,
redundant, renamed, or cut from v1.2.

## Reading rule

HydroFragments measures visible river surface-water extent, persistence,
fragmentation, morphology, clustering, dynamics, and connectivity from water-mask
time series. It does not directly measure discharge, streamflow, depth, water
quality, groundwater, or ecological condition.
