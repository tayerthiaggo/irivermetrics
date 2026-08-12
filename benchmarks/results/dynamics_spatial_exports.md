# Dynamics and spatial export benchmark (Task 12 gate)

Repository-owned synthetic fixtures and an optional read-only local monthly Zarr subset. Network-dependent DEA acquisition numbers from end_to_end_workflow are excluded from this gate.

- Schema: `1.0.0`
- Baseline commit: `4fab7df`
- Created: `2026-08-12T12:40:52.207502+00:00`

## Scenario medians

| Scenario | fixture | products | median total s | median peak RSS | metric parity |
| --- | --- | --- | ---: | ---: | :---: |
| baseline_export_off | compact_georef | off | 2.042 | 526,684,160 | no |
| candidate_export_off | compact_georef | off | 2.130 | 526,934,016 | yes |
| candidate_persistence_rasters | compact_georef | persistence_rasters | 2.235 | 539,840,512 | yes |
| candidate_monthly_pools | compact_georef | monthly_pools | 2.822 | 554,479,616 | yes |
| candidate_all_products | compact_georef | monthly_pools,persistence_rasters,temporal_rasters | 3.169 | 557,449,216 | yes |
| candidate_netcdf | n/a | n/a | skipped | n/a | n/a |
| long_480_memory | long_480_small | off | 65.241 | 555,429,888 | no |
| large_spatial_sparse | large_spatial_sparse | off | 1.155 | 525,905,920 | no |
| large_spatial_single_component | large_spatial_single_component | off | n/a | n/a | n/a |
| checkpoint_export_retry | compact_georef | persistence_rasters,monthly_pools | 2.624 | 556,482,560 | no |
| zarr_local_subset | n/a | n/a | skipped | n/a | n/a |

## Promotion gates

- `export_off_median_seconds_baseline`: 2.0417801999719813
- `export_off_median_seconds_candidate`: 2.1296700999955647
- `export_off_regression_fraction`: 0.04304572060439679
- `export_off_within_10pct_gate`: True
- `export_off_peak_rss_bytes_candidate_median`: 526934016
- `all_products_peak_rss_bytes_median`: 557449216
- `all_products_peak_rss_fraction_of_core`: 1.0579108561478787
- `all_products_peak_rss_within_125pct_gate`: True
- `metric_parity_on_off_holds`: True
- `checkpoint_retry_skips_source_reads`: True

