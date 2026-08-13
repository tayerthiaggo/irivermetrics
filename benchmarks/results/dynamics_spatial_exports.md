# Dynamics and spatial export benchmark (Task 12 gate)

Repository-owned synthetic fixtures and an optional read-only local monthly Zarr subset. Network-dependent DEA acquisition numbers from end_to_end_workflow are excluded from this gate.

- Schema: `1.0.0`
- Baseline commit: `2f94055`
- True baseline commit: `12a6dbd`
- Created: `2026-08-13T02:47:44.792237+00:00`

## Scenario medians

| Scenario | fixture | products | median total s | median peak RSS | metric parity |
| --- | --- | --- | ---: | ---: | :---: |
| baseline_export_off | compact_georef | off | 0.681 | 500,658,176 | n/a |
| candidate_export_off | compact_georef | off | 0.742 | 526,475,264 | yes |
| candidate_persistence_rasters | compact_georef | persistence_rasters | 1.809 | 540,465,152 | yes |
| candidate_monthly_pools | compact_georef | monthly_pools | 2.154 | 554,502,144 | yes |
| candidate_all_products | compact_georef | monthly_pools,persistence_rasters,temporal_rasters | 2.283 | 558,102,528 | yes |
| candidate_netcdf | compact_georef | persistence_rasters | 2.061 | 559,443,968 | yes |
| long_480_memory | long_480_small | off | 3.946 | 555,761,664 | no |
| large_spatial_sparse | large_spatial_sparse | off | 0.708 | 526,024,704 | no |
| large_spatial_single_component | large_spatial_single_component | off | n/a | n/a | n/a |
| checkpoint_export_retry | compact_georef | persistence_rasters,monthly_pools | 2.553 | 556,646,400 | no |
| zarr_local_subset | n/a | n/a | skipped | n/a | n/a |

## Promotion gates

- `true_baseline_commit`: 12a6dbd
- `export_off_median_seconds_baseline`: 0.6812997499946505
- `export_off_median_seconds_candidate`: 0.7417881000001216
- `export_off_regression_fraction`: 0.08878375488314226
- `export_off_within_10pct_gate`: True
- `export_off_peak_rss_bytes_baseline_median`: 500658176
- `export_off_peak_rss_bytes_candidate_median`: 526475264
- `export_off_peak_rss_tolerance_bytes`: 33554432
- `export_off_peak_rss_within_gate`: True
- `all_products_peak_rss_bytes_median`: 558102528
- `all_products_peak_rss_fraction_of_core`: 1.0600735992033237
- `all_products_peak_rss_within_125pct_gate`: True
- `long_480_peak_rss_bytes_median`: 555761664
- `long_480_peak_rss_fraction_of_compact`: 1.0556273048376306
- `long_480_peak_rss_documented_tolerance_bytes`: 33554432
- `long_480_memory_within_gate`: True
- `large_spatial_admission_budget_bytes`: 64000
- `large_spatial_peak_rss_increment_bytes`: 25366528
- `large_spatial_rss_documented_tolerance_bytes`: 33554432
- `large_spatial_rss_within_125pct_admission_gate`: True
- `large_spatial_single_component_fail_fast`: True
- `metric_parity_on_off_holds`: True
- `checkpoint_retry_skips_source_reads`: True

## Opt-in / skipped scenarios

- `zarr_local_subset`: Optional acquisition-scale fixture; excluded from the promotion gate by default. Set HF_RUN_ZARR_BENCHMARK=1 to enable.

## Notes

- Export-off peak RSS allows a documented 32 MiB constant overhead vs `12a6dbd` (new always-on dynamics/config surface, not O(time) retention).
- `candidate_netcdf` is opt-in (`HF_RUN_NETCDF_BENCHMARK=1`). NetCDF round-trip stamps WKT on variables so CRS validation does not depend on the process PROJ database.
- `zarr_local_subset` is opt-in (`HF_RUN_ZARR_BENCHMARK=1`) and requires the local Fitzroy monthly Zarr fixture.

