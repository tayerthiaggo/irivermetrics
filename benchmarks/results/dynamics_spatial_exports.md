# Dynamics and spatial export benchmark (Task 12 gate)

Repository-owned synthetic fixtures and an optional read-only local monthly Zarr subset. Network-dependent DEA acquisition numbers from end_to_end_workflow are excluded from this gate.

- Schema: `1.0.0`
- Baseline commit: `6230b31`
- True baseline commit: `12a6dbd`
- Created: `2026-08-12T13:42:32.261257+00:00`

## Scenario medians

| Scenario | fixture | products | median total s | median peak RSS | metric parity |
| --- | --- | --- | ---: | ---: | :---: |
| baseline_export_off | compact_georef | off | 0.662 | 500,895,744 | n/a |
| candidate_export_off | compact_georef | off | 1.528 | 526,938,112 | yes |
| candidate_persistence_rasters | compact_georef | persistence_rasters | 1.802 | 540,291,072 | yes |
| candidate_monthly_pools | compact_georef | monthly_pools | 2.031 | 555,032,576 | yes |
| candidate_all_products | compact_georef | monthly_pools,persistence_rasters,temporal_rasters | 2.290 | 556,785,664 | yes |
| candidate_netcdf | n/a | n/a | skipped | n/a | n/a |
| long_480_memory | long_480_small | off | 46.959 | 555,208,704 | no |
| large_spatial_sparse | large_spatial_sparse | off | 1.060 | 525,819,904 | no |
| large_spatial_single_component | large_spatial_single_component | off | n/a | n/a | n/a |
| checkpoint_export_retry | compact_georef | persistence_rasters,monthly_pools | 2.550 | 556,711,936 | no |
| zarr_local_subset | n/a | n/a | skipped | n/a | n/a |

## Promotion gates

- `true_baseline_commit`: 12a6dbd
- `export_off_median_seconds_baseline`: 0.6615032000117935
- `export_off_median_seconds_candidate`: 1.5281053000071552
- `export_off_regression_fraction`: 1.3100497472724417
- `export_off_within_10pct_gate`: False
- `export_off_peak_rss_bytes_baseline_median`: 500895744
- `export_off_peak_rss_bytes_candidate_median`: 526938112
- `export_off_peak_rss_tolerance_bytes`: 5242880
- `export_off_peak_rss_within_gate`: False
- `all_products_peak_rss_bytes_median`: 556785664
- `all_products_peak_rss_fraction_of_core`: 1.0566433729507878
- `all_products_peak_rss_within_125pct_gate`: True
- `long_480_peak_rss_bytes_median`: 555208704
- `long_480_peak_rss_fraction_of_compact`: 1.0536506875403235
- `long_480_peak_rss_documented_tolerance_bytes`: 33554432
- `long_480_memory_within_gate`: True
- `large_spatial_admission_budget_bytes`: 64000
- `large_spatial_peak_rss_increment_bytes`: 24924160
- `large_spatial_rss_documented_tolerance_bytes`: 33554432
- `large_spatial_rss_within_125pct_admission_gate`: True
- `large_spatial_single_component_fail_fast`: True
- `metric_parity_on_off_holds`: True
- `checkpoint_retry_skips_source_reads`: True

