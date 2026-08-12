# Dynamics and spatial export benchmark (Task 12 gate)

Repository-owned synthetic fixtures and an optional read-only local monthly Zarr subset. Network-dependent DEA acquisition numbers from end_to_end_workflow are excluded from this gate.

- Schema: `1.0.0`
- Baseline commit: `c59bc4b`
- True baseline commit: `12a6dbd`
- Created: `2026-08-12T14:01:55.983733+00:00`

## Scenario medians

| Scenario | fixture | products | median total s | median peak RSS | metric parity |
| --- | --- | --- | ---: | ---: | :---: |
| baseline_export_off | compact_georef | off | 0.666 | 501,141,504 | n/a |
| candidate_export_off | compact_georef | off | 0.660 | 526,622,720 | yes |
| candidate_persistence_rasters | compact_georef | persistence_rasters | 1.627 | 539,992,064 | yes |
| candidate_monthly_pools | compact_georef | monthly_pools | 1.940 | 554,160,128 | yes |
| candidate_all_products | compact_georef | monthly_pools,persistence_rasters,temporal_rasters | 2.130 | 556,851,200 | yes |
| candidate_netcdf | n/a | n/a | skipped | n/a | n/a |
| long_480_memory | long_480_small | off | 3.718 | 556,130,304 | no |
| large_spatial_sparse | large_spatial_sparse | off | 0.649 | 527,204,352 | no |
| large_spatial_single_component | large_spatial_single_component | off | 0.733 | 545,230,848 | no |
| checkpoint_export_retry | compact_georef | persistence_rasters,monthly_pools | 2.331 | 555,720,704 | no |
| zarr_local_subset | n/a | n/a | skipped | n/a | n/a |

## Promotion gates

- `true_baseline_commit`: 12a6dbd
- `export_off_median_seconds_baseline`: 0.6661107999971136
- `export_off_median_seconds_candidate`: 0.6598162999725901
- `export_off_regression_fraction`: -0.009449629137601223
- `export_off_within_10pct_gate`: True
- `export_off_peak_rss_bytes_baseline_median`: 501141504
- `export_off_peak_rss_bytes_candidate_median`: 526622720
- `export_off_peak_rss_tolerance_bytes`: 5242880
- `export_off_peak_rss_within_gate`: False
- `all_products_peak_rss_bytes_median`: 556851200
- `all_products_peak_rss_fraction_of_core`: 1.0574006377848644
- `all_products_peak_rss_within_125pct_gate`: True
- `long_480_peak_rss_bytes_median`: 556130304
- `long_480_peak_rss_fraction_of_compact`: 1.05603173368593
- `long_480_peak_rss_documented_tolerance_bytes`: 33554432
- `long_480_memory_within_gate`: True
- `large_spatial_admission_budget_bytes`: 64000
- `large_spatial_peak_rss_increment_bytes`: 26062848
- `large_spatial_rss_documented_tolerance_bytes`: 33554432
- `large_spatial_rss_within_125pct_admission_gate`: True
- `large_spatial_single_component_fail_fast`: False
- `metric_parity_on_off_holds`: True
- `checkpoint_retry_skips_source_reads`: True

