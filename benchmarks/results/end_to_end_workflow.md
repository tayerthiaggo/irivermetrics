# End-to-end workflow benchmark (W3.7 benchmark gate)

Controller-approved reduced scope: only Fitzroy has a local AOI/drainage fixture in this worktree. Gilbert and the large-catchment required cases are explicitly not run -- see their 'skipped' entries below, not omitted from this schema.

- Schema: `1.0.0`
- Created: `2026-07-31T06:55:37.933392+00:00`

## Fitzroy (compact) -- real, live-network run

| Candidate | factor | workers | cold total s | warm total s | warm speedup | regression vs serial | peak RSS (cold) | RSS vs serial | native-wet superset holds (cold) | superset coverage (cold) | valid obs. fraction (cold) | timing/RSS gates | all gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | :---: |
| factor4_workers1 | 4 | 1 | 43.521 | 16.359 | 62.4% | 0.0% | 778,051,584 | 100.0% | yes | 100.0% | 90.3% | yes | no |
| factor4_workers2 | 4 | 2 | 42.985 | 15.048 | 65.0% | -1.2% | 705,294,336 | 90.6% | yes | 100.0% | 90.3% | yes | no |
| factor4_workers4 | 4 | 4 | 43.765 | 15.890 | 63.7% | 0.6% | 707,088,384 | 90.9% | yes | 100.0% | 90.3% | yes | no |
| factor3_workers1 | 3 | 1 | 46.128 | 16.520 | 64.2% | 6.0% | 766,173,184 | 98.5% | yes | 100.0% | 90.3% | yes | no |
| factor3_workers2 | 3 | 2 | 43.655 | 16.436 | 62.4% | 0.3% | 784,420,864 | 100.8% | yes | 100.0% | 90.3% | yes | no |
| factor3_workers4 | 3 | 4 | 41.935 | 15.567 | 62.9% | -3.6% | 787,730,432 | 101.2% | yes | 100.0% | 90.3% | yes | no |

Promotion gates applied (Fitzroy-only subset -- see module docstring):
- `exact_metrics_table_and_per_metric_value_equality`: measured per-candidate as cold_warm_metrics_equal (cold vs warm rerun of the SAME candidate); cross-candidate full-AOI-vs-pruned equality was not separately run this session -- see report notes
- `n_water_equality_every_month`: measured per-candidate as cold_warm_n_water_equal
- `count_wet_planning_footprint_covers_100pct_of_native_wet_pixels`: measured per-candidate as planning_footprint_native_wet_pixel_superset_holds (native_mask <= expand(coarse_mask), the same superset property W1.5 proves in hydroseason)
- `cold_gilbert_at_least_30pct_faster_than_full_aoi`: None
- `warm_rerun_at_least_80pct_faster_than_cold_full_aoi`: Fitzroy's own warm-vs-cold speedup is reported per candidate as warm_speedup_fraction_vs_own_cold; the plan's 80% figure is stated against Gilbert specifically and is NOT claimed satisfied here even where Fitzroy's own number exceeds it
- `compact_fitzroy_regression_no_worse_than_10pct`: measured per-candidate as regression_within_10pct_gate, relative to the factor=4/workers=1 serial baseline
- `peak_rss_no_more_than_125pct_of_serial`: measured per-candidate as peak_rss_within_125pct_gate

## Recommendation

- **verdict**: no_passing_candidate
- **reason**: No candidate passed every measurable Fitzroy gate. Default settings are left unchanged.
- **timing_rss_only_note**: candidate=factor3_workers4 was fastest among candidates passing ONLY the timing/RSS gates (regression <=10%, peak RSS <=125%) -- informational, since it did NOT also pass the exact-equality/coverage gates required for all_measurable_gates_pass.

## Deferred (not run -- no local fixture)

### Gilbert -- thin/braided catchment, plan's 'cold Gilbert >= 30% faster than full AOI' gate

Status: `skipped`

Reason: No local AOI/drainage fixture for Gilbert exists in this worktree (only data/fitzroy_kimberley_*). The controller-approved scope for this task run is Fitzroy-only; sourcing a Gilbert geometry was explicitly out of scope, not attempted. This gate is NOT satisfied by any number in this report.

Gate fields (explicit null, not fabricated, not omitted):
- `cold_median_at_least_30pct_faster_than_full_aoi`: `None`
- `warm_rerun_at_least_80pct_faster_than_cold_full_aoi`: `None`
- `zero_stac_calls_on_warm_rerun`: `None`

### One large catchment -- plan's third required case

Status: `skipped`

Reason: No local AOI/drainage fixture for a large catchment exists in this worktree. The controller-approved scope for this task run is Fitzroy-only; sourcing a large-catchment geometry was explicitly out of scope, not attempted. This gate is NOT satisfied by any number in this report.

Gate fields (explicit null, not fabricated, not omitted):
- `cold_median_at_least_30pct_faster_than_full_aoi`: `None`
- `warm_rerun_at_least_80pct_faster_than_cold_full_aoi`: `None`
- `zero_stac_calls_on_warm_rerun`: `None`

