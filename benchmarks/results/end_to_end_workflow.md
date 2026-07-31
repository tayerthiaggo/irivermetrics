# End-to-end workflow benchmark (W3.7 benchmark gate)

Controller-approved reduced scope: only Fitzroy has a local AOI/drainage fixture in this worktree. Gilbert and the large-catchment required cases are explicitly not run -- see their 'skipped' entries below, not omitted from this schema.

- Schema: `1.0.0`
- Created: `2026-07-31T02:29:27.049497+00:00`

## Fitzroy (compact) -- real, live-network run

| Candidate | factor | workers | cold total s | warm total s | warm speedup | regression vs serial | peak RSS (cold) | RSS vs serial | coverage (cold) | timing/RSS gates | all gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| factor4_workers1 | 4 | 1 | 51.250 | 14.507 | 71.7% | 0.0% | 720,363,520 | 100.0% | 90.3% | yes | no |
| factor4_workers2 | 4 | 2 | 39.798 | 14.658 | 63.2% | -22.3% | 776,699,904 | 107.8% | 90.3% | yes | no |
| factor4_workers4 | 4 | 4 | 39.858 | 12.769 | 68.0% | -22.2% | 731,512,832 | 101.5% | 90.3% | yes | no |
| factor3_workers1 | 3 | 1 | 36.087 | 14.693 | 59.3% | -29.6% | 791,212,032 | 109.8% | 90.3% | yes | no |
| factor3_workers2 | 3 | 2 | 39.593 | 15.071 | 61.9% | -22.7% | 754,102,272 | 104.7% | 90.3% | yes | no |
| factor3_workers4 | 3 | 4 | 40.222 | 15.129 | 62.4% | -21.5% | 780,308,480 | 108.3% | 90.3% | yes | no |

Promotion gates applied (Fitzroy-only subset -- see module docstring):
- `exact_metrics_table_and_per_metric_value_equality`: measured per-candidate as cold_warm_metrics_equal (cold vs warm rerun of the SAME candidate); cross-candidate full-AOI-vs-pruned equality was not separately run this session -- see report notes
- `n_water_equality_every_month`: measured per-candidate as cold_warm_n_water_equal
- `native_wet_mask_coverage_exactly_100pct`: measured per-candidate
- `cold_gilbert_at_least_30pct_faster_than_full_aoi`: None
- `warm_rerun_at_least_80pct_faster_than_cold_full_aoi`: Fitzroy's own warm-vs-cold speedup is reported per candidate as warm_speedup_fraction_vs_own_cold; the plan's 80% figure is stated against Gilbert specifically and is NOT claimed satisfied here even where Fitzroy's own number exceeds it
- `compact_fitzroy_regression_no_worse_than_10pct`: measured per-candidate as regression_within_10pct_gate, relative to the factor=4/workers=1 serial baseline
- `peak_rss_no_more_than_125pct_of_serial`: measured per-candidate as peak_rss_within_125pct_gate

## Recommendation

- **verdict**: no_passing_candidate
- **reason**: No candidate passed every measurable Fitzroy gate. Default settings are left unchanged.
- **timing_rss_only_note**: candidate=factor3_workers1 was fastest among candidates passing ONLY the timing/RSS gates (regression <=10%, peak RSS <=125%) -- informational, since it did NOT also pass the exact-equality/coverage gates required for all_measurable_gates_pass.

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

