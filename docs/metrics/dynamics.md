# Surface-water dynamics

The `dynamics` profile emits hydrological-year metrics from monthly water masks, extent composites, and connectivity support series. Metric row schema version is **`1.1.0`**.

## Profile metrics

| Metric ID | Unit | Reportable when |
|-----------|------|-----------------|
| `extent_contraction` | percent_per_month | Dual APSEC composites (`max_water` and `median`) and HY anchors available |
| `reconnection_timing` | month | HY `end_dry` anchor detected; threshold crossing or explicit edge reason |
| `refuge_spatial_stability` | dimensionless | Consecutive HY pair with sufficient common-valid support |

Select the profile in configuration:

```python
"metric_profiles": ["dynamics"],
```

Or add individual IDs via `metric_overrides`.

## `extent_contraction`

Monthly surface-water extent slope over the drying limb of a hydrological year — **not** a hydrograph recession constant or discharge measurement.

Requires both `max_water` and `median` APSEC composites via `AnalysisInputs`. HydroFragments must skip this metric and record the reason when either composite is unavailable; it never fabricates `median` from a single monthly mask.

Each composite yields one OLS (or configured Theil–Sen) slope, finite monthly point count, low-degrees-of-freedom flag, HY confidence, and end-dry disagreement flag. Fewer than three usable points suppress the slope to `NaN`. Month coordinates use elapsed calendar months, so missing months are not silently compressed.

## `reconnection_timing`

Calendar-month lag from hydrological-year `end_dry` (exclusive) to the first month a connectivity metric crosses the configured threshold.

### Configuration

```python
"dynamics": {
    "reconnection_lpi_threshold_pct": 50.0,
    "reconnection_lpsec_threshold_pct": 50.0,
}
```

Both thresholds are **percentages** on `[0, 100]`, validated at parse time, and included in the scientific configuration hash.

### Provider precedence

1. **RC** — only when a future runtime supplies a real RC series (not implemented in 0.1.0).
2. **LPSEC** — when complete live channel-profile prerequisites exist for every cube month in the search interval.
3. **LPI** — computed as internal support even when LPI is not selected as an output row.

Do not fall back from a valid preferred series merely because it never crosses the threshold; that case is `no_threshold_crossing`, not a switch to an easier proxy. LPSEC and LPI both set `proxy_reconnection_flag=True` and `WarningFlag.PROXY_RECONNECTION`.

### Search window

For each detected hydrological year:

- Search starts after `end_dry` (exclusive).
- Search ends before the next HY `end_dry` (exclusive), or at record end for the final year.
- Missing months are not imputed; lag is calendar-month difference, not observation count.
- Threshold equality counts as crossing.

Emitted fields include `date` (end-dry), `hy`, `hy_anchor`, `hy_confidence`, `connected_wet_metric`, `connected_wet_threshold`, `reconnection_metric_used`, `proxy_reconnection_flag`, and `warning_flags`.

## `refuge_spatial_stability`

Scalar Jaccard stability between consecutive end-dry refuge footprints:

```text
common_valid = analysis_mask & valid_previous & valid_current
previous_refuge = water_previous & common_valid
current_refuge  = water_current  & common_valid
union = previous_refuge | current_refuge
stability = count(previous_refuge & current_refuge) / count(union)
```

### Validity rules

- **First hydrological year:** non-reportable (`no_previous_HY`).
- **Non-consecutive years:** pair not formed (`nonconsecutive_HY`).
- **Low common-valid fraction:** below `validity.min_valid_fraction_month` → `low_common_valid_support`.
- **Empty union (dry/dry):** undefined (`empty_refuge_union`), **not** zero.
- Partial invalidity is excluded through common support, not treated as dry.

Row `date` is the current end-dry date; `valid_fraction_month`, `n_valid_pixels`, and `n_water_pixels` map to common-valid fraction, common-valid count, and union-pixel count respectively.

### Machine-readable edge flags

Schema `1.1.0` adds `EdgeFlag` values without new columns: `missing_HY_anchor`, `no_previous_HY`, `nonconsecutive_HY`, `low_common_valid_support`, `empty_refuge_union`, `no_threshold_crossing`.

## Per-pixel refuge stability rasters

When `refuge_stability_rasters` is selected, separate raster products export per-pixel frequency:

```text
stable_count[p]   = eligible pairs where p is wet in both years
eligible_union[p] = eligible pairs where p is wet in either year
frequency_pct[p]  = 100 * stable_count / eligible_union
```

Pixels with `eligible_union == 0` are nodata. This percentage is **not** the scalar Jaccard metric. See [Spatial exports](../spatial_exports.md#refuge_stability_rasters).

## Input dependencies

| Input | Required for |
|-------|--------------|
| `AnalysisInputs.hydroyear_extent` | HY anchor detection |
| `AnalysisInputs.max_water_apsec` + `median_apsec` | `extent_contraction` |
| Channel profiles / `SpatialContext` | LPSEC-preferred reconnection |
| Georeferenced cube | Dynamics scalars only need masks; spatial refuge products need grid metadata |

Plain `analyze()` without these optional inputs skips dynamics metrics with explicit coverage reasons rather than fabricating values.

## Related documentation

- [Spatial exports](../spatial_exports.md)
- [Final metrics covered](../final_metrics_covered.md)
