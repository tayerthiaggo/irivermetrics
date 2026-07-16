# Surface-water dynamics

Milestone 9 exposes `extent_contraction` as a monthly surface-water extent
slope, not a hydrograph recession constant or discharge measurement.

The metric requires both `max_water` and `median` APSEC composites. The result
contains one OLS (or configured Theil–Sen) slope per composite, the number of
finite monthly points, a low-degrees-of-freedom flag, HY confidence, and an
end-dry disagreement flag. Fewer than three usable points suppresses each
slope to `NaN`. Month coordinates use elapsed calendar months, so missing
months are not silently compressed.

If both composites are unavailable, HydroFragments must skip this metric and
record the reason; it must never fabricate `median` from a single monthly
mask. The dual-composite sensitivity result remains a validation diagnostic,
not a demonstrated catchment-wide scientific effect, until V3 analysis is run
on real raw or dual-composite observations.

Reconnection timing currently uses an explicitly marked LPI proxy. RC/DCI
runtime metrics remain deferred to the connectivity tranche.
