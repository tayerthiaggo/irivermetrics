"""Persistence metrics: occurrence and Refuge Area.

Decision Gate 0 (U2/Q1, approved 2026-07-14) is the binding contract here:

- The denominator for any temporal aggregate is the **valid-observation count**
  (``valid_obs`` true), never the total number of timesteps. Unobserved months
  do not enter numerator or denominator.
- Occurrence (and every temporal aggregate) uses a **season-stratified**
  estimator: the per-calendar-month P-native ratio is computed separately, then
  the calendar-month ratios that have support are combined with equal weight.
  This corrects the confirmed seasonal missing-not-at-random (MNAR) pattern
  that a naive pooled ratio would bias
  (``docs/audit/evidence/validity_reliability_report.md`` §4).
- Occurrence is suppressed to NaN where a pixel's ``valid_count`` is below the
  ``validity.min_valid_obs`` floor (spec §8 guard 4).

Refuge Area then counts pixels whose occurrence meets the refuge threshold and
whose support clears the same valid-observation floor (spec §6.17).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from hydrofragments.config import HydroConfig


@dataclass(frozen=True)
class RefugeAreaResult:
    """Refuge Area magnitude and its supporting pixel count.

    ``value`` is an area in km^2 (registry unit for ``refuge_area``);
    ``n_refuge_pixels`` is the count of pixels meeting both the refuge
    threshold and the ``min_valid_obs`` support floor.
    """

    value: float
    n_refuge_pixels: int
    refuge_threshold: float
    cell_area_m2: float


@dataclass(frozen=True)
class RecurrenceResult:
    """Inter-annual recurrence (spec §6.12) and its valid-year support count.

    ``recurrence`` is the equal-weight mean of supported calendar-month wet
    fractions across years, expressed as a percentage. A calendar year with
    zero valid (observed) months is not a valid year and contributes to
    ``valid_year_count``; unsupported calendar months contribute no term.
    """

    recurrence: xr.DataArray
    valid_year_count: xr.DataArray


@dataclass(frozen=True)
class HydroperiodResult:
    """Within-year hydroperiod (spec §6.12), one value per calendar year.

    ``hydroperiod`` is ``valid wet months / valid observed months`` for each
    pixel and year; unobserved months are excluded from the denominator, not
    treated as dry. ``valid_observed_months`` is that denominator.
    """

    hydroperiod: xr.DataArray
    valid_observed_months: xr.DataArray


@dataclass(frozen=True)
class OccurrenceResult:
    """Per-pixel occurrence raster and its supporting valid-observation count.

    ``occurrence`` is a percentage in [0, 100] or NaN where support is below
    the ``min_valid_obs`` floor (or absent entirely). ``valid_count`` is the
    integer number of valid observations behind each pixel.
    """

    occurrence: xr.DataArray
    valid_count: xr.DataArray
    min_valid_obs: int


def _season_stratified_occurrence(
    water: xr.DataArray, valid_obs: xr.DataArray
) -> xr.DataArray:
    """Mean of per-calendar-month P-native ratios, equal-weighted.

    For each calendar month (1-12 present in the record) the ratio is
    ``sum(water & valid) / sum(valid)`` over the years for that calendar month.
    Calendar months with zero valid observations contribute no term. The final
    per-pixel value is the unweighted mean across the contributing calendar
    months, expressed as a percentage.
    """
    water_valid = (water & valid_obs).astype(np.float64)
    valid = valid_obs.astype(np.float64)

    grouped_water = water_valid.groupby("time.month").sum(dim="time")
    grouped_valid = valid.groupby("time.month").sum(dim="time")

    # Per-calendar-month ratio; NaN where that calendar month has no support so
    # it drops out of the mean rather than injecting a 0/0 term.
    ratio = grouped_water / grouped_valid.where(grouped_valid > 0)
    return ratio.mean(dim="month", skipna=True) * 100.0


def compute_occurrence(
    monthly: xr.Dataset, *, config: HydroConfig
) -> OccurrenceResult:
    """Compute the season-stratified P-native occurrence raster.

    ``monthly`` is the M4 monthly product with boolean ``water`` and
    ``valid_obs`` on ``(time, y, x)``.
    """
    water = monthly["water"].astype(bool)
    valid_obs = monthly["valid_obs"].astype(bool)

    valid_count = valid_obs.astype(np.int64).sum(dim="time")
    occurrence = _season_stratified_occurrence(water, valid_obs)

    min_valid_obs = config.validity.min_valid_obs
    supported = valid_count >= min_valid_obs
    occurrence = occurrence.where(supported)

    return OccurrenceResult(
        occurrence=occurrence,
        valid_count=valid_count,
        min_valid_obs=min_valid_obs,
    )


def compute_refuge_area(
    occurrence: OccurrenceResult,
    *,
    cell_area_m2: float,
    config: HydroConfig,
) -> RefugeAreaResult:
    """Compute Refuge Area from an occurrence surface (spec §6.17).

    ``RA_theta = cell_area * count(OCC_p >= theta and valid_count_p >=
    min_valid_obs)``. The occurrence surface is already NaN below the support
    floor (see :func:`compute_occurrence`), and the explicit ``valid_count``
    check keeps the semantics correct even when an occurrence surface is
    supplied directly.
    """
    threshold_pct = config.persistence.refuge_threshold * 100.0
    min_valid_obs = config.validity.min_valid_obs

    is_refuge = (occurrence.occurrence >= threshold_pct) & (
        occurrence.valid_count >= min_valid_obs
    )
    n_refuge_pixels = int(is_refuge.sum().item())
    area_km2 = n_refuge_pixels * cell_area_m2 / 1_000_000.0

    return RefugeAreaResult(
        value=area_km2,
        n_refuge_pixels=n_refuge_pixels,
        refuge_threshold=config.persistence.refuge_threshold,
        cell_area_m2=cell_area_m2,
    )


def compute_recurrence(monthly: xr.Dataset, *, config: HydroConfig) -> RecurrenceResult:
    """Compute pixel recurrence: inter-annual reliability of wetness (spec §6.12).

    The locked U2/Q1 estimator equal-weights supported calendar months to
    prevent seasonal missingness from reweighting the result toward months
    with better observation. ``valid_year_count`` remains available support
    diagnostics and excludes years with zero valid observations.
    """
    water = monthly["water"].astype(bool)
    valid_obs = monthly["valid_obs"].astype(bool)

    # U2/Q1: equal-weight supported calendar months so seasonal missingness
    # cannot reweight the estimate toward months with better observation.
    valid_months_per_year = valid_obs.astype(np.int64).groupby("time.year").sum(dim="time")
    is_valid_year = valid_months_per_year > 0

    valid_year_count = is_valid_year.sum(dim="year").astype(np.int64)

    grouped_wet = (water & valid_obs).astype(np.float64).groupby("time.month").sum(dim="time")
    grouped_valid = valid_obs.astype(np.float64).groupby("time.month").sum(dim="time")
    monthly_recurrence = grouped_wet / grouped_valid.where(grouped_valid > 0)
    recurrence = monthly_recurrence.mean(dim="month", skipna=True) * 100.0

    return RecurrenceResult(recurrence=recurrence, valid_year_count=valid_year_count)


def compute_hydroperiod(monthly: xr.Dataset, *, config: HydroConfig) -> HydroperiodResult:
    """Compute within-year hydroperiod per pixel and calendar year (spec §6.12).

    ``HP_{p,y} = valid wet months / valid observed months``. Unobserved months
    are excluded from the denominator so they never count as dry.
    """
    water = monthly["water"].astype(bool)
    valid_obs = monthly["valid_obs"].astype(bool)

    valid_observed_months = valid_obs.astype(np.int64).groupby("time.year").sum(dim="time")
    wet_months = (water & valid_obs).astype(np.int64).groupby("time.year").sum(dim="time")

    hydroperiod = wet_months / valid_observed_months.where(valid_observed_months > 0)

    return HydroperiodResult(
        hydroperiod=hydroperiod, valid_observed_months=valid_observed_months
    )


__all__ = [
    "HydroperiodResult",
    "OccurrenceResult",
    "RecurrenceResult",
    "RefugeAreaResult",
    "compute_hydroperiod",
    "compute_occurrence",
    "compute_recurrence",
    "compute_refuge_area",
]
