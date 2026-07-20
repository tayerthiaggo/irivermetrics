"""Baseline data-quality assessment and gapfill prescription (Section 2).

HydroFragments never gapfills data -- that behavior lives only in the
companion tool WaterMask-TSFill (not ported here, by project decision; see
``docs/input_format.md``). This module only *assesses* how much valid
observation coverage an input :class:`~hydrofragments.models.WaterCube`
actually has, and -- when coverage falls below the configured
``validity.min_valid_obs`` / ``validity.min_valid_fraction_month`` floors and
the caller has not declared ``config.gapfill = True`` -- *recommends*
pre-processing with WaterMask-TSFill before running HydroFragments.

The recommendation is purely advisory:

- It never mutates ``cube.water`` or ``cube.valid_obs``.
- It never gapfills, interpolates, or otherwise changes any data value.
- When ``config.gapfill`` is ``True`` the recommendation is suppressed
  outright -- coverage is not re-checked. HydroFragments trusts the user's
  declaration that the input was already gapfilled upstream.

Seasonal MNAR (missing-not-at-random) diagnostics reuse the season-stratified
estimator already locked in for occurrence
(:func:`hydrofragments.metrics.persistence._season_stratified_occurrence`,
Decision U2/Q1) rather than reimplementing calendar-month stratification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import _season_stratified_occurrence
from hydrofragments.models import WaterCube

WATERMASK_TSFILL_HINT = "WaterMask-TSFill"


def _has_monthly_time_coord(valid_obs: xr.DataArray) -> bool:
    """Whether ``valid_obs`` carries a real datetime ``time`` coordinate.

    ``validate_inputs`` (and callers of :func:`assess_baseline_quality`
    directly) may be exercised against minimal fixtures whose ``time``
    dimension has no attached coordinate values at all -- there is nothing to
    group by calendar month in that case, so callers fall back to an
    overall-coverage-only assessment instead of raising.
    """
    if "time" not in valid_obs.coords:
        return False
    return pd.api.types.is_datetime64_any_dtype(valid_obs["time"].dtype)


@dataclass(frozen=True)
class BaselineQualityReport:
    """Baseline valid-observation coverage assessment for one :class:`WaterCube`.

    Purely descriptive/advisory -- never mutates cube data. All fractions are
    in ``[0, 1]``; ``seasonal_occurrence_pct`` is a percentage in ``[0, 100]``
    matching :func:`hydrofragments.metrics.persistence.compute_occurrence`.
    """

    overall_valid_fraction: float
    valid_fraction_by_month: tuple[tuple[int, float], ...]
    seasonal_occurrence_pct: float
    below_min_valid_obs_fraction: float
    below_min_valid_fraction_month_fraction: float
    recommend_gapfill: bool
    reason: str | None


def _gapfill_reason() -> str:
    return (
        "insufficient baseline coverage; consider pre-processing with "
        f"{WATERMASK_TSFILL_HINT} before running HydroFragments, or set "
        "gapfill=true if already gapfilled"
    )


def assess_baseline_quality(
    cube: WaterCube, *, config: HydroConfig
) -> BaselineQualityReport:
    """Assess baseline valid-observation coverage and recommend action.

    Never mutates ``cube.water``/``cube.valid_obs``. When coverage is below
    the configured floors (``config.validity.min_valid_obs`` /
    ``config.validity.min_valid_fraction_month``) and ``config.gapfill`` is
    ``False``, the report recommends gapfilling upstream via
    WaterMask-TSFill. When ``config.gapfill`` is ``True`` the recommendation
    is suppressed outright -- coverage is still measured and reported, but
    the caller's declaration that the data was already gapfilled is trusted
    without re-verification.

    All intermediate reductions are assembled as lazy ``xr.DataArray``
    objects and materialised in a single batched ``Dataset.compute()`` call
    so a dask-backed cube triggers one graph execution here, not one
    ``.item()``/``.compute()`` per statistic (matching the batching
    discipline already established for temporal AOI summaries).
    """
    water = cube.water.astype(bool)
    valid_obs = cube.valid_obs.astype(bool)
    valid_f = valid_obs.astype(np.float64)

    scalars: dict[str, xr.DataArray] = {
        "overall_valid_fraction": valid_f.mean(),
    }

    has_monthly_time = _has_monthly_time_coord(valid_obs)
    if has_monthly_time:
        grouped_sum = valid_f.groupby("time.month").sum(dim="time")
        grouped_count = valid_f.groupby("time.month").count(dim="time")
        month_fraction = (grouped_sum / grouped_count).rename("month_fraction")
        pixel_dims = [dim for dim in month_fraction.dims if dim != "month"]

        valid_count = valid_obs.astype(np.int64).sum(dim="time")
        below_min_valid_obs = valid_count < config.validity.min_valid_obs
        below_month_floor = month_fraction < config.validity.min_valid_fraction_month

        scalars["below_min_valid_obs_fraction"] = (
            below_min_valid_obs.astype(np.float64).mean()
        )
        scalars["below_min_valid_fraction_month_fraction"] = (
            below_month_floor.astype(np.float64).mean()
        )
        scalars["seasonal_occurrence_pct"] = _season_stratified_occurrence(
            water, valid_obs
        ).mean(skipna=True)
        scalars["month_fraction_by_pixel"] = (
            month_fraction.mean(dim=pixel_dims) if pixel_dims else month_fraction
        )
    else:
        # No usable calendar-month coordinate to stratify by (e.g. a minimal
        # fixture cube with a bare `time` dimension and no coordinate
        # values). Fall back to overall coverage only; nothing "below the
        # month floor" can be claimed without calendar months to check.
        scalars["below_min_valid_obs_fraction"] = xr.DataArray(0.0)
        scalars["below_min_valid_fraction_month_fraction"] = xr.DataArray(0.0)
        scalars["seasonal_occurrence_pct"] = xr.DataArray(np.nan)

    # One batched compute for the whole assessment: a dask-backed cube must
    # not pay one graph execution per statistic (m8 batching discipline).
    computed = xr.Dataset(scalars).compute()

    overall_valid_fraction = float(computed["overall_valid_fraction"].values)
    below_obs_fraction = float(computed["below_min_valid_obs_fraction"].values)
    below_month_fraction = float(
        computed["below_min_valid_fraction_month_fraction"].values
    )
    seasonal_occurrence_pct = float(computed["seasonal_occurrence_pct"].values)

    if has_monthly_time:
        computed_month_fraction = computed["month_fraction_by_pixel"]
        valid_fraction_by_month = tuple(
            (int(month), float(value))
            for month, value in zip(
                computed_month_fraction["month"].values,
                computed_month_fraction.values,
            )
        )
    else:
        valid_fraction_by_month = ()

    coverage_is_low = below_obs_fraction > 0.0 or below_month_fraction > 0.0
    recommend_gapfill = coverage_is_low and not config.gapfill
    reason = _gapfill_reason() if recommend_gapfill else None

    return BaselineQualityReport(
        overall_valid_fraction=overall_valid_fraction,
        valid_fraction_by_month=valid_fraction_by_month,
        seasonal_occurrence_pct=seasonal_occurrence_pct,
        below_min_valid_obs_fraction=below_obs_fraction,
        below_min_valid_fraction_month_fraction=below_month_fraction,
        recommend_gapfill=recommend_gapfill,
        reason=reason,
    )


__all__ = [
    "WATERMASK_TSFILL_HINT",
    "BaselineQualityReport",
    "assess_baseline_quality",
]
