"""Dynamics metrics: surface-water extent contraction (spec §6.5).

Decision Gate 0 contract for this module:

- **Terminology guard (locked, spec §6.5 `[AUDIT FIX]`):** this is a
  monthly-extent contraction rate, never described as a hydrograph
  recession-constant analysis. No flow/recession language in any
  user-facing string this module emits.
- **Dual-composite requirement (U3/Q3, approved):** the contraction slope is
  computed on both the `max_water`- and `median`-composited monthly APSEC
  series. `max_water` is known to bias extent upward at end-dry, flattening
  the true contraction signal (spec §1.1.2); reporting only one composite
  hides that bias. A caller with only a single composite (e.g. already-monthly
  upstream input, no raw sub-monthly data) must be refused explicitly rather
  than have a second series fabricated from the one it has.
- **Method (Q11, approved 2026-07-16):** `linear` (OLS) slope of APSEC against
  month index between the HY's peak-wet and end-dry anchors;
  `minimum_points = 3`. Fewer usable points suppresses the slope (NaN) with a
  `low_df` diagnostic rather than reporting an unreliable regression.
- **HY anchor dependency:** anchors (`peak_month`, `end_dry_month`,
  `confidence`) come from `hydrofragments.temporal.hydroyear` (which itself
  delegates to the external `hydroseason` package). A missing anchor for a
  given HY skips the metric for that HY rather than guessing one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import theilslopes

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import ApsecRecord


class DualCompositeUnavailableError(ValueError):
    """Raised when extent contraction is requested without both composites."""


@dataclass(frozen=True)
class ExtentContractionResult:
    """One HY's surface-water extent contraction rate and its diagnostics.

    ``slope_pct_per_month`` is negative for drying, computed from the
    `max_water`-composited series (the conservative/default composite for
    general extent, per spec §1.1.2). ``composite_sensitive`` reports whether
    the `max_water` vs `median` end-dry APSEC disagree beyond the configured
    tolerance -- always computed, since the dual-composite input is mandatory.
    """

    hy: int
    slope_pct_per_month: float
    n_points: int
    low_df: bool
    median_slope_pct_per_month: float
    median_n_points: int
    median_low_df: bool
    composite_sensitive: bool
    end_dry_disagreement_pp: float
    hy_confidence: str
    metric_name: str = "extent_contraction"
    description: str = (
        "Monthly-extent contraction rate: slope of surface-water extent "
        "(APSEC) over the drying limb of a hydrological year. Not a "
        "hydrograph analysis or discharge-based measurement."
    )


def _as_datetime(value: "datetime | date") -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime(value.year, value.month, value.day)


def _month_number(value: datetime) -> int:
    return value.year * 12 + value.month


def _values_between(
    records: Sequence[ApsecRecord], start: "datetime | date", end: "datetime | date"
) -> list[tuple[float, float]]:
    start_dt = _as_datetime(start)
    end_dt = _as_datetime(end)
    origin = _month_number(start_dt)
    points = []
    for record in records:
        record_dt = _as_datetime(record.date)
        if start_dt <= record_dt <= end_dt and np.isfinite(record.value):
            points.append((_month_number(record_dt) - origin, float(record.value)))
    return sorted(points)


def _end_dry_value(records: Sequence[ApsecRecord], end: "datetime | date") -> float:
    end_dt = _as_datetime(end)
    for record in records:
        record_dt = _as_datetime(record.date)
        if (
            (record_dt.year, record_dt.month) == (end_dt.year, end_dt.month)
            and np.isfinite(record.value)
        ):
            return float(record.value)
    raise ValueError("end_dry_month has no matching APSEC record")


def _fit_slope(
    points: Sequence[tuple[float, float]], *, minimum_points: int, method: str
) -> tuple[float, bool]:
    low_df = len(points) < minimum_points
    if low_df:
        return float("nan"), True
    x = np.asarray([item[0] for item in points], dtype=float)
    y = np.asarray([item[1] for item in points], dtype=float)
    if method == "linear":
        slope, _ = np.polyfit(x, y, 1)
    elif method == "theil_sen":
        slope = theilslopes(y, x).slope
    else:
        raise ValueError(f"unsupported contraction method: {method}")
    return float(slope), False


def compute_extent_contraction(
    *,
    max_water: Sequence[ApsecRecord],
    median: Sequence[ApsecRecord] | None,
    anchor: Mapping[str, object],
    config: HydroConfig,
) -> ExtentContractionResult | None:
    """Compute one HY's extent-contraction slope from dual-composite APSEC.

    ``anchor`` supplies ``peak_month``, ``end_dry_month``, ``confidence``,
    (from :func:`hydrofragments.temporal.hydroyear.detect_hy_anchors`). A
    missing peak or end-dry anchor returns ``None`` (skip, not fabricate).
    """
    if median is None:
        raise DualCompositeUnavailableError(
            "extent_contraction requires both max_water and median monthly "
            "APSEC composites; only one was supplied. Skip this metric "
            "rather than fabricating a second composite from monthly masks."
        )

    peak_month = anchor.get("peak_month")
    end_dry_month = anchor.get("end_dry_month")
    if peak_month is None or end_dry_month is None:
        return None

    max_points = _values_between(max_water, peak_month, end_dry_month)
    median_points = _values_between(median, peak_month, end_dry_month)
    minimum_points = config.dynamics.minimum_points
    method = config.dynamics.contraction_method
    slope, low_df = _fit_slope(
        max_points, minimum_points=minimum_points, method=method
    )
    median_slope, median_low_df = _fit_slope(
        median_points, minimum_points=minimum_points, method=method
    )

    max_water_end_dry = _end_dry_value(max_water, end_dry_month)
    median_end_dry = _end_dry_value(median, end_dry_month)
    disagreement = abs(max_water_end_dry - median_end_dry)
    tolerance = config.dynamics.composite_sensitivity_tolerance_pp
    composite_sensitive = disagreement > tolerance

    return ExtentContractionResult(
        hy=int(anchor["hy"]),
        slope_pct_per_month=slope,
        n_points=len(max_points),
        low_df=low_df,
        median_slope_pct_per_month=median_slope,
        median_n_points=len(median_points),
        median_low_df=median_low_df,
        composite_sensitive=composite_sensitive,
        end_dry_disagreement_pp=disagreement,
        hy_confidence=str(anchor.get("confidence", "unassigned")),
    )


@dataclass(frozen=True)
class ReconnectionTimingResult:
    """Reconnection lag after end-dry (spec §6.15), with proxy provenance.

    Preference order (spec §6.15): **RC** (fixed connectivity graph
    available, Milestone 11) > **LPSEC** (real channel, no graph) > **LPI**
    (coarse proxy, last resort). ``reconnection_metric_used`` records which
    metric actually decided ``t_reconnect_months``; ``proxy_reconnection_flag``
    is ``False`` only for RC -- LPSEC and LPI are both proxies relative to
    the preferred network metric and must say so explicitly, never silently
    presenting a proxy as if it were RC.
    """

    reconnection_metric_used: str
    proxy_reconnection_flag: bool
    t_reconnect_months: int | None


def _first_crossing(
    series: Sequence[tuple["datetime | date", float]],
    *,
    end_dry_month: "datetime | date",
    threshold: float,
) -> int | None:
    for month, value in series:
        if month <= end_dry_month:
            continue
        if value >= threshold:
            return (month.year - end_dry_month.year) * 12 + (
                month.month - end_dry_month.month
            )
    return None


def compute_reconnection_timing(
    *,
    lpi_series: Sequence[tuple["datetime | date", float]],
    end_dry_month: "datetime | date",
    lpi_threshold: float,
    rc_series: Sequence[tuple["datetime | date", float]] | None = None,
    rc_threshold: float | None = None,
    lpsec_series: Sequence[tuple["datetime | date", float]] | None = None,
    lpsec_threshold: float | None = None,
) -> ReconnectionTimingResult:
    """Compute reconnection lag, preferring RC over LPSEC over LPI (spec §6.15).

    Each ``*_series`` is a monthly series after (and including) end-dry,
    ordered by date. The first metric that was actually supplied (RC, then
    LPSEC, then LPI) decides ``t_reconnect_months`` -- an unsupplied
    preferred metric falls through to the next one, but a *supplied*
    preferred metric that never crosses its threshold reports ``None``
    directly rather than silently falling back to a lower-preference proxy.
    """
    if rc_series is not None:
        if rc_threshold is None:
            raise ValueError("rc_threshold is required when rc_series is supplied")
        t_reconnect = _first_crossing(
            rc_series, end_dry_month=end_dry_month, threshold=rc_threshold
        )
        return ReconnectionTimingResult(
            reconnection_metric_used="RC",
            proxy_reconnection_flag=False,
            t_reconnect_months=t_reconnect,
        )

    if lpsec_series is not None:
        if lpsec_threshold is None:
            raise ValueError(
                "lpsec_threshold is required when lpsec_series is supplied"
            )
        t_reconnect = _first_crossing(
            lpsec_series, end_dry_month=end_dry_month, threshold=lpsec_threshold
        )
        return ReconnectionTimingResult(
            reconnection_metric_used="LPSEC",
            proxy_reconnection_flag=True,
            t_reconnect_months=t_reconnect,
        )

    t_reconnect = _first_crossing(
        lpi_series, end_dry_month=end_dry_month, threshold=lpi_threshold
    )
    return ReconnectionTimingResult(
        reconnection_metric_used="LPI",
        proxy_reconnection_flag=True,
        t_reconnect_months=t_reconnect,
    )


@dataclass(frozen=True)
class RefugeSpatialStabilityResult:
    """End-dry-footprint Jaccard overlap between consecutive HYs (spec §6.16).

    ``jaccard`` is ``None`` when there is no previous HY to compare against
    (first HY in the record). It is NaN (not 0.0) when both footprints are
    empty, since 0/0 overlap is undefined, not "no overlap."
    """

    jaccard: float | None


def compute_refuge_spatial_stability(
    *,
    current_end_dry_footprint: "npt.NDArray[np.bool_]",
    previous_end_dry_footprint: "npt.NDArray[np.bool_] | None",
) -> RefugeSpatialStabilityResult:
    """Compute ``J_y = |R_y ∩ R_{y-1}| / |R_y ∪ R_{y-1}|`` (spec §6.16 option 1).

    Each HY's footprint (``R_y``) is its own end-dry water mask, so stability
    varies by HY rather than comparing every year against one static
    long-term occurrence footprint.
    """
    if previous_end_dry_footprint is None:
        return RefugeSpatialStabilityResult(jaccard=None)

    intersection = np.logical_and(
        current_end_dry_footprint, previous_end_dry_footprint
    ).sum()
    union = np.logical_or(
        current_end_dry_footprint, previous_end_dry_footprint
    ).sum()

    if union == 0:
        return RefugeSpatialStabilityResult(jaccard=float("nan"))
    return RefugeSpatialStabilityResult(jaccard=float(intersection) / float(union))


__all__ = [
    "DualCompositeUnavailableError",
    "ExtentContractionResult",
    "ReconnectionTimingResult",
    "RefugeSpatialStabilityResult",
    "compute_extent_contraction",
    "compute_reconnection_timing",
    "compute_refuge_spatial_stability",
]
