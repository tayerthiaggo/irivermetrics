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
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import theilslopes

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import ApsecRecord
from hydrofragments.schema import EdgeFlag


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
    return float("nan")


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
    months = [m for m, _ in series]
    assert months == sorted(months), "reconnection series must be month-sorted"
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


@dataclass(frozen=True)
class EndDryState:
    """Water and validity at one hydrological year's end-dry anchor month."""

    hy: int
    date: datetime
    water: "npt.NDArray[np.bool_]"
    valid_obs: "npt.NDArray[np.bool_]"
    hy_confidence: str
    anchor_missing: bool = False


@dataclass(frozen=True)
class DynamicsSupport:
    """Aligned monthly LPI/LPSEC support for reconnection timing."""

    lpi_by_month: tuple[tuple[datetime, float], ...]
    lpsec_by_month: tuple[tuple[datetime, float], ...] | None = None


def _month_key(value: "datetime | date | pd.Timestamp") -> tuple[int, int]:
    ts = pd.Timestamp(value)
    return int(ts.year), int(ts.month)


def _as_month_datetime(value: "datetime | date | pd.Timestamp") -> datetime:
    ts = pd.Timestamp(value)
    return datetime(int(ts.year), int(ts.month), 1)


def _series_lookup(
    series: Sequence[tuple["datetime | date", float]],
) -> dict[tuple[int, int], float]:
    return {_month_key(month): float(value) for month, value in series}


def _sorted_series(
    lookup: Mapping[tuple[int, int], float],
) -> list[tuple[datetime, float]]:
    return [
        (_as_month_datetime(datetime(year, month, 1)), float(value))
        for (year, month), value in sorted(lookup.items())
    ]


def _search_window_months(
    *,
    end_dry_month: "datetime | date",
    next_end_dry_month: "datetime | date | None",
    cube_month_keys: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    end_key = _month_key(end_dry_month)
    next_key = _month_key(next_end_dry_month) if next_end_dry_month is not None else None
    window: list[tuple[int, int]] = []
    for month_key in sorted(cube_month_keys):
        if month_key <= end_key:
            continue
        if next_key is not None and month_key >= next_key:
            break
        window.append(month_key)
    return window


def _lpsec_complete_for_window(
    *,
    lpsec_lookup: Mapping[tuple[int, int], float],
    window_months: Sequence[tuple[int, int]],
) -> bool:
    if not window_months:
        return False
    for month_key in window_months:
        value = lpsec_lookup.get(month_key)
        if value is None or not np.isfinite(value):
            return False
    return True


def _slice_series_for_window(
    lookup: Mapping[tuple[int, int], float],
    *,
    window_months: Sequence[tuple[int, int]],
) -> list[tuple[datetime, float]]:
    return [
        (
            _as_month_datetime(datetime(month_key[0], month_key[1], 1)),
            float(lookup[month_key]),
        )
        for month_key in window_months
        if month_key in lookup and np.isfinite(lookup[month_key])
    ]


@dataclass(frozen=True)
class RefugeStabilityEvaluation:
    """Scalar refuge stability with machine-readable edge semantics."""

    jaccard: float | None
    edge_flag: EdgeFlag | None
    common_valid_fraction: float | None
    n_common_valid_pixels: int | None
    n_union_pixels: int | None


def evaluate_refuge_spatial_stability(
    *,
    current: EndDryState,
    previous: EndDryState | None,
    analysis_mask: "npt.NDArray[np.bool_] | None",
    min_valid_fraction: float,
) -> RefugeStabilityEvaluation:
    """Compute HY-pair refuge stability on common-valid support (spec §6.16)."""
    if previous is None:
        return RefugeStabilityEvaluation(
            jaccard=None,
            edge_flag=EdgeFlag.NO_PREVIOUS_HY,
            common_valid_fraction=None,
            n_common_valid_pixels=None,
            n_union_pixels=None,
        )
    if current.anchor_missing or previous.anchor_missing:
        return RefugeStabilityEvaluation(
            jaccard=None,
            edge_flag=EdgeFlag.MISSING_HY_ANCHOR,
            common_valid_fraction=None,
            n_common_valid_pixels=None,
            n_union_pixels=None,
        )
    if current.hy != previous.hy + 1:
        return RefugeStabilityEvaluation(
            jaccard=None,
            edge_flag=EdgeFlag.NONCONSECUTIVE_HY,
            common_valid_fraction=None,
            n_common_valid_pixels=None,
            n_union_pixels=None,
        )

    mask = (
        analysis_mask
        if analysis_mask is not None
        else np.ones_like(current.water, dtype=bool)
    )
    common_valid = mask & previous.valid_obs & current.valid_obs
    mask_count = int(mask.sum())
    common_count = int(common_valid.sum())
    common_fraction = float(common_count / mask_count) if mask_count else float("nan")
    if mask_count == 0 or common_fraction < min_valid_fraction:
        return RefugeStabilityEvaluation(
            jaccard=None,
            edge_flag=EdgeFlag.LOW_COMMON_VALID_SUPPORT,
            common_valid_fraction=common_fraction,
            n_common_valid_pixels=common_count,
            n_union_pixels=None,
        )

    previous_refuge = previous.water & common_valid
    current_refuge = current.water & common_valid
    union_count = int(np.logical_or(previous_refuge, current_refuge).sum())
    if union_count == 0:
        return RefugeStabilityEvaluation(
            jaccard=float("nan"),
            edge_flag=EdgeFlag.EMPTY_REFUGE_UNION,
            common_valid_fraction=common_fraction,
            n_common_valid_pixels=common_count,
            n_union_pixels=0,
        )

    result = compute_refuge_spatial_stability(
        current_end_dry_footprint=current_refuge,
        previous_end_dry_footprint=previous_refuge,
    )
    return RefugeStabilityEvaluation(
        jaccard=result.jaccard,
        edge_flag=None,
        common_valid_fraction=common_fraction,
        n_common_valid_pixels=common_count,
        n_union_pixels=union_count,
    )


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
    "DynamicsSupport",
    "EndDryState",
    "ExtentContractionResult",
    "ReconnectionTimingResult",
    "RefugeSpatialStabilityResult",
    "RefugeStabilityEvaluation",
    "compute_extent_contraction",
    "compute_reconnection_timing",
    "compute_refuge_spatial_stability",
    "evaluate_refuge_spatial_stability",
]
