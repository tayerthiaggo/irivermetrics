"""Fixed-denominator APSEC and real-channel LPSEC extent metrics.

Spec §6.17: ``APSEC_t = WA_t / A_ref * 100`` with
``WA_t = cell_area * count(W_t = 1)``.

The defining property is the **fixed denominator**. ``A_ref`` is the AOI
reference area supplied by the spatial context and is identical for every
month. Extent that exceeds ``A_ref`` (e.g. a clipped AOI, or braided high-flow
extent) yields APSEC above 100% rather than a silently renormalised value —
the denominator is never redefined as the wetted extent.

LPSEC is optional and accepts only a :class:`SpatialContext` backed by real
drainage. Wet-derived length references remain prohibited.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.spatial.context import SpatialContext


@dataclass(frozen=True)
class AnalysisMaskCoverageResult:
    """Monthly valid-coverage fraction denominated by ``analysis_mask``.

    Global constraint: "Monthly coverage denominator is approved as
    conservative potential-water ``analysis_mask``, not full catchment" --
    unlike APSEC/LPI (which stay pinned to the full ``aoi_mask``, see
    :func:`compute_apsec`), the monthly *coverage* fraction is deliberately
    denominated by the smaller, conservative potential-water footprint so a
    catchment with a large dry margin does not dilute its coverage signal
    with pixels that were never expected to be observed as water anyway.

    ``coverage_fraction``/``n_valid_pixels`` are one value per ``time``;
    ``n_mask_pixels`` is the (time-invariant) analysis_mask pixel count used
    as the shared denominator.
    """

    coverage_fraction: xr.DataArray
    n_valid_pixels: xr.DataArray
    n_mask_pixels: int


@dataclass(frozen=True)
class ApsecRecord:
    """One monthly APSEC value with its supporting counts and fixed A_ref."""

    date: datetime | date
    value: float
    n_water_pixels: int
    a_ref_m2: float
    cell_area_m2: float
    low_coverage_flag: bool = False


@dataclass(frozen=True)
class LpsecResult:
    """Fixed-reference longitudinal extent from a real channel context."""

    value: float
    wetted_length_m: float
    l_ref_m: float
    exceeds_reference: bool
    length_crs_caveat: bool = True


def compute_lpsec(
    wetted_length_m: float, *, context: SpatialContext
) -> LpsecResult:
    """Compute LPSEC without capping braided values above 100 percent."""
    if not np.isfinite(wetted_length_m) or wetted_length_m < 0:
        raise ValueError("wetted_length_m must be non-negative and finite")
    if not context.has_real_channel or context.l_ref_m is None:
        raise ValueError("LPSEC requires a real channel context with fixed L_ref")
    value = float(wetted_length_m / context.l_ref_m * 100.0)
    return LpsecResult(
        value=value,
        wetted_length_m=float(wetted_length_m),
        l_ref_m=float(context.l_ref_m),
        exceeds_reference=value > 100.0,
    )


def compute_apsec(
    monthly: xr.Dataset,
    *,
    a_ref_m2: float,
    cell_area_m2: float,
    config: HydroConfig,
    valid_obs: xr.DataArray | None = None,
    min_valid_fraction: float | None = None,
) -> list[ApsecRecord]:
    """Compute per-month APSEC against a fixed AOI reference area.

    ``monthly`` is the M4 monthly product with boolean ``water`` and
    ``valid_obs`` on ``(time, y, x)``. Water is counted only where the month is
    observed (``water & valid_obs``) so unobserved pixels never inflate extent.

    ``valid_obs`` / ``min_valid_fraction`` are optional additive metadata
    (m7): when both are supplied, the per-month fraction of valid pixels
    (mean of ``valid_obs`` over the spatial dims) is compared against
    ``min_valid_fraction`` and ``ApsecRecord.low_coverage_flag`` is set
    ``True`` for months below that floor. This never changes ``value`` --
    it purely annotates sparse months that were already being computed
    (mirrors the coverage floor occurrence already applies, but expressed
    as a per-month spatial fraction rather than a temporal support count).
    """
    if a_ref_m2 <= 0:
        raise ValueError("a_ref_m2 must be positive")

    water = monthly["water"].astype(bool)
    monthly_valid_obs = monthly["valid_obs"].astype(bool)
    observed_water = water & monthly_valid_obs

    spatial_dims = [dim for dim in observed_water.dims if dim != "time"]
    water_pixels = observed_water.sum(dim=spatial_dims).astype(np.int64)

    times = pd.to_datetime(monthly["time"].values)
    counts = np.asarray(water_pixels.values)

    low_coverage = np.zeros(len(times), dtype=bool)
    if valid_obs is not None and min_valid_fraction is not None:
        coverage_dims = [dim for dim in valid_obs.dims if dim != "time"]
        valid_fraction = valid_obs.astype(bool).mean(dim=coverage_dims)
        low_coverage = np.asarray(
            (valid_fraction < min_valid_fraction).values
        )

    records: list[ApsecRecord] = []
    for timestamp, n_water, flagged in zip(times, counts, low_coverage):
        n_water_int = int(n_water)
        wetted_area = n_water_int * cell_area_m2
        value = wetted_area / a_ref_m2 * 100.0
        records.append(
            ApsecRecord(
                date=timestamp.to_pydatetime(),
                value=value,
                n_water_pixels=n_water_int,
                a_ref_m2=a_ref_m2,
                cell_area_m2=cell_area_m2,
                low_coverage_flag=bool(flagged),
            )
        )
    return records


def compute_analysis_mask_coverage(
    valid_obs: xr.DataArray, *, analysis_mask: xr.DataArray
) -> AnalysisMaskCoverageResult:
    """Per-month valid-pixel coverage fraction denominated by ``analysis_mask``.

    ``valid_obs`` is ``(time, y, x)`` boolean; ``analysis_mask`` is a 2-D
    boolean mask aligned to ``valid_obs``'s spatial dims/sizes (the same
    contract :class:`hydrofragments.models.WaterCube` enforces for its
    ``analysis_mask`` field). The fraction is ``count(valid_obs & mask) /
    count(mask)`` -- pixels outside ``analysis_mask`` never enter either the
    numerator or the denominator, so they cannot dilute or inflate coverage,
    unlike a full-grid mean.

    Raises if ``analysis_mask`` does not spatially align with ``valid_obs``,
    or if ``analysis_mask`` has zero ``True`` pixels (an empty mask has no
    meaningful coverage denominator).
    """
    spatial_dims = tuple(dim for dim in valid_obs.dims if dim != "time")
    if tuple(analysis_mask.dims) != spatial_dims or dict(analysis_mask.sizes) != {
        dim: valid_obs.sizes[dim] for dim in spatial_dims
    }:
        raise ValueError(
            "analysis_mask must align with valid_obs's spatial dims/sizes"
        )

    mask = analysis_mask.astype(bool)
    n_mask_pixels = int(mask.sum().item())
    if n_mask_pixels == 0:
        raise ValueError("analysis_mask must contain at least one True pixel")

    masked_valid = valid_obs.astype(bool) & mask
    n_valid_pixels = masked_valid.sum(dim=spatial_dims).astype(np.int64)
    coverage_fraction = n_valid_pixels.astype(float) / float(n_mask_pixels)

    return AnalysisMaskCoverageResult(
        coverage_fraction=coverage_fraction,
        n_valid_pixels=n_valid_pixels,
        n_mask_pixels=n_mask_pixels,
    )


__all__ = [
    "AnalysisMaskCoverageResult",
    "ApsecRecord",
    "LpsecResult",
    "compute_analysis_mask_coverage",
    "compute_apsec",
    "compute_lpsec",
]
