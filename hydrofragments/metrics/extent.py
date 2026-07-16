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
class ApsecRecord:
    """One monthly APSEC value with its supporting counts and fixed A_ref."""

    date: datetime | date
    value: float
    n_water_pixels: int
    a_ref_m2: float
    cell_area_m2: float


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
) -> list[ApsecRecord]:
    """Compute per-month APSEC against a fixed AOI reference area.

    ``monthly`` is the M4 monthly product with boolean ``water`` and
    ``valid_obs`` on ``(time, y, x)``. Water is counted only where the month is
    observed (``water & valid_obs``) so unobserved pixels never inflate extent.
    """
    if a_ref_m2 <= 0:
        raise ValueError("a_ref_m2 must be positive")

    water = monthly["water"].astype(bool)
    valid_obs = monthly["valid_obs"].astype(bool)
    observed_water = water & valid_obs

    spatial_dims = [dim for dim in observed_water.dims if dim != "time"]
    water_pixels = observed_water.sum(dim=spatial_dims).astype(np.int64)

    times = pd.to_datetime(monthly["time"].values)
    counts = np.asarray(water_pixels.values)

    records: list[ApsecRecord] = []
    for timestamp, n_water in zip(times, counts):
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
            )
        )
    return records


__all__ = ["ApsecRecord", "LpsecResult", "compute_apsec", "compute_lpsec"]
