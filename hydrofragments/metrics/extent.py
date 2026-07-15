"""Extent metrics: APSEC (fixed-denominator surface-water extent).

Spec §6.17: ``APSEC_t = WA_t / A_ref * 100`` with
``WA_t = cell_area * count(W_t = 1)``.

The defining property is the **fixed denominator**. ``A_ref`` is the AOI
reference area supplied by the spatial context and is identical for every
month. Extent that exceeds ``A_ref`` (e.g. a clipped AOI, or braided high-flow
extent) yields APSEC above 100% rather than a silently renormalised value —
the denominator is never redefined as the wetted extent.

LPSEC and any wet-derived length reference are explicitly out of scope for the
minimal core (Decision Gate 0; ``adversarial_synthesis_2.md`` §7).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.config import HydroConfig


@dataclass(frozen=True)
class ApsecRecord:
    """One monthly APSEC value with its supporting counts and fixed A_ref."""

    date: datetime | date
    value: float
    n_water_pixels: int
    a_ref_m2: float
    cell_area_m2: float


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


__all__ = ["ApsecRecord", "compute_apsec"]
