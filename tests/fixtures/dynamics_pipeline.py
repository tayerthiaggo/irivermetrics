"""Analytic fixture for dynamics profile end-to-end tests (Task 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments import open_water_cube
from hydrofragments.metrics.extent import ApsecRecord
from hydrofragments.models import WaterCube
from hydrofragments.temporal.hydroyear import detect_hy_anchors

_REGULAR_CYCLE = np.tile(
    [70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55],
    3,
)


@dataclass(frozen=True)
class DynamicsPipelineFixture:
    cube: WaterCube
    hydroyear_extent: pd.Series
    max_water_apsec: tuple[ApsecRecord, ...]
    median_apsec: tuple[ApsecRecord, ...]
    expected_first_hy: int
    expected_end_dry_months: tuple[pd.Timestamp, ...]


def dynamics_pipeline_fixture() -> DynamicsPipelineFixture:
    """36-month, two-HY analytic cube with known dynamics behaviour.

    Grid is 4x4. A single connected refuge block at end-dry grows after the
    first HY's end-dry month so LPI crosses the default 50% threshold. Refuge
    footprints between consecutive HY end-dry months overlap on three of five
    wet pixels (Jaccard 0.6 on common-valid support).
  """
    times = pd.date_range("2001-01-01", periods=36, freq="MS")
    height, width = 4, 4
    water = np.zeros((36, height, width), dtype=bool)
    valid = np.ones((36, height, width), dtype=bool)

    refuge_previous = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    refuge_current = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    reconnected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    dry_pixel = np.zeros((height, width), dtype=bool)
    dry_pixel[0, 0] = True

    hy_extent = pd.Series(_REGULAR_CYCLE, index=times, name="extent_pct")
    hy_result = detect_hy_anchors(hy_extent)
    end_dry_keys = {
        (pd.Timestamp(row["end_dry_month"]).year, pd.Timestamp(row["end_dry_month"]).month)
        for _, row in hy_result.anchors.iterrows()
        if row["end_dry_month"] is not None and not pd.isna(row["end_dry_month"])
    }

    first_end_dry_key = min(end_dry_keys)
    second_end_dry_key = sorted(end_dry_keys)[1] if len(end_dry_keys) > 1 else None

    def _next_month(key: tuple[int, int]) -> tuple[int, int]:
        year, month = key
        if month == 12:
            return year + 1, 1
        return year, month + 1

    reconnect_key = _next_month(first_end_dry_key)

    for index, timestamp in enumerate(times):
        key = (timestamp.year, timestamp.month)
        if key == first_end_dry_key:
            water[index] = refuge_previous
            valid[index, 2:, 2:] = False
        elif second_end_dry_key is not None and key == second_end_dry_key:
            water[index] = refuge_current
            valid[index, 2:, 2:] = False
        elif key == reconnect_key:
            water[index] = reconnected
        else:
            water[index] = dry_pixel

    water_da = xr.DataArray(
        water.astype(np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times})
    cube = open_water_cube(
        water_da,
        valid_obs=valid_da,
        input_kind="generic_binary",
    )

    max_records = tuple(
        ApsecRecord(
            date=timestamp.to_pydatetime(),
            value=float(value),
            n_water_pixels=int(water[index].sum()),
            a_ref_m2=float(height * width) * 100.0,
            cell_area_m2=100.0,
        )
        for index, (timestamp, value) in enumerate(zip(times, _REGULAR_CYCLE))
    )
    median_records = tuple(
        ApsecRecord(
            date=record.date,
            value=float(record.value - 1.0),
            n_water_pixels=record.n_water_pixels,
            a_ref_m2=record.a_ref_m2,
            cell_area_m2=record.cell_area_m2,
        )
        for record in max_records
    )

    end_dry_months = tuple(
        pd.Timestamp(row["end_dry_month"])
        for _, row in hy_result.anchors.sort_values("hy").iterrows()
        if row["end_dry_month"] is not None and not pd.isna(row["end_dry_month"])
    )
    first_hy = int(hy_result.anchors.sort_values("hy").iloc[0]["hy"])

    return DynamicsPipelineFixture(
        cube=cube,
        hydroyear_extent=hy_extent,
        max_water_apsec=max_records,
        median_apsec=median_records,
        expected_first_hy=first_hy,
        expected_end_dry_months=end_dry_months,
    )
