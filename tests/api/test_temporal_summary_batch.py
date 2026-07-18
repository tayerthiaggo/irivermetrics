"""m8: temporal AOI summaries must materialize in one batched compute.

``_temporal_profile_records`` (hydrofragments/api.py) previously called
``.item()`` once for the AOI-mean recurrence value and once more per
hydroperiod year -- each ``.item()`` on a dask-backed DataArray triggers its
own independent graph execution. For N hydroperiod years that's 1 + N
separate materializations of overlapping (and redundant) parts of the same
underlying graph instead of a single batched compute.

This test builds a small dask-backed WaterCube (so ``.item()``/``.compute()``
calls are observable) spanning multiple hydroyears, counts how many times
``dask.array.Array.compute`` actually runs while ``analyze()`` executes the
``pixel_temporal`` profile (recurrence + hydroperiod), and asserts the count
is small and constant rather than growing with the number of years.
"""

from __future__ import annotations

from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze
from hydrofragments.models import WaterCube


def _dask_cube(n_years: int) -> WaterCube:
    """A dask-backed WaterCube spanning ``n_years`` of monthly data.

    12 months/year, 4x4 pixels, chunked so the underlying arrays are real
    ``dask.array.Array`` objects (not eagerly-evaluated numpy).
    """
    months = n_years * 12
    rng = np.random.default_rng(42)
    water_np = rng.random((months, 4, 4)) > 0.4
    valid_np = np.ones((months, 4, 4), dtype=bool)

    water_data = da.from_array(water_np, chunks=(3, 4, 4))
    valid_data = da.from_array(valid_np, chunks=(3, 4, 4))

    times = pd.date_range("2015-01-01", periods=months, freq="MS")
    water = xr.DataArray(water_data, dims=("time", "y", "x"), coords={"time": times})
    valid = xr.DataArray(valid_data, dims=("time", "y", "x"), coords={"time": times})

    return WaterCube(
        water=water,
        valid_obs=valid,
        source="synthetic_dask",
        cadence="monthly",
    )


def _pixel_temporal_config(tmp_path) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
            "metric_profiles": ["pixel_temporal"],
        }
    )


def _count_dask_computes(cube: WaterCube, config: HydroConfig) -> int:
    calls = {"n": 0}
    real_compute = da.Array.compute

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real_compute(self, *args, **kwargs)

    with mock.patch.object(da.Array, "compute", counting):
        analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    return calls["n"]


@pytest.mark.parametrize("n_years", [2, 5])
def test_temporal_summaries_materialize_in_bounded_calls(tmp_path, n_years):
    """Materialization count must not scale with the number of hydroperiod years.

    Before the m8 fix, each hydroperiod year triggered its own ``.item()``
    materialization on top of the one for recurrence, so the call count grew
    as ``1 + n_years``. After batching all summaries into a single
    ``xr.Dataset`` and calling ``.compute()`` once, the count must stay
    constant regardless of ``n_years``.
    """
    cube = _dask_cube(n_years)
    config = _pixel_temporal_config(tmp_path)
    calls = _count_dask_computes(cube, config)
    assert calls <= 2, f"expected a single batched materialization, got {calls} calls"


def test_temporal_summaries_materialization_does_not_scale_with_years(tmp_path):
    """Direct before/after comparator: 5-year run must not cost more computes than 2-year."""
    calls_2y = _count_dask_computes(_dask_cube(2), _pixel_temporal_config(tmp_path))
    calls_5y = _count_dask_computes(_dask_cube(5), _pixel_temporal_config(tmp_path))
    assert calls_2y == calls_5y, (
        f"materialization count scaled with year count ({calls_2y} vs {calls_5y}); "
        "temporal summaries are not batched into one compute"
    )
