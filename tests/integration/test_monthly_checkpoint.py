from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


def _counted_cube(calls: dict[str, int]) -> tuple[xr.DataArray, xr.DataArray]:
    from dask import delayed

    @delayed
    def load_water() -> np.ndarray:
        calls["loads"] += 1
        return np.array(
            [
                [[True, False], [False, True]],
                [[False, False], [True, True]],
                [[True, True], [False, False]],
                [[False, True], [False, True]],
            ],
            dtype=bool,
        )

    water_data = da.from_delayed(load_water(), shape=(4, 2, 2), dtype=bool)
    time = pd.to_datetime(
        ["2020-01-03", "2020-01-22", "2020-02-05", "2020-02-26"]
    )
    water = xr.DataArray(
        water_data.rechunk((2, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": time},
    )
    valid = xr.DataArray(
        da.ones((4, 2, 2), chunks=(2, 2, 2), dtype=bool),
        dims=water.dims,
        coords=water.coords,
    )
    return water, valid


def test_zarr_checkpoint_reuse_skips_upstream_recomputation(tmp_path) -> None:
    from hydrofragments.compute.policy import ComputePolicy
    from hydrofragments.pipeline import run_monthly_pipeline

    calls = {"loads": 0}
    water, valid = _counted_cube(calls)
    checkpoint_path = tmp_path / "monthly.zarr"
    policy = ComputePolicy(
        target_chunk_bytes=4096,
        live_array_multiplier=2.0,
        checkpoint="zarr",
    )

    first = run_monthly_pipeline(
        water,
        valid,
        input_cadence="submonthly",
        monthly_composite="max_water",
        composite_owner="hydrofragments",
        policy=policy,
        checkpoint_path=checkpoint_path,
        reuse_existing=True,
    )
    assert calls["loads"] == 1
    assert first.diagnostics.materialization_events[-1].action == "written"
    assert first.diagnostics.materialization_events[-1].materialization_occurred
    first_stages = {
        item["stage"] for item in first.manifest["compute"]["chunks"]
    }
    assert "input.water" in first_stages
    assert "checkpoint.water" in first_stages

    second = run_monthly_pipeline(
        water,
        valid,
        input_cadence="submonthly",
        monthly_composite="max_water",
        composite_owner="hydrofragments",
        policy=policy,
        checkpoint_path=checkpoint_path,
        reuse_existing=True,
    )

    assert calls["loads"] == 1
    assert second.diagnostics.materialization_events[-1].action == "reused"
    assert not second.diagnostics.materialization_events[-1].materialization_occurred
    assert second.manifest["compute"]["materialization"][-1]["action"] == "reused"
    assert all(isinstance(variable.data, da.Array) for variable in second.dataset.data_vars.values())
