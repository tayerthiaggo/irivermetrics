from __future__ import annotations

import ast
import inspect

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


def _lazy_observations() -> tuple[xr.DataArray, xr.DataArray, dict[str, int]]:
    from dask import delayed

    calls = {"loads": 0}

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
    valid_data = da.ones((4, 2, 2), chunks=(2, 2, 2), dtype=bool)
    time = pd.to_datetime(
        ["2020-01-03", "2020-01-22", "2020-02-05", "2020-02-26"]
    )
    water = xr.DataArray(
        water_data.rechunk((2, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": time},
    )
    valid = xr.DataArray(
        valid_data,
        dims=water.dims,
        coords=water.coords,
    )
    return water, valid, calls


def test_temporal_graph_stays_lazy_until_pipeline_checkpoint() -> None:
    from hydrofragments.compute.policy import ComputePolicy
    from hydrofragments.pipeline import assemble_monthly_pipeline

    water, valid, calls = _lazy_observations()
    result = assemble_monthly_pipeline(
        water,
        valid,
        input_cadence="submonthly",
        monthly_composite="max_water",
        composite_owner="hydrofragments",
        policy=ComputePolicy(
            target_chunk_bytes=4096,
            live_array_multiplier=2.0,
            checkpoint="zarr",
        ),
    )

    assert calls["loads"] == 0
    assert all(isinstance(variable.data, da.Array) for variable in result.dataset.data_vars.values())
    assert result.diagnostics.graph_task_count > 0
    assert result.diagnostics.materialization_events == ()
    assert result.manifest["compute"]["chunks"]


def test_temporal_scientific_modules_have_no_eager_dask_escape_hatches() -> None:
    from hydrofragments.temporal import cadence, composites

    forbidden: list[str] = []
    for module in (cadence, composites):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"compute", "values"}:
                forbidden.append(f"{module.__name__}.{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr == "asarray"
            ):
                forbidden.append(f"{module.__name__}.np.asarray")

    assert forbidden == []
