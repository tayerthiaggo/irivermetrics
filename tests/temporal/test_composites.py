from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _submonthly_cube() -> tuple[xr.DataArray, xr.DataArray]:
    time = pd.to_datetime(
        [
            "2020-01-02",
            "2020-01-15",
            "2020-01-29",
            "2020-02-03",
            "2020-02-16",
            "2020-02-27",
        ]
    )
    water_values = np.array(
        [
            [[0, 1, 0]],
            [[1, 1, 0]],
            [[1, 0, 0]],
            [[1, 0, 1]],
            [[0, 0, 1]],
            [[0, 1, 1]],
        ],
        dtype=bool,
    )
    valid_values = np.ones_like(water_values, dtype=bool)
    valid_values[2, 0, 1] = False
    valid_values[5, 0, 2] = False
    chunks = (3, 1, 3)
    water = xr.DataArray(
        da.from_array(water_values, chunks=chunks),
        dims=("time", "y", "x"),
        coords={"time": time},
    )
    valid = xr.DataArray(
        da.from_array(valid_values, chunks=chunks),
        dims=water.dims,
        coords=water.coords,
    )
    return water, valid


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("max_water", [[[1, 1, 0]], [[1, 1, 1]]]),
        ("median", [[[1, 1, 0]], [[0, 0, 1]]]),
        ("mode", [[[1, 1, 0]], [[0, 0, 1]]]),
        ("end_of_month_nearest", [[[1, 1, 0]], [[0, 1, 1]]]),
    ],
)
def test_submonthly_composites_are_exact_and_lazy(
    method: str, expected: list[list[list[int]]]
) -> None:
    from hydrofragments.temporal.composites import build_monthly_products

    water, valid = _submonthly_cube()
    monthly = build_monthly_products(
        water,
        valid,
        input_cadence="submonthly",
        monthly_composite=method,
        composite_owner="hydrofragments",
    )

    assert all(isinstance(variable.data, da.Array) for variable in monthly.data_vars.values())
    np.testing.assert_array_equal(monthly["water"].compute(), expected)
    np.testing.assert_array_equal(
        monthly["valid_count"].compute(),
        [[[3, 2, 3]], [[3, 3, 2]]],
    )
    np.testing.assert_allclose(
        monthly["valid_fraction"].compute(),
        [[[1.0, 2 / 3, 1.0]], [[1.0, 1.0, 2 / 3]]],
    )


def test_already_monthly_input_is_not_recomposited() -> None:
    from hydrofragments.temporal.composites import build_monthly_products

    values = da.from_array(
        np.array([[[True, False]], [[False, True]]]),
        chunks=(1, 1, 2),
    )
    time = pd.date_range("2020-01-01", periods=2, freq="MS")
    water = xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": time},
    )
    valid = xr.ones_like(water, dtype=bool)

    monthly = build_monthly_products(
        water,
        valid,
        input_cadence="monthly",
        monthly_composite="supplied",
        composite_owner="upstream",
    )

    assert monthly["water"].data is water.data
    assert monthly["water"].chunks == water.chunks
    assert monthly.attrs["monthly_composite"] == "supplied"
    assert monthly.attrs["composite_owner"] == "upstream"


def test_already_monthly_input_requires_composite_provenance() -> None:
    from hydrofragments.temporal.composites import CompositeError, build_monthly_products

    time = pd.date_range("2020-01-01", periods=2, freq="MS")
    water = xr.DataArray(
        da.zeros((2, 1, 1), chunks=(1, 1, 1), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": time},
    )

    with pytest.raises(CompositeError, match="provenance"):
        build_monthly_products(
            water,
            xr.ones_like(water, dtype=bool),
            input_cadence="monthly",
            monthly_composite=None,
            composite_owner=None,
        )
