"""Shared pytest fixtures for HydroFragments."""
import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from pathlib import Path

from hydrofragments import open_water_cube

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def da_wmask():
    """Water mask DataArray from the bundled NetCDF test file."""
    ds = xr.open_dataset(TEST_DIR / "wmask_ts.nc")
    return ds["water"]


@pytest.fixture(scope="session")
def rcor_extent_path():
    """Absolute path string to the bundled river corridor shapefile."""
    return str(TEST_DIR / "rcor_extent.shp")


@pytest.fixture(scope="session")
def rcor_extent_gdf(rcor_extent_path):
    """GeoDataFrame of the bundled river corridor shapefile."""
    return gpd.read_file(rcor_extent_path)


@pytest.fixture
def synthetic_cube():
    """6 months, 12x12, deterministic water + validity, as a real WaterCube."""
    rng = np.random.default_rng(1729)
    t, y, x = 6, 12, 12
    water = np.zeros((t, y, x), dtype=bool)
    water[:, 2:5, 2:5] = True  # stable patch A
    water[:, 7:10, 7:11] = True  # stable patch B
    water[::2, 5:7, 5:7] = True  # intermittent patch C (even months)
    valid = np.ones((t, y, x), dtype=bool)
    valid[3, :, :] = rng.random((y, x)) > 0.1  # one partially-invalid month
    valid |= water
    times = np.array(
        ["2015-01", "2015-02", "2015-03", "2015-04", "2015-05", "2015-06"],
        dtype="datetime64[M]",
    ).astype("datetime64[ns]")
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0

    water_da = xr.DataArray(
        water,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
    )
    valid_da = xr.DataArray(
        valid,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
    )

    return open_water_cube(
        water_da,
        valid_obs=valid_da,
        input_kind="generic_binary",
    )


@pytest.fixture
def tmp_zarr_path(tmp_path):
    """A small on-disk zarr store readable by ``open_water_cube(path)``."""
    rng = np.random.default_rng(2024)
    t, y, x = 6, 12, 12
    raw = np.zeros((t, y, x), dtype=np.uint8)
    raw[:, 2:5, 2:5] = 1  # stable wet patch
    raw[::2, 5:7, 5:7] = 1  # intermittent wet patch (even months)
    invalid_mask = rng.random((t, y, x)) > 0.97
    raw[invalid_mask] = 255  # sparse unobserved pixels

    times = np.array(
        ["2015-01", "2015-02", "2015-03", "2015-04", "2015-05", "2015-06"],
        dtype="datetime64[M]",
    ).astype("datetime64[ns]")
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0

    dataset = xr.Dataset(
        {
            "water_mask": (
                ("time", "y", "x"),
                raw,
            ),
        },
        coords={"time": times, "y": ys, "x": xs},
    )
    path = tmp_path / "cube.zarr"
    dataset.to_zarr(path, mode="w")
    return path
