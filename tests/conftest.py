"""Shared pytest fixtures for ecofragments test suite."""
import pytest
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path

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


@pytest.fixture(scope="session")
def expected_metrics():
    """Reference metrics CSV produced by the original codebase."""
    return pd.read_csv(
        TEST_DIR / "results_ecofragments" / "metrics" / "ecof_metrics.csv",
        index_col=0,
    )
