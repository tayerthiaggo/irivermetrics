"""Shared pytest fixtures for ecofragments test suite."""
import pytest
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
def legacy_baseline_csv_path():
    """Path to the legacy iRiverMetrics regression CSV.

    Quarantined (U7, approved 2026-07-11): this fixture may only back smoke
    comparisons of approved, purely-geometric invariant columns (e.g.
    ``section_area_km2``). It must never be treated as a v1.2 correctness oracle -
    see ``tests/contracts/test_legacy_baseline_quarantine.py`` and ``docs/testing.md``.
    """
    return TEST_DIR / "results_iRiverMetrics" / "metrics" / "irm_metrics.csv"
