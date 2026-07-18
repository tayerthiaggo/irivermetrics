"""Shared pytest fixtures for ecofragments test suite."""
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


@pytest.fixture(scope="session")
def legacy_baseline_csv_path():
    """Path to the legacy iRiverMetrics regression CSV.

    Quarantined (U7, approved 2026-07-11): this fixture may only back smoke
    comparisons of approved, purely-geometric invariant columns (e.g.
    ``section_area_km2``). It must never be treated as a v1.2 correctness oracle -
    see ``tests/contracts/test_legacy_baseline_quarantine.py`` and ``docs/testing.md``.
    """
    return TEST_DIR / "results_iRiverMetrics" / "metrics" / "irm_metrics.csv"


@pytest.fixture
def synthetic_cube():
    """6 months, 12x12, deterministic water + validity, as a real WaterCube.

    Duplicated here (also defined in tests/parity/conftest.py) so that
    sibling suites -- notably tests/gating/ -- can use it too: pytest
    conftest fixtures are visible to a directory and its descendants only,
    not to siblings, so a fixture defined solely under tests/parity/ would
    not reach tests/gating/. Task 1 (B1 safety net) and later tasks (2, 6,
    7) rely on this fixture being available from both locations.
    """
    rng = np.random.default_rng(1729)
    t, y, x = 6, 12, 12
    water = np.zeros((t, y, x), dtype=bool)
    water[:, 2:5, 2:5] = True  # stable patch A
    water[:, 7:10, 7:11] = True  # stable patch B
    water[::2, 5:7, 5:7] = True  # intermittent patch C (even months)
    valid = np.ones((t, y, x), dtype=bool)
    valid[3, :, :] = rng.random((y, x)) > 0.1  # one partially-invalid month
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
