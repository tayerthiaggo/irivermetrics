"""Shared fixtures for the parity test suite (Task 1+: analyze() snapshots).

``synthetic_cube`` builds a deterministic 6-month x 12x12 WaterCube with two
stable water patches and one intermittent patch, for use as a small, fast
input to ``hydrofragments.api.analyze()``. Reused by later tasks (2, 6, 7).

Note: this fixture is duplicated in ``tests/conftest.py`` so that
``tests/gating/`` (a sibling of ``tests/parity/``) can also see it -- pytest
conftest fixtures are only visible to the directory they live in and its
descendants, not to siblings, so a conftest under ``tests/parity/`` alone
would not reach ``tests/gating/``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hydrofragments import open_water_cube


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
