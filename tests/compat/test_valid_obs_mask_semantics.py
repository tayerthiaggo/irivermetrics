"""Water/valid_obs mask semantics: `water=True, valid_obs=False` is invalid input.

Superseded design note: an earlier iteration (M2 follow-up, Task 1) had
`section_compat_rows()` silently intersect `water & valid_obs`, masking out
any unobserved-but-flagged-water pixel before it reached patch/persistence
kernels. The current contract is stricter and matches `open_water_cube()`'s
public-entry invariant (`hydrofragments/api.py::_ensure_water_implies_valid_obs`):
`water=True` requires `valid_obs=True` for every pixel/month, enforced by
raising `_WATER_VALIDITY_ERROR` rather than silently dropping the pixel. A
caller with genuinely unobserved-but-flagged-water pixels must clean the
input upstream (e.g. via the WaterMask-TSFill gapfill/quality pipeline)
before calling `analyze()` or `section_compat_rows()` directly.

This test builds a `generic_binary` fixture with a deliberately-placed
`water=True, valid_obs=False` pixel inside an otherwise-square 2x2 water
patch and asserts `section_compat_rows` raises rather than silently masking
it out.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hydrofragments.compat import _WATER_VALIDITY_ERROR, section_compat_rows
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.patches import analyze_patch_metrics


def _feature_and_valid():
    """One month, 6x6 grid: a 2x2 water patch with one invalid-but-water pixel."""
    t, y, x = 1, 6, 6
    water = np.zeros((t, y, x), dtype=bool)
    water[0, 1:3, 1:3] = True  # 2x2 patch: (1,1) (1,2) (2,1) (2,2)
    valid = np.ones((t, y, x), dtype=bool)
    valid[0, 1, 1] = False  # water=True, valid_obs=False at one corner pixel

    times = np.array(["2015-01"], dtype="datetime64[M]").astype("datetime64[ns]")
    ys = np.arange(y, dtype=float) * -30.0
    xs = np.arange(x, dtype=float) * 30.0

    da_feature = xr.DataArray(
        water.astype("int8"),
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
    )
    valid_da = xr.DataArray(
        valid,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
    )
    return da_feature, valid_da, water, valid


def _config():
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
        }
    )


def test_section_compat_rows_rejects_water_without_valid_obs():
    """water=True, valid_obs=False anywhere in the input must raise, not be masked out.

    Reachable state: water=True, valid_obs=False at (1,1) inside an otherwise
    2x2 water patch. `section_compat_rows` must refuse this input outright
    (`_WATER_VALIDITY_ERROR`) rather than silently intersecting the mask and
    computing patch metrics over the remaining 3-pixel L-shape.
    """
    da_feature, valid_da, water, valid = _feature_and_valid()
    config = _config()
    pixel_size_m = 30.0
    section_area_km2 = float(water[0].size) * pixel_size_m**2 / 1_000_000.0

    with pytest.raises(ValueError, match=_WATER_VALIDITY_ERROR):
        section_compat_rows(
            da_feature,
            section="AOI",
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size_m,
            config=config,
            valid_obs=valid_da,
            selected_ids={"number_of_pools", "lpi", "awre", "awmsi"},
        )


def test_section_compat_rows_defaults_to_all_true_valid_obs_when_omitted():
    """Legacy callers (no valid_obs) keep today's water-only behavior.

    hydrofragments.compat.calculate_metrics_compat() has no separate
    valid_obs concept -- it only ever had a single water-mask array. Passing
    no ``valid_obs`` to section_compat_rows() must preserve that -- mask ==
    water, exactly like before this task's change.
    """
    da_feature, _valid_da, water, _valid = _feature_and_valid()
    config = _config()
    pixel_size_m = 30.0
    section_area_km2 = float(water[0].size) * pixel_size_m**2 / 1_000_000.0

    rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=pixel_size_m,
        config=config,
        selected_ids={"number_of_pools", "lpi", "awre", "awmsi"},
    )
    row = rows[0]

    expected = analyze_patch_metrics(
        water[0],
        pixel_size_m=pixel_size_m,
        a_total_m2=section_area_km2 * 1_000_000.0,
        connectivity=config.patches.connectivity_rule,
        min_patch_pixels=config.patches.min_patch_pixels,
    )
    assert row["n_patches"] == expected.number_of_pools
    assert row["LPI"] == expected.lpi
