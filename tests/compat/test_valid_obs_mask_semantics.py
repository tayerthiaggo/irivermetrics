"""Task 1 (M2 follow-up): pin the `water & valid_obs` mask decision (option b).

Human decision (see .superpowers/sdd/task-1-brief.md Step 1): option (b) --
unify onto `water & valid_obs` everywhere. Before this change,
``_monthly_dataset()`` in hydrofragments/compat.py hardcoded
``valid_obs = xr.ones_like(water, dtype=bool)``, so `section_compat_rows()`
computed patch metrics (`number_of_pools`/`lpi`/`awre`/`awmsi`) and
persistence metrics (`occurrence`/`refuge_area`) over `water` alone, ignoring
any real `valid_obs` the caller had. This is a deliberate behavior change:
unobserved-but-flagged-water pixels (`water=True, valid_obs=False`) must now
be excluded from every mask fed to patch/persistence kernels, matching
`_pool_width_records()`'s existing `water & valid_obs` mask in
hydrofragments/api.py.

This test builds a `generic_binary` fixture with a deliberately-placed
`water=True, valid_obs=False` pixel inside an otherwise-square 2x2 water
patch, so removing that one pixel changes the patch's pixel count/shape
(and therefore lpi/awre/awmsi) in a way we can predict independently via
`analyze_patch_metrics()` -- the reference implementation this task must not
change (see brief: "already correct -- reference only").
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hydrofragments.compat import section_compat_rows
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


def test_section_compat_rows_masks_water_with_valid_obs():
    """Patch metrics from section_compat_rows must use water & valid_obs.

    Reachable state: water=True, valid_obs=False at (1,1). Under option (b),
    that pixel must be excluded from the mask fed to analyze_patch_metrics,
    shrinking the 2x2 patch to a 3-pixel L-shape (still one 8-connected
    component). If section_compat_rows instead used `water` alone, the row
    values below would reflect the full 2x2 (4-pixel) patch instead.
    """
    da_feature, valid_da, water, valid = _feature_and_valid()
    config = _config()
    pixel_size_m = 30.0
    section_area_km2 = float(water[0].size) * pixel_size_m**2 / 1_000_000.0

    rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=pixel_size_m,
        config=config,
        valid_obs=valid_da,
        selected_ids={"number_of_pools", "lpi", "awre", "awmsi"},
    )
    assert len(rows) == 1
    row = rows[0]

    # Independently computed expectation using the untouched reference
    # implementation (analyze_patch_metrics), fed the *intersected* mask --
    # this is what "correct under option (b)" means, not a regenerated
    # snapshot.
    expected_mask = water[0] & valid[0]
    assert int(expected_mask.sum()) == 3  # 4-pixel patch minus the invalid corner
    expected = analyze_patch_metrics(
        expected_mask,
        pixel_size_m=pixel_size_m,
        a_total_m2=section_area_km2 * 1_000_000.0,
        connectivity=config.patches.connectivity_rule,
        min_patch_pixels=config.patches.min_patch_pixels,
    )

    assert row["n_patches"] == expected.number_of_pools == 1
    assert row["LPI"] == expected.lpi
    assert row["AWRe"] == expected.awre
    assert row["AWMSI"] == expected.awmsi

    # Sanity: pin against the *wrong* (water-only) mask to prove the two
    # differ for this fixture, i.e. the test would have failed before the
    # fix (water=True regardless of valid_obs used to be handed straight to
    # analyze_patch_metrics).
    water_only = analyze_patch_metrics(
        water[0],
        pixel_size_m=pixel_size_m,
        a_total_m2=section_area_km2 * 1_000_000.0,
        connectivity=config.patches.connectivity_rule,
        min_patch_pixels=config.patches.min_patch_pixels,
    )
    assert water_only.lpi != expected.lpi


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
