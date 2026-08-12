"""Task 1: analyze_patch_bundle() must be wired into api.py's analyze() path.

Origin: Task 4 of the parent plan added ``analyze_patch_bundle()`` to
hydrofragments/metrics/patches.py (one label/crop/measure pass shared by
core patch metrics and pool_width), but nothing called it -- ``analyze()``
still ran two independent labeling passes per month whenever both a
patch-family metric (number_of_pools/lpi/awre/awmsi) and pool_width were
selected: one inside ``section_compat_rows()`` via ``analyze_patch_metrics``,
one inside ``_pool_width_records()`` via ``analyze_pool_width_distribution``.

This test pins the fix directly at the label-components layer (the
expensive op both paths ultimately call): selecting both a patch-family
profile (``contracts_core``) and ``secondary`` (which resolves to
``mesh``/``pool_width``) together must invoke ``label_components`` exactly
once per month, not twice.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.patches import labels as labels_module


def _cube_two_months():
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    mask = np.zeros((2, 6, 6), dtype=bool)
    mask[:, 1:4, 1:4] = True
    water = xr.DataArray(mask, dims=("time", "y", "x"), coords={"time": times})
    return open_water_cube(water, input_kind="generic_binary")


def _config(tmp_path):
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core", "secondary"],
            "input": {"kind": "generic_binary"},
            "patches": {
                "connectivity_rule": 8,
                "min_patch_pixels": 1,
                "width_resolution_floor_pixels": 1.0,
            },
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )


def test_patch_family_and_pool_width_share_one_bundle_call_per_month(tmp_path):
    """number_of_pools/lpi/awre/awmsi + pool_width selected together => 1 label pass/month."""
    cube = _cube_two_months()
    config = _config(tmp_path)
    with mock.patch.object(
        labels_module, "label_components", wraps=labels_module.label_components
    ) as spy:
        result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    # 2 months in the fixture -> exactly one label_components call per month,
    # not two (one for patch metrics, one for pool_width).
    assert spy.call_count == 2

    metrics = set(result.metrics_table["metric"])
    assert {"number_of_pools", "lpi", "awre", "awmsi", "pool_width"} <= metrics


def test_pool_width_alone_still_works_without_patch_family(tmp_path):
    """pool_width selected without any patch-family metric: standalone path still works."""
    cube = _cube_two_months()
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["secondary"],
            "input": {"kind": "generic_binary"},
            "patches": {
                "connectivity_rule": 8,
                "min_patch_pixels": 1,
                "width_resolution_floor_pixels": 1.0,
            },
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )
    with mock.patch.object(
        labels_module, "label_components", wraps=labels_module.label_components
    ) as spy:
        result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    assert spy.call_count == 2  # one per month, standalone pool_width path
    assert set(result.metrics_table["metric"]) == {"pool_width"}
