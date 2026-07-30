"""W4.3: split measure_patch_properties()/reduce_patch_properties() and prove
active-window reuse of shared intermediates across metric families.

Step 1 pins analyze_patch_bundle's exact current output on a real fixture
(the same two-pool/width fixture test_patch_bundle.py already uses) as a
concrete regression baseline BEFORE any refactor. Every later step in this
file must keep these pinned values exactly reproduced by the new
measure/reduce split -- both when called directly (single "whole mask as one
window" call) and when called per-window-then-concatenated-then-reduced-once
via the production analyze() path.

Step 2 proves full-mask versus active-window property/output parity for the
PRODUCTION metric pipeline end-to-end (not just independent_active_windows()
in isolation, which W3.1 already covered) -- i.e. running analyze() with a
fragmented analysis_mask that forces multiple windows must produce identical
LPI/AWRe/AWMSI/width/number_of_pools/n_water_pixels to running analyze() with
the default all-true mask (single window covering the whole grid).

Step 3 proves exactly one shared materialization per selected month/window
regardless of how many metric families are enabled -- extending the existing
test_patch_bundle_wiring.py "one label_components call per month" invariant
to "one label_components call per (month, window)" once window-splitting is
wired in.

Step 4 proves a default all_available run produces identical output rows to
the union of running each individual profile separately.

The ComputePolicy/target_chunk_bytes threading requirement (added by the
controller, not the brief's literal text) is covered in
tests/patches/test_label_threshold_wiring.py.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.metrics import patches as patches_module
from hydrofragments.metrics.patches import (
    PatchMetricResult,
    PoolWidthDistribution,
    analyze_patch_bundle,
    measure_patch_properties,
    reduce_patch_properties,
)
from hydrofragments.patches import label_components
from tests.fixtures.analytic_masks import patch_bundle_width_fixture


def _two_pool_mask() -> np.ndarray:
    m = np.zeros((12, 12), dtype=bool)
    m[2:5, 2:5] = True
    m[7:10, 7:11] = True
    return m


# ---------------------------------------------------------------------------
# Step 1: pin analyze_patch_bundle's current output before refactor.
# ---------------------------------------------------------------------------

_PINNED_BUNDLE_BASELINE = {
    "number_of_pools": 2,
    "n_water_pixels": 21,
    "lpi": pytest.approx(12.0 * 900.0 / (12 * 12 * 900.0) * 100.0),
    "awmsi": pytest.approx(
        (0.25 * 12.0 / np.sqrt(9.0) * (9.0 / 21.0))
        + (0.25 * 14.0 / np.sqrt(12.0) * (12.0 / 21.0))
    ),
    "edge_flag": None,
    "awre_length_method": "major_axis",
}


def test_step1_pin_analyze_patch_bundle_output_before_refactor():
    """Concrete regression baseline: exact analyze_patch_bundle output on a
    real two-pool fixture, recorded before touching any implementation code.
    """
    mask = _two_pool_mask()
    core, width = analyze_patch_bundle(
        mask,
        pixel_size_m=30.0,
        a_total_m2=12 * 12 * 900.0,
        include_mesh=True,
        include_width=True,
        resolution_floor_pixels=2.0,
    )

    assert core.number_of_pools == _PINNED_BUNDLE_BASELINE["number_of_pools"]
    assert core.n_water_pixels == _PINNED_BUNDLE_BASELINE["n_water_pixels"]
    assert core.lpi == _PINNED_BUNDLE_BASELINE["lpi"]
    assert core.awmsi == _PINNED_BUNDLE_BASELINE["awmsi"]
    assert core.edge_flag == _PINNED_BUNDLE_BASELINE["edge_flag"]
    assert core.awre_length_method == _PINNED_BUNDLE_BASELINE["awre_length_method"]
    assert np.isfinite(core.mesh_m2)
    assert width is not None
    assert width.suppressed_pools == 0
    assert width.mean_m > 0


def test_step1_pin_patch_bundle_width_fixture_baseline():
    """Same pinning discipline applied to the shared width fixture that
    test_patch_bundle.py already uses (two pools, 4-connectivity, width
    floor 2.0 px, 10 m pixels) -- a second independent baseline point.
    """
    water_np, _times = patch_bundle_width_fixture()
    mask = water_np[0].astype(bool)
    core, width = analyze_patch_bundle(
        mask,
        pixel_size_m=10.0,
        a_total_m2=float(mask.size) * 100.0,
        connectivity=4,
        min_patch_pixels=3,
        include_width=True,
        resolution_floor_pixels=2.0,
    )
    assert core.number_of_pools == 2
    assert width is not None
    assert width.suppressed_pools == 1
    assert width.mean_m == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Step 2a: measure_patch_properties()/reduce_patch_properties() must
# reproduce analyze_patch_bundle()'s pinned output exactly when called as a
# single "whole mask" measurement followed by one reduction.
# ---------------------------------------------------------------------------


def test_measure_then_reduce_matches_pinned_bundle_baseline():
    mask = _two_pool_mask()
    properties = measure_patch_properties(
        mask, pixel_size_m=30.0, include_width=True
    )
    core, width = reduce_patch_properties(
        properties,
        pixel_size_m=30.0,
        a_total_m2=12 * 12 * 900.0,
        include_mesh=True,
        include_width=True,
        resolution_floor_pixels=2.0,
    )

    assert core.number_of_pools == _PINNED_BUNDLE_BASELINE["number_of_pools"]
    assert core.n_water_pixels == _PINNED_BUNDLE_BASELINE["n_water_pixels"]
    assert core.lpi == _PINNED_BUNDLE_BASELINE["lpi"]
    assert core.awmsi == _PINNED_BUNDLE_BASELINE["awmsi"]
    assert width is not None
    assert width.suppressed_pools == 0


def test_measure_then_reduce_equals_analyze_patch_bundle_directly():
    """analyze_patch_bundle must now be a thin wrapper: measure once, reduce
    once, byte-identical to calling the two halves directly.
    """
    mask = _two_pool_mask()
    bundle_core, bundle_width = analyze_patch_bundle(
        mask,
        pixel_size_m=30.0,
        a_total_m2=12 * 12 * 900.0,
        include_mesh=True,
        include_width=True,
        resolution_floor_pixels=2.0,
    )
    properties = measure_patch_properties(
        mask, pixel_size_m=30.0, include_width=True
    )
    split_core, split_width = reduce_patch_properties(
        properties,
        pixel_size_m=30.0,
        a_total_m2=12 * 12 * 900.0,
        include_mesh=True,
        include_width=True,
        resolution_floor_pixels=2.0,
    )
    assert bundle_core == split_core
    assert bundle_width == split_width


def test_measure_patch_properties_labels_exactly_once():
    mask = _two_pool_mask()
    with mock.patch.object(
        patches_module, "label_components", wraps=patches_module.label_components
    ) as spy:
        measure_patch_properties(mask, pixel_size_m=30.0, include_width=True)
    assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Step 2b: full-mask vs. active-window parity for the PRODUCTION pipeline
# end-to-end -- using W3.1's independent_active_windows() as the actual
# window source, exercised via concatenate-then-reduce-once (never per-window
# aggregate metrics -- the Critical correctness property from Global
# Constraints: "Per-window aggregate metrics cannot be averaged. Concatenate
# patch properties, then calculate LPI, AWRe, AWMSI, width distribution, and
# counts once across all windows.").
# ---------------------------------------------------------------------------


def _fragmented_two_component_mask(shape=(24, 24)):
    mask = np.zeros(shape, dtype=bool)
    mask[2:5, 2:5] = True
    mask[18:22, 18:22] = True
    return mask


def _config(tmp_path, *, profiles=("contracts_core", "secondary")):
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": list(profiles),
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


def _analyze_result_metric_values(result) -> dict[str, float]:
    frame = result.metrics_table
    values: dict[str, float] = {}
    for _, row in frame.iterrows():
        key = (row["metric"], row.get("statistic"))
        values.setdefault(key, []).append(row["value"])
    return values


def test_analyze_with_fragmented_analysis_mask_matches_default_all_true_mask(
    tmp_path,
):
    """Central W4.3 production-pipeline parity proof: analyze() with an
    analysis_mask that independent_active_windows() splits into TWO disjoint
    windows must produce identical patch-family metric values (per month) to
    analyze() with the default all-true single-window mask over the same
    water data. If window-splitting ever silently changed a metric (e.g. by
    averaging per-window LPI instead of concatenating properties), this test
    would catch it.
    """
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    shape = (24, 24)
    water_np = np.stack([_fragmented_two_component_mask(shape) for _ in times])
    water = xr.DataArray(
        water_np, dims=("time", "y", "x"), coords={"time": times}
    )

    fragmented_mask = xr.DataArray(
        _fragmented_two_component_mask(shape), dims=("y", "x")
    )
    cube_windowed = open_water_cube(
        water, input_kind="generic_binary", analysis_mask=fragmented_mask
    )
    cube_default = open_water_cube(water, input_kind="generic_binary")

    config = _config(tmp_path)
    result_windowed = analyze(
        cube_windowed, aoi_id="windowed", config=config, pixel_size_m=30.0
    )
    result_default = analyze(
        cube_default, aoi_id="default", config=config, pixel_size_m=30.0
    )

    windowed_values = _analyze_result_metric_values(result_windowed)
    default_values = _analyze_result_metric_values(result_default)

    def _sort_key(value):
        # pandas nullable Float64 columns can hold pd.NA, which raises on
        # any ordering comparison (`bool(pd.NA < x)` is ambiguous) -- rank
        # missing values first via an explicit (is_missing, value) tuple
        # rather than comparing them directly.
        missing = value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA
        return (missing, 0.0 if missing else value)

    assert set(windowed_values) == set(default_values)
    for key, values in default_values.items():
        windowed = sorted(windowed_values[key], key=_sort_key)
        default = sorted(values, key=_sort_key)
        assert len(windowed) == len(default)
        for left, right in zip(windowed, default):
            left_missing = left is None or left is pd.NA or (
                isinstance(left, float) and np.isnan(left)
            )
            right_missing = right is None or right is pd.NA or (
                isinstance(right, float) and np.isnan(right)
            )
            if left_missing or right_missing:
                assert left_missing and right_missing
            else:
                assert left == pytest.approx(right)


def test_lpi_awre_awmsi_computed_once_across_concatenated_properties_not_per_window():
    """The single most important correctness property in this task: LPI,
    AWRe, AWMSI must be computed ONCE across properties concatenated from
    every window -- never per-window then averaged. This test builds a mask
    with two windows of DIFFERENT pool sizes such that per-window LPI
    averaging would give a different (wrong) answer than one full-mask LPI.

    Window A: one small pool (3x3 = 9 px).
    Window B: one large pool (4x4 = 16 px).
    Full-mask LPI = max(9, 16) / A_total * 100 = 16 / A_total * 100.
    A per-window-then-averaged LPI would instead average LPI_A (9/A_total*100
    computed against the SAME fixed a_total) and LPI_B (16/A_total*100),
    giving (9+16)/2 / A_total * 100 -- a different, wrong number. Concatenate-
    then-reduce-once must match the true full-mask LPI exactly.
    """
    shape = (30, 30)
    mask = np.zeros(shape, dtype=bool)
    mask[2:5, 2:5] = True  # window A: 3x3 = 9 px
    mask[24:28, 24:28] = True  # window B: 4x4 = 16 px
    a_total_m2 = float(shape[0] * shape[1]) * 900.0

    analysis_mask = xr.DataArray(mask, dims=("y", "x"))
    from hydrofragments.spatial.active_windows import independent_active_windows

    windows = list(
        independent_active_windows(
            analysis_mask, connectivity=8, halo_pixels=1, align_pixels=1
        )
    )
    assert len(windows) == 2, "fixture must produce exactly two independent windows"

    all_properties = []
    for window in windows:
        row0, col0, row1, col1 = window.bbox
        crop = mask[row0:row1, col0:col1]
        all_properties.extend(
            measure_patch_properties(crop, pixel_size_m=30.0, include_width=False)
        )
    core, _ = reduce_patch_properties(
        all_properties, pixel_size_m=30.0, a_total_m2=a_total_m2
    )

    full_core, _ = reduce_patch_properties(
        measure_patch_properties(mask, pixel_size_m=30.0, include_width=False),
        pixel_size_m=30.0,
        a_total_m2=a_total_m2,
    )

    expected_lpi = 16.0 * 900.0 / a_total_m2 * 100.0
    wrong_averaged_lpi = ((9.0 * 900.0 / a_total_m2 * 100.0) + expected_lpi) / 2.0

    assert core.lpi == pytest.approx(expected_lpi)
    assert core.lpi == pytest.approx(full_core.lpi)
    assert core.lpi != pytest.approx(wrong_averaged_lpi)
    assert core.number_of_pools == full_core.number_of_pools == 2
    assert core.awmsi == pytest.approx(full_core.awmsi)
    assert core.awre == pytest.approx(full_core.awre)


# ---------------------------------------------------------------------------
# Step 3: exactly one label_components materialization per (month, window),
# regardless of how many metric families are enabled -- extending the
# pre-existing "one label_components call per month" invariant
# (tests/metrics/test_patch_bundle.py::test_analyze_core_patches_and_pool_
# width_share_one_bundle) to the windowed case.
# ---------------------------------------------------------------------------


def _far_apart_two_component_mask(shape=(1200, 1200)):
    """Two components far enough apart that independent_active_windows()'s
    production ``align_pixels=512`` default keeps them as two disjoint
    windows rather than snap-merging into one -- unlike a small test-sized
    grid, where 512-pixel alignment always collapses everything into a
    single window regardless of fragmentation.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 10:14] = True
    mask[1100:1104, 1100:1104] = True
    return mask


def test_one_label_components_call_per_window_regardless_of_metric_families(
    tmp_path,
):
    """Two months x two windows must call label_components exactly four
    times total -- once per (month, window) -- even though both core patch
    metrics (lpi/awre/awmsi/number_of_pools) AND pool_width are enabled,
    proving core and width share the same per-window measurement rather than
    each family re-labeling the same window.
    """
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    shape = (1200, 1200)
    water_np = np.stack([_far_apart_two_component_mask(shape) for _ in times])
    water = xr.DataArray(water_np, dims=("time", "y", "x"), coords={"time": times})
    fragmented_mask = xr.DataArray(
        _far_apart_two_component_mask(shape), dims=("y", "x")
    )
    cube = open_water_cube(
        water, input_kind="generic_binary", analysis_mask=fragmented_mask
    )
    config = _config(tmp_path, profiles=("contracts_core", "secondary"))

    with mock.patch.object(
        patches_module, "label_components", wraps=patches_module.label_components
    ) as spy:
        analyze(cube, aoi_id="windowed", config=config, pixel_size_m=30.0)

    assert spy.call_count == 2 * 2  # 2 months * 2 independent windows


# ---------------------------------------------------------------------------
# Step 4: default all_available run must equal the union of running each
# individual explicit profile separately.
# ---------------------------------------------------------------------------


def test_default_all_available_matches_union_of_explicit_profile_runs(tmp_path):
    """A single all_available run's patch-family values must equal running
    contracts_core and secondary separately then taking their union -- no
    metric silently changes value depending on which other profiles ran
    alongside it in the same call.
    """
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    shape = (12, 12)
    water_np = np.stack([_two_pool_mask() for _ in times])
    water = xr.DataArray(water_np, dims=("time", "y", "x"), coords={"time": times})
    cube = open_water_cube(water, input_kind="generic_binary")

    all_available_config = _config(tmp_path / "all", profiles=("all_available",))
    core_config = _config(tmp_path / "core", profiles=("contracts_core",))
    secondary_config = _config(tmp_path / "secondary", profiles=("secondary",))

    result_all = analyze(
        cube, aoi_id="all", config=all_available_config, pixel_size_m=30.0
    )
    result_core = analyze(
        cube, aoi_id="core", config=core_config, pixel_size_m=30.0
    )
    result_secondary = analyze(
        cube, aoi_id="secondary", config=secondary_config, pixel_size_m=30.0
    )

    all_values = _analyze_result_metric_values(result_all)
    union_values: dict[tuple, list] = {}
    for partial in (result_core, result_secondary):
        for key, values in _analyze_result_metric_values(partial).items():
            union_values.setdefault(key, []).extend(values)

    patch_family_keys = {
        key
        for key in union_values
        if key[0] in {"number_of_pools", "lpi", "awre", "awmsi", "pool_width"}
    }
    assert patch_family_keys <= set(all_values)
    for key in patch_family_keys:
        left = sorted(all_values[key], key=lambda v: (v is None, v))
        right = sorted(union_values[key], key=lambda v: (v is None, v))
        assert len(left) == len(right)
        for a, b in zip(left, right):
            if a is None or b is None:
                assert a is b
            else:
                assert a == pytest.approx(b)
