"""M2: single per-month patch bundle must match the two separate legacy calls."""

from __future__ import annotations

from unittest import mock

import numpy as np

from hydrofragments.metrics import patches


def _mask():
    m = np.zeros((12, 12), dtype=bool)
    m[2:5, 2:5] = True
    m[7:10, 7:11] = True
    return m


def test_bundle_matches_separate_calls():
    mask = _mask()
    core_ref = patches.analyze_patch_metrics(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0, include_mesh=True
    )
    width_ref = patches.analyze_pool_width_distribution(
        mask, pixel_size_m=30.0, resolution_floor_pixels=2.0
    )
    core, width = patches.analyze_patch_bundle(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
        include_mesh=True, include_width=True, resolution_floor_pixels=2.0,
    )
    assert core == core_ref
    assert width == width_ref


def test_bundle_labels_once():
    mask = _mask()
    with mock.patch.object(
        patches, "label_components", wraps=patches.label_components
    ) as spy:
        patches.analyze_patch_bundle(
            mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
            include_mesh=False, include_width=True, resolution_floor_pixels=2.0,
        )
    assert spy.call_count == 1
