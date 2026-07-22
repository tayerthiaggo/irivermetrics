from __future__ import annotations

import math

import numpy as np
import pytest

from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import measure_components
from tests.fixtures.analytic_masks import long_bar_mask, mask_with_hole


def _measure(mask: np.ndarray, *, pixel_size_m: float = 1.0):
    labels = label_components(mask, connectivity=4, min_patch_pixels=1).labels
    return measure_components(
        extract_component_crops(labels), pixel_size_m=pixel_size_m
    )


def test_hole_contributes_inner_and_outer_pixel_edge_perimeter() -> None:
    (patch,) = _measure(mask_with_hole(), pixel_size_m=30.0)

    assert patch.area_pixels == 24
    assert patch.area_m2 == 24 * 30.0**2
    assert patch.perimeter_m == 24 * 30.0  # outer 20 + inner 4 edges
    assert patch.length_method == "major_axis"


def test_long_bar_major_axis_matches_second_moment_analytic_truth() -> None:
    (patch,) = _measure(long_bar_mask(length=4), pixel_size_m=10.0)

    assert patch.area_pixels == 4
    assert patch.perimeter_m == 10 * 10.0  # 1 x 4 rectangle
    assert patch.major_axis_length_m == pytest.approx(math.sqrt(20.0) * 10.0)


def test_component_measurement_is_unchanged_by_surrounding_canvas() -> None:
    bar = long_bar_mask(length=8)
    padded = np.pad(bar, ((4, 7), (3, 9)))

    (small,) = _measure(bar, pixel_size_m=30.0)
    (large,) = _measure(padded, pixel_size_m=30.0)

    assert large.area_m2 == small.area_m2
    assert large.perimeter_m == small.perimeter_m
    assert large.major_axis_length_m == pytest.approx(
        small.major_axis_length_m
    )


def test_pixel_size_must_be_positive() -> None:
    labels = label_components(long_bar_mask(length=3), min_patch_pixels=1).labels
    crops = extract_component_crops(labels)

    with pytest.raises(ValueError, match="pixel_size_m"):
        measure_components(crops, pixel_size_m=0)


def test_core_morphology_does_not_compute_optional_width() -> None:
    labels = label_components(long_bar_mask(length=3), min_patch_pixels=1).labels
    crops = extract_component_crops(labels)

    (patch,) = measure_components(crops, pixel_size_m=30.0)

    assert np.isnan(patch.width_m)
    assert np.isnan(patch.width_pixels)
