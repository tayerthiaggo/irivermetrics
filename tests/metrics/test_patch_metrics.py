from __future__ import annotations

import math

import numpy as np
import pytest

from hydrofragments.metrics.patches import (
    analyze_patch_metrics,
    compute_patch_metrics,
)
from hydrofragments.patches.components import ComponentCrop
from hydrofragments.patches.morphology import measure_components
from hydrofragments.schema import EdgeFlag


def test_n_and_lpi_use_filtered_patches_and_fixed_aoi_area() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:2, 0:2] = True  # 4 pixels
    mask[4, 4:7] = True  # 3 pixels
    mask[5, 7] = True  # noise removed by global MMU

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=10.0,
        a_total_m2=2_000.0,
        connectivity=4,
        min_patch_pixels=3,
    )

    assert result.number_of_pools == 2
    assert result.lpi == pytest.approx(20.0)  # 400 / fixed 2000 * 100
    assert result.n_water_pixels == 7


def test_analysis_accepts_bounded_component_work_target() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4, 4:7] = True

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=10.0,
        a_total_m2=2_000.0,
        connectivity=4,
        min_patch_pixels=3,
        target_component_pixels=4,
    )

    assert result.number_of_pools == 2


def test_lpi_denominator_is_not_total_wet_area() -> None:
    mask = np.ones((2, 2), dtype=bool)

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=10.0,
        a_total_m2=1_000.0,
        min_patch_pixels=3,
    )

    assert result.lpi == pytest.approx(40.0)


def test_awre_major_axis_fallback_matches_long_bar_truth() -> None:
    mask = np.ones((1, 4), dtype=bool)

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=10.0,
        a_total_m2=400.0,
        connectivity=4,
        min_patch_pixels=3,
    )

    expected = 2.0 * math.sqrt(400.0 / math.pi) / (math.sqrt(20.0) * 10.0)
    assert result.awre == pytest.approx(expected)
    assert result.awre_length_method == "major_axis"


def test_awre_major_axis_fallback_matches_bent_pool_truth() -> None:
    mask = np.array([[True, False], [True, True]], dtype=bool)

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=1.0,
        a_total_m2=3.0,
        connectivity=4,
        min_patch_pixels=3,
    )

    # L-triomino major axis = 4/sqrt(3) from its coordinate covariance.
    expected = 3.0 / (2.0 * math.sqrt(math.pi))
    assert result.awre == pytest.approx(expected)
    assert result.awre_length_method == "major_axis"


def test_awmsi_matches_exact_pixel_edge_shape_indices() -> None:
    square = np.ones((2, 2), dtype=bool)  # area 4, perimeter 8, SI 1
    bar = np.ones((1, 4), dtype=bool)  # area 4, perimeter 10, SI 1.25
    crops = (
        ComponentCrop(1, (0, 0, 2, 2), np.pad(square, 1)),
        ComponentCrop(2, (3, 0, 4, 4), np.pad(bar, 1)),
    )
    properties = measure_components(crops, pixel_size_m=1.0)

    result = compute_patch_metrics(properties, a_total_m2=8.0)

    assert result.awmsi == pytest.approx(1.125)


def test_hole_awmsi_counts_inner_boundary() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=1.0,
        a_total_m2=25.0,
        min_patch_pixels=3,
    )

    expected = 0.25 * 24.0 / math.sqrt(24.0)
    assert result.awmsi == pytest.approx(expected)


def test_all_dry_n0_has_zero_count_and_nan_patch_metrics() -> None:
    result = analyze_patch_metrics(
        np.zeros((4, 4), dtype=bool),
        pixel_size_m=30.0,
        a_total_m2=14_400.0,
    )

    assert result.number_of_pools == 0
    assert result.n_water_pixels == 0
    assert np.isnan(result.lpi)
    assert np.isnan(result.awre)
    assert np.isnan(result.awmsi)
    assert result.edge_flag is EdgeFlag.N0


def test_all_dry_still_rejects_nonpositive_pixel_size() -> None:
    with pytest.raises(ValueError, match="pixel_size_m"):
        analyze_patch_metrics(
            np.zeros((4, 4), dtype=bool),
            pixel_size_m=0.0,
            a_total_m2=1.0,
        )


def test_one_patch_n1_keeps_shape_and_extent_metrics_valid() -> None:
    result = analyze_patch_metrics(
        np.ones((1, 3), dtype=bool),
        pixel_size_m=30.0,
        a_total_m2=2_700.0,
    )

    assert result.number_of_pools == 1
    assert result.lpi == pytest.approx(100.0)
    assert np.isfinite(result.awre)
    assert np.isfinite(result.awmsi)
    assert result.edge_flag is EdgeFlag.N1


@pytest.mark.parametrize("a_total_m2", [0.0, -1.0])
def test_fixed_area_denominator_must_be_positive(a_total_m2: float) -> None:
    with pytest.raises(ValueError, match="a_total_m2"):
        analyze_patch_metrics(
            np.ones((2, 2), dtype=bool),
            pixel_size_m=1.0,
            a_total_m2=a_total_m2,
        )


def test_result_contains_no_mesh_width_or_lineage_fields() -> None:
    result = analyze_patch_metrics(
        np.ones((2, 2), dtype=bool),
        pixel_size_m=1.0,
        a_total_m2=4.0,
    )

    assert set(result.__dataclass_fields__) == {
        "number_of_pools",
        "n_water_pixels",
        "lpi",
        "awre",
        "awmsi",
        "edge_flag",
        "awre_length_method",
    }
