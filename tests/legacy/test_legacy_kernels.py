"""Legacy kernel characterisation tests (Milestone 1).

Scope: unchanged low-level areas, perimeters, skeleton paths, and EDT behaviour from
``ecofragments.utils.calc_metrics``, exercised on tiny Tier A analytic masks
(``tests/fixtures/analytic_masks.py``) with hand-calculable ground truth.

These tests characterise CURRENT legacy kernel behaviour only, including quirks (see
``TestFindConnectedComponents.test_min_patch_size_removal_is_inclusive_of_equal_size``).
They must NEVER be used to validate v1.2 occurrence, schema, or metric correctness, and
they must NEVER compare against ``tests/results_iRiverMetrics/metrics/irm_metrics.csv``
(U7, quarantined — see ``tests/contracts/test_legacy_baseline_quarantine.py`` and
``docs/testing.md``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from ecofragments.utils.calc_metrics import (
    compute_area_and_perimeter_df,
    compute_length_single_graph,
    distance_transform,
    find_connected_components,
    skeletonize_label,
)
from tests.fixtures.analytic_masks import (
    chunk_crossing_mask,
    diagonal_pair_mask,
    empty_mask,
    full_mask,
    long_bar_mask,
    mask_with_hole,
    one_pixel_noise_mask,
    padded_square_mask,
)

PIXEL_SIZE_M = 30.0

# The installed skimage version's remove_small_objects treats min_size as an inclusive
# "<=" threshold rather than the classic "<" threshold; this triggers a FutureWarning on
# every call. It is legacy behaviour under characterisation here, not something this
# milestone may fix (no v1.2 kernel changes), so the warning is suppressed deliberately.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Parameter `min_size` is deprecated:FutureWarning"
)


class TestFindConnectedComponents:
    def test_empty_mask_has_no_components(self):
        labels = find_connected_components(empty_mask(), min_patch_size=1)
        assert labels.max() == 0

    def test_full_mask_is_one_component_covering_every_pixel(self):
        mask = full_mask((6, 6))
        labels = find_connected_components(mask, min_patch_size=1)
        assert labels.max() == 1
        assert (labels > 0).sum() == mask.size

    def test_diagonal_pixels_are_one_component_under_8_connectivity(self):
        labels = find_connected_components(diagonal_pair_mask(), min_patch_size=1)
        assert labels.max() == 1
        assert (labels > 0).sum() == 2

    def test_min_patch_size_removal_is_inclusive_of_equal_size(self):
        """Documents a legacy quirk: min_patch_size=N removes objects of size <= N,
        not size < N. A 1-pixel speck is removed even at min_patch_size=1."""
        mask = one_pixel_noise_mask()
        labels = find_connected_components(mask, min_patch_size=1)
        assert labels.max() == 1
        assert (labels > 0).sum() == 4  # only the 2x2 block survives

    def test_min_patch_size_zero_retains_all_components(self):
        mask = one_pixel_noise_mask()
        labels = find_connected_components(mask, min_patch_size=0)
        assert labels.max() == 2
        assert (labels > 0).sum() == mask.sum() == 5

    def test_hole_does_not_split_component(self):
        labels = find_connected_components(mask_with_hole(), min_patch_size=1)
        assert labels.max() == 1
        assert (labels > 0).sum() == 24

    @pytest.mark.parametrize("n_chunks", [2, 4])
    def test_chunk_crossing_component_is_single_label_on_whole_array(self, n_chunks):
        mask = chunk_crossing_mask(n_chunks=n_chunks, chunk_size=4)
        labels = find_connected_components(mask, min_patch_size=1)
        assert labels.max() == 1
        assert (labels > 0).sum() == mask.sum() == n_chunks * 4


class TestAreaAndPerimeter:
    def test_full_mask_area_matches_pixel_count(self):
        mask = full_mask((4, 4))
        labels = find_connected_components(mask, min_patch_size=1)
        df = compute_area_and_perimeter_df(labels, pixel_size=PIXEL_SIZE_M)
        expected_area_km2 = 16 * (PIXEL_SIZE_M**2) / 1e6
        assert df.loc[0, "area_km2"] == pytest.approx(expected_area_km2)

    def test_hole_reduces_area_by_exactly_one_pixel(self):
        mask = mask_with_hole()
        labels = find_connected_components(mask, min_patch_size=1)
        df = compute_area_and_perimeter_df(labels, pixel_size=PIXEL_SIZE_M)
        expected_area_km2 = 24 * (PIXEL_SIZE_M**2) / 1e6
        assert df.loc[0, "area_km2"] == pytest.approx(expected_area_km2)
        assert df.loc[0, "perimeter_km"] > 0


class TestSkeletonAndEDT:
    def test_long_bar_skeleton_preserves_all_pixels(self):
        mask = long_bar_mask(length=20)
        labels = find_connected_components(mask, min_patch_size=1)
        skel = skeletonize_label(labels)
        # A single-pixel-wide bar is already its own morphological skeleton.
        assert (skel > 0).sum() == mask.sum() == 20

    def test_long_bar_longest_path_length_in_pixel_units(self):
        mask = long_bar_mask(length=20)
        labels = find_connected_components(mask, min_patch_size=1)
        skel = skeletonize_label(labels)
        length_df = compute_length_single_graph(skel, pixel_size=1.0)
        # 20 collinear pixels -> 19 unit steps -> 0.019 "km" at pixel_size=1.0.
        assert length_df.loc[0, "length_km"] == pytest.approx(19 / 1e3)

    def test_padded_square_edt_maximum_sits_at_the_centre(self):
        mask = padded_square_mask(square_size=5, pad=2)
        labels = find_connected_components(mask, min_patch_size=1)
        edt = distance_transform(labels)
        centre = (mask.shape[0] // 2, mask.shape[1] // 2)
        assert edt[centre] == edt.max()
        assert edt[centre] == pytest.approx(3.0)
