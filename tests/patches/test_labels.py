from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from hydrofragments.patches.labels import label_components
from tests.fixtures.analytic_masks import diagonal_pair_mask, mask_with_hole


def test_connectivity_rule_distinguishes_corner_touching_pixels() -> None:
    mask = diagonal_pair_mask()

    four = label_components(mask, connectivity=4, min_patch_pixels=1)
    eight = label_components(mask, connectivity=8, min_patch_pixels=1)

    assert four.count == 2
    assert eight.count == 1


@pytest.mark.parametrize("chunks", [(6, 2), (2, 3), (3, 6)])
def test_normalized_labels_are_invariant_to_chunk_layout(
    chunks: tuple[int, int],
) -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1:5] = True
    mask[2, 4] = True
    mask[4:6, 0:2] = True
    expected = label_components(mask, connectivity=4, min_patch_pixels=1)

    actual = label_components(
        da.from_array(mask, chunks=chunks),
        connectivity=4,
        min_patch_pixels=1,
    )

    np.testing.assert_array_equal(actual.labels, expected.labels)
    assert actual.count == expected.count == 2
    assert actual.labels.dtype == np.int32


def test_minimum_mapping_unit_is_applied_after_global_reconciliation() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[1, 1:5] = True  # four-pixel patch split 2 + 2 by x chunks
    mask[4, 5] = True  # globally undersized noise

    result = label_components(
        da.from_array(mask, chunks=(5, 3)),
        connectivity=4,
        min_patch_pixels=3,
    )

    assert result.count == 1
    assert np.count_nonzero(result.labels) == 4
    assert result.labels[1, 1] == result.labels[1, 4] == 1
    assert result.labels[4, 5] == 0


@pytest.mark.parametrize("chunks", [(2, 2), (1, 2), (2, 1)])
def test_eight_neighbor_diagonal_reconciles_across_chunk_corners(
    chunks: tuple[int, int],
) -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    expected = label_components(
        mask, connectivity=8, min_patch_pixels=1
    )

    actual = label_components(
        da.from_array(mask, chunks=chunks),
        connectivity=8,
        min_patch_pixels=1,
    )

    np.testing.assert_array_equal(actual.labels, expected.labels)
    assert actual.count == 1
    assert actual.labels.dtype == np.int32


def test_hole_remains_background_without_splitting_patch() -> None:
    result = label_components(
        mask_with_hole(), connectivity=8, min_patch_pixels=3
    )

    assert result.count == 1
    assert np.count_nonzero(result.labels) == 24
    assert result.labels[2, 2] == 0


@pytest.mark.parametrize("connectivity", [0, 6, 12])
def test_unsupported_connectivity_is_rejected(connectivity: int) -> None:
    with pytest.raises(ValueError, match="connectivity"):
        label_components(np.ones((2, 2), dtype=bool), connectivity=connectivity)


def test_only_two_dimensional_masks_are_accepted() -> None:
    with pytest.raises(ValueError, match="2-D"):
        label_components(np.ones((1, 2, 2), dtype=bool))
