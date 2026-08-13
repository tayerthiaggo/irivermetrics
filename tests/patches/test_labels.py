from __future__ import annotations

from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest

from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.patches import labels as labels_module
from hydrofragments.patches.labels import label_components, label_components_to_checkpoint
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


# --- Size-threshold routing (W3.4) ---------------------------------------
#
# Below the byte threshold, a Dask-backed mask must be routed through
# ``np.asarray`` + ``scipy.ndimage.label`` (the eager path) instead of
# ``dask_image.ndmeasure.label``. Above it, the dask-image cross-chunk path
# is retained. Both paths must produce byte-identical normalized labels for
# the same mask regardless of original chunk layout, because
# ``_filter_and_normalize`` normalizes by first row-major occurrence.


def _fragmented_mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, 6:9] = True
    mask[10:14, 2:5] = True
    mask[12, 12] = True  # singleton, dropped by min_patch_pixels=3 downstream
    return mask


@pytest.mark.parametrize("chunks", [(4, 4), (8, 8), (16, 4), (4, 16), (5, 7)])
def test_small_dask_mask_matches_eager_across_chunk_layouts(
    chunks: tuple[int, int],
) -> None:
    """Parity claim: SciPy-eager and dask-image paths must agree bit-for-bit,
    for every chunk layout of the same small mask, once normalized.
    """
    mask = _fragmented_mask()
    eager = label_components(mask, connectivity=8, min_patch_pixels=3)

    dask_backed = da.from_array(mask, chunks=chunks)
    routed = label_components(
        dask_backed,
        connectivity=8,
        min_patch_pixels=3,
        local_label_threshold_bytes=mask.nbytes,  # forces the local branch
    )

    np.testing.assert_array_equal(routed.labels, eager.labels)
    assert routed.count == eager.count
    assert routed.labels.dtype == np.int32


def test_small_dask_mask_below_threshold_uses_scipy_label(monkeypatch) -> None:
    """Below the threshold, a Dask-backed mask must route to
    ``scipy.ndimage.label``, not ``dask_image.ndmeasure.label`` -- proven by
    spying on both entry points rather than only checking output values.
    """
    mask = _fragmented_mask()
    dask_backed = da.from_array(mask, chunks=(4, 4))

    with patch(
        "hydrofragments.patches.labels.ndimage.label",
        wraps=labels_module.ndimage.label,
    ) as scipy_label, patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        label_components(
            dask_backed,
            connectivity=8,
            min_patch_pixels=3,
            local_label_threshold_bytes=mask.nbytes,
        )

    scipy_label.assert_called_once()
    dask_image_label.assert_not_called()


def test_large_dask_mask_above_threshold_uses_dask_image_label(monkeypatch) -> None:
    """Above the threshold, the existing dask-image cross-chunk path must
    still be used.

    ``dask_image.ndmeasure.label`` is the unambiguous signal of which
    routing branch ``labels.py`` took: it is only ever called from the
    "keep the dask-image path" branch. It is NOT safe to also assert
    ``ndimage.label`` was never called here, because dask-image's own
    cross-chunk implementation calls ``ndimage.label`` internally per block
    -- that would make the assertion about our routing, not dask-image's.
    """
    mask = _fragmented_mask()
    dask_backed = da.from_array(mask, chunks=(4, 4))

    with patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        label_components(
            dask_backed,
            connectivity=8,
            min_patch_pixels=3,
            local_label_threshold_bytes=mask.nbytes - 1,
        )

    dask_image_label.assert_called_once()


def test_threshold_boundary_is_inclusive_below_and_exclusive_above() -> None:
    """Exactly at the threshold routes to SciPy (<=); one byte above stays on
    dask-image (>). This pins the boundary comparison operator itself.
    """
    mask = _fragmented_mask()
    dask_backed = da.from_array(mask, chunks=(4, 4))

    with patch(
        "hydrofragments.patches.labels.ndimage.label",
        wraps=labels_module.ndimage.label,
    ) as scipy_label, patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        label_components(
            dask_backed,
            connectivity=8,
            min_patch_pixels=3,
            local_label_threshold_bytes=mask.nbytes,
        )
    scipy_label.assert_called_once()
    dask_image_label.assert_not_called()

    with patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        label_components(
            dask_backed,
            connectivity=8,
            min_patch_pixels=3,
            local_label_threshold_bytes=mask.nbytes - 1,
        )
    dask_image_label.assert_called_once()


def test_numpy_eager_mask_is_unaffected_by_threshold() -> None:
    """A plain (non-Dask) mask always uses the eager branch regardless of the
    threshold value -- the threshold only governs Dask-backed routing.
    """
    mask = _fragmented_mask()
    tiny_threshold = label_components(
        mask, connectivity=8, min_patch_pixels=3, local_label_threshold_bytes=1
    )
    huge_threshold = label_components(
        mask,
        connectivity=8,
        min_patch_pixels=3,
        local_label_threshold_bytes=mask.nbytes * 1000,
    )
    np.testing.assert_array_equal(tiny_threshold.labels, huge_threshold.labels)


def test_default_threshold_is_derived_from_compute_policy_target_chunk_bytes() -> None:
    """The default byte threshold must come from the existing compute memory
    policy (``ComputePolicy.target_chunk_bytes``), not a new independent
    config value -- this is a Global Constraints / brief requirement.
    """
    assert (
        labels_module._default_local_label_threshold_bytes()
        == ComputePolicy().target_chunk_bytes
    )


def test_large_mask_checkpoint_does_not_materialize_full_labels(
    tmp_path, monkeypatch
) -> None:
    mask = da.ones((64, 64), chunks=(16, 16), dtype=bool)

    def _boom(*_args, **_kwargs):
        raise AssertionError("must not materialize full label raster")

    monkeypatch.setattr(labels_module, "_materialize_global_labels", _boom)
    result, checkpoint = label_components_to_checkpoint(
        mask,
        connectivity=8,
        min_patch_pixels=1,
        local_label_threshold_bytes=512,
        spill_dir=tmp_path,
    )
    assert result is None
    assert checkpoint is not None
    stored = __import__("zarr").open(checkpoint.path, mode="r")
    assert tuple(stored.shape) == (64, 64)
    assert checkpoint.count == 1
    assert int(np.asarray(stored[:]).max()) == 1


def test_label_checkpoint_matches_eager_labels(tmp_path) -> None:
    mask = _fragmented_mask()
    eager = label_components(mask, connectivity=8, min_patch_pixels=3)
    _, checkpoint = label_components_to_checkpoint(
        da.from_array(mask, chunks=(4, 4)),
        connectivity=8,
        min_patch_pixels=3,
        local_label_threshold_bytes=mask.nbytes - 1,
        spill_dir=tmp_path,
    )
    assert checkpoint is not None
    stored = np.asarray(__import__("zarr").open(checkpoint.path, mode="r")[:])
    np.testing.assert_array_equal(stored, eager.labels)


def test_default_threshold_routes_small_mask_to_scipy_without_override() -> None:
    """End-to-end proof that the *default* (no explicit override) threshold
    is large enough to route a small monthly-mask-sized Dask array to SciPy.
    """
    mask = _fragmented_mask()
    dask_backed = da.from_array(mask, chunks=(4, 4))

    with patch(
        "hydrofragments.patches.labels.ndimage.label",
        wraps=labels_module.ndimage.label,
    ) as scipy_label, patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        label_components(dask_backed, connectivity=8, min_patch_pixels=3)

    scipy_label.assert_called_once()
    dask_image_label.assert_not_called()
