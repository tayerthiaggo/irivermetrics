"""3.6 parity gate: skipping ``np.minimum.at`` for the SciPy branch of
``_filter_and_normalize`` must produce byte-identical ``LabelResult`` to the
unconditional (dask-image-safe) reordering path, on masks that exercise both
the SciPy (in-memory / small-Dask) and dask-image (cross-chunk) branches.

Structural claim under test: ``scipy.ndimage.label`` assigns raw component
IDs by scanning the raster in row-major order and incrementing an ID counter
the first time each new component is encountered. That means, for ANY mask
labeled by ``scipy.ndimage.label``, raw ID ``k`` always first occurs earlier
in row-major flattened order than raw ID ``k + 1`` -- i.e. raw IDs are
already sorted by first-occurrence position, by construction. Consequently
``first[retained_ids]`` (computed by ``np.minimum.at``) equals
``retained_ids`` itself minus 1 position semantics: the identity that matters
is that ``np.argsort(first[retained_ids])`` is already the identity
permutation, so the whole ``np.minimum.at`` reordering computation is
provably a no-op for SciPy-branch input and can be skipped.

This is explicitly NOT assumed true without proof -- task 3.5 in this same
plan found an almost-identical "this is obviously true" structural claim
(about ``skimage.morphology.medial_axis``) to be FALSE under test. This file
starts by proving the claim empirically on real and synthetic masks (Section
1) *before* trusting the bypass, then locks in end-to-end parity (Section 2).
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from hydrofragments.patches.labels import (
    _filter_and_normalize,
    _materialize_global_labels,
    _structure,
    label_components,
)
from tests.fixtures.analytic_masks import (
    diagonal_pair_mask,
    mask_with_hole,
    one_pixel_noise_mask,
)

WMASK_PATH = Path(__file__).parent.parent / "wmask_ts.nc"


def _real_masks() -> list[np.ndarray]:
    ds = xr.open_dataset(WMASK_PATH)
    var = list(ds.data_vars)[0]
    data = ds[var].values
    return [data[t] == 1 for t in range(data.shape[0])]


# --- Section 1: structural claim, proven empirically, not assumed ---------


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize(
    "mask_factory",
    [diagonal_pair_mask, mask_with_hole, one_pixel_noise_mask],
)
def test_scipy_raw_ids_are_already_row_major_first_occurrence_order(
    mask_factory, connectivity: int
) -> None:
    """For scipy.ndimage.label output, raw ID k's first row-major occurrence
    must come before raw ID k+1's, for every k -- i.e. IDs are already in
    ascending first-occurrence order, with no permutation needed.
    """
    mask = mask_factory()
    raw, _ = ndimage.label(mask, structure=_structure(connectivity))
    flat = raw.reshape(-1)
    raw_ids = np.unique(flat)
    raw_ids = raw_ids[raw_ids != 0]
    if raw_ids.size < 2:
        pytest.skip(
            "fixture has fewer than 2 components; ordering claim is vacuous"
        )

    first_occurrence = [
        int(np.flatnonzero(flat == rid)[0]) for rid in raw_ids
    ]
    assert first_occurrence == sorted(first_occurrence), (
        "scipy.ndimage.label raw IDs are NOT already row-major "
        "first-occurrence ordered for this mask -- bypass claim is false"
    )


def test_scipy_raw_ids_row_major_across_all_real_fixture_timesteps() -> None:
    """Exhaustive sweep: every one of the real Fitzroy monthly masks, under
    8-connectivity SciPy labeling, must already have row-major-ordered raw
    IDs. This is the strongest empirical proof available before trusting the
    bypass in production.
    """
    masks = _real_masks()
    checked_with_multiple_components = 0
    for mask in masks:
        raw, n = ndimage.label(mask, structure=_structure(8))
        if n < 2:
            continue
        checked_with_multiple_components += 1
        flat = raw.reshape(-1)
        raw_ids = np.arange(1, n + 1)
        first_occurrence = [
            int(np.flatnonzero(flat == rid)[0]) for rid in raw_ids
        ]
        assert first_occurrence == sorted(first_occurrence), (
            "counterexample found in real fixture data: scipy raw IDs are "
            "NOT row-major first-occurrence ordered"
        )
    assert checked_with_multiple_components > 10, (
        "fixture sweep did not exercise enough multi-component masks; "
        "test would be weak evidence"
    )


def test_dask_image_branch_can_still_violate_row_major_order() -> None:
    """Sanity check that the bypass claim is NOT vacuously true for every
    labeling path -- the dask-image cross-chunk branch genuinely can and
    does permute raw IDs out of row-major order (already proven in
    tests/parity/test_label_normalization_parity.py). Re-asserted here so
    this file's own reasoning about "SciPy-only" is self-contained.
    """
    from tests.fixtures.analytic_masks import chunk_crossing_mask

    mask = np.zeros((12, 12), dtype=bool)
    mask[0:4, :] = chunk_crossing_mask(n_chunks=3, chunk_size=4)
    mask[6, 1] = True
    mask[5, 9] = True
    chunked = da.from_array(mask, chunks=(4, 4))

    raw, already_row_major = _materialize_global_labels(
        chunked, structure=_structure(8), local_label_threshold_bytes=0
    )
    assert already_row_major is False
    flat = raw.reshape(-1)
    raw_ids = np.unique(flat)
    raw_ids = raw_ids[raw_ids != 0]
    first_occurrence = [
        int(np.flatnonzero(flat == rid)[0]) for rid in raw_ids
    ]
    assert first_occurrence != sorted(first_occurrence), (
        "fixture no longer exercises a dask-image ID permutation; "
        "the contrast case for this file is vacuous"
    )


# --- Section 2: end-to-end LabelResult parity between bypass and full path ---


def test_bypass_matches_unconditional_reorder_on_scipy_labels() -> None:
    """Direct unit-level proof: calling _filter_and_normalize with the bypass
    flag on SciPy-branch raw labels gives the identical LabelResult as the
    unconditional (dask-image-safe) path, for every real fixture timestep.
    """
    for mask in _real_masks():
        raw, _ = ndimage.label(mask, structure=_structure(8))
        unconditional = _filter_and_normalize(
            raw, min_patch_pixels=1, already_row_major=False
        )
        bypassed = _filter_and_normalize(
            raw, min_patch_pixels=1, already_row_major=True
        )
        np.testing.assert_array_equal(bypassed.labels, unconditional.labels)
        assert bypassed.count == unconditional.count


@pytest.mark.parametrize("min_patch_pixels", [1, 3, 10])
def test_label_components_scipy_branch_matches_forced_dask_branch(
    min_patch_pixels: int,
) -> None:
    """End-to-end through the public label_components entry point: a mask
    small enough to route through SciPy (bypass active) must produce
    byte-identical LabelResult to the SAME mask forced through the
    dask-image branch (bypass inactive, full np.minimum.at reorder).
    """
    for mask in _real_masks()[:10]:
        scipy_branch = label_components(
            mask, connectivity=8, min_patch_pixels=min_patch_pixels
        )
        dask_backed = da.from_array(mask, chunks=(37, 83))
        dask_branch = label_components(
            dask_backed,
            connectivity=8,
            min_patch_pixels=min_patch_pixels,
            local_label_threshold_bytes=0,  # force dask-image path
        )
        np.testing.assert_array_equal(scipy_branch.labels, dask_branch.labels)
        assert scipy_branch.count == dask_branch.count


def test_label_components_default_routing_is_unaffected_end_to_end() -> None:
    """The default (no override) call path -- a plain in-memory mask -- must
    give the exact same LabelResult before and after the bypass is wired in.
    Frozen against a fragmented multi-component mask.
    """
    mask = np.zeros((16, 16), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, 6:9] = True
    mask[10:14, 2:5] = True
    mask[12, 12] = True

    result = label_components(mask, connectivity=8, min_patch_pixels=3)
    assert result.count == 3
    assert result.labels[1, 1] == 1
    assert result.labels[1, 6] == 2
    assert result.labels[10, 2] == 3
    assert result.labels[12, 12] == 0
