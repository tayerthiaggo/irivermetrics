"""3.6 parity gate: a batched (vectorised-across-a-bucket) pixel-edge
perimeter computation must match the pre-rewrite per-crop
``_pixel_edge_perimeter(mask)`` loop bit-for-bit (exact integer equality --
perimeter is a pixel-edge count, not a float, so there is no tolerance to
argue about).

The batching strategy mirrors ``_bulk_major_axis_lengths``: every crop in a
bucket is placed into disjoint blocks of one shared boolean composite,
stacked down the rows with each block given the widest crop's column
extent (composite is bool, not int64 -- perimeter counting only needs
foreground/background, not per-label identity). One ``np.pad`` + two
diff-and-count calls are then done ONCE on the whole composite instead of
once per crop; the result is split back out per crop's own row range.

Correctness is not assumed: every ``ComponentCrop.mask`` already carries its
own 1px false border (``iter_component_crops``'s ``padding=1`` default), so
adjacent blocks in the composite always have a false row directly above and
below them -- vertical diffs across a block boundary compare False vs False
(no edge), reproducing exactly what per-crop padding would have produced.
This file proves that with a randomized stress test (Section 1) before
trusting the production kernel (Section 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import (
    _bulk_pixel_edge_perimeters,
    _pixel_edge_perimeter,
)

WMASK_PATH = Path(__file__).parent.parent / "wmask_ts.nc"


def _random_bool_crop(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(1, 14))
    w = int(rng.integers(1, 18))
    body = rng.random((h, w)) > 0.35
    return np.pad(body, 1, constant_values=False)


class _FakeCrop:
    """Minimal stand-in with the two attributes _bulk_pixel_edge_perimeters
    needs (``label``, ``mask``), avoiding a dependency on real labeling for
    the randomized stress test."""

    def __init__(self, label: int, mask: np.ndarray) -> None:
        self.label = label
        self.mask = mask


# --- Section 1: randomized stress test, varied shapes/sizes ---------------


@pytest.mark.parametrize("seed", range(20))
def test_bulk_perimeter_matches_per_crop_loop_randomized(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 9))
    crops = [_FakeCrop(i + 1, _random_bool_crop(rng)) for i in range(n)]

    expected = {c.label: _pixel_edge_perimeter(c.mask) for c in crops}
    actual = _bulk_pixel_edge_perimeters(tuple(crops))

    assert actual == expected


def test_bulk_perimeter_matches_loop_for_all_true_and_single_pixel() -> None:
    """Degenerate shapes: an all-wet crop with real background padding and a
    single wet pixel, mixed into the same bucket as ordinary shapes.
    """
    all_true = np.pad(np.ones((3, 3), dtype=bool), 1, constant_values=False)
    single_px = np.pad(np.ones((1, 1), dtype=bool), 1, constant_values=False)
    crops = [
        _FakeCrop(1, all_true),
        _FakeCrop(2, single_px),
        _FakeCrop(3, _random_bool_crop(np.random.default_rng(7))),
    ]
    expected = {c.label: _pixel_edge_perimeter(c.mask) for c in crops}
    actual = _bulk_pixel_edge_perimeters(tuple(crops))
    assert actual == expected


# --- Section 2: production entry point, real components -------------------


def test_bulk_perimeter_matches_per_crop_loop_on_real_fitzroy_components() -> None:
    """Exhaustive proof over every real connected component across all 63
    monthly timesteps of tests/wmask_ts.nc (real Fitzroy catchment data).

    ``_bulk_pixel_edge_perimeters`` keys its result by ``crop.label``, which
    repeats across timesteps (each timestep's labels start at 1), so crops
    are bucketed per timestep -- exactly matching how
    ``bucket_component_crops`` groups crops from a single labeling call in
    production (``measure_patch_properties`` labels and buckets one month
    at a time).
    """
    ds = xr.open_dataset(WMASK_PATH)
    var = list(ds.data_vars)[0]
    data = ds[var].values

    checked = 0
    for t in range(data.shape[0]):
        wet = data[t] == 1
        labels = label_components(
            wet, connectivity=8, min_patch_pixels=1
        ).labels
        timestep_crops = extract_component_crops(labels)
        if not timestep_crops:
            continue
        bulk = _bulk_pixel_edge_perimeters(timestep_crops)
        for crop in timestep_crops:
            assert bulk[crop.label] == _pixel_edge_perimeter(crop.mask)
            checked += 1
    assert checked > 500, "fixture sweep too small to be meaningful"
