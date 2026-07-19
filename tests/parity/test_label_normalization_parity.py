"""m3 parity gate: bincount label normalization must match np.unique/searchsorted.

This test pins ``label_components``'s output on a fragmented multi-component
mask before ``_filter_and_normalize`` is rewritten to use ``np.bincount`` +
a lookup table instead of ``np.unique(sort)`` + ``searchsorted``. The golden
fixture (``labels_golden.npy``) was frozen against the pre-rewrite
implementation; see ``.superpowers/sdd/task-11-report.md`` for provenance.

Row-major first-pixel-occurrence ID ordering is a load-bearing contract for
the rest of the patch/morphology pipeline (component IDs must be
deterministic), so this asserts bit-identical arrays, not just matching
counts.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np

from hydrofragments.patches.labels import (
    _materialize_global_labels,
    _structure,
    label_components,
)
from tests.fixtures.analytic_masks import chunk_crossing_mask

GOLDEN_PATH = Path(__file__).parent / "labels_golden.npy"


def _fragmented():
    m = np.zeros((16, 16), dtype=bool)
    m[1:3, 1:3] = True
    m[1:3, 6:9] = True
    m[10:14, 2:5] = True
    m[12, 12] = True  # singleton -> dropped by min_patch
    return m


def _non_monotonic_raw_id_mask() -> np.ndarray:
    """A 12x12 mask whose ``ndmeasure.label`` raw IDs, under (4, 4) chunking,
    are NOT monotonic with row-major first-occurrence order.

    Built from ``chunk_crossing_mask`` (a single component straddling all
    three chunk boundaries along axis 1, rows 0-3) plus two extra singleton
    pixels placed in the row-3-to-7 chunk band: one in block (1, 0) at
    ``(6, 1)`` and one in block (1, 2) at ``(5, 9)``. Dask-image's block
    labeling assigns per-block label offsets in row-major *block* order
    ``(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), ...``, so the singleton in
    block (1, 0) receives a smaller raw ID than the singleton in block
    (1, 2) even though the latter's pixel occurs earlier in row-major pixel
    order (row 5 < row 6). This reproduces exactly the scenario
    ``_filter_and_normalize`` must handle: raw component IDs from
    cross-chunk reconciliation are not guaranteed to agree with row-major
    occurrence order.
    """
    mask = np.zeros((12, 12), dtype=bool)
    mask[0:4, :] = chunk_crossing_mask(n_chunks=3, chunk_size=4)
    mask[6, 1] = True
    mask[5, 9] = True
    return mask


def test_label_ids_row_major_stable():
    m = _fragmented()
    res = label_components(m, connectivity=8, min_patch_pixels=3)
    # freeze: np.save("tests/parity/labels_golden.npy", res.labels)
    golden = np.load(GOLDEN_PATH)
    np.testing.assert_array_equal(res.labels, golden)
    assert res.count == int(golden.max())


def test_label_dask_matches_dense():
    m = _fragmented()
    dense = label_components(m, min_patch_pixels=3).labels
    chunked = label_components(
        da.from_array(m, chunks=(8, 8)), min_patch_pixels=3
    ).labels
    np.testing.assert_array_equal(dense, chunked)


def test_normalization_handles_non_monotonic_raw_ids_across_chunks():
    """Regression test for Task 11 review: ``_filter_and_normalize`` must not
    rely on raw label IDs being monotonic with row-major first-occurrence
    order. Raw IDs from ``dask_image.ndmeasure.label``'s cross-chunk
    reconciliation are not guaranteed to satisfy that ordering (only dense
    ``scipy.ndimage.label`` guarantees it), so this exercises a genuinely
    chunked, non-monotonic-raw-ID input end to end and asserts the
    normalized output is still correct row-major-ordered IDs matching the
    dense (ground-truth) result.
    """
    mask = _non_monotonic_raw_id_mask()
    chunked = da.from_array(mask, chunks=(4, 4))

    # Proof this scenario is non-vacuous: the raw (pre-normalization) IDs
    # from cross-chunk reconciliation are genuinely out of row-major order.
    raw = _materialize_global_labels(chunked, structure=_structure(8))
    flat_raw = raw.reshape(-1)
    raw_ids = np.unique(flat_raw)
    raw_ids = raw_ids[raw_ids != 0]
    first_occurrence = [
        (int(raw_id), int(np.flatnonzero(flat_raw == raw_id)[0]))
        for raw_id in raw_ids
    ]
    ids_by_id_order = [raw_id for raw_id, _ in first_occurrence]
    ids_by_position_order = [
        raw_id for raw_id, _ in sorted(first_occurrence, key=lambda t: t[1])
    ]
    assert ids_by_id_order != ids_by_position_order, (
        "fixture no longer exercises non-monotonic raw IDs; "
        "test would be vacuous"
    )

    dense = label_components(mask, connectivity=8, min_patch_pixels=1)
    actual = label_components(chunked, connectivity=8, min_patch_pixels=1)

    np.testing.assert_array_equal(actual.labels, dense.labels)
    assert actual.count == dense.count == 3
    # Row-major first-occurrence contract: component at (2, *) is 1, the
    # singleton at (5, 9) is 2 (occurs before (6, 1) in row-major order),
    # and the singleton at (6, 1) is 3.
    assert actual.labels[2, 0] == 1
    assert actual.labels[5, 9] == 2
    assert actual.labels[6, 1] == 3
