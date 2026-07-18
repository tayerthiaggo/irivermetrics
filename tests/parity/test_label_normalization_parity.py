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

from hydrofragments.patches.labels import label_components

GOLDEN_PATH = Path(__file__).parent / "labels_golden.npy"


def _fragmented():
    m = np.zeros((16, 16), dtype=bool)
    m[1:3, 1:3] = True
    m[1:3, 6:9] = True
    m[10:14, 2:5] = True
    m[12, 12] = True  # singleton -> dropped by min_patch
    return m


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
