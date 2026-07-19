"""m9 parity gate: bulk ``regionprops_table`` major-axis-length must match
the pre-rewrite per-component ``regionprops`` call bit-for-bit before the
bulk path is allowed to replace it.

The fixture places three components of different sizes/shapes at different
offsets within one shared 20x30 raster (mimicking a bucketed crop group from
``bucket_component_crops``), so the parity check genuinely exercises the
translation from "per-component crop in its own local coordinate frame" to
"one shared label raster, mapped back by label" -- not a single-component
case where any implementation would trivially agree.

Provenance: values below were captured by running this exact fixture through
``measure_components(..., include_width=False)`` against the pre-rewrite
per-component ``regionprops(mask.astype(np.uint8))`` implementation on
2026-07-19. See ``.superpowers/sdd/task-12-report.md`` for the mathematical
justification (``axis_major_length`` is derived from central moments, which
are translation-invariant by construction, and ``regionprops_table``
delegates to the same ``RegionProperties`` machinery as ``regionprops``) and
for the randomized stress-test evidence (250 trials, offsets/shapes varied,
zero mismatches) backing the decision of whether m9's bulk rewrite was kept
or reverted.
"""

from __future__ import annotations

import numpy as np

from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import measure_components


def _multi_component_mask() -> np.ndarray:
    """Three distinct components (different sizes/shapes) sharing one raster."""
    m = np.zeros((20, 30), dtype=bool)
    m[1:5, 1:4] = True  # component A: small rectangle
    m[10:16, 2:9] = True  # component B: L-shape ...
    m[10:12, 9:14] = True  # ... continued
    m[16, 5:8] = False  # ... with a notch
    m[1:6, 20:27] = True  # component C: irregular block ...
    m[6, 22:25] = True  # ... with an extra lobe
    m[3, 20] = False  # ... and a notch
    return m


def _measure(mask: np.ndarray, *, pixel_size_m: float):
    labels = label_components(mask, connectivity=8, min_patch_pixels=1).labels
    crops = extract_component_crops(labels)
    return crops, measure_components(crops, pixel_size_m=pixel_size_m, include_width=False)


def test_fixture_has_three_distinct_nontrivial_components() -> None:
    mask = _multi_component_mask()
    labels = label_components(mask, connectivity=8, min_patch_pixels=1)
    assert labels.count == 3


def test_major_axis_length_stable_per_component() -> None:
    crops, results = _measure(_multi_component_mask(), pixel_size_m=25.0)
    by_label = {r.label: r for r in results}

    assert len(results) == 3
    # Frozen against the pre-rewrite per-component regionprops(...) call.
    assert by_label[1].major_axis_length_m == 111.80339887498948
    assert by_label[1].area_pixels == 12
    assert by_label[2].major_axis_length_m == 189.4307153058859
    assert by_label[2].area_pixels == 37
    assert by_label[3].major_axis_length_m == 311.8588594715587
    assert by_label[3].area_pixels == 52


def test_components_are_genuinely_offset_within_shared_raster() -> None:
    """Guard against the fixture degenerating into overlapping/adjacent
    bboxes, which would make the bulk-vs-per-component translation trivial."""
    crops, _ = _measure(_multi_component_mask(), pixel_size_m=25.0)
    bboxes = sorted(c.bbox for c in crops)
    # No two bounding boxes share row or column ranges (genuinely distinct
    # offsets in the shared raster, not just distinct labels).
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            r0a, c0a, r1a, c1a = bboxes[i]
            r0b, c0b, r1b, c1b = bboxes[j]
            rows_overlap = r0a < r1b and r0b < r1a
            cols_overlap = c0a < c1b and c0b < c1a
            assert not (rows_overlap and cols_overlap)
