"""m4 parity gate: single-EDT ``medial_axis(..., return_distance=True)`` width
must match the pre-rewrite two-call pattern (separate ``medial_axis`` +
``distance_transform_edt``) bit-for-bit.

The fixture is a single irregular, branching (T-shaped-with-a-notch)
component -- not a rectangle or other degenerate shape -- so the medial axis
genuinely has to resolve ridges from more than one arm and the frozen
``width_pixels``/``width_m`` actually exercise the multi-arm skeleton +
distance-transform path this rewrite touches, not a trivial case where any
implementation would agree.

Provenance: values below were captured by running this exact fixture through
``measure_components(..., include_width=True)`` against the pre-rewrite
implementation (two ``ndi``/``skimage`` calls: ``medial_axis(mask)`` for the
skeleton, ``distance_transform_edt(mask)`` for distances) on 2026-07-19,
before ``hydrofragments/patches/morphology.py`` was changed to use
``medial_axis(mask, return_distance=True)``. See
``.superpowers/sdd/task-12-report.md`` for the verification that
``medial_axis``'s ``dist`` output is bit-identical to a standalone
``distance_transform_edt`` call by construction (skimage internally computes
``distance = ndi.distance_transform_edt(masked_image)`` and returns a
``.copy()`` of that exact array as ``dist``), not merely "happens to match".
"""

from __future__ import annotations

import numpy as np

from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import measure_components


def _irregular_branching_mask() -> np.ndarray:
    """Single connected T-shaped component with an interior notch.

    Vertical arm + horizontal arm + an off-axis lobe, with a notch carved
    into the joint so the shape is not a simple rectangle/cross -- the
    medial axis must resolve ridges across multiple arms and the notch,
    genuinely stressing the skeleton + distance-transform computation.
    """
    m = np.zeros((14, 18), dtype=bool)
    m[2:10, 2:6] = True  # vertical arm
    m[6:9, 2:14] = True  # horizontal arm
    m[3:6, 9:13] = True  # extra lobe off the horizontal arm
    m[7, 6:9] = False  # notch carved into the joint interior
    return m


def _measure(mask: np.ndarray, *, pixel_size_m: float):
    labels = label_components(mask, connectivity=8, min_patch_pixels=1).labels
    crops = extract_component_crops(labels)
    assert len(crops) == 1, "fixture must stay a single connected component"
    return measure_components(crops, pixel_size_m=pixel_size_m, include_width=True)


def test_fixture_is_single_nontrivial_component() -> None:
    """Guard against the fixture silently degenerating (e.g. splitting into
    two components, or losing the notch) across future edits."""
    mask = _irregular_branching_mask()
    labels = label_components(mask, connectivity=8, min_patch_pixels=1)
    assert labels.count == 1
    assert int(np.count_nonzero(mask)) == 65


def test_width_pixels_and_width_m_stable() -> None:
    (p,) = _measure(_irregular_branching_mask(), pixel_size_m=30.0)

    # Frozen against the pre-rewrite two-call implementation
    # (medial_axis(mask) + distance_transform_edt(mask) separately).
    assert p.width_pixels == 4.47213595499958
    assert p.width_m == 134.1640786499874


def test_area_perimeter_major_axis_unaffected_by_width_rewrite() -> None:
    """These fields don't touch the m4 code path but are frozen too, so any
    accidental cross-talk from the rewrite would be caught."""
    (p,) = _measure(_irregular_branching_mask(), pixel_size_m=30.0)

    assert p.area_pixels == 65
    assert p.area_m2 == 58500.0
    assert p.perimeter_m == 1620.0
    assert p.major_axis_length_m == 438.4797018429283


def test_width_pixels_deterministic_across_repeated_calls() -> None:
    """medial_axis takes an ``rng`` for corner-processing tiebreaks; this
    fixture's max-distance skeleton pixel must not depend on tie order, or
    the frozen value above would be flaky."""
    mask = _irregular_branching_mask()
    labels = label_components(mask, connectivity=8, min_patch_pixels=1).labels
    crops = extract_component_crops(labels)

    widths = {
        measure_components(crops, pixel_size_m=30.0, include_width=True)[0].width_pixels
        for _ in range(10)
    }
    assert widths == {4.47213595499958}
