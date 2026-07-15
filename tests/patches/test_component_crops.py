from __future__ import annotations

import numpy as np

from hydrofragments.patches.components import (
    ComponentCrop,
    bucket_component_crops,
    extract_component_crops,
)
from hydrofragments.patches.labels import label_components
from tests.fixtures.analytic_masks import long_bar_mask, mask_with_hole


def test_crop_has_one_cell_background_padding_at_raster_edges() -> None:
    labels = label_components(mask_with_hole(), min_patch_pixels=3).labels

    (crop,) = extract_component_crops(labels, padding=1)

    assert crop.label == 1
    assert crop.bbox == (0, 0, 5, 5)
    assert crop.mask.shape == (7, 7)
    assert not crop.mask[0].any()
    assert not crop.mask[-1].any()
    assert not crop.mask[:, 0].any()
    assert not crop.mask[:, -1].any()
    assert np.count_nonzero(crop.mask) == 24
    assert not crop.mask[3, 3]


def test_long_bar_crop_is_bounded_to_component_not_full_raster() -> None:
    mask = np.pad(long_bar_mask(length=12), ((5, 5), (7, 7)))
    labels = label_components(mask, connectivity=4, min_patch_pixels=3).labels

    (crop,) = extract_component_crops(labels, padding=1)

    assert crop.bbox == (6, 7, 7, 19)
    assert crop.mask.shape == (3, 14)
    assert np.count_nonzero(crop.mask) == 12


def test_crops_do_not_retain_path_or_geometry_objects() -> None:
    labels = label_components(mask_with_hole(), min_patch_pixels=3).labels

    (crop,) = extract_component_crops(labels)

    assert set(crop.__dataclass_fields__) == {"label", "bbox", "mask"}


def test_component_crops_are_bucketed_by_bounded_pixel_work() -> None:
    crops = (
        ComponentCrop(1, (0, 0, 1, 2), np.ones((1, 2), dtype=bool)),
        ComponentCrop(2, (1, 0, 2, 3), np.ones((1, 3), dtype=bool)),
        ComponentCrop(3, (2, 0, 3, 4), np.ones((1, 4), dtype=bool)),
    )

    buckets = tuple(bucket_component_crops(crops, target_pixels=5))

    assert tuple(tuple(crop.label for crop in bucket) for bucket in buckets) == (
        (1, 2),
        (3,),
    )


def test_bucketing_streams_without_materializing_all_component_masks() -> None:
    first = ComponentCrop(1, (0, 0, 1, 2), np.ones((1, 2), dtype=bool))
    second = ComponentCrop(2, (1, 0, 2, 3), np.ones((1, 3), dtype=bool))

    def crops():
        yield first
        yield second
        raise AssertionError("bucketing consumed beyond first bounded unit")

    buckets = bucket_component_crops(crops(), target_pixels=2)

    assert tuple(crop.label for crop in next(buckets)) == (1,)
