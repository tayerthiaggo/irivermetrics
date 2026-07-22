from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.spatial.zones import build_zones


def test_no_drainage_emits_only_zones_2_to_4() -> None:
    occurrence = np.array([[0.9, 0.5], [0.1, 0.01]])
    result = build_zones(
        occurrence,
        max_wet_mask=np.ones((2, 2), dtype=bool),
        valid_count=np.full((2, 2), 20),
        t_persist=0.5,
        t_season=0.1,
    )

    assert result.emitted_zones == (2, 3, 4)
    assert set(np.unique(result.mask)) == {2, 3, 4}
    assert result.mask.tolist() == [[2, 3], [3, 4]]
    assert result.has_zone_1 is False


def test_drainage_zone_1_replaces_adjacent_high_frequency_pixels_only() -> None:
    occurrence = np.array(
        [
            [0.9, 0.9, 0.0],
            [0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0],
        ]
    )
    drainage = np.zeros((3, 3), dtype=bool)
    drainage[1, 1] = True

    result = build_zones(
        occurrence,
        max_wet_mask=occurrence > 0,
        valid_count=np.full((3, 3), 20),
        drainage_mask=drainage,
        t_persist=0.5,
        t_season=0.1,
    )

    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.mask[1, 1] == 1
    assert result.mask[0, 0] == 1
    assert result.mask[0, 1] == 1
    assert result.mask[1, 0] == 3
    assert result.mask[1, 2] == 3


def test_zone_inputs_must_share_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_zones(
            np.ones((2, 2)),
            max_wet_mask=np.ones((3, 3), dtype=bool),
            valid_count=np.ones((2, 2)),
        )


def test_pixels_below_min_valid_obs_are_not_zoned() -> None:
    result = build_zones(
        np.array([[0.9, 0.2]]),
        max_wet_mask=np.ones((1, 2), dtype=bool),
        valid_count=np.array([[19, 20]]),
        min_valid_obs=20,
    )
    assert result.mask.tolist() == [[0, 3]]


def test_morphology_proxy_argument_does_not_exist() -> None:
    with pytest.raises(TypeError):
        build_zones(
            np.ones((2, 2)),
            max_wet_mask=np.ones((2, 2), dtype=bool),
            valid_count=np.full((2, 2), 20),
            morphology_proxy=True,
        )
