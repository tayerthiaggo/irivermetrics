from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hydrofragments.config import ValidityConfig, ZonesConfig
from hydrofragments.io.dea import WoStatistics
from hydrofragments.spatial.zones import ZoneResult, build_zones, zones_from_wo_statistics


def test_no_drainage_emits_only_zones_2_to_4() -> None:
    # Percent-scale occurrence (0-100), matching compute_occurrence's contract.
    occurrence = np.array([[90.0, 50.0], [10.0, 1.0]])
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
            [90.0, 90.0, 0.0],
            [20.0, 20.0, 20.0],
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
        np.array([[90.0, 20.0]]),
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


# --- Percentage-boundary tests (W1.2 threshold-unit fix) --------------------
#
# build_zones receives occurrence as PERCENT (0-100), matching
# compute_occurrence/WoStatistics.frequency's contract throughout the rest of
# the codebase. t_persist/t_season are validated by config.py as FRACTIONS in
# [0, 1] (default t_persist=0.50, t_season=0.10). build_zones must convert
# the fraction thresholds to percent ONCE at its own boundary before
# comparing against the percent-scale occurrence array. These tests pin every
# boundary explicitly so the conversion can never silently regress.


def _single_pixel_zone(occurrence_value: float, *, t_persist: float = 0.50, t_season: float = 0.10) -> int:
    result = build_zones(
        np.array([[occurrence_value]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[20]]),
        t_persist=t_persist,
        t_season=t_season,
    )
    return int(result.mask[0, 0])


def test_boundary_below_t_season_is_zone_4() -> None:
    # 9.9% < 10% (t_season fraction 0.10 -> percent 10.0)
    assert _single_pixel_zone(9.9) == 4


def test_boundary_at_t_season_is_zone_3() -> None:
    # 10% == t_season boundary; build_zones treats >= t_season as zone 3.
    assert _single_pixel_zone(10.0) == 3


def test_boundary_mid_range_45_percent_is_zone_3_under_default_t_persist() -> None:
    # The plan's explicit required boundary: 45% pixel under default
    # t_persist=0.50 (-> 50.0 percent after conversion) must land in zone 3.
    assert _single_pixel_zone(45.0) == 3


def test_boundary_at_t_persist_is_zone_3() -> None:
    # 50% == t_persist boundary; build_zones treats <= t_persist as zone 3.
    assert _single_pixel_zone(50.0) == 3


def test_boundary_above_t_persist_is_zone_2() -> None:
    # 50.1% > 50% (t_persist fraction 0.50 -> percent 50.0)
    assert _single_pixel_zone(50.1) == 2


def test_boundary_support_floor_excludes_pixel_from_zoning() -> None:
    # A pixel with valid_count below min_valid_obs must be unzoned (0),
    # regardless of its percent-scale occurrence value.
    result = build_zones(
        np.array([[90.0]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[19]]),
        min_valid_obs=20,
    )
    assert result.mask[0, 0] == 0


def test_boundary_nodata_occurrence_excludes_pixel_from_zoning() -> None:
    # NaN occurrence (nodata) must never be assigned to a zone.
    result = build_zones(
        np.array([[np.nan]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[20]]),
    )
    assert result.mask[0, 0] == 0


def test_boundary_no_wet_pixels_excludes_pixel_from_zoning() -> None:
    # max_wet_mask False means the pixel was never observed wet; even a high
    # percent-scale occurrence value must not be zoned.
    result = build_zones(
        np.array([[90.0]]),
        max_wet_mask=np.array([[False]]),
        valid_count=np.array([[20]]),
    )
    assert result.mask[0, 0] == 0


def test_threshold_fraction_bounds_are_still_enforced_after_conversion() -> None:
    # config.py validates t_season < t_persist as fractions in [0, 1]; that
    # invariant must still be checked before any percent conversion happens.
    with pytest.raises(ValueError, match="t_season"):
        build_zones(
            np.array([[50.0]]),
            max_wet_mask=np.array([[True]]),
            valid_count=np.array([[20]]),
            t_persist=0.5,
            t_season=0.5,
        )


# --- ZoneResult.source -------------------------------------------------------


def test_zone_result_source_defaults_to_occurrence() -> None:
    result = build_zones(
        np.array([[90.0]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[20]]),
    )
    assert result.source == "occurrence"


def test_zone_result_source_is_overridable() -> None:
    result = ZoneResult(mask=np.zeros((1, 1), dtype=np.uint8), emitted_zones=(), has_zone_1=False, source="custom")
    assert result.source == "custom"


# --- zones_from_wo_statistics adapter ---------------------------------------


def _wo_statistics(
    *,
    frequency: np.ndarray,
    count_wet: np.ndarray,
    count_clear: np.ndarray,
    product: str = "ga_ls_wo_fq_myear_3",
) -> WoStatistics:
    return WoStatistics(
        frequency=frequency,
        count_wet=count_wet,
        count_clear=count_clear,
        product=product,
        version="1.2.3",
        crs="EPSG:3577",
        time_span="2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        provenance={"product": product},
    )


def _config(*, t_persist: float = 0.50, t_season: float = 0.10, min_valid_obs: int = 20):
    return SimpleNamespace(
        zones=ZonesConfig(t_persist=t_persist, t_season=t_season),
        validity=ValidityConfig(min_valid_obs=min_valid_obs),
    )


def test_zones_from_wo_statistics_maps_fields_per_plan_spec() -> None:
    stats = _wo_statistics(
        frequency=np.array([[90.0, 45.0], [10.0, 5.0]]),
        count_wet=np.array([[5, 5], [1, 0]]),
        count_clear=np.array([[20, 20], [20, 20]]),
    )

    result = zones_from_wo_statistics(stats, config=_config())

    # frequency -> occurrence, count_clear -> valid_count,
    # (count_wet > 0) -> max_wet_mask
    assert result.mask.tolist() == [[2, 3], [3, 0]]
    assert result.has_zone_1 is False


def test_zones_from_wo_statistics_stamps_source_with_stats_product() -> None:
    stats = _wo_statistics(
        frequency=np.array([[90.0]]),
        count_wet=np.array([[5]]),
        count_clear=np.array([[20]]),
        product="ga_ls_wo_fq_myear_3",
    )

    result = zones_from_wo_statistics(stats, config=_config())

    assert result.source == "ga_ls_wo_fq_myear_3"


def test_zones_from_wo_statistics_passes_config_thresholds_and_support_floor() -> None:
    stats = _wo_statistics(
        frequency=np.array([[9.9, 10.0]]),
        count_wet=np.array([[1, 1]]),
        count_clear=np.array([[5, 20]]),
    )

    # min_valid_obs=20 excludes the first pixel (count_clear=5) entirely;
    # the second pixel sits exactly at the (converted) t_season boundary.
    result = zones_from_wo_statistics(stats, config=_config(min_valid_obs=20))

    assert result.mask.tolist() == [[0, 3]]


def test_zones_from_wo_statistics_passes_drainage_mask_through() -> None:
    stats = _wo_statistics(
        frequency=np.array([[90.0, 90.0], [20.0, 20.0]]),
        count_wet=np.array([[5, 5], [5, 5]]),
        count_clear=np.array([[20, 20], [20, 20]]),
    )
    drainage = np.array([[True, False], [False, False]])

    result = zones_from_wo_statistics(stats, config=_config(), drainage_mask=drainage)

    assert result.has_zone_1 is True
    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.mask[0, 0] == 1


def test_zones_from_wo_statistics_accepts_dask_backed_stats() -> None:
    da = pytest.importorskip("dask.array")

    frequency = da.from_array(np.array([[90.0, 45.0], [10.0, 5.0]]), chunks=(1, 2))
    count_wet = da.from_array(np.array([[5, 5], [1, 0]]), chunks=(1, 2))
    count_clear = da.from_array(np.array([[20, 20], [20, 20]]), chunks=(1, 2))
    stats = _wo_statistics(frequency=frequency, count_wet=count_wet, count_clear=count_clear)

    result = zones_from_wo_statistics(stats, config=_config())

    assert result.mask.tolist() == [[2, 3], [3, 0]]


def test_zones_from_wo_statistics_accepts_dask_backed_xarray_stats() -> None:
    # Production shape: WoStatistics fields are xr.DataArray wrapping Dask
    # arrays (per open_wo_statistics_for_zoning), not raw dask arrays --
    # confirm the adapter doesn't need any special handling for that either.
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")

    frequency = xr.DataArray(
        da.from_array(np.array([[90.0, 45.0], [10.0, 5.0]]), chunks=(1, 2)),
        dims=("y", "x"),
    )
    count_wet = xr.DataArray(
        da.from_array(np.array([[5, 5], [1, 0]]), chunks=(1, 2)), dims=("y", "x")
    )
    count_clear = xr.DataArray(
        da.from_array(np.array([[20, 20], [20, 20]]), chunks=(1, 2)), dims=("y", "x")
    )
    stats = _wo_statistics(frequency=frequency, count_wet=count_wet, count_clear=count_clear)

    result = zones_from_wo_statistics(stats, config=_config())

    assert result.mask.tolist() == [[2, 3], [3, 0]]
