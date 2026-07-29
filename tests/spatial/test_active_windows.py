"""W3.1: WaterCube aoi_mask/analysis_mask fields and independent_active_windows().

Two families of coverage:

1. Mask-alignment / backward-compatibility tests: every existing WaterCube
   constructor path must keep defaulting both new masks to all-true over the
   spatial grid (unpruned behaviour unchanged), and explicitly supplied masks
   must be validated for alignment against ``water``.
2. Window-equivalence property tests: the central correctness property this
   task must prove -- concatenated per-window patch properties (measured
   independently on each ``AnalysisWindow``'s crop) must equal patch
   properties measured on the FULL mask in one pass. Exercised over random
   masks plus the named thin/diagonal-channel edge cases, using the SAME
   ``analyze_patch_bundle`` measurement path production code actually uses
   (see hydrofragments/compat.py's per-month loop).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hydrofragments.api import open_water_cube
from hydrofragments.metrics.patches import analyze_patch_bundle
from hydrofragments.models import WaterCube
from hydrofragments.spatial.active_windows import (
    AnalysisWindow,
    independent_active_windows,
)

# ---------------------------------------------------------------------------
# 1. WaterCube aoi_mask / analysis_mask defaults and validation
# ---------------------------------------------------------------------------


def _water_cube(shape: tuple[int, int, int] = (2, 6, 6)) -> WaterCube:
    water = xr.DataArray(
        np.zeros(shape, dtype=bool),
        dims=("time", "y", "x"),
    )
    valid_obs = xr.DataArray(
        np.ones(shape, dtype=bool),
        dims=("time", "y", "x"),
    )
    return WaterCube(water=water, valid_obs=valid_obs, source="test", cadence="monthly")


def test_watercube_defaults_both_masks_to_all_true_over_spatial_grid():
    cube = _water_cube(shape=(3, 5, 7))

    assert cube.aoi_mask is not None
    assert cube.analysis_mask is not None
    assert cube.aoi_mask.dims == ("y", "x")
    assert cube.analysis_mask.dims == ("y", "x")
    assert cube.aoi_mask.shape == (5, 7)
    assert cube.analysis_mask.shape == (5, 7)
    assert bool(cube.aoi_mask.all())
    assert bool(cube.analysis_mask.all())


def test_watercube_default_masks_are_boolean_dtype():
    cube = _water_cube()
    assert cube.aoi_mask.dtype == bool
    assert cube.analysis_mask.dtype == bool


def test_watercube_accepts_explicit_aligned_masks():
    shape = (2, 4, 4)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))
    valid_obs = xr.DataArray(np.ones(shape, dtype=bool), dims=("time", "y", "x"))
    aoi_mask = xr.DataArray(np.ones((4, 4), dtype=bool), dims=("y", "x"))
    analysis_mask = xr.DataArray(
        np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        dims=("y", "x"),
    )

    cube = WaterCube(
        water=water,
        valid_obs=valid_obs,
        source="test",
        cadence="monthly",
        aoi_mask=aoi_mask,
        analysis_mask=analysis_mask,
    )

    assert bool(cube.aoi_mask.all())
    assert int(cube.analysis_mask.sum()) == 4


def test_watercube_rejects_misaligned_analysis_mask():
    shape = (2, 4, 4)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))
    valid_obs = xr.DataArray(np.ones(shape, dtype=bool), dims=("time", "y", "x"))
    wrong_shape_mask = xr.DataArray(np.ones((5, 5), dtype=bool), dims=("y", "x"))

    with pytest.raises(ValueError, match="align"):
        WaterCube(
            water=water,
            valid_obs=valid_obs,
            source="test",
            cadence="monthly",
            analysis_mask=wrong_shape_mask,
        )


def test_watercube_rejects_misaligned_aoi_mask():
    shape = (2, 4, 4)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))
    valid_obs = xr.DataArray(np.ones(shape, dtype=bool), dims=("time", "y", "x"))
    wrong_dims_mask = xr.DataArray(np.ones((4, 4), dtype=bool), dims=("row", "col"))

    with pytest.raises(ValueError, match="align"):
        WaterCube(
            water=water,
            valid_obs=valid_obs,
            source="test",
            cadence="monthly",
            aoi_mask=wrong_dims_mask,
        )


def test_open_water_cube_defaults_masks_to_all_true():
    """open_water_cube() (the public entry point every real caller uses) must
    also default both masks when the caller supplies neither -- exercising
    the actual public API surface, not just the WaterCube dataclass directly.
    """
    shape = (2, 5, 5)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))

    cube = open_water_cube(water, input_kind="generic_binary")

    assert bool(cube.aoi_mask.all())
    assert bool(cube.analysis_mask.all())
    assert cube.aoi_mask.shape == (5, 5)


def test_open_water_cube_passes_through_explicit_masks():
    shape = (2, 5, 5)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))
    aoi_mask = xr.DataArray(np.ones((5, 5), dtype=bool), dims=("y", "x"))
    analysis_mask = xr.DataArray(
        np.zeros((5, 5), dtype=bool), dims=("y", "x")
    )
    analysis_mask[0:2, 0:2] = True

    cube = open_water_cube(
        water,
        input_kind="generic_binary",
        aoi_mask=aoi_mask,
        analysis_mask=analysis_mask,
    )

    assert bool(cube.aoi_mask.all())
    assert int(cube.analysis_mask.sum()) == 4


def test_watercube_legacy_construction_without_masks_is_unaffected():
    """Regression: constructing WaterCube exactly as every pre-existing
    caller does (no aoi_mask/analysis_mask kwargs at all) must keep working
    and produce all-true masks -- i.e. this feature is purely additive.
    """
    shape = (4, 3, 3)
    water = xr.DataArray(np.zeros(shape, dtype=bool), dims=("time", "y", "x"))
    valid_obs = xr.DataArray(np.ones(shape, dtype=bool), dims=("time", "y", "x"))

    cube = WaterCube(water=water, valid_obs=valid_obs, source="legacy", cadence="monthly")

    assert bool(cube.aoi_mask.all())
    assert bool(cube.analysis_mask.all())


# ---------------------------------------------------------------------------
# 2. independent_active_windows(): window-equivalence property tests
# ---------------------------------------------------------------------------

_A_TOTAL_M2 = 1_000_000.0  # arbitrary fixed placeholder denominator for equality checks
_PIXEL_SIZE_M = 30.0
_MIN_PATCH_PIXELS = 1  # keep every component so window-splitting is the only variable


def _patch_result(mask: np.ndarray, *, connectivity: int):
    core, _ = analyze_patch_bundle(
        mask,
        pixel_size_m=_PIXEL_SIZE_M,
        a_total_m2=_A_TOTAL_M2,
        connectivity=connectivity,
        min_patch_pixels=_MIN_PATCH_PIXELS,
        include_width=False,
    )
    return core


def _combined_patch_result_from_windows(
    wet_mask: np.ndarray,
    windows: "list[AnalysisWindow]",
    *,
    connectivity: int,
):
    """Independently measure each window's crop and concatenate raw patch
    properties before ONE final reduction -- mirroring what a correct W4.3
    measure/reduce split would do, and exactly what this task's correctness
    property must hold for (a naive per-window compute_patch_metrics() call
    would double- or mis-count boundary effects, whereas concatenating
    PatchProperties before one aggregation does not).
    """
    from hydrofragments.metrics.patches import compute_patch_metrics
    from hydrofragments.patches import (
        bucket_component_crops,
        iter_component_crops,
        label_components,
        measure_components,
    )

    all_properties = []
    for window in windows:
        row0, col0, row1, col1 = window.bbox
        crop = wet_mask[row0:row1, col0:col1]
        if not crop.any():
            continue
        labels = label_components(
            crop, connectivity=connectivity, min_patch_pixels=_MIN_PATCH_PIXELS
        )
        crops = iter_component_crops(labels.labels)
        for bucket in bucket_component_crops(crops, target_pixels=1_000_000):
            all_properties.extend(
                measure_components(bucket, pixel_size_m=_PIXEL_SIZE_M, include_width=False)
            )

    return compute_patch_metrics(all_properties, a_total_m2=_A_TOTAL_M2)


def _assert_windows_equivalent_to_full_mask(
    analysis_mask: np.ndarray, wet_mask: np.ndarray, *, connectivity: int
):
    """The core correctness property: concatenated window patch properties
    (using windows derived from ``analysis_mask``) must equal patch
    properties computed on the full ``wet_mask`` in one pass -- proving no
    retained wet component was ever split across window boundaries.
    """
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))
    windows = list(
        independent_active_windows(mask_da, connectivity=connectivity, halo_pixels=1)
    )
    assert windows, "expected at least one window for a non-empty analysis_mask"

    full = _patch_result(wet_mask, connectivity=connectivity)
    combined = _combined_patch_result_from_windows(wet_mask, windows, connectivity=connectivity)

    assert combined.number_of_pools == full.number_of_pools
    assert combined.n_water_pixels == full.n_water_pixels
    assert combined.lpi == pytest.approx(full.lpi, nan_ok=True)
    assert combined.awre == pytest.approx(full.awre, nan_ok=True)
    assert combined.awmsi == pytest.approx(full.awmsi, nan_ok=True)


def test_windows_partition_disjoint_components_without_splitting():
    analysis_mask = np.zeros((20, 20), dtype=bool)
    analysis_mask[2:5, 2:5] = True
    analysis_mask[14:18, 14:18] = True
    wet_mask = analysis_mask.copy()

    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=8)


def test_windows_equivalence_thin_channel():
    """A single-pixel-wide winding channel: any bbox-only windowing without
    halo/merge care could clip or split the channel's connectivity.
    """
    analysis_mask = np.zeros((30, 30), dtype=bool)
    # A thin zig-zag channel one pixel wide.
    for i in range(30):
        analysis_mask[i, (i * 3) % 28 : (i * 3) % 28 + 1] = True
    wet_mask = analysis_mask.copy()

    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=8)
    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=4)


def test_windows_equivalence_diagonal_channel_connectivity_8():
    """A strictly diagonal chain of pixels: connected under 8-connectivity
    only. Must not be split into separate windows under connectivity=8.
    """
    analysis_mask = np.zeros((16, 16), dtype=bool)
    for i in range(14):
        analysis_mask[i, i] = True
    wet_mask = analysis_mask.copy()

    _assert_windows_equivalence_or_raise = _assert_windows_equivalent_to_full_mask
    _assert_windows_equivalence_or_raise(analysis_mask, wet_mask, connectivity=8)


def test_windows_equivalence_diagonal_pixels_disconnect_under_connectivity_4():
    """Same diagonal chain, but under connectivity=4 each pixel is its own
    component -- windows may legitimately separate them, and the property
    must still hold (every single-pixel component measured independently).
    """
    analysis_mask = np.zeros((16, 16), dtype=bool)
    for i in range(14):
        analysis_mask[i, i] = True
    wet_mask = analysis_mask.copy()

    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=4)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_windows_equivalence_random_masks(seed: int):
    rng = np.random.default_rng(seed)
    analysis_mask = rng.random((40, 40)) > 0.7
    wet_mask = analysis_mask.copy()

    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=8)
    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=4)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_windows_equivalence_random_masks_with_halo_2(seed: int):
    """Also prove equivalence holds with a larger halo, since a bigger halo
    is more likely to trigger the overlap-merge path.
    """
    rng = np.random.default_rng(seed)
    analysis_mask = rng.random((48, 48)) > 0.75
    wet_mask = analysis_mask.copy()

    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))
    windows = list(
        independent_active_windows(mask_da, connectivity=8, halo_pixels=2)
    )
    if not windows:
        return
    full = _patch_result(wet_mask, connectivity=8)
    combined = _combined_patch_result_from_windows(wet_mask, windows, connectivity=8)
    assert combined.number_of_pools == full.number_of_pools
    assert combined.n_water_pixels == full.n_water_pixels
    assert combined.lpi == pytest.approx(full.lpi, nan_ok=True)


def test_windows_equivalence_wet_mask_is_subset_of_analysis_mask():
    """analysis_mask is a conservative POTENTIAL-water footprint -- the
    actual monthly wet_mask is typically a strict subset. Windows derived
    from analysis_mask must still be valid (no splitting) for any wet_mask
    subset of it, not just wet_mask == analysis_mask.
    """
    analysis_mask = np.zeros((25, 25), dtype=bool)
    analysis_mask[3:10, 3:10] = True
    analysis_mask[15:22, 15:22] = True

    rng = np.random.default_rng(99)
    wet_mask = analysis_mask & (rng.random((25, 25)) > 0.4)

    _assert_windows_equivalent_to_full_mask(analysis_mask, wet_mask, connectivity=8)


def test_empty_analysis_mask_returns_no_windows():
    mask_da = xr.DataArray(np.zeros((10, 10), dtype=bool), dims=("y", "x"))
    windows = list(independent_active_windows(mask_da, connectivity=8))
    assert windows == []


def test_full_analysis_mask_returns_single_window_covering_whole_grid():
    mask_da = xr.DataArray(np.ones((10, 12), dtype=bool), dims=("y", "x"))
    windows = list(independent_active_windows(mask_da, connectivity=8))
    assert len(windows) == 1
    assert windows[0].bbox == (0, 0, 10, 12)


def test_window_ids_are_stable_and_row_major_ordered():
    analysis_mask = np.zeros((20, 20), dtype=bool)
    analysis_mask[14:16, 14:16] = True  # bottom-right component
    analysis_mask[1:3, 1:3] = True  # top-left component
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    windows = list(
        independent_active_windows(
            mask_da, connectivity=8, halo_pixels=0, align_pixels=1
        )
    )

    assert len(windows) == 2
    # Row-major: the top-left component's window must sort first.
    assert windows[0].bbox[0] <= windows[1].bbox[0]
    ids = [w.window_id for w in windows]
    assert len(set(ids)) == len(ids)


def test_close_components_merge_when_halo_overlaps():
    """Two components close enough that expanding by halo_pixels causes
    their padded boxes to overlap must be merged into ONE window -- not
    returned as two overlapping windows (which would double-count the
    overlap region if measured independently per window).
    """
    analysis_mask = np.zeros((20, 20), dtype=bool)
    analysis_mask[5, 5] = True
    analysis_mask[5, 8] = True  # 2 px gap; halo_pixels=2 makes boxes touch/overlap
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    windows = list(independent_active_windows(mask_da, connectivity=8, halo_pixels=2))

    assert len(windows) == 1


def test_far_apart_components_remain_separate_windows():
    analysis_mask = np.zeros((100, 100), dtype=bool)
    analysis_mask[1, 1] = True
    analysis_mask[90, 90] = True
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    windows = list(
        independent_active_windows(
            mask_da, connectivity=8, halo_pixels=1, align_pixels=1
        )
    )

    assert len(windows) == 2


def test_analysis_mask_must_be_2d():
    mask_da = xr.DataArray(np.ones((2, 4, 4), dtype=bool), dims=("time", "y", "x"))
    with pytest.raises(ValueError):
        list(independent_active_windows(mask_da, connectivity=8))


def test_connectivity_must_be_4_or_8():
    mask_da = xr.DataArray(np.ones((4, 4), dtype=bool), dims=("y", "x"))
    with pytest.raises(ValueError):
        list(independent_active_windows(mask_da, connectivity=6))  # type: ignore[arg-type]


def test_windows_stay_within_raster_bounds():
    analysis_mask = np.zeros((10, 10), dtype=bool)
    analysis_mask[0, 0] = True
    analysis_mask[9, 9] = True
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    windows = list(
        independent_active_windows(mask_da, connectivity=8, halo_pixels=5)
    )
    for window in windows:
        row0, col0, row1, col1 = window.bbox
        assert 0 <= row0 < row1 <= 10
        assert 0 <= col0 < col1 <= 10


def test_align_pixels_snaps_window_bounds_to_grid():
    """align_pixels rounds each window's bounds outward to multiples of
    align_pixels (clamped to the raster), without breaking equivalence.
    """
    analysis_mask = np.zeros((64, 64), dtype=bool)
    analysis_mask[30:34, 30:34] = True
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    windows = list(
        independent_active_windows(
            mask_da, connectivity=8, halo_pixels=1, align_pixels=16
        )
    )
    assert len(windows) == 1
    row0, col0, row1, col1 = windows[0].bbox
    assert row0 % 16 == 0
    assert col0 % 16 == 0
    assert row1 % 16 == 0 or row1 == 64
    assert col1 % 16 == 0 or col1 == 64
