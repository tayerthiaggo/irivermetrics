"""Tests for hydrofragments.io.cache_footprints: the HydroFragments-side
re-rasterize/verify consumer over hydroseason's persisted cache-footprint
metadata (task W2.3).

hydroseason.verify_cache_footprints is monkeypatched as a module attribute
for every test here (via ``monkeypatch.setattr(hydroseason,
"verify_cache_footprints", ..., raising=False)``), exactly like
``tests/io/test_dea.py`` monkeypatches ``hydroseason.open_wo_statistics`` --
this task's hydroseason-side implementation lives in a sibling worktree that
is not what this environment's installed ``hydroseason`` package resolves
to. The contract under test is entirely this consumer's: call hydroseason's
verification first (and propagate its failures unchanged), then
independently re-rasterize both persisted geometries on this side of the
boundary and assert the re-rasterized pixel counts agree.

Two properties are load-bearing for this task (mirrored from the
hydroseason-side test suite, but exercised here through the HydroFragments
consumer specifically):

1. A pruned and an unpruned cache's re-rasterized ``aoi_mask``/
   ``aoi_pixel_count`` are identical; ``analysis_mask``/
   ``analysis_pixel_count`` may differ.
2. Tampered geometry/digest is rejected -- here, because
   ``hydroseason.verify_cache_footprints`` itself would raise, and this
   consumer must propagate that failure rather than swallow it.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")

import hydroseason
from shapely import wkb
from shapely.geometry import box

from hydrofragments.io.cache_footprints import (
    CacheFootprintVerificationError,
    VerifiedCacheFootprints,
    open_verified_cache_footprints,
)

# A 120 m x 120 m grid at 30 m resolution -> 4x4 = 16 pixels.
_SHAPE = (4, 4)
_TRANSFORM = (30.0, 0.0, 0.0, 0.0, -30.0, 120.0)
_CRS = "EPSG:3577"


@dataclass(frozen=True)
class _FakeCacheFootprints:
    """A stand-in for hydroseason._io_wofs_zarr.CacheFootprints."""

    aoi_geometry_wkb_hex: str
    analysis_geometry_wkb_hex: str
    crs: str
    shape: tuple
    transform: tuple
    aoi_pixel_count: int
    analysis_pixel_count: int
    aoi_digest: str
    analysis_digest: str


def _wkb_hex(geom) -> str:
    return wkb.dumps(geom).hex()


def _full_aoi_geom():
    # Matches _TRANSFORM's origin (0, 120) with y decreasing by 30/row --
    # the raster covers x in [0, 120], y in [0, 120].
    return box(0.0, 0.0, 120.0, 120.0)


def _half_analysis_geom():
    """Left half of the full AOI: 60x120 m -> 2x4 = 8 pixels."""
    return box(0.0, 0.0, 60.0, 120.0)


def _footprints(*, analysis_geom, aoi_pixel_count=16, analysis_pixel_count=8):
    return _FakeCacheFootprints(
        aoi_geometry_wkb_hex=_wkb_hex(_full_aoi_geom()),
        analysis_geometry_wkb_hex=_wkb_hex(analysis_geom),
        crs=_CRS,
        shape=_SHAPE,
        transform=_TRANSFORM,
        aoi_pixel_count=aoi_pixel_count,
        analysis_pixel_count=analysis_pixel_count,
        aoi_digest="a" * 64,
        analysis_digest="b" * 64,
    )


def _handle():
    return SimpleNamespace(path="fake/store.zarr", identity="id", request_digest="digest")


def test_calls_hydroseason_verify_cache_footprints(monkeypatch):
    calls = []

    def fake_verify(handle):
        calls.append(handle)
        return _footprints(analysis_geom=_half_analysis_geom())

    monkeypatch.setattr(hydroseason, "verify_cache_footprints", fake_verify, raising=False)

    handle = _handle()
    open_verified_cache_footprints(handle)

    assert calls == [handle]


def test_returns_verified_cache_footprints_dataclass(monkeypatch):
    monkeypatch.setattr(
        hydroseason,
        "verify_cache_footprints",
        lambda handle: _footprints(analysis_geom=_half_analysis_geom()),
        raising=False,
    )

    result = open_verified_cache_footprints(_handle())

    assert isinstance(result, VerifiedCacheFootprints)
    assert result.aoi_mask.dtype == np.bool_
    assert result.analysis_mask.dtype == np.bool_
    assert result.aoi_mask.shape == _SHAPE
    assert result.analysis_mask.shape == _SHAPE


def test_aoi_pixel_count_identical_between_pruned_and_unpruned(monkeypatch):
    """The central W2.3 correctness property, exercised through this
    consumer: an unpruned cache (analysis footprint == full AOI) and a
    pruned cache (analysis footprint is a strict subset) must yield the
    SAME re-rasterized aoi_pixel_count, while analysis_pixel_count differs."""

    def fake_verify_unpruned(handle):
        return _footprints(analysis_geom=_full_aoi_geom(), analysis_pixel_count=16)

    def fake_verify_pruned(handle):
        return _footprints(analysis_geom=_half_analysis_geom(), analysis_pixel_count=8)

    monkeypatch.setattr(hydroseason, "verify_cache_footprints", fake_verify_unpruned, raising=False)
    unpruned = open_verified_cache_footprints(_handle())

    monkeypatch.setattr(hydroseason, "verify_cache_footprints", fake_verify_pruned, raising=False)
    pruned = open_verified_cache_footprints(_handle())

    assert unpruned.aoi_pixel_count == pruned.aoi_pixel_count == 16
    assert np.array_equal(unpruned.aoi_mask, pruned.aoi_mask)

    assert unpruned.analysis_pixel_count == 16
    assert pruned.analysis_pixel_count == 8
    assert unpruned.analysis_pixel_count != pruned.analysis_pixel_count
    assert not np.array_equal(unpruned.analysis_mask, pruned.analysis_mask)


def test_analysis_mask_is_true_subset_of_aoi_mask_when_pruned(monkeypatch):
    monkeypatch.setattr(
        hydroseason,
        "verify_cache_footprints",
        lambda handle: _footprints(analysis_geom=_half_analysis_geom()),
        raising=False,
    )

    result = open_verified_cache_footprints(_handle())

    assert np.all(result.aoi_mask[result.analysis_mask])
    assert result.analysis_mask.sum() < result.aoi_mask.sum()


def test_hydroseason_verification_failure_propagates_unchanged(monkeypatch):
    """A cache hydroseason itself does not trust (missing metadata, corrupted
    WKB, mismatched digest/count -- task W2.3 Step 2's tamper detection)
    must never be silently accepted here; the ValueError must propagate."""

    def fake_verify_raises(handle):
        raise ValueError("cache footprint 'analysis' digest mismatch at fake/store.zarr")

    monkeypatch.setattr(hydroseason, "verify_cache_footprints", fake_verify_raises, raising=False)

    with pytest.raises(ValueError, match="digest mismatch"):
        open_verified_cache_footprints(_handle())


def test_hydroseason_missing_metadata_failure_propagates_unchanged(monkeypatch):
    def fake_verify_raises(handle):
        raise FileNotFoundError("no manifest found")

    monkeypatch.setattr(hydroseason, "verify_cache_footprints", fake_verify_raises, raising=False)

    with pytest.raises(FileNotFoundError):
        open_verified_cache_footprints(_handle())


def test_local_pixel_count_mismatch_is_independently_rejected(monkeypatch):
    """Even when hydroseason's own verification passes, a persisted
    aoi_pixel_count that disagrees with what THIS repository's rasterizer
    actually produces must be rejected -- this is the second, independent
    verification layer this module adds on top of hydroseason's own check."""
    monkeypatch.setattr(
        hydroseason,
        "verify_cache_footprints",
        # aoi_pixel_count is wrong on purpose: the true rasterized full-AOI
        # box at this grid/transform is 16 pixels, not 999.
        lambda handle: _footprints(analysis_geom=_half_analysis_geom(), aoi_pixel_count=999),
        raising=False,
    )

    with pytest.raises(CacheFootprintVerificationError, match="aoi_pixel_count"):
        open_verified_cache_footprints(_handle())


def test_local_analysis_pixel_count_mismatch_is_independently_rejected(monkeypatch):
    monkeypatch.setattr(
        hydroseason,
        "verify_cache_footprints",
        lambda handle: _footprints(analysis_geom=_half_analysis_geom(), analysis_pixel_count=999),
        raising=False,
    )

    with pytest.raises(CacheFootprintVerificationError, match="analysis_pixel_count"):
        open_verified_cache_footprints(_handle())


def test_verified_footprints_carry_crs_shape_transform_and_digests(monkeypatch):
    monkeypatch.setattr(
        hydroseason,
        "verify_cache_footprints",
        lambda handle: _footprints(analysis_geom=_half_analysis_geom()),
        raising=False,
    )

    result = open_verified_cache_footprints(_handle())

    assert result.crs == _CRS
    assert tuple(result.shape) == _SHAPE
    assert tuple(result.transform) == _TRANSFORM
    assert result.aoi_digest == "a" * 64
    assert result.analysis_digest == "b" * 64
