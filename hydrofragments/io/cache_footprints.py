"""Consume hydroseason's persisted cache-footprint metadata (task W2.3).

hydroseason owns writing full-AOI and analysis-footprint geometry, grid
transform, pixel counts, and digests into a WOfS cache store's root
``manifest.json`` (:mod:`hydroseason._io_wofs_zarr`'s
``record_cache_footprints``/``read_cache_footprints``/
``verify_cache_footprints``, exposed publicly as
``hydroseason.read_cache_footprints``/``hydroseason.verify_cache_footprints``).
This module's job is the HydroFragments side of that contract: given a
:class:`hydroseason.WOfSCacheHandle` (as returned by
``hydroseason.acquire_wofs_cache``/looked up via
``hydroseason.open_completed_mask_cache``), independently re-rasterize both
persisted geometries onto the cache's own grid and verify the result before
handing back the two masks/counts a caller needs.

This is deliberately a SECOND, independent verification layer on top of
``hydroseason.verify_cache_footprints`` (which already re-rasterizes from the
persisted WKB and cross-checks the persisted pixel count/digest) -- this
module re-rasterizes AGAIN, on this side of the repository boundary, using
this repository's own ``rasterio``/``shapely`` versions, and additionally
asserts the resulting boolean masks' true-pixel counts match the persisted
counts. A cache that fails hydroseason's own verification is never even
handed to this module's rasterizer (see :func:`open_verified_cache_footprints`,
which calls ``hydroseason.verify_cache_footprints`` first and propagates its
``ValueError``/``FileNotFoundError`` unchanged).

Scope note: this module produces the *masks and counts* a later task
(W3.1, not implemented here) would attach to ``WaterCube`` as its
``aoi_mask``/``analysis_mask`` fields -- it does not itself touch
``WaterCube`` or build independent active windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import geometry_mask

import hydroseason


@dataclass(frozen=True)
class VerifiedCacheFootprints:
    """Re-rasterized, verified full-AOI and analysis-footprint masks.

    ``aoi_mask`` is the fixed reference-area mask APSEC/LPI/reference-area
    denominators must use (global constraint: "APSEC/LPI/reference-area
    denominators remain full catchment ``aoi_mask``"). ``analysis_mask`` is
    the conservative potential-water footprint mask approved as the monthly
    coverage denominator (global constraint: "Monthly coverage denominator is
    approved as conservative potential-water ``analysis_mask``, not full
    catchment"). Both are boolean 2-D ``numpy`` arrays on the cache's own
    grid (``shape``/``transform``/``crs``).

    ``aoi_pixel_count``/``analysis_pixel_count`` are each mask's exact
    ``True`` pixel count, independently recomputed from the re-rasterized
    array here (NOT merely copied from the persisted metadata) -- see
    :func:`open_verified_cache_footprints` for the assertion that this count
    matches what hydroseason persisted.
    """

    aoi_mask: "np.ndarray"
    analysis_mask: "np.ndarray"
    crs: str
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]
    aoi_pixel_count: int
    analysis_pixel_count: int
    aoi_digest: str
    analysis_digest: str


class CacheFootprintVerificationError(ValueError):
    """Raised when a re-rasterized mask does not match hydroseason's persisted counts.

    Distinct from whatever ``hydroseason.verify_cache_footprints`` itself
    raises (propagated unchanged as a plain ``ValueError``/
    ``FileNotFoundError``, see :func:`open_verified_cache_footprints`): this
    is raised only when hydroseason's own verification already passed, but
    THIS repository's independent re-rasterization still disagrees --
    e.g. a ``rasterio``/``shapely`` version skew between the two
    repositories' environments that changes rasterization behaviour at the
    boundary. Either way, a disagreement is never silently accepted.
    """


def _rasterize_mask(
    wkb_hex: str, *, shape: tuple[int, int], transform: tuple[float, ...]
) -> "np.ndarray":
    """Re-rasterize persisted canonical WKB hex onto ``shape``/``transform``.

    Uses ``rasterio.features.geometry_mask`` with ``invert=True``
    (``True`` inside the geometry) and ``all_touched=True`` -- the same
    rasterization primitive and touch convention hydroseason's own
    ``_rasterize_pixel_count``/``_inside_aoi_mask_like`` use, so a geometry
    persisted by hydroseason rasterizes to the identical pixel set on this
    side of the boundary (modulo any ``rasterio``/``shapely`` version skew,
    which :func:`open_verified_cache_footprints`'s pixel-count assertion
    would then catch).
    """
    from shapely import wkb

    geometry = wkb.loads(bytes.fromhex(wkb_hex))
    height, width = int(shape[0]), int(shape[1])
    affine_transform = Affine(*transform)
    return geometry_mask(
        [geometry],
        out_shape=(height, width),
        transform=affine_transform,
        invert=True,
        all_touched=True,
    )


def open_verified_cache_footprints(handle: Any) -> VerifiedCacheFootprints:
    """Open, re-rasterize, and verify a hydroseason cache's AOI/analysis footprints.

    ``handle`` is a :class:`hydroseason.WOfSCacheHandle` (as returned by
    ``hydroseason.acquire_wofs_cache`` or resolved via
    ``hydroseason.open_completed_mask_cache``'s companion lookups).

    Steps:

    1. Call ``hydroseason.verify_cache_footprints(handle)`` -- hydroseason's
       own tamper check: it re-rasterizes each persisted geometry from its
       WKB and cross-checks both the digest and the pixel count it
       persisted. Any ``ValueError``/``FileNotFoundError`` this raises
       (missing metadata, corrupted WKB, mismatched digest or pixel count)
       propagates unchanged -- this function never proceeds past a cache
       hydroseason itself does not trust.
    2. Independently re-rasterize both persisted geometries again, on this
       side of the repository boundary (:func:`_rasterize_mask`), producing
       the actual boolean masks this module hands back.
    3. Assert each re-rasterized mask's true-pixel count matches the
       (already-verified) persisted pixel count. This is a second,
       independent check -- not a redundant repeat of step 1 -- because it
       exercises THIS repository's own ``rasterio``/``shapely``
       installation rather than hydroseason's; raises
       :class:`CacheFootprintVerificationError` on any disagreement.

    Returns a :class:`VerifiedCacheFootprints` carrying both masks, ready
    for a caller to attach as ``aoi_mask``/``analysis_mask`` (task W3.1,
    not implemented here).
    """
    footprints = hydroseason.verify_cache_footprints(handle)

    aoi_mask = _rasterize_mask(
        footprints.aoi_geometry_wkb_hex, shape=footprints.shape, transform=footprints.transform
    )
    analysis_mask = _rasterize_mask(
        footprints.analysis_geometry_wkb_hex,
        shape=footprints.shape,
        transform=footprints.transform,
    )

    aoi_pixel_count = int(np.count_nonzero(aoi_mask))
    analysis_pixel_count = int(np.count_nonzero(analysis_mask))

    if aoi_pixel_count != footprints.aoi_pixel_count:
        raise CacheFootprintVerificationError(
            "HydroFragments-side re-rasterization of the persisted AOI "
            f"geometry produced {aoi_pixel_count} pixels, which does not "
            f"match hydroseason's persisted aoi_pixel_count "
            f"{footprints.aoi_pixel_count} (cache handle path="
            f"{getattr(handle, 'path', '<unknown>')!s}); refusing to trust "
            "a mismatched mask rather than silently using either count."
        )
    if analysis_pixel_count != footprints.analysis_pixel_count:
        raise CacheFootprintVerificationError(
            "HydroFragments-side re-rasterization of the persisted analysis "
            f"footprint geometry produced {analysis_pixel_count} pixels, "
            "which does not match hydroseason's persisted "
            f"analysis_pixel_count {footprints.analysis_pixel_count} (cache "
            f"handle path={getattr(handle, 'path', '<unknown>')!s}); "
            "refusing to trust a mismatched mask rather than silently using "
            "either count."
        )

    return VerifiedCacheFootprints(
        aoi_mask=aoi_mask,
        analysis_mask=analysis_mask,
        crs=footprints.crs,
        shape=footprints.shape,
        transform=footprints.transform,
        aoi_pixel_count=aoi_pixel_count,
        analysis_pixel_count=analysis_pixel_count,
        aoi_digest=footprints.aoi_digest,
        analysis_digest=footprints.analysis_digest,
    )


__all__ = [
    "CacheFootprintVerificationError",
    "VerifiedCacheFootprints",
    "open_verified_cache_footprints",
]
