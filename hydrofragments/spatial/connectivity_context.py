"""Reach/water-mask intersection for the RC/TCF fixed graph (U9, approved 2026-07-17).

Decides, per drainage reach, whether the reach was ever wet across a
monthly series -- the ``wet_any_month`` input consumed by
:func:`hydrofragments.metrics.connectivity.build_fixed_graph`.

Method (U9): seeded-skeleton, not a raw buffer-stamp. For each month,
skeletonize the full water mask (:func:`skimage.morphology.medial_axis`,
the same function already used per-component in
``hydrofragments/patches/morphology.py``), buffer each reach line by
``reach_buffer_m`` (default 2 pixel widths), and flag the reach wet that
month iff the skeleton has >=1 pixel inside its buffer. A reach is
``wet_any_month`` if this holds for at least one month in the series.
Chosen over a plain buffer-stamp specifically to avoid two failure modes: a
too-narrow buffer silently losing a real but slightly offset channel, and a
too-wide buffer/no-skeleton-filter falsely crediting an unrelated nearby
channel's water to the wrong reach.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine
from rasterio.features import geometry_mask
from skimage.morphology import medial_axis

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


def _raster_transform(y_coords: "np.ndarray", x_coords: "np.ndarray") -> Affine:
    """Build a pixel-center-convention affine transform from coord arrays.

    ``x_coords``/``y_coords`` hold pixel-center coordinates (the xarray
    convention). ``geometry_mask``/rasterio transforms describe the
    pixel's top-left corner, so each origin is shifted back by half a
    pixel step.
    """
    x_step = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
    y_step = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
    return Affine(
        x_step, 0.0, float(x_coords[0]) - x_step / 2.0,
        0.0, y_step, float(y_coords[0]) - y_step / 2.0,
    )


def _reach_buffer_mask(
    buffer_polygon, *, transform: Affine, y_coords: "np.ndarray", x_coords: "np.ndarray"
) -> "np.ndarray":
    return geometry_mask(
        [buffer_polygon],
        out_shape=(len(y_coords), len(x_coords)),
        transform=transform,
        invert=True,
    )


def _build_reach_label_raster(
    drainage: "gpd.GeoDataFrame",
    *,
    buffer_m: float,
    transform: Affine,
    y_coords: "np.ndarray",
    x_coords: "np.ndarray",
) -> tuple["np.ndarray", dict[tuple[int, int], list[int]]]:
    """Rasterize every reach buffer into one multilabel raster, once.

    Returns ``(label_raster, overlap_pixels)``:

    - ``label_raster`` holds ``i + 1`` (1-based reach index) for pixels
      claimed by exactly one reach buffer, and a sentinel ``-1`` for pixels
      claimed by more than one (last-writer-wins would silently drop all
      but the last reach for those pixels).
    - ``overlap_pixels`` maps each ``-1``-sentinel ``(row, col)`` to the
      *full* list of 0-based reach indices whose buffer covers that pixel,
      so no reach attribution is lost for shared pixels.

    Adjacent reaches sharing an endpoint routinely produce overlapping
    buffers when ``buffer_m`` is comparable to reach spacing near
    junctions -- this is the normal case for a drainage network, not an
    edge case, so it is handled exactly rather than approximated.
    """
    shape = (len(y_coords), len(x_coords))
    label_raster = np.zeros(shape, dtype=np.int32)
    overlap_pixels: dict[tuple[int, int], list[int]] = {}

    for i, (_, reach) in enumerate(drainage.iterrows()):
        mask = _reach_buffer_mask(
            reach.geometry.buffer(buffer_m),
            transform=transform,
            y_coords=y_coords,
            x_coords=x_coords,
        )
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            existing = label_raster[r, c]
            if existing == 0:
                label_raster[r, c] = i + 1
            elif existing > 0:
                # Second claimant for this pixel: promote to the overlap
                # map, seeded with the first reach's index plus this one.
                overlap_pixels[(r, c)] = [existing - 1, i]
                label_raster[r, c] = -1
            else:
                # Already an overlap pixel (>=2 prior claimants): append.
                overlap_pixels[(r, c)].append(i)

    return label_raster, overlap_pixels


def reach_wet_any_month(
    drainage: "gpd.GeoDataFrame",
    water: "xr.DataArray",
    *,
    buffer_m: float,
) -> dict[str, bool]:
    """Return, per reach ``HydroID``, whether its skeleton-seeded buffer was wet in >=1 month.

    For each month, the full water mask is skeletonized with
    :func:`skimage.morphology.medial_axis`; a reach is wet that month iff
    the skeleton has at least one pixel inside the reach line's
    ``buffer_m``-metre buffer. ``wet_any_month`` is the OR of the monthly
    flag across the whole series.

    Implementation note (M4): reach buffers are rasterized once into a
    single multilabel raster instead of one full boolean mask per reach,
    and each month's skeleton is intersected against that raster via
    ``np.unique`` instead of a per-reach ``&``. Pixels shared by more than
    one reach's buffer (common near junctions) are tracked separately so
    every reach whose buffer covers a wet skeleton pixel is still credited
    -- matching the semantics of the old per-reach independent check.
    """
    y_coords = water["y"].values
    x_coords = water["x"].values
    transform = _raster_transform(y_coords, x_coords)
    reach_ids = [str(reach["HydroID"]) for _, reach in drainage.iterrows()]
    label_raster, overlap_pixels = _build_reach_label_raster(
        drainage, buffer_m=buffer_m, transform=transform,
        y_coords=y_coords, x_coords=x_coords,
    )

    result: dict[str, bool] = {reach_id: False for reach_id in reach_ids}
    for month_index in range(water.sizes["time"]):
        if all(result.values()):
            break
        month_mask = np.asarray(water.isel(time=month_index).values, dtype=bool)
        if not month_mask.any():
            continue
        skeleton = medial_axis(month_mask)

        for label in np.unique(label_raster[skeleton]):
            if label > 0:
                result[reach_ids[label - 1]] = True

        if overlap_pixels:
            overlap_rows, overlap_cols = zip(*overlap_pixels.keys())
            hit = skeleton[np.array(overlap_rows), np.array(overlap_cols)]
            if hit.any():
                for (r, c), is_hit in zip(overlap_pixels.keys(), hit):
                    if is_hit:
                        for reach_index in overlap_pixels[(r, c)]:
                            result[reach_ids[reach_index]] = True
    return result


__all__ = ["reach_wet_any_month"]
