"""Spatial raster products for the persistence core (Milestone 5).

Builds the occurrence, valid-observation-count, and refuge-mask rasters the
minimal core must emit (plan §5), each carrying the validity/refuge provenance
that makes the numbers interpretable. Writing these to Zarr/GeoTIFF is owned by
the Milestone 7 output tranche; this module only assembles the in-memory
xarray products so they stay reusable and cheap to test.
"""
from __future__ import annotations
from pathlib import Path

import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import OccurrenceResult


def build_persistence_rasters(
    occurrence: OccurrenceResult, *, config: HydroConfig
) -> xr.Dataset:
    """Assemble occurrence, valid-count, and refuge-mask rasters.

    The refuge mask marks pixels whose occurrence meets the refuge threshold
    and whose support clears ``min_valid_obs`` — the same rule as
    :func:`hydrofragments.metrics.persistence.compute_refuge_area`, so the
    raster and the scalar RA always agree.
    """
    threshold_pct = config.persistence.refuge_threshold * 100.0
    min_valid_obs = config.validity.min_valid_obs

    refuge_mask = (occurrence.occurrence >= threshold_pct) & (
        occurrence.valid_count >= min_valid_obs
    )

    rasters = xr.Dataset(
        {
            "occurrence": occurrence.occurrence,
            "valid_count": occurrence.valid_count,
            "refuge_mask": refuge_mask,
        }
    )
    rasters.attrs.update(
        {
            "refuge_threshold": config.persistence.refuge_threshold,
            "min_valid_obs": min_valid_obs,
            "validity_policy": config.validity.policy,
        }
    )
    return rasters


def write_persistence_rasters(
    rasters: xr.Dataset, output_dir: str | Path
) -> dict[str, Path]:
    """Write minimal-core rasters as independent, reopenable Zarr stores."""

    required = ("occurrence", "valid_count", "refuge_mask")
    missing = [name for name in required if name not in rasters.data_vars]
    if missing:
        raise ValueError(f"missing persistence raster: {missing[0]}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for name in required:
        path = root / name
        rasters[[name]].to_zarr(path, mode="w")
        artifacts[name] = path
    return artifacts


__all__ = ["build_persistence_rasters", "write_persistence_rasters"]
