"""Spatial raster products for the persistence core (Milestone 5).

Builds the occurrence, valid-observation-count, and refuge-mask rasters the
minimal core must emit (plan §5), each carrying the validity/refuge provenance
that makes the numbers interpretable. Checkpoint-backed accumulators in
``hydrofragments.output.checkpoints`` materialize the counters during the
monthly pass; this module assembles grid-bearing xarray products for tests and
future GeoTIFF writers.
"""
from __future__ import annotations
from pathlib import Path

import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import (
    HydroperiodResult,
    OccurrenceResult,
    RecurrenceResult,
)
from hydrofragments.output.checkpoints import (
    SpatialRasterCheckpoint,
    SpatialRasterCheckpointAccumulator,
)


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


def build_persistence_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble persistence rasters from a completed spatial checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    occurrence = accumulator.finalize_occurrence()
    return build_persistence_rasters(occurrence, config=config)


def build_temporal_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble recurrence and hydroperiod rasters from a completed checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    recurrence = accumulator.finalize_recurrence()
    hydroperiod = accumulator.finalize_hydroperiod()
    return xr.Dataset(
        {
            "recurrence": recurrence.recurrence,
            "recurrence_valid_year_count": recurrence.valid_year_count,
            "hydroperiod": hydroperiod.hydroperiod,
            "hydroperiod_valid_month_count": hydroperiod.valid_observed_months,
        }
    ).assign_attrs(
        {
            "min_valid_obs": config.validity.min_valid_obs,
            "validity_policy": config.validity.policy,
        }
    )


def build_refuge_stability_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble refuge-stability frequency rasters from a completed checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    return accumulator.finalize_refuge_stability_rasters()


def build_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Build every raster product present in ``checkpoint``."""

    checkpoint.validate_complete()
    products = set(checkpoint.metadata.products)
    datasets: list[xr.Dataset] = []
    if "persistence_rasters" in products:
        datasets.append(
            build_persistence_rasters_from_checkpoint(checkpoint, config=config)
        )
    if "temporal_rasters" in products:
        datasets.append(
            build_temporal_rasters_from_checkpoint(checkpoint, config=config)
        )
    if "refuge_stability_rasters" in products:
        datasets.append(
            build_refuge_stability_rasters_from_checkpoint(checkpoint, config=config)
        )
    if not datasets:
        return xr.Dataset()
    merged = xr.merge(datasets)
    merged.attrs["scientific_config_hash"] = checkpoint.metadata.scientific_config_hash
    merged.attrs["algorithm_version"] = checkpoint.metadata.algorithm_version
    return merged


__all__ = [
    "build_persistence_rasters",
    "build_persistence_rasters_from_checkpoint",
    "build_rasters_from_checkpoint",
    "build_refuge_stability_rasters_from_checkpoint",
    "build_temporal_rasters_from_checkpoint",
    "write_persistence_rasters",
]
