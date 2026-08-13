"""Single owner for validated result-bundle publication from both public workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio import features as rio_features
from shapely.geometry import shape as shapely_shape

from hydrofragments._version import __version__
from hydrofragments.config import HydroConfig
from hydrofragments.models import AnalysisInputs, HydroResult, WaterCube
from hydrofragments.output.bundle import (
    ArtifactRegistration,
    BundleError,
    assert_output_dir_available,
    open_bundle_transaction,
)
from hydrofragments.output.core import CoreAnalysisResult, build_in_memory_manifest
from hydrofragments.output.manifest import build_run_manifest, validate_result_bundle
from hydrofragments.output.rasters import export_rasters_from_checkpoint
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.output.tables import write_metric_coverage, write_output_tables
from hydrofragments.output.vectors import (
    MONTHLY_POOLS_LAYER,
    SPATIAL_GPKG_NAME,
    export_vectors_from_checkpoint,
)
from hydrofragments.spatial import SpatialContext, ZoneResult
from hydrofragments.spatial.zones import ZoneResult as ZoneResultType
from hydrofragments.temporal.hydroyear import HyAnchorResult

ZONES_LAYER = "zones"
REACHES_LAYER = "reaches"
REACH_WET_MONTHLY_LAYER = "reach_wet_monthly"

_ZONE_NAMES = {
    1: "channel_connected",
    2: "persistent",
    3: "seasonal",
    4: "ephemeral",
}


class SpatialProductUnavailable(ValueError):
    """Raised when a requested spatial product lacks runtime prerequisites."""


def _resolve_zone_result(
    inputs: AnalysisInputs,
    *,
    zone_result: ZoneResult | None,
) -> ZoneResult | None:
    if zone_result is not None:
        return zone_result
    if inputs.zones is not None:
        return inputs.zones
    return None


def preflight_spatial_outputs(
    config: HydroConfig,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs,
    hydroyear_result: HyAnchorResult | None,
    zone_result: ZoneResult | None = None,
) -> SpatialGrid | None:
    """Validate spatial products, source grid, writers, and output paths."""

    config.validate_output_preflight()
    products = config.output.spatial_products
    if not products and config.output.output_dir is None:
        return None

    grid: SpatialGrid | None = None
    if products:
        template = cube.water.isel(time=0)
        grid = SpatialGrid.from_dataarray(template, require_georeference=True)

    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    drainage = inputs.drainage
    has_channel = isinstance(drainage, SpatialContext) and drainage.has_real_channel

    for product in products:
        if product == "monthly_pools":
            if grid is None:
                raise SpatialProductUnavailable(
                    "monthly_pools requires a georeferenced water cube grid"
                )
            if config.patches.min_patch_pixels < 1:
                raise SpatialProductUnavailable(
                    "monthly_pools requires patches.min_patch_pixels >= 1"
                )
        elif product == "zones":
            if resolved_zones is None:
                raise SpatialProductUnavailable(
                    "zones requires an explicit zone input or DEA zone derivation"
                )
            if resolved_zones.grid is None:
                raise SpatialProductUnavailable(
                    "zones requires a georeferenced zone mask grid"
                )
        elif product == "persistence_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "persistence_rasters requires a georeferenced water cube grid"
                )
        elif product == "temporal_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "temporal_rasters requires a georeferenced water cube grid"
                )
        elif product == "refuge_stability_rasters":
            if grid is None:
                raise SpatialProductUnavailable(
                    "refuge_stability_rasters requires a georeferenced water cube grid"
                )
            if hydroyear_result is None or len(hydroyear_result.anchors) < 2:
                raise SpatialProductUnavailable(
                    "refuge_stability_rasters requires at least two hydrological-year anchors"
                )
        elif product == "reach_profiles":
            if not has_channel:
                raise SpatialProductUnavailable(
                    "reach_profiles requires a real channel SpatialContext and drainage"
                )
            if (
                inputs.channel_wet_profiles is None
                or inputs.channel_segment_lengths_m is None
            ):
                raise SpatialProductUnavailable(
                    "reach_profiles requires channel_wet_profiles and "
                    "channel_segment_lengths_m"
                )
        else:
            raise SpatialProductUnavailable(f"unsupported spatial product: {product}")

    if config.output.output_dir is not None:
        assert_output_dir_available(Path(config.output.output_dir))
        output_root = Path(config.output.output_dir)
        collision_paths = [
            output_root / "run_manifest.json",
            output_root / "config.json",
            output_root / "metrics",
            output_root / "vectors" / SPATIAL_GPKG_NAME,
            output_root / "rasters",
        ]
        for path in collision_paths:
            if path.exists():
                raise BundleError(f"refusing to overwrite existing artifact: {path}")

    return grid


def _zones_geodataframe(zone_result: ZoneResultType, *, pixel_size_m: float) -> gpd.GeoDataFrame:
    if zone_result.grid is None:
        raise SpatialProductUnavailable("zones vector export requires a georeferenced zone mask")
    grid = zone_result.grid
    mask = np.asarray(zone_result.mask, dtype=np.uint8)
    cell_area_m2 = float(pixel_size_m) ** 2
    rows: list[dict[str, object]] = []
    for zone_id in sorted(set(int(value) for value in np.unique(mask) if int(value) > 0)):
        zone_pixels = mask == zone_id
        shapes = rio_features.shapes(
            zone_pixels.astype(np.uint8),
            mask=zone_pixels,
            transform=grid.transform,
        )
        geometries = [shapely_shape(geom) for geom, value in shapes if int(value) == 1]
        if not geometries:
            continue
        geometry = geometries[0] if len(geometries) == 1 else gpd.GeoSeries(geometries).union_all()
        area_m2 = float(zone_pixels.sum()) * cell_area_m2
        rows.append(
            {
                "zone_id": zone_id,
                "zone_name": _ZONE_NAMES.get(zone_id, f"zone_{zone_id}"),
                "area_km2": area_m2 / 1_000_000.0,
                "source": zone_result.source,
                "geometry": geometry,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            columns=["zone_id", "zone_name", "area_km2", "source", "geometry"],
            geometry="geometry",
            crs=grid.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=grid.crs)


def _reach_wet_monthly_table(
    *,
    context: SpatialContext,
    cube: WaterCube,
    wet_profiles: Sequence[Sequence[bool]],
    segment_lengths_m: Sequence[float],
) -> pd.DataFrame:
    drainage = context.drainage
    if drainage is None:
        raise SpatialProductUnavailable("reach_profiles requires drainage geometry")
    reach_ids = drainage["HydroID"].astype(str).tolist()
    lengths = np.asarray(segment_lengths_m, dtype=float)
    wet = np.asarray(wet_profiles, dtype=bool)
    times = pd.to_datetime(cube.water["time"].values)
    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(times):
        month_wet = wet[time_index]
        wetted_length_m = float(lengths[month_wet].sum())
        total_length_m = float(lengths.sum())
        lpsec_pct = (
            100.0 * wetted_length_m / total_length_m if total_length_m > 0 else float("nan")
        )
        for reach_index, reach_id in enumerate(reach_ids):
            rows.append(
                {
                    "reach_id": reach_id,
                    "date": pd.Timestamp(timestamp),
                    "is_wet": bool(month_wet[reach_index]),
                    "length_m": float(lengths[reach_index]),
                    "lpsec_contribution_pct": lpsec_pct if month_wet[reach_index] else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _write_spatial_vectors(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    cube: WaterCube,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    pixel_size_m: float,
) -> list[ArtifactRegistration]:
    products = set(config.output.spatial_products)
    if not products:
        return []

    registrations: list[ArtifactRegistration] = []
    vectors_dir = staging_root / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = vectors_dir / SPATIAL_GPKG_NAME
    if gpkg_path.exists():
        raise BundleError(f"refusing to overwrite existing artifact: {gpkg_path}")

    if "monthly_pools" in products:
        if core.pool_checkpoint_root is None:
            raise SpatialProductUnavailable(
                "monthly_pools export requires a completed pool vector checkpoint"
            )
        export_vectors_from_checkpoint(core.pool_checkpoint_root, vectors_dir)
        registrations.append(
            ArtifactRegistration(
                name="spatial_vectors",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    if "zones" in products and resolved_zones is not None:
        zones_gdf = _zones_geodataframe(resolved_zones, pixel_size_m=pixel_size_m)
        pyogrio.write_dataframe(
            zones_gdf,
            gpkg_path,
            layer=ZONES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=gpkg_path.exists(),
        )
        registrations.append(
            ArtifactRegistration(
                name="zones_vector",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    if "reach_profiles" in products:
        context = inputs.drainage
        if not isinstance(context, SpatialContext) or context.drainage is None:
            raise SpatialProductUnavailable("reach_profiles requires channel drainage geometry")
        reaches = context.drainage.copy()
        reaches["reach_id"] = reaches["HydroID"].astype(str)
        reach_table = _reach_wet_monthly_table(
            context=context,
            cube=cube,
            wet_profiles=inputs.channel_wet_profiles or (),
            segment_lengths_m=inputs.channel_segment_lengths_m or (),
        )
        pyogrio.write_dataframe(
            reaches,
            gpkg_path,
            layer=REACHES_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=gpkg_path.exists(),
        )
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(reach_table, geometry=None),
            gpkg_path,
            layer=REACH_WET_MONTHLY_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=True,
        )
        registrations.append(
            ArtifactRegistration(
                name="reach_profiles",
                relative_path=f"vectors/{SPATIAL_GPKG_NAME}",
                media_type="application/geopackage+sqlite3",
            )
        )

    return registrations


def _write_spatial_rasters(
    staging_root: Path,
    *,
    config: HydroConfig,
    core: CoreAnalysisResult,
    inputs: AnalysisInputs,
    zone_result: ZoneResult | None,
    analysis_mask: np.ndarray | None,
) -> list[ArtifactRegistration]:
    raster_products = {
        item
        for item in config.output.spatial_products
        if item
        in {
            "persistence_rasters",
            "temporal_rasters",
            "refuge_stability_rasters",
            "zones",
        }
    }
    if not raster_products and "zones" not in config.output.spatial_products:
        return []
    if core.raster_checkpoint is None and "zones" not in config.output.spatial_products:
        return []

    raster_dir = staging_root / "rasters"
    registrations: list[ArtifactRegistration] = []
    zone_mask = None
    resolved_zones = _resolve_zone_result(inputs, zone_result=zone_result)
    if "zones" in config.output.spatial_products and resolved_zones is not None:
        zone_mask = np.asarray(resolved_zones.mask, dtype=np.uint8)

    if core.raster_checkpoint is not None:
        artifacts = export_rasters_from_checkpoint(
            core.raster_checkpoint,
            raster_dir,
            config=config,
            raster_formats=config.output.raster_formats,
            zone_mask=zone_mask if "zones" in config.output.spatial_products else None,
            analysis_mask=analysis_mask,
        )
        for name, path in artifacts.items():
            registrations.append(
                ArtifactRegistration(
                    name=name,
                    relative_path=str(path.relative_to(staging_root)).replace("\\", "/"),
                    media_type="image/tiff" if path.suffix.lower() in {".tif", ".tiff"} else None,
                )
            )
    elif zone_mask is not None:
        from hydrofragments.output.rasters import RASTER_PRODUCT_CONTRACTS, write_zones_geotiff

        if core.spatial_grid is None:
            raise SpatialProductUnavailable("zones raster export requires a spatial grid")
        zones_path = raster_dir / RASTER_PRODUCT_CONTRACTS["zones"].filename
        write_zones_geotiff(
            zone_mask,
            zones_path,
            grid=core.spatial_grid,
            metadata={
                "algorithm_version": "1.0.0",
                "scientific_config_hash": config.config_hash,
            },
        )
        registrations.append(
            ArtifactRegistration(
                name="zones",
                relative_path=f"rasters/{RASTER_PRODUCT_CONTRACTS['zones'].filename}",
                media_type="image/tiff",
            )
        )

    return registrations


def finalize_analysis_bundle(
    config: HydroConfig,
    core: CoreAnalysisResult,
    *,
    cube: WaterCube,
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    zone_result: ZoneResult | None = None,
    dea_provenance: Mapping[str, object] | None = None,
    timings_seconds: Mapping[str, float] | None = None,
    peak_rss_bytes: int | None = None,
) -> HydroResult:
    """Publish tables, spatial products, and one validated manifest atomically."""

    inputs = inputs or AnalysisInputs()
    output_dir = Path(config.output.output_dir)
    analysis_mask_np = None
    if cube.analysis_mask is not None:
        analysis_mask_np = np.asarray(cube.analysis_mask.values, dtype=bool)

    write_started = time.perf_counter()
    transaction = open_bundle_transaction(
        output_dir, run_id=core.run_id, config=config
    )
    try:
        write_output_tables(
            core.metrics_table,
            transaction.root,
            formats=config.output.formats,
            export_csv="csv" in config.output.formats,
        )
        transaction.register_artifact(
            ArtifactRegistration(name="metrics", relative_path="metrics")
        )
        write_metric_coverage(core.metric_coverage, transaction.root)
        transaction.register_artifact(
            ArtifactRegistration(
                name="metric_coverage",
                relative_path="metric_coverage.csv",
                media_type="text/csv",
            )
        )

        vector_regs = _write_spatial_vectors(
            transaction.root,
            config=config,
            core=core,
            cube=cube,
            inputs=inputs,
            zone_result=zone_result,
            pixel_size_m=pixel_size_m,
        )
        for registration in vector_regs:
            transaction.register_artifact(registration)

        raster_regs = _write_spatial_rasters(
            transaction.root,
            config=config,
            core=core,
            inputs=inputs,
            zone_result=zone_result,
            analysis_mask=analysis_mask_np,
        )
        for registration in raster_regs:
            transaction.register_artifact(registration)

        transaction.write_config()
        transaction.register_artifact(
            ArtifactRegistration(
                name="config",
                relative_path="config.json",
                media_type="application/json",
            )
        )

        resolved_timings = dict(timings_seconds or {})
        resolved_timings["output_write"] = time.perf_counter() - write_started
        if resolved_timings.keys() - {"total"}:
            resolved_timings["total"] = sum(
                value
                for key, value in resolved_timings.items()
                if key != "total"
            )

        artifacts = transaction.finalize(
            package_version=__version__,
            git_sha=core.git_sha,
            input_fingerprint=core.input_fingerprint,
            planned_backend=str(core.execution_plan_mapping.get("planned_backend", "cpu")),
            actual_backend_by_stage=dict(
                core.execution_plan_mapping.get("actual_backend_by_stage", {})
            ),
            backend_capabilities=dict(
                core.execution_plan_mapping.get("backend_capabilities", {})
            ),
            skipped_metrics=[
                {"metric_id": metric_id, "reason": reason}
                for metric_id, reason in core.skipped_metrics
            ],
            warnings=list(core.report_warnings),
            comparison_context=core.comparison_context,
            timings_seconds=resolved_timings,
            dea_provenance=dict(dea_provenance) if dea_provenance else None,
            peak_rss_bytes=peak_rss_bytes,
        )
    except Exception:
        transaction.abort()
        raise

    manifest = validate_result_bundle(output_dir)
    manifest_dict = dict(manifest)
    manifest_dict["manifest_path"] = str(output_dir / "run_manifest.json")
    return HydroResult(
        metrics_table=core.metrics_table,
        manifest=manifest_dict,
        output_dir=output_dir,
        run_id=core.run_id,
        metric_coverage=core.metric_coverage,
    )


__all__ = [
    "CoreAnalysisResult",
    "SpatialProductUnavailable",
    "build_in_memory_manifest",
    "finalize_analysis_bundle",
    "preflight_spatial_outputs",
]
