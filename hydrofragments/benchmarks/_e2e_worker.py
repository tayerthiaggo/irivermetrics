"""Isolated-subprocess worker for the W3.7 end-to-end benchmark.

Reads one JSON payload from stdin describing a single candidate run
(factor/workers/executor_kind, AOI/drainage paths, date range, cache/output
dirs, cold-vs-warm mode), runs the real pipeline primitives
:func:`hydrofragments.workflow.analyze_from_dea` itself uses -- but called
directly rather than through that orchestrator, because the benchmark needs
to vary the planning-footprint ``factor`` and the metric-processing
``config.compute.workers`` knobs that ``analyze_from_dea`` intentionally
keeps fixed at its own opinionated defaults (see this task's report for why
modifying ``workflow.py`` itself is out of scope) -- and writes one JSON
result line to stdout.

``executor_kind`` is accepted in the payload for schema completeness but is
NOT actually threaded into :func:`hydrofragments.api.analyze` below: that
function calls ``analyze_section_rows`` with no ``executor_kind`` override,
so it is always ``"thread"`` on this call path regardless of what a
candidate requests (confirmed by reading ``hydrofragments/api.py``'s call
site directly, not assumed) -- reaching ``"process"`` would require calling
``analyze_section_rows`` directly instead of ``analyze()``, which would
duplicate ``analyze()``'s own orchestration (metric-plan resolution, other
metric records, output assembly) well beyond what this benchmark needs.
W3.2's own benchmark already gated serial-vs-thread-vs-process at the
``analyze_section_rows`` level directly on synthetic data (see its report);
this benchmark's ``workers`` axis is ``config.compute.workers`` in
``{1, 2, 4}`` through the real, only-reachable-from-``analyze()`` thread
pool, not a second process-pool re-test.

Run as ``python -m hydrofragments.benchmarks._e2e_worker`` so each candidate
gets a fresh interpreter (clean Dask/xarray/GDAL state for peak-RSS
measurement) and its own process for the parent benchmark runner to poll via
``psutil``.

This module intentionally duplicates only the ORCHESTRATION shape of
``analyze_from_dea`` (phase boundaries, call order) with the two knobs
parameterised; every scientific computation still comes from the exact same
functions ``analyze_from_dea`` calls (``open_wo_statistics_for_zoning``,
``hydroseason.acquire_wofs_cache``, ``open_verified_cache_footprints``,
``open_water_cube``, ``analyze``) -- no metric formula, threshold, or
pruning algorithm is reimplemented here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Unsigned-S3/GDAL COG reads at WHOLE-PROCESS scope: hydroseason's own
# open_wo_statistics() only applies hydroseason._io_geo._configure_cog_read_env()
# narrowly, inside a try/finally that restores the caller's original env
# once open_wo_statistics() returns (hydroseason/_io_dea_stats.py). But
# odc.stac.load() is lazy/Dask-backed -- the actual GDAL S3 reads and CRS
# transforms only happen later, when THIS benchmark calls
# build_wet_planning_footprint()'s _eager_values(...).compute(), which runs
# AFTER open_wo_statistics() already restored the caller's original
# (broken, on this machine: a stray PostGIS-bundled PROJ_LIB) environment.
# Confirmed via two real runs: this produced a genuine
# RasterioIOError/InvalidCredentials (missing AWS_NO_SIGN_REQUEST), then
# after that fix, a genuine pyproj.exceptions.ProjError (PROJ_LIB pointed
# at an incompatible PROJ data dir) -- not hypotheticals. This is a real,
# load-bearing hydroseason gap (its narrow env scope doesn't cover a
# consumer's later lazy .compute()), out of this task's file scope to fix
# (hydroseason/_io_geo.py is a sibling repo) -- flagged in the task report
# as a concern. Calling hydroseason's own real helper unconditionally at
# whole-process/import scope (never reimplementing its env-variable
# choices) is the workaround this benchmark needs; setdefault/explicit
# rasterio-bundled-PROJ-data reassignment matches _configure_cog_read_env's
# own documented behaviour exactly.
def _apply_cog_read_env() -> None:
    from hydroseason._io_geo import _configure_cog_read_env

    _configure_cog_read_env()


_apply_cog_read_env()


def planning_footprint_native_wet_pixel_superset_metrics(footprint: Any) -> dict[str, Any]:
    """Prove (or explicitly decline to prove) the plan's actual coverage gate.

    The plan's Global Constraints require: "count_wet > 0 planning footprint
    must cover 100% of native wet pixels" -- i.e. every pixel that is
    genuinely wet in the native DEA statistics grid must fall inside the
    coarse planning footprint once expanded back to native resolution:
    ``native_mask <= expand(coarse_mask)``.

    This is the exact superset property W1.5 already proves in the sibling
    hydroseason repo -- see ``hydroseason._io_dea_stats.build_wet_planning_footprint``'s
    own docstring and ``tests/test_io_dea_stats.py``, which establish this
    proof using this exact expansion technique
    (``coarse.repeat(factor, axis=0).repeat(factor, axis=1)``). This helper
    intentionally reuses that identical technique rather than inventing a
    different one, so a divergent implementation here can never silently
    disagree with what W1.5 already proved.

    ``footprint`` is a ``hydroseason._io_dea_stats.WetPlanningFootprint`` (or
    any object exposing ``.native_mask``/``.coarse_mask`` xarray DataArrays
    and an integer ``.factor``), or ``None`` when DEA statistics were
    unavailable this run and acquisition fell open to full-AOI -- in which
    case there is no planning footprint to prove anything about, so this
    returns explicit ``None``s plus a reason rather than a fabricated
    vacuous pass.

    Returns a dict with keys ``planning_footprint_native_wet_pixel_superset_holds``
    (bool | None), ``planning_footprint_native_wet_pixel_coverage_fraction``
    (float | None), and ``planning_footprint_superset_reason`` (str | None).
    """

    if footprint is None:
        return {
            "planning_footprint_native_wet_pixel_superset_holds": None,
            "planning_footprint_native_wet_pixel_coverage_fraction": None,
            "planning_footprint_superset_reason": (
                "footprint is None (DEA statistics were unavailable this run, "
                "acquisition fell open to full-AOI; no planning footprint "
                "exists to prove the native-wet-pixel superset property about)"
            ),
        }

    native = np.asarray(footprint.native_mask.values, dtype=bool)
    coarse = np.asarray(footprint.coarse_mask.values, dtype=bool)
    factor = footprint.factor
    expanded = coarse.repeat(factor, axis=0).repeat(factor, axis=1)[
        : native.shape[0], : native.shape[1]
    ]
    native_wet_pixel_count = int(native.sum())
    covered_wet_pixel_count = int(np.count_nonzero(native & expanded))
    superset_holds = bool(np.all(native <= expanded))
    coverage_fraction = (
        float(covered_wet_pixel_count) / float(native_wet_pixel_count)
        if native_wet_pixel_count > 0
        else 1.0  # vacuously true: no native wet pixels to cover
    )
    return {
        "planning_footprint_native_wet_pixel_superset_holds": superset_holds,
        "planning_footprint_native_wet_pixel_coverage_fraction": coverage_fraction,
        "planning_footprint_superset_reason": None,
    }


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("benchmark_kind") == "spatial_export":
        return _run_spatial_export(payload)
    return _run_dea_workflow(payload)


def _run_dea_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import rioxarray  # noqa: F401 -- registers xarray's .rio accessor as a side
    # effect of import. hydrofragments.io.dea.open_wo_statistics_for_zoning
    # calls `dataset.rio.crs` but never imports rioxarray itself (a real,
    # pre-existing gap outside this task's file scope, see task-3.7 report);
    # every existing test that reaches this code path masks the same gap via
    # `pytest.importorskip("rioxarray")`'s import side effect. This worker
    # needs the identical side effect to run the real pipeline standalone.

    import hydroseason
    import hydroseason._io_dea_stats  # noqa: F401 -- module attrs accessed below
    from hydroseason._io_dea_stats import DEAStatsUnavailable, WoStatisticsUnavailable

    from hydrofragments.api import analyze, open_water_cube
    from hydrofragments.config import HydroConfig
    from hydrofragments.io.cache_footprints import open_verified_cache_footprints
    from hydrofragments.io.dea import open_wo_statistics_for_zoning
    from hydrofragments.models import AnalysisInputs
    from hydrofragments.workflow import _dual_extent_inputs, _load_geometry, _wo_statistics_as_dataset

    _DEA_PRODUCT = "ga_ls_wo_fq_myear_3"
    _DEA_STAC_URL = "https://explorer.sandbox.dea.ga.gov.au/stac"

    factor = int(payload["factor"])
    workers = int(payload["workers"])
    executor_kind = payload["executor_kind"]
    aoi_path = payload["aoi_path"]
    drainage_path = payload.get("drainage_path")
    start_date = payload["start_date"]
    end_date = payload["end_date"]
    aoi_id = payload["aoi_id"]
    cache_dir = payload["cache_dir"]
    output_dir = payload["output_dir"]
    mode = payload["mode"]

    resolution = 30.0
    dea_crs = "EPSG:3577"

    timings: dict[str, float] = {}

    resolved_config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "output": {"output_dir": str(output_dir)},
            "compute": {"workers": workers},
        }
    )

    aoi_gdf = _load_geometry(aoi_path)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    requested_years = list(range(start.year, end.year + 1))

    # --- Phase 1: DEA planning (native stats + coarse planning footprint) --
    t0 = time.perf_counter()
    try:
        stats = open_wo_statistics_for_zoning(
            aoi_gdf, product=_DEA_PRODUCT, stac_url=_DEA_STAC_URL,
            resolution=resolution, crs=dea_crs,
        )
        stats_dataset = _wo_statistics_as_dataset(stats)
        footprint = hydroseason._io_dea_stats.build_wet_planning_footprint(
            stats_dataset, requested_years=requested_years, factor=factor, safety_cells=1,
        )
    except (WoStatisticsUnavailable, DEAStatsUnavailable):
        footprint = None
    timings["dea_planning"] = time.perf_counter() - t0

    # --- Phase 2/3: WOfS query + acquisition -------------------------------
    t0 = time.perf_counter()
    handle = hydroseason.acquire_wofs_cache(
        _DEA_STAC_URL,
        "ga_ls_wo_3",
        aoi_gdf,
        start_date,
        end_date,
        cache_root=cache_dir,
        crs=dea_crs,
        resolution=resolution,
        wet_mask="dea_stats" if footprint is not None else "off",
        planning_footprint=footprint,
        composite_bundle="hydrofragments_v1",
    )
    timings["wofs_query_and_acquisition"] = time.perf_counter() - t0

    # --- Phase 4: local metric processing -----------------------------------
    t0 = time.perf_counter()
    verified_footprints = open_verified_cache_footprints(handle)
    mask_cube = hydroseason.open_completed_mask_cache(handle, start_date, end_date)

    aoi_mask_da = _as_spatial_mask(verified_footprints.aoi_mask, mask_cube)
    analysis_mask_da = _as_spatial_mask(verified_footprints.analysis_mask, mask_cube)

    cube = open_water_cube(
        mask_cube,
        input_kind="watermask_tsfill",
        aoi_mask=aoi_mask_da,
        analysis_mask=analysis_mask_da,
    )

    dual_counts = hydroseason.open_completed_dual_extent_counts(handle, start_date, end_date)
    hydroyear_extent, max_water_apsec, median_apsec = _dual_extent_inputs(dual_counts)

    channel_context = None
    channel_wet_profiles = None
    channel_segment_lengths_m = None
    if drainage_path is not None:
        from hydrofragments.spatial import create_channel_context, reach_monthly_wet_profile

        drainage_gdf = _load_geometry(drainage_path)
        channel_context = create_channel_context(
            aoi_id, aoi_gdf, drainage_gdf, drainage_id="workflow",
            target_crs=resolved_config.spatial.target_crs,
        )
        channel_wet_profiles = reach_monthly_wet_profile(
            channel_context.drainage, cube.water, buffer_m=60.0,
        )
        channel_segment_lengths_m = channel_context.drainage.geometry.length.tolist()

    inputs = AnalysisInputs(
        drainage=channel_context,
        hydroyear_extent=hydroyear_extent,
        max_water_apsec=max_water_apsec,
        median_apsec=median_apsec,
        channel_wet_profiles=channel_wet_profiles,
        channel_segment_lengths_m=channel_segment_lengths_m,
    )
    result = analyze(
        cube, aoi_id, config=resolved_config, inputs=inputs, pixel_size_m=resolution,
    )
    timings["metric_processing"] = time.perf_counter() - t0

    # --- Phase 5: output write ------------------------------------------------
    t0 = time.perf_counter()
    from hydrofragments.output.tables import write_metric_coverage, write_output_tables

    write_output_tables(result.metrics_table, result.output_dir)
    write_metric_coverage(result.metric_coverage, result.output_dir)
    timings["output_write"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())

    metrics_table = result.metrics_table
    digest_columns = sorted(
        c for c in metrics_table.columns if c not in ("run_id", "config_hash")
    )
    digest_frame = metrics_table[digest_columns].sort_values(
        by=[c for c in ("metric", "date", "zone", "window_id") if c in digest_columns]
    )
    metrics_digest = hashlib.sha256(
        digest_frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()

    n_water_by_month: dict[str, int] = {}
    if "n_water_pixels" in metrics_table.columns and "date" in metrics_table.columns:
        for date_value, group in metrics_table.groupby("date"):
            values = group["n_water_pixels"].dropna().unique()
            if len(values) == 1:
                n_water_by_month[str(date_value)] = int(values[0])

    # Data-completeness / valid-observation-rate metric: the fraction of AOI
    # pixel-months in the raw native mask cube hydroseason.open_completed_mask_cache
    # returns that are a VALID observation. Confirmed by direct inspection of a
    # real acquired cube (not assumed): this cube's actual convention is
    # {-1, 0, 1, NaN} (see hydroseason._io_wofs_zarr.open_completed_mask_cache's
    # own docstring: "gaps become -1 invalid, not -2 outside" plus a NaN fill
    # from complete_monthly_axis for any calendar month with literally zero
    # source observations) -- NOT the {0,1,254,255} watermask_tsfill
    # convention documented in hydrofragments/io/validity.py, which describes
    # the SEPARATE on-disk product format that convention was written for.
    # Only 0/1 count as a valid observation here; -1 and NaN are both
    # invalid, matched via ==1 short-circuiting NaN (NaN != NaN and
    # NaN != -1 are both true, so the OR chain below correctly excludes NaN
    # without a separate isnan check).
    #
    # NOTE: this is a real Landsat/Sentinel cloud-free-observation-rate signal,
    # but it is NOT the plan's "count_wet > 0 planning footprint must cover
    # 100% of native wet pixels" gate -- it never references the planning
    # footprint at all. See planning_footprint_native_wet_pixel_superset_holds
    # below for that proof.
    aoi_mask_arr = np.asarray(verified_footprints.aoi_mask)
    native_wet_mask = np.asarray(mask_cube.values) if hasattr(mask_cube, "values") else None
    analysis_mask_valid_observation_fraction = None
    if native_wet_mask is not None:
        valid_mask = (native_wet_mask == 0) | (native_wet_mask == 1)
        aoi_true = np.count_nonzero(aoi_mask_arr)
        if aoi_true > 0:
            inside_covered = np.count_nonzero(valid_mask & aoi_mask_arr[np.newaxis, :, :])
            analysis_mask_valid_observation_fraction = (
                float(inside_covered) / float(aoi_true * native_wet_mask.shape[0])
            )

    # The plan's ACTUAL gate: "count_wet > 0 planning footprint must cover
    # 100% of native wet pixels" -- i.e. every pixel that is genuinely wet in
    # the native DEA statistics grid must fall inside the coarse planning
    # footprint once expanded back to native resolution:
    # native_mask <= expand(coarse_mask). This is the exact superset property
    # W1.5 already proves in hydroseason (build_wet_planning_footprint's own
    # docstring and tests/test_io_dea_stats.py), reusing the identical
    # expansion technique (coarse.repeat(factor, ...)) so this can't silently
    # disagree with what W1.5 proved.
    #
    # When footprint is None (DEA stats were unavailable this run, fell open
    # to full-AOI acquisition), there is no planning footprint to prove
    # anything about -- report None with a reason rather than a fabricated
    # vacuous pass.
    superset_metrics = planning_footprint_native_wet_pixel_superset_metrics(footprint)

    return {
        "status": "ok",
        "mode": mode,
        "candidate_id": payload["candidate_id"],
        "timings_seconds": timings,
        "metrics_digest": metrics_digest,
        "metrics_row_count": int(len(metrics_table)),
        "n_water_by_month": n_water_by_month,
        "analysis_mask_valid_observation_fraction": analysis_mask_valid_observation_fraction,
        **superset_metrics,
        "footprint_factor": factor,
        "footprint_used": footprint is not None,
        "cache_path": str(getattr(handle, "path", cache_dir)),
    }


def _as_spatial_mask(mask, reference):
    import xarray as xr

    spatial_dims = tuple(dim for dim in reference.dims if dim != "time")
    coords = {
        name: coord
        for name, coord in reference.coords.items()
        if set(coord.dims) <= set(spatial_dims)
    }
    return xr.DataArray(mask, dims=spatial_dims, coords=coords)


def _metrics_digest(metrics_table) -> str:
    digest_columns = sorted(
        c for c in metrics_table.columns if c not in ("run_id", "config_hash")
    )
    digest_frame = metrics_table[digest_columns].sort_values(
        by=[c for c in ("metric", "date", "zone", "window_id") if c in digest_columns]
    )
    return hashlib.sha256(digest_frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _coverage_digest(coverage_table) -> str:
    return hashlib.sha256(coverage_table.to_csv(index=False).encode("utf-8")).hexdigest()


def _build_spatial_fixture(fixture_id: str, *, zarr_path: str | None = None):
    """Build a repository-owned synthetic cube for spatial-export benchmarks."""

    import pandas as pd
    import rioxarray  # noqa: F401
    import xarray as xr

    from hydrofragments.api import open_water_cube

    if fixture_id == "zarr_local_subset":
        if not zarr_path:
            raise ValueError("zarr_local_subset requires zarr_path")
        cube = open_water_cube(zarr_path, chunks={"time": 12, "y": 128, "x": 128})
        water = cube.water.isel(time=slice(0, 12), y=slice(0, 64), x=slice(0, 64))
        valid = cube.valid_obs.isel(time=slice(0, 12), y=slice(0, 64), x=slice(0, 64))
        return open_water_cube(water, valid_obs=valid, input_kind="watermask_tsfill")

    if fixture_id == "compact_georef":
        months, height, width = 12, 65, 65
        times = pd.date_range("2020-01-01", periods=months, freq="MS")
        y = 240.0 - np.arange(height) * 30.0 - 15.0
        x = np.arange(width) * 30.0 + 15.0
        rng = np.random.default_rng(1201)
        water = (rng.random((months, height, width)) < 0.3).astype(np.uint8)
        water_da = xr.DataArray(
            water,
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x},
        ).rio.write_crs("EPSG:3577")
        return open_water_cube(water_da, input_kind="generic_binary")

    if fixture_id == "long_480_small":
        months, height, width = 480, 16, 16
        times = pd.date_range("1986-01-01", periods=months, freq="MS")
        y = 240.0 - np.arange(height) * 30.0 - 15.0
        x = np.arange(width) * 30.0 + 15.0
        rng = np.random.default_rng(1480)
        water = (rng.random((months, height, width)) < 0.2).astype(np.uint8)
        water_da = xr.DataArray(
            water,
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x},
        ).rio.write_crs("EPSG:3577")
        return open_water_cube(water_da, input_kind="generic_binary")

    if fixture_id == "large_spatial_sparse":
        months, height, width = 4, 128, 128
        times = pd.date_range("2020-01-01", periods=months, freq="MS")
        y = 240.0 - np.arange(height) * 30.0 - 15.0
        x = np.arange(width) * 30.0 + 15.0
        rng = np.random.default_rng(8128)
        water = (rng.random((months, height, width)) < 0.02).astype(np.uint8)
        water_da = xr.DataArray(
            water,
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x},
        ).rio.write_crs("EPSG:3577")
        return open_water_cube(water_da, input_kind="generic_binary")

    if fixture_id == "large_spatial_single_component":
        import dask.array as da

        months, height, width = 2, 96, 96
        times = pd.date_range("2020-01-01", periods=months, freq="MS")
        y = 240.0 - np.arange(height) * 30.0 - 15.0
        x = np.arange(width) * 30.0 + 15.0
        water = np.zeros((months, height, width), dtype=np.int8)
        water[:, 4:92, 4:92] = 1
        valid = np.ones((months, height, width), dtype=bool)
        water_da = xr.DataArray(
            da.from_array(water, chunks=(1, 48, 48)),
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x},
        ).rio.write_crs("EPSG:3577")
        valid_da = xr.DataArray(
            da.from_array(valid, chunks=(1, 48, 48)),
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x},
        )
        return open_water_cube(water_da, valid_obs=valid_da, input_kind="generic_binary")

    raise ValueError(f"unsupported spatial-export fixture_id: {fixture_id!r}")


def _spatial_export_config(
    *,
    output_dir: Path,
    spatial_products: tuple[str, ...],
    raster_formats: tuple[str, ...],
    workers: int,
):
    from hydrofragments.config import HydroConfig

    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
            "compute": {
                "workers": workers,
                "target_chunk_bytes": 256_000,
                "worker_memory_fraction": 0.25,
            },
            "output": {
                "output_dir": str(output_dir),
                "spatial_products": list(spatial_products),
                "raster_formats": list(raster_formats),
            },
        }
    )


def _artifact_bytes(output_dir: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if not output_dir.exists():
        return sizes
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(output_dir).as_posix()
            sizes[rel] = path.stat().st_size
    return sizes


def _worker_peak_rss_bytes() -> int | None:
    try:
        import psutil

        process = psutil.Process()
        peak = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                peak = max(peak, child.memory_info().rss)
            except Exception:
                pass
        return int(peak)
    except Exception:
        return None


def _run_spatial_export(payload: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import rioxarray  # noqa: F401

    from hydrofragments.api import analyze
    from hydrofragments.output.checkpoints import (
        CheckpointMetadata,
        SpatialRasterCheckpoint,
    )
    from hydrofragments.output.finalize import CoreAnalysisResult, finalize_analysis_bundle
    from hydrofragments.output.manifest import validate_result_bundle
    from hydrofragments.output.spatial import SpatialGrid

    scenario_id = payload["scenario_id"]
    fixture_id = payload["fixture_id"]
    spatial_products = tuple(payload.get("spatial_products") or ())
    raster_formats = tuple(payload.get("raster_formats") or ("geotiff",))
    workers = int(payload.get("workers", 1))
    output_dir = Path(payload["output_dir"])
    phase = payload.get("phase", "full")
    check_metric_parity = bool(payload.get("check_metric_parity"))
    expect_failure = bool(payload.get("expect_failure"))
    zarr_path = payload.get("zarr_path")

    if phase == "export_retry":
        checkpoint_state = payload.get("checkpoint_state") or {}
        retry_output = output_dir
        retry_output.mkdir(parents=True, exist_ok=True)
        config = _spatial_export_config(
            output_dir=retry_output,
            spatial_products=spatial_products,
            raster_formats=raster_formats,
            workers=workers,
        )
        cube = _build_spatial_fixture(fixture_id, zarr_path=zarr_path)
        grid = SpatialGrid.from_dataarray(cube.water.isel(time=0), require_georeference=True)
        raster_root = Path(checkpoint_state["raster_checkpoint_root"])
        metadata = CheckpointMetadata.from_json(
            (raster_root / "metadata.json").read_text(encoding="utf-8")
        )
        raster_checkpoint = SpatialRasterCheckpoint(root=raster_root, metadata=metadata)
        raster_checkpoint.validate_complete()
        metrics_table = pd.read_parquet(checkpoint_state["metrics_parquet"])
        coverage_table = pd.read_csv(checkpoint_state["coverage_csv"])
        core = CoreAnalysisResult(
            metrics_table=metrics_table,
            metric_coverage=coverage_table,
            run_id=str(checkpoint_state["run_id"]),
            git_sha=str(checkpoint_state["git_sha"]),
            report_warnings=tuple(checkpoint_state.get("report_warnings", ())),
            skipped_metrics=tuple(
                (item["metric_id"], item["reason"])
                for item in checkpoint_state.get("skipped_metrics", [])
            ),
            execution_plan_mapping=checkpoint_state.get("execution_plan_mapping", {}),
            input_fingerprint=checkpoint_state.get("input_fingerprint", {}),
            comparison_context=checkpoint_state.get("comparison_context", {}),
            raster_checkpoint=raster_checkpoint,
            pool_checkpoint_root=(
                Path(checkpoint_state["pool_checkpoint_root"])
                if checkpoint_state.get("pool_checkpoint_root")
                else None
            ),
            spatial_grid=grid,
        )
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        finalize_analysis_bundle(
            config,
            core,
            cube=cube,
            pixel_size_m=30.0,
        )
        timings["output_finalize"] = time.perf_counter() - t0
        timings["total"] = timings["output_finalize"]
        validate_result_bundle(retry_output)
        return {
            "status": "ok",
            "scenario_id": scenario_id,
            "phase": phase,
            "timings_seconds": timings,
            "source_materializations": 0,
            "label_passes": 0,
            "metrics_digest": _metrics_digest(metrics_table),
            "coverage_digest": _coverage_digest(coverage_table),
            "output_bytes_by_product": _artifact_bytes(retry_output),
            "peak_rss_bytes": _worker_peak_rss_bytes(),
        }

    if "netcdf" in raster_formats:
        try:
            import h5netcdf  # noqa: F401
        except ImportError:
            return {
                "status": "skipped",
                "scenario_id": scenario_id,
                "skipped_reason": "netcdf extra (h5netcdf) is not installed",
            }

    source_materializations = {"count": 0}
    label_passes = {"count": 0}

    from hydrofragments.analysis import window_stream

    original_materialize = window_stream._materialize_window_month

    def counted_materialize(*args, **kwargs):
        source_materializations["count"] += 1
        return original_materialize(*args, **kwargs)

    from hydrofragments.metrics import patches as patch_metrics

    original_measure = patch_metrics.measure_patch_properties

    def counted_measure(*args, **kwargs):
        label_passes["count"] += 1
        return original_measure(*args, **kwargs)

    timings = {}
    metric_parity_holds = None
    metrics_digest_off = None
    coverage_digest_off = None

    try:
        cube = _build_spatial_fixture(fixture_id, zarr_path=zarr_path)
        window_stream._materialize_window_month = counted_materialize
        patch_metrics.measure_patch_properties = counted_measure

        if check_metric_parity:
            from hydrofragments.config import HydroConfig

            off_config = HydroConfig.from_mapping(
                {
                    "config_schema_version": "1.1.0",
                    "input": {"kind": "generic_binary"},
                    "temporal": {
                        "input_cadence": "monthly",
                        "monthly_composite": "supplied",
                        "composite_owner": "caller",
                    },
                    "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
                    "compute": {"workers": workers},
                }
            )
            t0 = time.perf_counter()
            off_result = analyze(cube, "benchmark", config=off_config, pixel_size_m=30.0)
            timings["parity_off_seconds"] = time.perf_counter() - t0
            metrics_digest_off = _metrics_digest(off_result.metrics_table)
            coverage_digest_off = _coverage_digest(off_result.metric_coverage)

        config = _spatial_export_config(
            output_dir=output_dir,
            spatial_products=spatial_products,
            raster_formats=raster_formats,
            workers=workers,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        result = analyze(cube, "benchmark", config=config, pixel_size_m=30.0)
        timings["core_analysis"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        validate_result_bundle(output_dir)
        timings["bundle_validation"] = time.perf_counter() - t0
        timings["total"] = sum(
            value for key, value in timings.items() if key not in {"parity_off_seconds"}
        )

        metrics_digest = _metrics_digest(result.metrics_table)
        coverage_digest = _coverage_digest(result.metric_coverage)
        if metrics_digest_off is not None:
            metric_parity_holds = (
                metrics_digest_off == metrics_digest and coverage_digest_off == coverage_digest
            )

        checkpoint_state = None
        if spatial_products:
            durable_root = output_dir.parent / f"{output_dir.name}.spatial_checkpoints"
            raster_root = durable_root / "spatial_rasters"
            pool_root = durable_root / "pool_vectors"
            manifest = dict(result.manifest)
            backend = dict(manifest.get("backend", {}))
            checkpoint_state = {
                "run_id": result.run_id,
                "git_sha": str(manifest.get("git_sha", "unknown")),
                "metrics_parquet": str(output_dir / "metrics"),
                "coverage_csv": str(output_dir / "metric_coverage.csv"),
                "raster_checkpoint_root": str(raster_root),
                "pool_checkpoint_root": str(pool_root) if pool_root.exists() else None,
                "report_warnings": list(manifest.get("warnings", [])),
                "skipped_metrics": list(manifest.get("skipped_metrics", [])),
                "execution_plan_mapping": {
                    "planned_backend": backend.get("planned", "cpu"),
                    "actual_backend_by_stage": dict(backend.get("actual_by_stage", {})),
                    "backend_capabilities": dict(backend.get("capabilities", {})),
                },
                "input_fingerprint": dict(manifest.get("input_fingerprint", {})),
                "comparison_context": dict(manifest.get("comparison", {})),
            }

        return {
            "status": "ok",
            "scenario_id": scenario_id,
            "phase": phase,
            "fixture_id": fixture_id,
            "spatial_products": list(spatial_products),
            "timings_seconds": timings,
            "metrics_digest": metrics_digest,
            "coverage_digest": coverage_digest,
            "metrics_digest_off": metrics_digest_off,
            "coverage_digest_off": coverage_digest_off,
            "metric_parity_holds": metric_parity_holds,
            "source_materializations": source_materializations["count"],
            "label_passes": label_passes["count"],
            "cube_shape": dict(cube.water.sizes),
            "output_bytes_by_product": _artifact_bytes(output_dir),
            "checkpoint_state": checkpoint_state,
            "peak_rss_bytes": _worker_peak_rss_bytes(),
        }
    except Exception as exc:
        if expect_failure:
            return {
                "status": "expected_failure",
                "scenario_id": scenario_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        raise
    finally:
        if "original_materialize" in locals():
            window_stream._materialize_window_month = original_materialize
        if "original_measure" in locals():
            patch_metrics.measure_patch_properties = original_measure


def main() -> int:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    try:
        result = _run(payload)
    except Exception as exc:  # noqa: BLE001 -- report the failure as data, not a traceback-only crash
        import traceback

        result = {
            "status": "error",
            "mode": payload.get("mode"),
            "candidate_id": payload.get("candidate_id"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    return 0 if result.get("status") in {"ok", "skipped", "expected_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
