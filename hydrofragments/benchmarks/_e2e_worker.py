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
function calls ``section_compat_rows`` with no ``executor_kind`` override,
so it is always ``"thread"`` on this call path regardless of what a
candidate requests (confirmed by reading ``hydrofragments/api.py``'s call
site directly, not assumed) -- reaching ``"process"`` would require calling
``section_compat_rows`` directly instead of ``analyze()``, which would
duplicate ``analyze()``'s own orchestration (metric-plan resolution, other
metric records, output assembly) well beyond what this benchmark needs.
W3.2's own benchmark already gated serial-vs-thread-vs-process at the
``section_compat_rows`` level directly on synthetic data (see its report);
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


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
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
            "config_schema_version": "1.2.0",
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

    # "native-wet mask coverage exactly 100%" -- every AOI pixel-month in the
    # raw native mask cube hydroseason.open_completed_mask_cache returns must
    # be a VALID observation. Confirmed by direct inspection of a real
    # acquired cube (not assumed): this cube's actual convention is
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
    aoi_mask_arr = np.asarray(verified_footprints.aoi_mask)
    native_wet_mask = np.asarray(mask_cube.values) if hasattr(mask_cube, "values") else None
    coverage_fraction = None
    if native_wet_mask is not None:
        valid_mask = (native_wet_mask == 0) | (native_wet_mask == 1)
        aoi_true = np.count_nonzero(aoi_mask_arr)
        if aoi_true > 0:
            inside_covered = np.count_nonzero(valid_mask & aoi_mask_arr[np.newaxis, :, :])
            coverage_fraction = float(inside_covered) / float(aoi_true * native_wet_mask.shape[0])

    return {
        "status": "ok",
        "mode": mode,
        "candidate_id": payload["candidate_id"],
        "timings_seconds": timings,
        "metrics_digest": metrics_digest,
        "metrics_row_count": int(len(metrics_table)),
        "n_water_by_month": n_water_by_month,
        "native_wet_mask_coverage_fraction": coverage_fraction,
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
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
