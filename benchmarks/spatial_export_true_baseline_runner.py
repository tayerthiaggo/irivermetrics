"""Run export-off compact_georef timing against a frozen HydroFragments commit.

Invoked from :func:`hydrofragments.benchmarks.end_to_end_workflow._run_true_baseline_export_off_trial`
with ``PYTHONPATH`` pointing at the detached worktree for
``SPATIAL_EXPORT_TRUE_BASELINE_COMMIT``. Reads one JSON object from stdin and
writes one JSON result object to stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


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


def _build_compact_georef_cube():
    import numpy as np
    import pandas as pd
    import rioxarray  # noqa: F401
    import xarray as xr

    from hydrofragments.api import open_water_cube

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


def run(payload: dict[str, Any]) -> dict[str, Any]:
    from hydrofragments.api import analyze
    from hydrofragments.config import HydroConfig

    output_dir = Path(payload["output_dir"])
    workers = int(payload.get("workers", 1))
    output_dir.mkdir(parents=True, exist_ok=True)

    cube = _build_compact_georef_cube()
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
            "compute": {"workers": workers},
            "output": {"output_dir": str(output_dir)},
        }
    )

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    analyze(cube, "benchmark", config=config, pixel_size_m=30.0)
    timings["core_analysis"] = time.perf_counter() - t0
    timings["bundle_validation"] = 0.0
    timings["total"] = timings["core_analysis"]

    return {
        "status": "ok",
        "scenario_id": "baseline_export_off",
        "phase": "full",
        "fixture_id": "compact_georef",
        "spatial_products": [],
        "timings_seconds": timings,
        "source_materializations": None,
        "label_passes": None,
        "code_commit": payload.get("code_commit"),
        "peak_rss_bytes": _worker_peak_rss_bytes(),
    }


def main() -> int:
    payload = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(run(payload)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
