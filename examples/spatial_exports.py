#!/usr/bin/env python
"""Offline spatial export example.

Runs a small synthetic georeferenced cube through ``analyze()`` with persistence
rasters enabled, validates the result bundle, and prints a readback summary.

No DEA credentials or private data paths are required::

    python examples/spatial_exports.py

Optional output directory (must not exist)::

    python examples/spatial_exports.py --output-dir examples/spatial_exports_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.output.manifest import validate_result_bundle


def build_synthetic_cube(*, months: int = 6, shape: tuple[int, int] = (12, 12)) -> xr.DataArray:
    """Deterministic georeferenced monthly water mask on EPSG:3577."""
    times = pd.date_range("2020-01-01", periods=months, freq="MS")
    y = 240.0 - np.arange(shape[0]) * 30.0 - 15.0
    x = np.arange(shape[1]) * 30.0 + 15.0
    rng = np.random.default_rng(0)
    water = (rng.random((months, *shape)) < 0.35).astype(np.uint8)
    return xr.DataArray(
        water,
        dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
        name="water",
    ).rio.write_crs("EPSG:3577")


def run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_dir}")

    cube = open_water_cube(build_synthetic_cube(), input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(output_dir),
                "spatial_products": ["persistence_rasters"],
            },
        }
    )

    result = analyze(cube, aoi_id="spatial_exports_demo", config=config, pixel_size_m=30.0)
    print(f"run_id: {result.run_id}")
    print(f"metrics rows: {len(result.metrics_table)}")
    print(result.metrics_table[["metric", "value"]].head())

    manifest = validate_result_bundle(output_dir)
    print(f"manifest schema: {manifest['manifest_schema_version']}")
    inventory = {item["relative_path"] for item in manifest["artifact_inventory"]}
    print("artifacts:", ", ".join(sorted(inventory)))

    occurrence_path = output_dir / "rasters" / "occurrence.tif"
    occurrence = xr.open_dataarray(occurrence_path).squeeze("band", drop=True)
    finite = occurrence.where(np.isfinite(occurrence))
    if int(finite.count()) > 0:
        print(
            "occurrence readback:",
            f"shape={tuple(occurrence.shape)}",
            f"min={float(finite.min()):.1f}%",
            f"max={float(finite.max()):.1f}%",
            f"crs={occurrence.rio.crs}",
        )
    else:
        print(
            "occurrence readback:",
            f"shape={tuple(occurrence.shape)}",
            "all nodata",
            f"crs={occurrence.rio.crs}",
        )
    occurrence.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/spatial_exports_out"),
        help="Final run directory (created by analyze; must not pre-exist)",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
