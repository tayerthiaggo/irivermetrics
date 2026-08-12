"""End-to-end integration tests for spatial export finalization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("rioxarray")

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.output.finalize import SpatialProductUnavailable
from hydrofragments.output.manifest import validate_result_bundle


def _georef_cube(*, months: int = 6, shape: tuple[int, int] = (8, 8)) -> "xr.DataArray":
    times = pd.date_range("2020-01-01", periods=months, freq="MS")
    y = 240.0 - np.arange(shape[0]) * 30.0 - 15.0
    x = np.arange(shape[1]) * 30.0 + 15.0
    rng = np.random.default_rng(0)
    water = (rng.random((months, *shape)) < 0.3).astype(np.uint8)
    return xr.DataArray(
        water,
        dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
    ).rio.write_crs("EPSG:3577")


def _config(tmp_path: Path, *, spatial_products: tuple[str, ...] = ()) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {
                "output_dir": str(tmp_path),
                "spatial_products": list(spatial_products),
            },
        }
    )


def test_export_off_and_on_metrics_are_byte_identical(tmp_path: Path) -> None:
    water = _georef_cube()
    cube = open_water_cube(water, input_kind="generic_binary")

    off = analyze(
        cube,
        "demo",
        config=_config(tmp_path / "off"),
        pixel_size_m=30.0,
    )
    on = analyze(
        cube,
        "demo",
        config=_config(tmp_path / "on", spatial_products=("persistence_rasters",)),
        pixel_size_m=30.0,
    )

    left = off.metrics_table.drop(columns=["run_id", "config_hash"])
    right = on.metrics_table.drop(columns=["run_id", "config_hash"])
    pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(off.metric_coverage, on.metric_coverage)


def test_export_on_writes_bundle_and_validates_manifest(tmp_path: Path) -> None:
    water = _georef_cube()
    cube = open_water_cube(water, input_kind="generic_binary")
    output_dir = tmp_path / "bundle"

    analyze(
        cube,
        "demo",
        config=_config(output_dir, spatial_products=("persistence_rasters",)),
        pixel_size_m=30.0,
    )

    manifest = validate_result_bundle(output_dir)
    assert manifest["manifest_schema_version"] == "1.1.0"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics").is_dir()
    assert (output_dir / "metric_coverage.csv").exists()
    assert (output_dir / "rasters" / "occurrence.tif").exists()
    inventory_paths = {item["relative_path"] for item in manifest["artifact_inventory"]}
    assert "rasters/occurrence.tif" in inventory_paths


def test_export_off_does_not_call_spatial_writers(tmp_path: Path) -> None:
    water = _georef_cube()
    cube = open_water_cube(water, input_kind="generic_binary")

    with mock.patch(
        "hydrofragments.output.rasters.export_rasters_from_checkpoint"
    ) as export_rasters, mock.patch(
        "hydrofragments.output.vectors.export_vectors_from_checkpoint"
    ) as export_vectors:
        analyze(
            cube,
            "demo",
            config=_config(tmp_path / "off"),
            pixel_size_m=30.0,
        )
        export_rasters.assert_not_called()
        export_vectors.assert_not_called()


def test_unavailable_zones_product_fails_preflight(tmp_path: Path) -> None:
    water = _georef_cube()
    cube = open_water_cube(water, input_kind="generic_binary")

    with pytest.raises(SpatialProductUnavailable, match="zones requires"):
        analyze(
            cube,
            "demo",
            config=_config(tmp_path / "zones", spatial_products=("zones",)),
            pixel_size_m=30.0,
        )


def test_single_and_multi_worker_metrics_match(tmp_path: Path) -> None:
    water = _georef_cube(months=4, shape=(6, 6))
    cube = open_water_cube(water, input_kind="generic_binary")

    single = analyze(
        cube,
        "demo",
        config=HydroConfig.from_mapping(
            {
                "config_schema_version": "1.1.0",
                "input": {"kind": "generic_binary"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "supplied",
                    "composite_owner": "caller",
                },
                "compute": {"workers": 1},
                "output": {"output_dir": str(tmp_path / "w1")},
            }
        ),
        pixel_size_m=30.0,
    )
    multi = analyze(
        cube,
        "demo",
        config=HydroConfig.from_mapping(
            {
                "config_schema_version": "1.1.0",
                "input": {"kind": "generic_binary"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "supplied",
                    "composite_owner": "caller",
                },
                "compute": {"workers": 2},
                "output": {"output_dir": str(tmp_path / "w2")},
            }
        ),
        pixel_size_m=30.0,
    )

    pd.testing.assert_frame_equal(
        single.metrics_table.drop(columns=["run_id", "config_hash"]),
        multi.metrics_table.drop(columns=["run_id", "config_hash"]),
    )


def test_failed_output_leaves_no_final_manifest(tmp_path: Path) -> None:
    water = _georef_cube()
    cube = open_water_cube(water, input_kind="generic_binary")
    output_dir = tmp_path / "bundle"

    with mock.patch(
        "hydrofragments.output.bundle.BundleTransaction.finalize",
        side_effect=RuntimeError("manifest publication failed"),
    ):
        with pytest.raises(RuntimeError, match="manifest publication failed"):
            analyze(
                cube,
                "demo",
                config=_config(output_dir, spatial_products=("persistence_rasters",)),
                pixel_size_m=30.0,
            )

    assert not output_dir.exists()
    assert not (output_dir / "run_manifest.json").exists()
