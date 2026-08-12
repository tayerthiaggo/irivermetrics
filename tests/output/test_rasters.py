"""Milestone 5 — spatial raster products for the persistence core.

Plan §5 requires the minimal core to emit ``rasters/occurrence``,
``valid_count``, and a configured refuge mask, each carrying provenance. This
module builds those xarray products (writing to disk is exercised by the M7
output tranche).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import compute_occurrence
from hydrofragments.output.rasters import (
    build_persistence_rasters,
    build_persistence_rasters_from_checkpoint,
)


def _config(refuge_threshold: float = 0.90, min_valid_obs: int = 1) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "upstream",
            },
            "persistence": {"refuge_threshold": refuge_threshold},
            "validity": {"min_valid_obs": min_valid_obs},
        }
    )


def _monthly() -> xr.Dataset:
    # 1x2 grid: pixel (0,0) perennial, pixel (0,1) dry. 5 Januaries.
    times = pd.to_datetime([f"200{y}-01-01" for y in range(1, 6)])
    water = np.array([[[1, 0]]] * 5)
    valid = np.ones_like(water)
    dims = ("time", "y", "x")
    return xr.Dataset(
        {
            "water": xr.DataArray(
                water.astype(bool), dims=dims, coords={"time": times}
            ),
            "valid_obs": xr.DataArray(
                valid.astype(bool), dims=dims, coords={"time": times}
            ),
        }
    )


def _rasters():
    config = _config(min_valid_obs=3)
    occ = compute_occurrence(_monthly(), config=config)
    return build_persistence_rasters(occ, config=config), occ


def test_builds_occurrence_valid_count_and_refuge_rasters():
    rasters, _ = _rasters()
    assert "occurrence" in rasters.data_vars
    assert "valid_count" in rasters.data_vars
    assert "refuge_mask" in rasters.data_vars


def test_occurrence_raster_matches_source_surface():
    rasters, occ = _rasters()
    xr.testing.assert_equal(rasters["occurrence"], occ.occurrence)


def test_refuge_mask_marks_pixels_at_or_above_threshold():
    rasters, _ = _rasters()
    # Perennial pixel occurrence 100% >= 90% -> refuge; dry pixel not.
    assert bool(rasters["refuge_mask"].isel(y=0, x=0).item()) is True
    assert bool(rasters["refuge_mask"].isel(y=0, x=1).item()) is False


def test_rasters_carry_provenance_attrs():
    rasters, _ = _rasters()
    assert rasters.attrs["refuge_threshold"] == pytest.approx(0.90)
    assert rasters.attrs["min_valid_obs"] == 3
    assert rasters.attrs["validity_policy"] == "p_native_season_stratified_v1"


def test_refuge_mask_excludes_thin_support_even_if_high_occurrence():
    # Build a surface directly: high occurrence but valid_count below the floor.
    config = _config(refuge_threshold=0.90, min_valid_obs=20)
    from hydrofragments.metrics.persistence import OccurrenceResult

    occ = OccurrenceResult(
        occurrence=xr.DataArray(np.array([[99.0]]), dims=("y", "x")),
        valid_count=xr.DataArray(np.array([[5]]), dims=("y", "x")),
        min_valid_obs=20,
    )
    rasters = build_persistence_rasters(occ, config=config)
    assert bool(rasters["refuge_mask"].isel(y=0, x=0).item()) is False


def test_writes_reopenable_self_contained_raster_artifacts(tmp_path: Path):
    from hydrofragments.output.rasters import write_persistence_rasters

    rasters, _ = _rasters()

    artifacts = write_persistence_rasters(rasters, tmp_path / "rasters")

    assert artifacts == {
        "occurrence": tmp_path / "rasters" / "occurrence",
        "valid_count": tmp_path / "rasters" / "valid_count",
        "refuge_mask": tmp_path / "rasters" / "refuge_mask",
    }
    for name, path in artifacts.items():
        reopened = xr.open_zarr(path)
        xr.testing.assert_equal(reopened[name], rasters[name])
        assert reopened.attrs == rasters.attrs
        reopened.close()


def test_zarr_is_declared_for_canonical_raster_output():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"zarr>=' in pyproject


def test_build_persistence_rasters_from_checkpoint_matches_direct_builder(tmp_path: Path):
    from hydrofragments.output.checkpoints import SpatialRasterCheckpointAccumulator, grid_from_dataarray

    config = _config(min_valid_obs=3)
    occ = compute_occurrence(_monthly(), config=config)
    direct = build_persistence_rasters(occ, config=config)

    monthly = _monthly()
    template = monthly["water"].isel(time=0)
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("persistence_rasters",),
        input_fingerprint="rasters",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    for time_index, timestamp in enumerate(monthly["time"].values):
        ts = pd.Timestamp(timestamp)
        accumulator.add_month(
            calendar_month=int(ts.month),
            calendar_year=int(ts.year),
            water=np.asarray(monthly["water"].isel(time=time_index).values, dtype=bool),
            valid_obs=np.asarray(monthly["valid_obs"].isel(time=time_index).values, dtype=bool),
            timestamp=ts,
        )
    checkpoint = accumulator.finalize_checkpoint()
    from_checkpoint = build_persistence_rasters_from_checkpoint(checkpoint, config=config)

    np.testing.assert_allclose(
        from_checkpoint["occurrence"].values,
        direct["occurrence"].values,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        from_checkpoint["valid_count"].values,
        direct["valid_count"].values,
    )
    np.testing.assert_array_equal(
        from_checkpoint["refuge_mask"].values,
        direct["refuge_mask"].values,
    )
