"""Bounded spatial raster checkpoint accumulator tests (task 7)."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import (
    compute_hydroperiod,
    compute_occurrence,
    compute_recurrence,
)
from hydrofragments.output.checkpoints import (
    CheckpointError,
    SpatialRasterCheckpointAccumulator,
    grid_from_dataarray,
    try_open_completed_checkpoint,
)
from hydrofragments.output.rasters import (
    build_persistence_rasters_from_checkpoint,
    build_refuge_stability_rasters_from_checkpoint,
    build_temporal_rasters_from_checkpoint,
)


def _config(tmp_path=None, **changes: object) -> HydroConfig:
    output: dict[str, object] = {}
    if tmp_path is not None:
        output["output_dir"] = str(tmp_path)
    mapping: dict[str, object] = {
        "config_schema_version": "1.1.0",
        "input": {"kind": "watermask_tsfill"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "upstream",
        },
        "persistence": {"refuge_threshold": 0.90},
        "validity": {"min_valid_obs": 1, "min_valid_fraction_month": 0.1},
        "output": output,
    }
    mapping.update(changes)
    if "output" in changes and isinstance(changes["output"], dict):
        merged_output = dict(output)
        merged_output.update(changes["output"])
        if merged_output.get("spatial_products") and not merged_output.get("output_dir"):
            merged_output["output_dir"] = str(tmp_path or ".")
        mapping["output"] = merged_output
    return HydroConfig.from_mapping(mapping)


def _monthly_dataset(
    *,
    years: int = 3,
    shape: tuple[int, int] = (2, 3),
    start_year: int = 2020,
) -> xr.Dataset:
    months = []
    water_rows = []
    valid_rows = []
    for year in range(start_year, start_year + years):
        for month in range(1, 13):
            months.append(pd.Timestamp(year=year, month=month, day=1))
            base = (year + month) % 2
            water = np.full(shape, base, dtype=bool)
            valid = np.ones(shape, dtype=bool)
            if month == 2:
                water[0, 0] = False
                valid[0, 1] = False
            water_rows.append(water)
            valid_rows.append(valid)
    return xr.Dataset(
        {
            "water": xr.DataArray(
                np.stack(water_rows),
                dims=("time", "y", "x"),
                coords={"time": months},
            ),
            "valid_obs": xr.DataArray(
                np.stack(valid_rows),
                dims=("time", "y", "x"),
                coords={"time": months},
            ),
        }
    )


def _feed_monthly(
    accumulator: SpatialRasterCheckpointAccumulator,
    monthly: xr.Dataset,
) -> None:
    for time_index, timestamp in enumerate(monthly["time"].values):
        water = np.asarray(monthly["water"].isel(time=time_index).values, dtype=bool)
        valid = np.asarray(monthly["valid_obs"].isel(time=time_index).values, dtype=bool)
        ts = pd.Timestamp(timestamp)
        accumulator.add_month(
            calendar_month=int(ts.month),
            calendar_year=int(ts.year),
            water=water,
            valid_obs=valid,
            timestamp=ts,
        )


def _end_dry_anchors(years: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for hy, year in enumerate(years, start=1):
        rows.append(
            {
                "hy": hy,
                "end_dry_month": pd.Timestamp(year=year, month=4, day=1),
                "confidence": "high",
            }
        )
    return pd.DataFrame(rows)


def test_occurrence_checkpoint_matches_eager_reference(tmp_path) -> None:
    monthly = _monthly_dataset(years=3)
    config = _config(tmp_path, output={"spatial_products": ["persistence_rasters"]})
    reference = compute_occurrence(monthly, config=config)

    template = monthly["water"].isel(time=0)
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("persistence_rasters",),
        input_fingerprint="demo",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_monthly(accumulator, monthly)
    checkpoint = accumulator.finalize_checkpoint()
    result = build_persistence_rasters_from_checkpoint(checkpoint, config=config)

    np.testing.assert_allclose(
        result["occurrence"].values,
        reference.occurrence.values,
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_array_equal(result["valid_count"].values, reference.valid_count.values)


def test_recurrence_and_hydroperiod_match_eager_reference(tmp_path) -> None:
    monthly = _monthly_dataset(years=3)
    config = _config(tmp_path, output={"spatial_products": ["temporal_rasters"]})
    reference_recurrence = compute_recurrence(monthly, config=config)
    reference_hydroperiod = compute_hydroperiod(monthly, config=config)

    template = monthly["water"].isel(time=0)
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("temporal_rasters",),
        input_fingerprint="demo",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_monthly(accumulator, monthly)
    checkpoint = accumulator.finalize_checkpoint()
    temporal = build_temporal_rasters_from_checkpoint(checkpoint, config=config)

    np.testing.assert_allclose(
        temporal["recurrence"].values,
        reference_recurrence.recurrence.values,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        temporal["recurrence_valid_year_count"].values,
        reference_recurrence.valid_year_count.values,
    )
    np.testing.assert_allclose(
        temporal["hydroperiod"].values,
        reference_hydroperiod.hydroperiod.values,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        temporal["hydroperiod_valid_month_count"].values,
        reference_hydroperiod.valid_observed_months.values,
    )


def test_refuge_stability_frequency_for_alternating_and_dry_pairs(tmp_path) -> None:
    shape = (2, 2)
    config = _config(
        tmp_path,
        output={"spatial_products": ["refuge_stability_rasters"]},
        validity={"min_valid_obs": 1, "min_valid_fraction_month": 0.1},
    )
    template = xr.DataArray(np.zeros(shape), dims=("y", "x"))
    grid = grid_from_dataarray(template)
    anchors = _end_dry_anchors((2020, 2021, 2022, 2023))

    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid,
        config=config,
        products=("refuge_stability_rasters",),
        input_fingerprint="refuge",
        template=template,
        end_dry_anchors=anchors,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )

    states = {
        2020: (np.array([[True, False], [False, False]], dtype=bool), np.ones(shape, bool)),
        2021: (np.array([[True, True], [False, False]], dtype=bool), np.ones(shape, bool)),
        2022: (np.array([[False, False], [False, False]], dtype=bool), np.ones(shape, bool)),
        2023: (np.array([[False, False], [False, False]], dtype=bool), np.ones(shape, bool)),
    }
    for year, (water, valid) in states.items():
        accumulator.add_month(
            calendar_month=4,
            calendar_year=year,
            water=water,
            valid_obs=valid,
            timestamp=pd.Timestamp(year=year, month=4, day=1),
        )

    checkpoint = accumulator.finalize_checkpoint()
    rasters = build_refuge_stability_rasters_from_checkpoint(checkpoint, config=config)

    frequency = rasters["refuge_stability_frequency"].values
    union_count = rasters["refuge_stability_union_pair_count"].values
    assert frequency[0, 0] == pytest.approx(50.0)
    assert union_count[0, 0] == 2
    assert frequency[0, 1] == pytest.approx(0.0)
    assert union_count[0, 1] == 2


def test_refuge_stability_skips_nonconsecutive_and_missing_anchor_pairs(tmp_path) -> None:
    shape = (1, 1)
    config = _config(
        tmp_path,
        output={"spatial_products": ["refuge_stability_rasters"]},
        validity={"min_valid_obs": 1, "min_valid_fraction_month": 0.1},
    )
    template = xr.DataArray(np.zeros(shape), dims=("y", "x"))
    grid = grid_from_dataarray(template)
    anchors = pd.DataFrame(
        [
            {"hy": 1, "end_dry_month": pd.NaT, "confidence": "low"},
            {"hy": 3, "end_dry_month": pd.Timestamp("2022-04-01"), "confidence": "high"},
        ]
    )
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid,
        config=config,
        products=("refuge_stability_rasters",),
        input_fingerprint="edges",
        template=template,
        end_dry_anchors=anchors,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    water = np.array([[True]], dtype=bool)
    valid = np.array([[True]], dtype=bool)
    accumulator.add_month(
        calendar_month=4,
        calendar_year=2022,
        water=water,
        valid_obs=valid,
        timestamp=pd.Timestamp("2022-04-01"),
    )
    checkpoint = accumulator.finalize_checkpoint()
    rasters = build_refuge_stability_rasters_from_checkpoint(checkpoint, config=config)
    assert np.all(np.isnan(rasters["refuge_stability_frequency"].values))


def test_uint32_overflow_guard_raises(tmp_path) -> None:
    config = _config(tmp_path)
    template = xr.DataArray(np.zeros((1, 1)), dims=("y", "x"))
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("persistence_rasters",),
        input_fingerprint="overflow",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    max_value = np.iinfo(np.uint32).max
    accumulator._group["valid_count_total"][...] = np.uint32(max_value)
    with pytest.raises(OverflowError, match="uint32"):
        accumulator.add_month(
            calendar_month=1,
            calendar_year=2020,
            water=np.array([[True]], dtype=bool),
            valid_obs=np.array([[True]], dtype=bool),
            timestamp=pd.Timestamp("2020-01-01"),
        )


def test_completed_checkpoint_can_be_reused(tmp_path) -> None:
    monthly = _monthly_dataset(years=2)
    config = _config(tmp_path, output={"spatial_products": ["persistence_rasters"]})
    template = monthly["water"].isel(time=0)
    grid = grid_from_dataarray(template)
    root = tmp_path / "checkpoint"

    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid,
        config=config,
        products=("persistence_rasters",),
        input_fingerprint="reuse",
        template=template,
        root=root,
        export_enabled=True,
    )
    _feed_monthly(accumulator, monthly)
    completed = accumulator.finalize_checkpoint()

    reopened = try_open_completed_checkpoint(
        root,
        grid=grid,
        scientific_config_hash=config.config_hash,
        products=("persistence_rasters",),
        input_fingerprint="reuse",
    )
    assert reopened is not None
    assert reopened.metadata.completed is True
    assert (root / "COMPLETED").exists()

    mismatched = try_open_completed_checkpoint(
        root,
        grid=grid,
        scientific_config_hash="different",
        products=("persistence_rasters",),
        input_fingerprint="reuse",
    )
    assert mismatched is None

    rebuilt = build_persistence_rasters_from_checkpoint(completed, config=config)
    assert "occurrence" in rebuilt.data_vars


def test_incomplete_checkpoint_cannot_be_exported(tmp_path) -> None:
    monthly = _monthly_dataset(years=1)
    config = _config(tmp_path, output={"spatial_products": ["persistence_rasters"]})
    template = monthly["water"].isel(time=0)
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("persistence_rasters",),
        input_fingerprint="incomplete",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_monthly(accumulator, monthly)
    from hydrofragments.output.checkpoints import CheckpointMetadata, SpatialRasterCheckpoint

    incomplete = SpatialRasterCheckpoint(
        root=accumulator.root,
        metadata=CheckpointMetadata.from_json(
            (accumulator.root / "metadata.json").read_text(encoding="utf-8")
        ),
    )
    with pytest.raises(CheckpointError, match="incomplete"):
        build_persistence_rasters_from_checkpoint(incomplete, config=config)


def test_large_spatial_fixture_retains_bounded_live_arrays(tmp_path) -> None:
    n_y, n_x = 64, 64
    months = pd.date_range("2020-01-01", periods=24, freq="MS")
    water = np.random.default_rng(0).integers(0, 2, size=(len(months), n_y, n_x), dtype=bool)
    valid = np.ones_like(water, dtype=bool)
    monthly = xr.Dataset(
        {
            "water": (("time", "y", "x"), water),
            "valid_obs": (("time", "y", "x"), valid),
        },
        coords={"time": months},
    )
    config = _config(
        tmp_path,
        compute={"target_chunk_bytes": 16_384},
        output={"spatial_products": ["persistence_rasters", "temporal_rasters"]},
    )
    template = monthly["water"].isel(time=0)
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid_from_dataarray(template),
        config=config,
        products=("persistence_rasters", "temporal_rasters"),
        input_fingerprint="memory",
        template=template,
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    weak = weakref.ref(accumulator)
    _feed_monthly(accumulator, monthly)
    checkpoint = accumulator.finalize_checkpoint()
    del accumulator
    gc.collect()

    assert weak() is None
    assert checkpoint.metadata.chunk_inventory
    assert len(checkpoint.metadata.chunk_inventory) < 24
