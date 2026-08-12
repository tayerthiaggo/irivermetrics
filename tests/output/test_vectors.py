"""Checkpoint-only monthly pool vector export tests (task 8)."""

from __future__ import annotations

import gc
import weakref

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
from affine import Affine
from rasterio.crs import CRS

from hydrofragments.analysis.window_stream import MeasuredPatchBundle, WindowMonthResult
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.patches import label_and_measure_window
from hydrofragments.output.checkpoints import PoolCheckpointConsumer
from hydrofragments.output.vectors import (
    GPKG_BATCH_TARGET_BYTES,
    MONTHLY_POOLS_LAYER,
    POOL_VECTOR_COLUMNS,
    SPATIAL_GPKG_NAME,
    VectorExportError,
    build_pool_id,
    export_vectors_from_checkpoint,
    polygonize_pool_features,
    validate_pool_geodataframe,
)
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.patches import label_components


def _config(tmp_path, **changes: object) -> HydroConfig:
    output: dict[str, object] = {"spatial_products": ["monthly_pools"]}
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
    return HydroConfig.from_mapping(mapping)


_TEST_CRS = CRS.from_string(
    "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def _grid(shape: tuple[int, int] = (6, 8), pixel_size: float = 30.0) -> SpatialGrid:
    height, width = shape
    return SpatialGrid(
        crs=_TEST_CRS,
        transform=Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, height * pixel_size),
        height=height,
        width=width,
        y_dim="y",
        x_dim="x",
        y=np.arange(height, dtype=float) * pixel_size,
        x=np.arange(width, dtype=float) * pixel_size,
    )


def _block(
    *,
    water: np.ndarray,
    grid: SpatialGrid,
    date: pd.Timestamp,
    window_id: str,
    row_slice: slice,
    col_slice: slice,
    pixel_size_m: float = 30.0,
) -> WindowMonthResult:
    label_measure, _checkpoint = label_and_measure_window(
        water,
        pixel_size_m=pixel_size_m,
        connectivity=8,
        min_patch_pixels=1,
        include_width=False,
        window_id=window_id,
    )
    if label_measure is None:
        bundle = None
    else:
        bundle = MeasuredPatchBundle(
            properties=label_measure.properties,
            labels=label_measure.labels,
            window_id=window_id,
        )
    return WindowMonthResult(
        time_index=0,
        date=date,
        window_id=window_id,
        row_slice=row_slice,
        col_slice=col_slice,
        estimated_live_bytes=water.nbytes,
        metric_partials={},
        water=water,
        valid_obs=np.ones_like(water, dtype=bool),
        patch_bundle=bundle,
    )


def _feed_consumer(
    consumer: PoolCheckpointConsumer,
    blocks: list[WindowMonthResult],
) -> None:
    proxy = consumer.as_consumer()
    for block in blocks:
        proxy.consume(block)
    proxy.finalize()


def test_pool_ids_are_date_scoped_and_stable(tmp_path) -> None:
    grid = _grid()
    config = _config(tmp_path)
    water = np.zeros((4, 4), dtype=bool)
    water[1:3, 1:3] = True
    block = _block(
        water=water,
        grid=grid,
        date=pd.Timestamp("2021-06-01"),
        window_id="window-0001",
        row_slice=slice(0, 4),
        col_slice=slice(0, 4),
    )

    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="ids",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, [block])
    checkpoint = consumer.finalize_checkpoint()

    rows = polygonize_pool_features(
        labels=block.patch_bundle.labels,
        properties=block.patch_bundle.properties,
        date=block.date,
        window_id=block.window_id,
        grid=grid,
        row_slice=block.row_slice,
        col_slice=block.col_slice,
    )
    assert rows[0]["pool_id"] == build_pool_id(
        date=block.date, window_id="window-0001", label_id=1
    )
    exported = export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        exported / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    assert frame.loc[0, "pool_id"] == "2021-06-01:window-0001:1"


def test_window_offsets_are_applied_to_geometry(tmp_path) -> None:
    grid = _grid(shape=(8, 8))
    config = _config(tmp_path)
    water = np.zeros((3, 3), dtype=bool)
    water[1, 1] = True
    block = _block(
        water=water,
        grid=grid,
        date=pd.Timestamp("2020-03-01"),
        window_id="window-0002",
        row_slice=slice(2, 5),
        col_slice=slice(3, 6),
    )

    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="offset",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, [block])
    checkpoint = consumer.finalize_checkpoint()
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    bounds = frame.geometry.iloc[0].bounds
    assert bounds[0] >= 90.0
    assert bounds[2] <= 180.0


def test_multiple_windows_do_not_duplicate_boundary_components(tmp_path) -> None:
    grid = _grid(shape=(6, 6))
    config = _config(tmp_path)
    left_water = np.zeros((3, 3), dtype=bool)
    left_water[1, 1] = True
    right_water = np.zeros((3, 3), dtype=bool)
    right_water[1, 1] = True
    date = pd.Timestamp("2020-05-01")
    blocks = [
        _block(
            water=left_water,
            grid=grid,
            date=date,
            window_id="window-0001",
            row_slice=slice(0, 3),
            col_slice=slice(0, 3),
        ),
        _block(
            water=right_water,
            grid=grid,
            date=date,
            window_id="window-0002",
            row_slice=slice(3, 6),
            col_slice=slice(3, 6),
        ),
    ]

    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="windows",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, blocks)
    checkpoint = consumer.finalize_checkpoint()
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    assert len(frame) == 2
    assert len(frame["pool_id"].unique()) == 2


def test_polygonized_area_and_pixels_match_patch_properties(tmp_path) -> None:
    grid = _grid()
    config = _config(tmp_path)
    water = np.zeros((5, 5), dtype=bool)
    water[1:4, 1:4] = True
    block = _block(
        water=water,
        grid=grid,
        date=pd.Timestamp("2022-01-01"),
        window_id="window-0001",
        row_slice=slice(0, 5),
        col_slice=slice(0, 5),
    )
    prop = block.patch_bundle.properties[0]

    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="parity",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, [block])
    checkpoint = consumer.finalize_checkpoint()
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )

    assert int(frame.loc[0, "n_pixels"]) == prop.area_pixels
    assert frame.loc[0, "area_m2"] == pytest.approx(prop.area_m2, rel=0.02)
    rasterized_pixels = int(round(frame.geometry.iloc[0].area / (30.0 * 30.0)))
    assert rasterized_pixels == prop.area_pixels


def test_filtered_labels_are_not_polygonized(tmp_path) -> None:
    grid = _grid(shape=(4, 4))
    water = np.zeros((4, 4), dtype=bool)
    water[0, 0] = True
    water[2:4, 2:4] = True
    label_result = label_components(water, connectivity=8, min_patch_pixels=2)
    assert label_result.count == 1

    from hydrofragments.metrics.patches import measure_components
    from hydrofragments.patches import iter_component_crops

    properties = measure_components(
        tuple(iter_component_crops(label_result.labels)),
        pixel_size_m=30.0,
        include_width=False,
    )
    rows = polygonize_pool_features(
        labels=label_result.labels,
        properties=properties,
        date=pd.Timestamp("2020-01-01"),
        window_id="window-0001",
        grid=grid,
        row_slice=slice(0, 4),
        col_slice=slice(0, 4),
    )
    assert len(rows) == 1
    assert int(rows[0]["n_pixels"]) == 4


def test_all_dry_month_writes_empty_layer_with_exact_schema(tmp_path) -> None:
    grid = _grid()
    config = _config(tmp_path)
    date = pd.Timestamp("2019-12-01")
    block = WindowMonthResult(
        time_index=0,
        date=date,
        window_id="window-0001",
        row_slice=slice(0, 6),
        col_slice=slice(0, 8),
        estimated_live_bytes=48,
        metric_partials={},
        water=np.zeros((6, 8), dtype=bool),
        valid_obs=np.ones((6, 8), dtype=bool),
        patch_bundle=None,
    )
    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="dry",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, [block])
    checkpoint = consumer.finalize_checkpoint()
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")

    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    validate_pool_geodataframe(frame, crs=grid.crs)
    assert tuple(frame.columns) == POOL_VECTOR_COLUMNS
    assert len(frame) == 0


def test_memory_remains_bounded_across_many_months(tmp_path) -> None:
    grid = _grid(shape=(8, 8))
    config = _config(tmp_path)
    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="memory",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    weak = weakref.ref(consumer)
    proxy = consumer.as_consumer()
    for month in range(24):
        water = np.zeros((4, 4), dtype=bool)
        if month % 2 == 0:
            water[1:3, 1:3] = True
        block = _block(
            water=water,
            grid=grid,
            date=pd.Timestamp(year=2020 + month // 12, month=(month % 12) + 1, day=1),
            window_id="window-0001",
            row_slice=slice(0, 4),
            col_slice=slice(0, 4),
        )
        proxy.consume(block)
        del block
        gc.collect()
    checkpoint = proxy.finalize()
    del consumer, proxy
    gc.collect()
    assert weak() is None
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    assert len(frame) == 12


def test_export_rejects_in_memory_geodataframe(tmp_path) -> None:
    grid = _grid()
    empty = gpd.GeoDataFrame(geometry=[], crs=grid.crs)
    with pytest.raises(VectorExportError, match="in-memory GeoDataFrame"):
        export_vectors_from_checkpoint(empty, tmp_path / "vectors")


def test_incomplete_checkpoint_cannot_be_exported(tmp_path) -> None:
    grid = _grid()
    config = _config(tmp_path)
    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="incomplete",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    with pytest.raises(VectorExportError, match="incomplete"):
        export_vectors_from_checkpoint(consumer.root, tmp_path / "vectors")


def test_gpkg_batch_target_is_64_mib() -> None:
    assert GPKG_BATCH_TARGET_BYTES == 64 * 1024 * 1024


def test_multiple_month_partitions_round_trip(tmp_path) -> None:
    grid = _grid()
    config = _config(tmp_path)
    blocks = [
        _block(
            water=np.pad(np.array([[True]], dtype=bool), 1, constant_values=False),
            grid=grid,
            date=pd.Timestamp("2020-01-01"),
            window_id="window-0001",
            row_slice=slice(0, 3),
            col_slice=slice(0, 3),
        ),
        _block(
            water=np.pad(np.array([[True, True]], dtype=bool), 1, constant_values=False),
            grid=grid,
            date=pd.Timestamp("2020-02-01"),
            window_id="window-0001",
            row_slice=slice(1, 4),
            col_slice=slice(1, 5),
        ),
    ]
    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="months",
        root=tmp_path / "checkpoint",
        export_enabled=True,
    )
    _feed_consumer(consumer, blocks)
    checkpoint = consumer.finalize_checkpoint()
    export_vectors_from_checkpoint(checkpoint, tmp_path / "vectors")
    frame = pyogrio.read_dataframe(
        tmp_path / "vectors" / SPATIAL_GPKG_NAME,
        layer=MONTHLY_POOLS_LAYER,
    )
    assert set(frame["date"].dt.strftime("%Y-%m-%d")) == {"2020-01-01", "2020-02-01"}
