"""Spatial raster builders and verified GeoTIFF / NetCDF writer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("rioxarray")

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import compute_occurrence
from hydrofragments.output.checkpoints import (
    SpatialRasterCheckpointAccumulator,
    grid_from_dataarray,
)
from hydrofragments.output.rasters import (
    NETCDF_SPATIAL_FILENAME,
    RASTER_PRODUCT_CONTRACTS,
    RASTER_TILE_SIZE,
    RasterExportError,
    _crs_equal,
    build_persistence_rasters,
    build_persistence_rasters_from_checkpoint,
    export_rasters_from_checkpoint,
    preflight_raster_artifacts,
    write_geotiff_from_dataarray,
    write_verified_geotiff,
    write_verified_netcdf,
    write_zones_geotiff,
)
from hydrofragments.output.spatial import SpatialGrid


def _config(refuge_threshold: float = 0.90, min_valid_obs: int = 1, **changes: object) -> HydroConfig:
    mapping: dict[str, object] = {
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
    mapping.update(changes)
    return HydroConfig.from_mapping(mapping)


def _monthly() -> xr.Dataset:
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


def _georef_template(shape: tuple[int, int] = (3, 4)) -> tuple[xr.DataArray, SpatialGrid]:
    y = np.linspace(100.0, 40.0, shape[0])
    x = np.linspace(10.0, 70.0, shape[1])
    da = xr.DataArray(np.zeros(shape, dtype=float), dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:3577")
    grid = SpatialGrid.from_dataarray(da, require_georeference=True)
    assert grid is not None
    return da, grid


def _metadata() -> dict[str, str]:
    return {
        "algorithm_version": "1.0.0",
        "scientific_config_hash": "abc123",
    }


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
    assert bool(rasters["refuge_mask"].isel(y=0, x=0).item()) is True
    assert bool(rasters["refuge_mask"].isel(y=0, x=1).item()) is False


def test_rasters_carry_provenance_attrs():
    rasters, _ = _rasters()
    assert rasters.attrs["refuge_threshold"] == pytest.approx(0.90)
    assert rasters.attrs["min_valid_obs"] == 3
    assert rasters.attrs["validity_policy"] == "p_native_season_stratified_v1"


def test_refuge_mask_excludes_thin_support_even_if_high_occurrence():
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


@pytest.mark.parametrize(
    ("product_key", "values"),
    [
        (
            "occurrence",
            np.array([[100.0, 0.0], [50.0, np.nan]], dtype=np.float32),
        ),
        (
            "valid_observation_count",
            np.array([[12, 0], [3, 4294967294]], dtype=np.uint32),
        ),
        (
            "refuge_mask",
            np.array([[1, 0], [1, 0]], dtype=np.uint8),
        ),
        (
            "recurrence",
            np.array([[80.0, 20.0], [np.nan, 0.0]], dtype=np.float32),
        ),
        (
            "recurrence_valid_year_count",
            np.array([[3, 1], [0, 65534]], dtype=np.uint16),
        ),
        (
            "refuge_stability_frequency",
            np.array([[75.0, np.nan], [0.0, 100.0]], dtype=np.float32),
        ),
        (
            "refuge_stability_union_pair_count",
            np.array([[2, 0], [1, 65534]], dtype=np.uint16),
        ),
    ],
)
def test_geotiff_roundtrip_single_band_products(
    tmp_path: Path,
    product_key: str,
    values: np.ndarray,
) -> None:
    template, grid = _georef_template(values.shape)
    data = template.copy(data=values)
    contract = RASTER_PRODUCT_CONTRACTS[product_key]
    path = tmp_path / contract.filename
    write_geotiff_from_dataarray(
        data,
        path,
        grid=grid,
        contract=contract,
        metadata=_metadata(),
        source_name=product_key,
    )

    import rasterio

    with rasterio.open(path) as dataset:
        assert dataset.is_tiled
        assert dataset.block_shapes[0] == (
            min(RASTER_TILE_SIZE, values.shape[0]) if values.shape[0] >= 16 else 16,
            min(RASTER_TILE_SIZE, values.shape[1]) if values.shape[1] >= 16 else 16,
        )
        assert dataset.compression.name == "deflate"
        assert _crs_equal(dataset.crs, grid.crs)
        assert dataset.transform == grid.transform
        assert dataset.descriptions[0] == product_key
        reopened = dataset.read(1)
        if contract.dtype == np.dtype(np.float32):
            np.testing.assert_allclose(
                reopened,
                values,
                rtol=0,
                atol=1e-5,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(reopened, values)


def test_geotiff_roundtrip_hydroperiod_stack(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 3))
    years = [2020, 2021]
    hydroperiod = np.array(
        [
            [[0.5, 1.0, np.nan], [0.0, 0.25, 0.75]],
            [[1.0, 0.0, 0.5], [np.nan, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )
    valid_months = np.array(
        [
            [[12, 6, 0], [3, 12, 8]],
            [[10, 0, 4], [255, 7, 1]],
        ],
        dtype=np.uint8,
    )
    hydro_da = xr.DataArray(
        hydroperiod,
        dims=("year", "y", "x"),
        coords={"year": years, "y": template.y, "x": template.x},
    ).rio.set_spatial_dims(y_dim="y", x_dim="x").rio.write_crs("EPSG:3577")
    valid_da = xr.DataArray(
        valid_months,
        dims=("year", "y", "x"),
        coords={"year": years, "y": template.y, "x": template.x},
    ).rio.set_spatial_dims(y_dim="y", x_dim="x").rio.write_crs("EPSG:3577")

    hydro_path = tmp_path / RASTER_PRODUCT_CONTRACTS["hydroperiod"].filename
    valid_path = tmp_path / RASTER_PRODUCT_CONTRACTS["hydroperiod_valid_month_count"].filename
    write_geotiff_from_dataarray(
        hydro_da,
        hydro_path,
        grid=grid,
        contract=RASTER_PRODUCT_CONTRACTS["hydroperiod"],
        metadata=_metadata(),
        source_name="hydroperiod",
    )
    write_geotiff_from_dataarray(
        valid_da,
        valid_path,
        grid=grid,
        contract=RASTER_PRODUCT_CONTRACTS["hydroperiod_valid_month_count"],
        metadata=_metadata(),
        source_name="hydroperiod_valid_month_count",
    )

    import rasterio

    with rasterio.open(hydro_path) as dataset:
        assert dataset.count == 2
        assert list(dataset.descriptions) == ["calendar_year=2020", "calendar_year=2021"]
        np.testing.assert_allclose(dataset.read(1), hydroperiod[0], equal_nan=True)
        np.testing.assert_allclose(dataset.read(2), hydroperiod[1], equal_nan=True)

    with rasterio.open(valid_path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), valid_months[0])
        np.testing.assert_array_equal(dataset.read(2), valid_months[1])


def test_geotiff_roundtrip_refuge_overlap_stack(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 2))
    overlap = np.array(
        [
            [[0, 1], [2, 3]],
            [[255, 255], [255, 255]],
        ],
        dtype=np.uint8,
    )
    labels = ["HY1-HY2 end_dry=2020-04-01 end_dry=2021-04-01", "HY2-HY3 unsupported"]
    overlap_da = xr.DataArray(
        overlap,
        dims=("hy_pair", "y", "x"),
        coords={"hy_pair": labels, "y": template.y, "x": template.x},
        attrs={"band_descriptions": labels},
    ).rio.set_spatial_dims(y_dim="y", x_dim="x").rio.write_crs("EPSG:3577")
    path = tmp_path / RASTER_PRODUCT_CONTRACTS["refuge_overlap"].filename
    write_geotiff_from_dataarray(
        overlap_da,
        path,
        grid=grid,
        contract=RASTER_PRODUCT_CONTRACTS["refuge_overlap"],
        metadata=_metadata(),
        source_name="refuge_overlap",
    )

    import rasterio

    with rasterio.open(path) as dataset:
        assert list(dataset.descriptions) == labels
        np.testing.assert_array_equal(dataset.read(1), overlap[0])
        np.testing.assert_array_equal(dataset.read(2), overlap[1])


def test_geotiff_roundtrip_zones(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 2))
    zones = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    path = tmp_path / RASTER_PRODUCT_CONTRACTS["zones"].filename
    write_zones_geotiff(zones, path, grid=grid, metadata=_metadata())

    import rasterio

    with rasterio.open(path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), zones)


def test_preflight_rejects_existing_final_artifact(tmp_path: Path) -> None:
    path = tmp_path / RASTER_PRODUCT_CONTRACTS["occurrence"].filename
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(RasterExportError, match="refusing to overwrite"):
        preflight_raster_artifacts(
            tmp_path,
            filenames=[RASTER_PRODUCT_CONTRACTS["occurrence"].filename],
        )


def test_shifted_grid_input_is_rejected(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 2))
    shifted = template.copy(data=np.ones((2, 2), dtype=np.float32))
    shifted = shifted.assign_coords(x=shifted.x + 30.0)
    path = tmp_path / "occurrence.tif"
    with pytest.raises(ValueError, match="align"):
        write_geotiff_from_dataarray(
            shifted,
            path,
            grid=grid,
            contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
            metadata=_metadata(),
            source_name="occurrence",
        )


def test_missing_crs_input_is_rejected(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 2))
    da = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": template.y, "x": template.x},
    )
    path = tmp_path / "occurrence.tif"
    with pytest.raises(ValueError, match="CRS|grid contract"):
        write_geotiff_from_dataarray(
            da,
            path,
            grid=grid,
            contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
            metadata=_metadata(),
            source_name="occurrence",
        )


def test_truncated_geotiff_fails_validation(tmp_path: Path) -> None:
    template, grid = _georef_template((2, 2))
    values = np.ones((2, 2), dtype=np.float32)
    path = tmp_path / RASTER_PRODUCT_CONTRACTS["occurrence"].filename
    write_geotiff_from_dataarray(
        template.copy(data=values),
        path,
        grid=grid,
        contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
        metadata=_metadata(),
        source_name="occurrence",
    )
    truncated = path.read_bytes()[:32]
    path.write_bytes(truncated)
    from hydrofragments.output.rasters import validate_geotiff

    with pytest.raises(Exception):
        validate_geotiff(
            path,
            grid=grid,
            contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
            band_descriptions=["occurrence"],
            metadata={**_metadata(), "source_name": "occurrence"},
        )


def test_export_rasters_from_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _config(
        min_valid_obs=1,
        validity={"min_valid_obs": 1, "min_valid_fraction_month": 0.1},
    )
    template, grid = _georef_template((2, 2))
    times = pd.to_datetime([f"200{y}-01-01" for y in range(1, 6)])
    water = np.array([[[1, 0], [0, 1]]] * 5)
    valid = np.ones_like(water)
    monthly = xr.Dataset(
        {
            "water": xr.DataArray(
                water.astype(bool),
                dims=("time", "y", "x"),
                coords={"time": times, "y": template.y, "x": template.x},
            ).rio.set_spatial_dims(y_dim="y", x_dim="x").rio.write_crs("EPSG:3577"),
            "valid_obs": xr.DataArray(
                valid.astype(bool),
                dims=("time", "y", "x"),
                coords={"time": times, "y": template.y, "x": template.x},
            ).rio.set_spatial_dims(y_dim="y", x_dim="x").rio.write_crs("EPSG:3577"),
        }
    )
    anchors = pd.DataFrame(
        [
            {"hy": 1, "end_dry_month": pd.Timestamp("2001-01-01"), "confidence": "high"},
            {"hy": 2, "end_dry_month": pd.Timestamp("2002-01-01"), "confidence": "high"},
        ]
    )
    accumulator = SpatialRasterCheckpointAccumulator.create(
        grid=grid,
        config=config,
        products=(
            "persistence_rasters",
            "temporal_rasters",
            "refuge_stability_rasters",
        ),
        input_fingerprint="export",
        template=template,
        end_dry_anchors=anchors,
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
    zones = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    artifacts = export_rasters_from_checkpoint(
        checkpoint,
        tmp_path / "rasters",
        config=config,
        raster_formats=("geotiff",),
        zone_mask=zones,
    )

    expected = {
        "occurrence",
        "valid_observation_count",
        "refuge_mask",
        "recurrence",
        "recurrence_valid_year_count",
        "hydroperiod",
        "hydroperiod_valid_month_count",
        "refuge_overlap",
        "refuge_stability_frequency",
        "refuge_stability_union_pair_count",
        "zones",
    }
    assert expected == set(artifacts.keys())
    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_netcdf_optional_extra_declared() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "h5netcdf>=1.4" in pyproject


def _require_netcdf_backend() -> None:
    """Skip unless h5py and h5netcdf are healthy in this pytest session."""
    import importlib
    import sys

    try:
        import h5py  # noqa: F401
    except ImportError:
        pytest.skip("h5py not installed")

    for name in list(sys.modules):
        if name == "h5netcdf" or name.startswith("h5netcdf."):
            del sys.modules[name]
    try:
        h5netcdf = importlib.import_module("h5netcdf")
    except ImportError:
        pytest.skip("h5netcdf not installed")
    if getattr(h5netcdf.core, "no_h5py", False):
        pytest.skip("h5py backend not available for h5netcdf")


def test_netcdf_roundtrip_when_extra_installed(tmp_path: Path) -> None:
    _require_netcdf_backend()
    template, grid = _georef_template((2, 2))
    dataset = xr.Dataset(
        {
            "occurrence": (("y", "x"), np.array([[100.0, 0.0], [50.0, np.nan]], dtype=np.float32)),
            "refuge_mask": (("y", "x"), np.array([[1, 0], [0, 1]], dtype=np.uint8)),
        },
        coords={"y": template.y, "x": template.x},
    )
    path = tmp_path / NETCDF_SPATIAL_FILENAME
    write_verified_netcdf(dataset, path, grid=grid, metadata=_metadata())
    reopened = xr.open_dataset(path)
    xr.testing.assert_allclose(reopened["occurrence"], dataset["occurrence"], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(reopened["refuge_mask"].values, dataset["refuge_mask"].values)
    reopened.close()


def test_netcdf_missing_extra_raises_actionable_error(tmp_path: Path) -> None:
    import builtins
    import sys

    template, grid = _georef_template((2, 2))
    dataset = xr.Dataset(
        {"occurrence": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"y": template.y, "x": template.x},
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5netcdf":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    path = tmp_path / NETCDF_SPATIAL_FILENAME
    with pytest.raises(RasterExportError, match="netcdf extra"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(builtins, "__import__", fake_import)
            if "h5netcdf" in sys.modules:
                monkeypatch.delitem(sys.modules, "h5netcdf", raising=False)
            write_verified_netcdf(dataset, path, grid=grid, metadata=_metadata())

    _require_netcdf_backend()
