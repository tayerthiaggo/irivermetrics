"""Tests for hydrofragments.io.dea: the WoStatistics adapter over
hydroseason's native open_wo_statistics loader (task W1.1).

hydroseason.open_wo_statistics is monkeypatched as a module attribute for
every test here (via ``monkeypatch.setattr(hydroseason, "open_wo_statistics",
..., raising=False)``) rather than exercised directly, because this task's
hydroseason-side counterpart lives in a sibling worktree that is not what
this environment's installed `hydroseason` package resolves to. The
contract under test is entirely this adapter's: convert whatever Dataset
open_wo_statistics returns into the frozen WoStatistics dataclass, and
enforce the CRS guard the brief requires this layer (not hydroseason) to own.
"""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
da = pytest.importorskip("dask.array")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

import hydroseason
from shapely.geometry import box

from hydrofragments.guards.scientific import ScientificGuardError
from hydrofragments.io.dea import WoStatistics, open_wo_statistics_for_zoning


def _aoi():
    return gpd.GeoDataFrame({"geometry": [box(0.0, -90.0, 90.0, 0.0)]}, crs="EPSG:3577")


def _dataset(*, crs="EPSG:3577", nodata=-1, shape=(3, 3), count_wet=3, count_clear=10):
    ny, nx = shape
    y = -np.arange(ny) * 30.0
    x = np.arange(nx) * 30.0

    wet = np.full((ny, nx), count_wet, dtype=np.int16)
    clear = np.full((ny, nx), count_clear, dtype=np.int16)
    wet[0, 0] = nodata
    clear[0, 0] = nodata

    wet_da = da.from_array(wet, chunks=(2, 2))
    clear_da = da.from_array(clear, chunks=(2, 2))
    frequency = 100.0 * wet_da.astype("float32") / clear_da.astype("float32")

    ds = xr.Dataset(
        {
            "count_wet": (("y", "x"), wet_da),
            "count_clear": (("y", "x"), clear_da),
            "frequency": (("y", "x"), frequency),
        },
        coords={"y": y, "x": x},
    ).rio.write_crs(crs)
    ds.attrs["provenance"] = {
        "product": "ga_ls_wo_fq_myear_3",
        "stac_url": "https://example.test/stac",
        "item_ids": ["item-1", "item-2"],
        "crs": crs,
        "resolution": 30.0,
        "time_span": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        "frequency": {
            "derivation": "100 * count_wet / count_clear",
            "count_wet": "count_wet",
            "count_clear": "count_clear",
        },
    }
    return ds


def test_calls_hydroseason_open_wo_statistics_once(monkeypatch):
    calls = []

    def fake_loader(aoi, **kwargs):
        calls.append(kwargs)
        return _dataset()

    monkeypatch.setattr(hydroseason, "open_wo_statistics", fake_loader, raising=False)

    open_wo_statistics_for_zoning(_aoi())

    assert len(calls) == 1


def test_returns_frozen_wostatistics_dataclass(monkeypatch):
    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", lambda aoi, **kwargs: _dataset(), raising=False
    )

    result = open_wo_statistics_for_zoning(_aoi())

    assert isinstance(result, WoStatistics)
    with pytest.raises(Exception):
        result.product = "changed"  # frozen dataclass must refuse mutation


def test_wostatistics_carries_required_fields(monkeypatch):
    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", lambda aoi, **kwargs: _dataset(), raising=False
    )

    result = open_wo_statistics_for_zoning(_aoi(), product="ga_ls_wo_fq_myear_3")

    assert result.product == "ga_ls_wo_fq_myear_3"
    assert result.crs is not None
    assert result.version == hydroseason.__version__
    assert result.time_span is not None
    assert isinstance(result.provenance, dict)
    assert "item_ids" in result.provenance


def test_frequency_is_0_to_100_float32_and_lazy(monkeypatch):
    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", lambda aoi, **kwargs: _dataset(), raising=False
    )

    result = open_wo_statistics_for_zoning(_aoi())

    assert isinstance(result.frequency.data, da.Array)
    assert result.frequency.dtype == np.float32
    computed = result.frequency.compute()
    valid = computed.values[1:, 1:]
    assert np.allclose(valid, 100.0 * 3.0 / 10.0)
    assert np.nanmax(computed.values) <= 100.0


def test_count_wet_and_count_clear_are_preserved_dask_backed(monkeypatch):
    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", lambda aoi, **kwargs: _dataset(), raising=False
    )

    result = open_wo_statistics_for_zoning(_aoi())

    assert isinstance(result.count_wet.data, da.Array)
    assert isinstance(result.count_clear.data, da.Array)


def test_provenance_records_frequency_derivation(monkeypatch):
    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", lambda aoi, **kwargs: _dataset(), raising=False
    )

    result = open_wo_statistics_for_zoning(_aoi())

    assert result.provenance["frequency"]["derivation"] == "100 * count_wet / count_clear"
    assert result.provenance["product"] == "ga_ls_wo_fq_myear_3"
    assert result.provenance["item_ids"] == ["item-1", "item-2"]


def test_geographic_crs_is_hard_rejected_via_guard_area_metric_crs(monkeypatch):
    """The brief requires calling guard_area_metric_crs, not duplicating the
    check. Since hydroseason has no pyproj dependency and cannot import this
    guard (one-way dependency rule), this adapter -- the HydroFragments side
    -- is the one that must call it and hard-fail on a geographic CRS."""
    monkeypatch.setattr(
        hydroseason,
        "open_wo_statistics",
        lambda aoi, **kwargs: _dataset(crs="EPSG:4326"),
        raising=False,
    )

    with pytest.raises(ScientificGuardError):
        open_wo_statistics_for_zoning(_aoi())


def test_native_resolution_is_not_a_scientific_resolution_knob(monkeypatch):
    """resolution selects the native grid explicitly; it must not become a
    scientific-resolution knob for downstream zoning (global constraint /
    brief requirement). The adapter must pass resolution straight through to
    hydroseason's loader rather than reinterpreting or defaulting it to
    something coarser."""
    captured = {}

    def fake_loader(aoi, **kwargs):
        captured.update(kwargs)
        return _dataset()

    monkeypatch.setattr(hydroseason, "open_wo_statistics", fake_loader, raising=False)

    open_wo_statistics_for_zoning(_aoi(), resolution=30.0)

    assert captured["resolution"] == 30.0


def test_wostatistics_object_is_frozen_dataclass_instance():
    import dataclasses

    assert dataclasses.is_dataclass(WoStatistics)
    fields = {f.name for f in dataclasses.fields(WoStatistics)}
    assert fields >= {
        "frequency", "count_wet", "count_clear", "product", "version",
        "crs", "time_span", "provenance",
    }


def test_loader_failure_propagates_as_wo_statistics_unavailable_equivalent(monkeypatch):
    """A hydroseason-side loader failure (e.g. WoStatisticsUnavailable) must
    not be swallowed -- callers using this as a zoning source need to see the
    failure to fall back to their local-cube zoning path."""

    class _Boom(RuntimeError):
        pass

    def fake_loader(aoi, **kwargs):
        raise _Boom("STAC unreachable")

    monkeypatch.setattr(hydroseason, "open_wo_statistics", fake_loader, raising=False)

    with pytest.raises(_Boom):
        open_wo_statistics_for_zoning(_aoi())
