"""Integration tests for ``hydrofragments.workflow.analyze_from_dea`` (W4.2).

``analyze_from_dea`` is the single public entry point wiring together
everything workstreams 1-3 and W4.1/W4.3 already built: it calls
hydroseason's public DEA-statistics/acquisition/cache APIs, builds verified
``aoi_mask``/``analysis_mask``, opens the canonical cache, derives
hydro-year/dual-composite inputs automatically, wires channel inputs when
drainage is supplied, calls ``analyze()`` exactly once, and writes final
artifacts (metrics table, metric-coverage table, DEA-enriched manifest).

Following the established convention in ``tests/io/test_dea.py`` and
``tests/io/test_cache_footprints.py``: every hydroseason entry point this
orchestrator calls is monkeypatched as a module attribute on the real
``hydroseason`` package (``monkeypatch.setattr(hydroseason, "<name>", fake,
raising=False)``), because the hydroseason implementation for this plan lives
in a sibling worktree this environment's installed package does not resolve
to. The contract under test is entirely the orchestrator's: call order,
argument forwarding, mask semantics, dependency discovery (drainage
optional), and output files -- not hydroseason's own internals (those have
their own test suites in the hydroseason repo).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("rioxarray")

import hydroseason
from hydrofragments.config import HydroConfig
from shapely import wkb
from shapely.geometry import box, LineString

from hydrofragments.workflow import _dual_extent_inputs, analyze_from_dea


_SHAPE = (4, 4)
_TRANSFORM = (30.0, 0.0, 0.0, 0.0, -30.0, 120.0)
_CRS = "EPSG:3577"


def _aoi() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 120.0, 120.0)]}, crs=_CRS)


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
        },
        geometry=[LineString([(10.0, 10.0), (100.0, 100.0)])],
        crs=_CRS,
    )


def _wo_statistics_dataset(*, crs: str = _CRS) -> "xr.Dataset":
    ny, nx = _SHAPE
    y = 120.0 - np.arange(ny) * 30.0 - 15.0
    x = np.arange(nx) * 30.0 + 15.0

    wet = np.full((ny, nx), 6, dtype=np.int16)
    clear = np.full((ny, nx), 10, dtype=np.int16)
    frequency = 100.0 * wet.astype("float32") / clear.astype("float32")

    ds = xr.Dataset(
        {
            "count_wet": (("y", "x"), wet),
            "count_clear": (("y", "x"), clear),
            "frequency": (("y", "x"), frequency),
        },
        coords={"y": y, "x": x},
    ).rio.write_crs(crs)
    ds.attrs["provenance"] = {
        "product": "ga_ls_wo_fq_myear_3",
        "stac_url": "https://example.test/stac",
        "item_ids": ["item-1"],
        "crs": crs,
        "resolution": 30.0,
        "time_span": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        "frequency": {"derivation": "100 * count_wet / count_clear"},
    }
    return ds


def _handle(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=str(tmp_path / "cache" / "aoi.zarr"),
        identity="identity-digest",
        request_digest="request-digest",
    )


def _footprints() -> SimpleNamespace:
    full = box(0.0, 0.0, 120.0, 120.0)
    return SimpleNamespace(
        aoi_geometry_wkb_hex=wkb.dumps(full).hex(),
        analysis_geometry_wkb_hex=wkb.dumps(full).hex(),
        crs=_CRS,
        shape=_SHAPE,
        transform=_TRANSFORM,
        aoi_pixel_count=16,
        analysis_pixel_count=16,
        aoi_digest="a" * 64,
        analysis_digest="b" * 64,
    )


def _mask_cube(*, months: int = 4) -> "xr.DataArray":
    """A tiny canonical {-2,-1,0,1} monthly water-mask cube."""
    time = pd.date_range("2020-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2, size=(months, *_SHAPE)).astype(np.int16)
    return xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={
            "time": time,
            "y": 120.0 - np.arange(_SHAPE[0]) * 30.0 - 15.0,
            "x": np.arange(_SHAPE[1]) * 30.0 + 15.0,
        },
    )


def _dual_extent_counts(*, months: int = 4) -> "pd.DataFrame":
    time = pd.date_range("2020-01-01", periods=months, freq="MS")
    return pd.DataFrame(
        {
            "aoi_pixel_count": [16] * months,
            "analysis_mask_pixel_count": [16] * months,
            "n_max_water": [8] * months,
            "n_median_water": [6] * months,
            "n_valid_analysis": [16] * months,
        },
        index=pd.DatetimeIndex(time),
    )


class _Recorder:
    """Records every call made to each patched hydroseason entry point."""

    def __init__(self) -> None:
        self.open_wo_statistics_calls: list[dict] = []
        self.build_wet_planning_footprint_calls: list[dict] = []
        self.acquire_wofs_cache_calls: list[dict] = []
        self.open_completed_mask_cache_calls: list[dict] = []
        self.verify_cache_footprints_calls: list[object] = []
        self.open_completed_dual_extent_counts_calls: list[dict] = []


def _install_happy_path(monkeypatch, recorder: _Recorder, tmp_path: Path, *, months: int = 4):
    def fake_open_wo_statistics(aoi, **kwargs):
        recorder.open_wo_statistics_calls.append(kwargs)
        return _wo_statistics_dataset()

    def fake_build_wet_planning_footprint(stats, **kwargs):
        recorder.build_wet_planning_footprint_calls.append(kwargs)
        return SimpleNamespace(
            native_mask=None,
            coarse_mask=None,
            active_windows=(),
            factor=kwargs.get("factor", 4),
            safety_cells=kwargs.get("safety_cells", 1),
            digest="footprint-digest",
            covered_years=list(kwargs.get("requested_years", [])),
            source_collection="ga_ls_wo_fq_myear_3",
            source_version="2.1.0",
            source_lineage="ga_ls_wo_3",
        )

    def fake_acquire_wofs_cache(*args, **kwargs):
        recorder.acquire_wofs_cache_calls.append(kwargs)
        return _handle(tmp_path)

    def fake_open_completed_mask_cache(handle, start_date, end_date, **kwargs):
        recorder.open_completed_mask_cache_calls.append(
            {"handle": handle, "start_date": start_date, "end_date": end_date, **kwargs}
        )
        return _mask_cube(months=months)

    def fake_verify_cache_footprints(handle):
        recorder.verify_cache_footprints_calls.append(handle)
        return _footprints()

    def fake_open_completed_dual_extent_counts(handle, start_date, end_date):
        recorder.open_completed_dual_extent_counts_calls.append(
            {"handle": handle, "start_date": start_date, "end_date": end_date}
        )
        return _dual_extent_counts(months=months)

    monkeypatch.setattr(hydroseason, "open_wo_statistics", fake_open_wo_statistics, raising=False)
    monkeypatch.setattr(
        hydroseason._io_dea_stats,
        "build_wet_planning_footprint",
        fake_build_wet_planning_footprint,
        raising=False,
    )
    monkeypatch.setattr(hydroseason, "acquire_wofs_cache", fake_acquire_wofs_cache, raising=False)
    monkeypatch.setattr(
        hydroseason, "open_completed_mask_cache", fake_open_completed_mask_cache, raising=False
    )
    monkeypatch.setattr(
        hydroseason, "verify_cache_footprints", fake_verify_cache_footprints, raising=False
    )
    monkeypatch.setattr(
        hydroseason,
        "open_completed_dual_extent_counts",
        fake_open_completed_dual_extent_counts,
        raising=False,
    )
    return recorder


# ---------------------------------------------------------------------------
# Step 1: call order, argument forwarding, mask semantics, output files
# ---------------------------------------------------------------------------


def test_analyze_from_dea_calls_hydroseason_apis_in_order(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
        config=None,
    )

    assert len(recorder.open_wo_statistics_calls) == 1
    assert len(recorder.build_wet_planning_footprint_calls) == 1
    assert len(recorder.acquire_wofs_cache_calls) == 1
    assert len(recorder.verify_cache_footprints_calls) == 1
    assert len(recorder.open_completed_mask_cache_calls) == 1
    assert len(recorder.open_completed_dual_extent_counts_calls) == 1


def test_analyze_from_dea_forwards_planning_footprint_into_acquisition(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    call = recorder.acquire_wofs_cache_calls[0]
    assert call["wet_mask"] == "dea_stats"
    assert call["planning_footprint"] is not None
    assert call["planning_footprint"].digest == "footprint-digest"
    assert call["composite_bundle"] == "hydrofragments_v1"


def test_analyze_from_dea_returns_hydro_result_with_metrics(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    assert not result.metrics_table.empty
    assert not result.metric_coverage.empty
    assert result.run_id


def test_analyze_from_dea_writes_output_files(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    output_dir = Path(result.output_dir)
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics").exists()
    assert (output_dir / "metric_coverage.csv").exists()


def test_analyze_from_dea_manifest_carries_dea_provenance(monkeypatch, tmp_path):
    import json

    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    manifest = json.loads((Path(result.output_dir) / "run_manifest.json").read_text())
    assert "dea_provenance" in manifest
    assert manifest["dea_provenance"]["product"] == "ga_ls_wo_fq_myear_3"
    assert manifest["dea_provenance"]["planning_footprint"]["digest"] == "footprint-digest"


def test_analyze_from_dea_records_phase_timings(monkeypatch, tmp_path):
    import json

    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    manifest = json.loads((Path(result.output_dir) / "run_manifest.json").read_text())
    timings = manifest["timings_seconds"]
    assert "dea_planning" in timings
    assert "wofs_query_and_acquisition" in timings
    assert "metric_processing" in timings
    assert "output_write" in timings
    assert "total" in timings
    assert timings["total"] >= sum(
        v for k, v in timings.items() if k != "total"
    ) - 1e-6


def test_analyze_from_dea_returned_result_manifest_is_dea_enriched(monkeypatch, tmp_path):
    """``result.manifest`` in memory must match the DEA-enriched manifest
    write_run_metadata() writes to disk -- not analyze()'s own
    pre-enrichment manifest.

    A caller who reads ``result.manifest`` directly (rather than reopening
    ``run_manifest.json`` from disk, as the two tests above both do) must
    see the same ``timings_seconds``/``dea_provenance`` this module's own
    docstring promises ("Phase timings recorded in the run manifest's
    timings_seconds"). Regression test for a real bug: analyze_from_dea()
    called write_run_metadata() to write the enriched manifest, then
    returned analyze()'s original ``result`` object unchanged, so
    ``result.manifest`` in memory silently lacked both fields even though
    the on-disk file was correct.
    """
    import json

    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    assert "timings_seconds" in result.manifest
    assert "dea_planning" in result.manifest["timings_seconds"]
    assert "dea_provenance" in result.manifest
    assert result.manifest["dea_provenance"]["product"] == "ga_ls_wo_fq_myear_3"

    on_disk = json.loads((Path(result.output_dir) / "run_manifest.json").read_text())
    in_memory = dict(result.manifest)
    in_memory.pop("manifest_path", None)
    assert in_memory == on_disk


def test_analyze_from_dea_without_drainage_skips_channel_metrics(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
        drainage=None,
    )

    coverage = result.metric_coverage
    lpsec_row = coverage.loc[coverage["metric"] == "lpsec"]
    assert not lpsec_row.empty
    assert lpsec_row.iloc[0]["status"] != "computed"


def test_analyze_from_dea_with_drainage_computes_channel_metrics(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
        drainage=_drainage(),
    )

    coverage = result.metric_coverage
    lpsec_row = coverage.loc[coverage["metric"] == "lpsec"]
    assert not lpsec_row.empty
    assert lpsec_row.iloc[0]["status"] == "computed"


# ---------------------------------------------------------------------------
# Step 3: cache hit makes zero STAC calls, same metrics/coverage
# ---------------------------------------------------------------------------


def test_second_call_delegates_resume_decision_to_acquire_wofs_cache(
    monkeypatch, tmp_path
):
    """This orchestrator never second-guesses hydroseason's own cache-hit
    short-circuiting: it always calls ``acquire_wofs_cache`` exactly once per
    ``analyze_from_dea`` invocation and lets THAT function decide internally
    whether any STAC/remote work is actually needed (its own documented
    resumability contract -- "Queries STAC exactly once for the whole
    interval ... writes one annual Zarr group per calendar year not already
    completed"). This test proves the orchestrator's half of that contract:
    it adds no redundant DEA-statistics query of its own across repeated
    calls, and never bypasses ``acquire_wofs_cache`` on a would-be cache hit
    (e.g. via some independent "is this cached?" pre-check). The actual
    "zero STAC calls on a real cache hit" property is hydroseason's own
    internal behaviour and is exercised by hydroseason's own test suite
    (``tests/test_io_wofs_acquire.py``) -- it cannot be proven here, since
    this test's ``acquire_wofs_cache`` is a fake that always returns
    immediately regardless of cache state."""
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    analyze_from_dea(
        _aoi(), "2020-01-01", "2020-04-30", aoi_id="test_aoi", cache_dir=tmp_path / "wofs_cache"
    )
    analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
        config=HydroConfig.from_mapping(
            {
                "config_schema_version": "1.0.0",
                "input": {"kind": "watermask_tsfill"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "max_water",
                    "composite_owner": "upstream",
                },
                "output": {"output_dir": str(tmp_path / "output_second")},
            }
        ),
    )

    # One open_wo_statistics + one build_wet_planning_footprint call per
    # analyze_from_dea invocation (planning is not itself cached by this
    # orchestrator) -- exactly two each, never more, and acquire_wofs_cache
    # is called exactly once per invocation regardless of whether it does
    # any real work internally.
    assert len(recorder.open_wo_statistics_calls) == 2
    assert len(recorder.acquire_wofs_cache_calls) == 2


def test_cache_hit_emits_same_metrics_and_coverage_tables(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    first = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )
    second = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
        config=HydroConfig.from_mapping(
            {
                "config_schema_version": "1.0.0",
                "input": {"kind": "watermask_tsfill"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "max_water",
                    "composite_owner": "upstream",
                },
                "output": {"output_dir": str(tmp_path / "output_repeat")},
            }
        ),
    )

    pd.testing.assert_frame_equal(
        first.metrics_table.drop(columns=["run_id", "config_hash"]),
        second.metrics_table.drop(columns=["run_id", "config_hash"]),
    )
    pd.testing.assert_frame_equal(first.metric_coverage, second.metric_coverage)


# ---------------------------------------------------------------------------
# Step 4: failure-mode tests
# ---------------------------------------------------------------------------


def test_stats_unavailable_falls_open_to_full_aoi_acquisition(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    def fake_open_wo_statistics_raises(aoi, **kwargs):
        raise hydroseason._io_dea_stats.WoStatisticsUnavailable("STAC unreachable")

    monkeypatch.setattr(
        hydroseason, "open_wo_statistics", fake_open_wo_statistics_raises, raising=False
    )

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    assert not result.metrics_table.empty
    call = recorder.acquire_wofs_cache_calls[0]
    assert call["wet_mask"] == "off"
    assert call.get("planning_footprint") is None
    assert len(recorder.build_wet_planning_footprint_calls) == 0


def test_dea_stats_unavailable_from_footprint_build_falls_open(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    def fake_build_raises(stats, **kwargs):
        raise hydroseason._io_dea_stats.DEAStatsUnavailable("no wet pixels")

    monkeypatch.setattr(
        hydroseason._io_dea_stats,
        "build_wet_planning_footprint",
        fake_build_raises,
        raising=False,
    )

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    assert not result.metrics_table.empty
    call = recorder.acquire_wofs_cache_calls[0]
    assert call["wet_mask"] == "off"
    assert call.get("planning_footprint") is None


def test_invalid_mask_digest_propagates_and_is_not_swallowed(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    def fake_verify_raises(handle):
        raise ValueError("cache footprint 'analysis' digest mismatch")

    monkeypatch.setattr(
        hydroseason, "verify_cache_footprints", fake_verify_raises, raising=False
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        analyze_from_dea(
            _aoi(),
            "2020-01-01",
            "2020-04-30",
            aoi_id="test_aoi",
            cache_dir=tmp_path / "wofs_cache",
        )


def test_dual_extent_counts_unavailable_skips_dynamics_without_crashing(monkeypatch, tmp_path):
    recorder = _Recorder()
    _install_happy_path(monkeypatch, recorder, tmp_path)

    monkeypatch.setattr(
        hydroseason,
        "open_completed_dual_extent_counts",
        lambda handle, start_date, end_date: None,
        raising=False,
    )

    result = analyze_from_dea(
        _aoi(),
        "2020-01-01",
        "2020-04-30",
        aoi_id="test_aoi",
        cache_dir=tmp_path / "wofs_cache",
    )

    assert not result.metrics_table.empty
    coverage = result.metric_coverage
    extent_contraction_row = coverage.loc[coverage["metric"] == "extent_contraction"]
    assert not extent_contraction_row.empty
    assert extent_contraction_row.iloc[0]["status"] != "computed"


# ---------------------------------------------------------------------------
# Regression: dual-composite APSEC/hydro-year-extent must be denominated by
# the full-catchment ``aoi_pixel_count``, never the conservative
# ``analysis_mask_pixel_count`` -- these are two intentionally distinct
# denominators (see the plan's Global Constraints and
# ``hydrofragments.metrics.extent``'s module docstring / ``compute_apsec``).
# The fixtures used everywhere else in this file set both columns to the same
# value (16), which makes this particular bug invisible; this test uses
# deliberately different values so a wrong-column read produces a wrong
# number.
# ---------------------------------------------------------------------------


def test_dual_extent_inputs_denominates_by_aoi_pixel_count_not_analysis_mask():
    months = 2
    time = pd.date_range("2020-01-01", periods=months, freq="MS")
    aoi_pixel_count = 20
    analysis_mask_pixel_count = 16
    n_max_water = [8, 10]
    n_median_water = [6, 7]
    dual_counts = pd.DataFrame(
        {
            "aoi_pixel_count": [aoi_pixel_count] * months,
            "analysis_mask_pixel_count": [analysis_mask_pixel_count] * months,
            "n_max_water": n_max_water,
            "n_median_water": n_median_water,
            "n_valid_analysis": [analysis_mask_pixel_count] * months,
        },
        index=pd.DatetimeIndex(time),
    )

    hydroyear_extent, max_water_apsec, median_apsec = _dual_extent_inputs(dual_counts)

    # Expected values computed by hand against aoi_pixel_count (20), NOT
    # analysis_mask_pixel_count (16) -- would fail against the pre-fix code,
    # which divided by analysis_mask_pixel_count instead.
    expected_extent_pct = [
        100.0 * n / aoi_pixel_count for n in n_max_water
    ]
    assert hydroyear_extent is not None
    assert hydroyear_extent.tolist() == pytest.approx(expected_extent_pct)

    assert max_water_apsec is not None
    for record, n in zip(max_water_apsec, n_max_water):
        assert record.value == pytest.approx(100.0 * n / aoi_pixel_count)
        assert record.a_ref_m2 == pytest.approx(float(aoi_pixel_count))

    assert median_apsec is not None
    for record, n in zip(median_apsec, n_median_water):
        assert record.value == pytest.approx(100.0 * n / aoi_pixel_count)
        assert record.a_ref_m2 == pytest.approx(float(aoi_pixel_count))
