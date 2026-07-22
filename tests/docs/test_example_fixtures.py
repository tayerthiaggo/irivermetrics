"""TDD for the tiny hand-calculable fixture used by examples/01_quickstart.ipynb.

Section 5 -- the quickstart notebook needs a bundled fixture that is small,
fast, grid/CRS-consistent, and has documented ground truth (per the brief's
pointer at tests/fixtures/analytic_masks.py's pattern), wrapped with a real
``time`` dimension so it can flow through ``open_water_cube`` -> ``analyze``.
This test module drives that fixture-construction helper (importable from
``examples._fixtures`` so both the notebook and this test call the exact
same code, not a copy-pasted variant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _import_fixtures():
    import sys
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import _fixtures  # noqa: PLC0415

    return _fixtures


def test_quickstart_cube_has_a_real_time_dimension() -> None:
    fixtures = _import_fixtures()
    water = fixtures.quickstart_water_timeseries()

    assert "time" in water.dims
    assert water.sizes["time"] >= 3


def test_quickstart_cube_is_projected_and_crs_defined() -> None:
    fixtures = _import_fixtures()
    water = fixtures.quickstart_water_timeseries()

    assert water.rio.crs is not None
    assert not water.rio.crs.is_geographic


def test_quickstart_cube_ground_truth_wet_pixel_counts() -> None:
    """Ground truth: a shrinking wet square, 1 fewer ring of pixels per month.

    Matches the fixture's own docstring -- locked here so any accidental
    edit to the generator is caught by a failing test, the same contract
    tests/fixtures/analytic_masks.py fixtures carry.
    """
    fixtures = _import_fixtures()
    water = fixtures.quickstart_water_timeseries()

    wet_counts = water.sum(dim=("y", "x")).values.tolist()
    assert wet_counts == fixtures.QUICKSTART_WET_PIXEL_COUNTS


def test_quickstart_cube_feeds_open_water_cube_end_to_end() -> None:
    from hydrofragments.api import open_water_cube

    fixtures = _import_fixtures()
    water = fixtures.quickstart_water_timeseries()

    cube = open_water_cube(water, input_kind="generic_binary")

    assert cube.water.dtype == bool
    assert cube.valid_obs.all()


def test_quickstart_minimal_config_builds() -> None:
    from hydrofragments.config import HydroConfig

    fixtures = _import_fixtures()
    config = HydroConfig.from_mapping(fixtures.quickstart_config(output_dir="ignored"))

    assert config.input.kind == "generic_binary"


def test_walkthrough_water_timeseries_has_multiple_years() -> None:
    fixtures = _import_fixtures()
    water = fixtures.walkthrough_water_timeseries()

    assert water.sizes["time"] == 36
    assert water.rio.crs is not None and not water.rio.crs.is_geographic


def test_walkthrough_fixture_drives_all_four_metric_families(tmp_path) -> None:
    """The notebook 03 fixture must genuinely produce every metric_family
    the brief's 4 groupings map onto (extent, persistence, morphology,
    fragmentation, clustering, dynamics) through the real analyze() path --
    not a hand-picked subset. Connectivity (realised_connectivity/tcf) is
    intentionally excluded: those require FIXED_NODES/GRAPH dependencies
    analyze()'s AnalysisInputs has no way to supply, so they never appear
    in real analyze() output today (see design decision in section-5
    report).
    """
    from hydrofragments.api import analyze, open_water_cube
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import AnalysisInputs
    from hydrofragments.schema import MetricFamily

    fixtures = _import_fixtures()
    water = fixtures.walkthrough_water_timeseries()
    cube = open_water_cube(water, input_kind="generic_binary")
    context = fixtures.walkthrough_channel_context()
    extent = fixtures.walkthrough_hydroyear_extent(water)
    max_records, median_records = fixtures.walkthrough_apsec_composites(water, extent)

    config = HydroConfig.from_mapping(
        fixtures.walkthrough_config(output_dir=str(tmp_path))
    )
    channel_wet_profiles = (water.isel(y=3) > 0).values
    segment_lengths_m = [30.0] * water.sizes["x"]

    result = analyze(
        cube,
        aoi_id="walkthrough_demo",
        config=config,
        inputs=AnalysisInputs(
            drainage=context,
            channel_wet_profiles=channel_wet_profiles,
            channel_segment_lengths_m=segment_lengths_m,
            hydroyear_extent=extent,
            max_water_apsec=max_records,
            median_apsec=median_records,
        ),
        pixel_size_m=30.0,
    )

    families = set(result.metrics_table["metric_family"])
    expected = {
        MetricFamily.EXTENT.value,
        MetricFamily.PERSISTENCE.value,
        MetricFamily.MORPHOLOGY.value,
        MetricFamily.FRAGMENTATION.value,
        MetricFamily.CLUSTERING.value,
        MetricFamily.DYNAMICS.value,
    }
    assert expected <= families


def test_synthetic_tsfill_zarr_round_trips_through_open_water_cube(tmp_path) -> None:
    """Notebook 02's fixture must route through the real watermask_tsfill adapter."""
    from hydrofragments.api import open_water_cube

    fixtures = _import_fixtures()
    path = tmp_path / "synthetic_tsfill.zarr"
    fixtures.write_synthetic_tsfill_zarr(path)

    cube = open_water_cube(path)

    assert dict(cube.provenance).get("adapter") == "watermask_tsfill"
    # Sentinel pixels must never be counted as valid observations.
    assert not bool(cube.valid_obs.isel(y=0, x=0).any())
    assert not bool(cube.valid_obs.isel(y=0, x=1).any())
