"""Read-only fixture characterisation for release readiness evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.contracts.fixture_inspector import (
    inspect_drainage_geopackage,
    inspect_shapefile,
    inspect_water_mask_netcdf,
    inspect_water_mask_zarr,
)

TEST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_DIR.parent
WMASK_PATH = TEST_DIR / "wmask_ts.nc"
RCOR_PATH = TEST_DIR / "rcor_extent.shp"
FITZROY_ZARR_PATH = REPO_ROOT / "data" / "wofs_monthly_masks_1986_2026.zarr"
FITZROY_DRAINAGE_PATH = REPO_ROOT / "data" / "fitzroy_kimberley_drainage.gpkg"


@pytest.fixture(scope="module")
def wmask_report():
    return inspect_water_mask_netcdf(WMASK_PATH)


@pytest.fixture(scope="module")
def rcor_report():
    return inspect_shapefile(RCOR_PATH)


@pytest.fixture(scope="module")
def fitzroy_zarr_report():
    return inspect_water_mask_zarr(FITZROY_ZARR_PATH)


@pytest.fixture(scope="module")
def fitzroy_drainage_report():
    return inspect_drainage_geopackage(FITZROY_DRAINAGE_PATH)


def test_wmask_fixture_exists(wmask_report):
    assert wmask_report["exists"] is True
    assert wmask_report["checksum_sha256"]


def test_wmask_reports_dimensions_crs_and_cadence(wmask_report):
    assert wmask_report["dimensions"]["time"] == 63
    assert wmask_report["dimensions"]["y"] == 145
    assert wmask_report["dimensions"]["x"] == 331
    assert "28351" in wmask_report["crs"]
    assert wmask_report["cadence"]["inferred"] == "sub_monthly_irregular"
    assert wmask_report["time_range"]["start"] == "2018-01-01"
    assert wmask_report["time_range"]["end"] == "2020-12-16"


def test_wmask_reports_value_domain_and_wet_variability(wmask_report):
    domain = wmask_report["value_domain"]
    assert set(domain.keys()) == {"-1", "0", "1"}
    stats = wmask_report["wet_fraction_per_timestep"]
    assert stats["unique_count"] >= 10
    assert stats["min"] < stats["max"]
    assert wmask_report["sentinel_presence"]["uint8_254"] is False
    assert wmask_report["sentinel_presence"]["uint8_255"] is False
    assert wmask_report["sentinel_presence"]["legacy_nodata_-1"] is True


def test_wmask_checksum_is_stable(wmask_report):
    assert (
        wmask_report["checksum_sha256"]
        == "8866c7737a33fa078a9daf74b8f08435ada24096188ee1f76c3cc1487ef698d6"
    )


def test_rcor_reports_seven_polygon_sections(rcor_report):
    assert rcor_report["exists"] is True
    assert rcor_report["feature_count"] == 7
    assert "28351" in rcor_report["crs"]
    assert rcor_report["geometry_types"] == ["Polygon"]


def test_fitzroy_zarr_exists_and_checksum_is_stable(fitzroy_zarr_report):
    assert fitzroy_zarr_report["exists"] is True
    assert (
        fitzroy_zarr_report["zmetadata_checksum_sha256"]
        == "c69f7e8b07064969f7f3785420c794bf459fa9c157e73446ba2e4ced1a36e790"
    )


def test_fitzroy_zarr_reports_dimensions_and_crs(fitzroy_zarr_report):
    assert fitzroy_zarr_report["dimensions"] == {"time": 480, "y": 539, "x": 1117}
    assert fitzroy_zarr_report["crs"] == "EPSG:3577"
    assert fitzroy_zarr_report["dtype"] == "int8"


def test_fitzroy_zarr_value_domain_and_sentinels(fitzroy_zarr_report):
    assert set(fitzroy_zarr_report["value_domain"].keys()) == {"-2", "-1", "0", "1"}
    assert fitzroy_zarr_report["sentinel_presence"]["outside_aoi_-2"] is True
    assert fitzroy_zarr_report["sentinel_presence"]["unobserved_-1"] is True
    assert fitzroy_zarr_report["has_per_pixel_confidence_or_method_flag"] is False


def test_fitzroy_zarr_wet_fraction_variability(fitzroy_zarr_report):
    stats = fitzroy_zarr_report["wet_fraction_per_timestep"]
    assert stats["unique_count"] > 100
    assert stats["min"] < stats["max"]


def test_fitzroy_zarr_reliability_diagnostic_matches_evidence_report(fitzroy_zarr_report):
    obs = fitzroy_zarr_report["observed_frac_of_aoi"]
    assert obs["min"] == 0.0
    assert 0.97 < obs["median"] < 0.99
    assert obs["n_zero_coverage_months"] == 18


def test_fitzroy_drainage_exists_and_checksum_is_stable(fitzroy_drainage_report):
    assert fitzroy_drainage_report["exists"] is True
    assert (
        fitzroy_drainage_report["checksum_sha256"]
        == "004442d0a65a7eeb51a335dbaa621e281f610080b31e7ae05ee9980a46dc3b3a"
    )


def test_fitzroy_zarr_missingness_is_seasonal_mnar(fitzroy_zarr_report):
    """Assert coverage dips when wetness peaks in Fitzroy validation Zarr.

    Monsoon peak (Jan-Mar) wetness is >3x dry season wetness; the stratified
    estimator corrects the naive pooled under-estimation bias.
    """
    mnar = fitzroy_zarr_report["seasonal_mnar"]
    wet_frac_by_month = mnar["per_calendar_month_wet_frac"]

    monsoon_peak_wet_frac = max(wet_frac_by_month[m] for m in (1, 2, 3))
    dry_season_wet_frac = max(wet_frac_by_month[m] for m in (6, 7, 8, 9, 10, 11))
    assert monsoon_peak_wet_frac > dry_season_wet_frac * 3

    assert mnar["stratified_minus_naive"] > 0
    relative_bias = mnar["stratified_minus_naive"] / mnar["naive_pooled_wet_frac"]
    assert 0.03 < relative_bias < 0.15


def test_fitzroy_drainage_is_centreline_with_complete_topology(fitzroy_drainage_report):
    assert fitzroy_drainage_report["feature_count"] == 291
    assert "3577" in fitzroy_drainage_report["crs"]
    assert fitzroy_drainage_report["geometry_types"] == ["MultiLineString"]
    assert fitzroy_drainage_report["has_drainage_centreline"] is True
    assert all(n == 0 for n in fitzroy_drainage_report["topology_null_counts"].values())
