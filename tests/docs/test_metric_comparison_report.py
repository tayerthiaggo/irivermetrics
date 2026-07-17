"""Completeness and offline contract for the metric comparison report."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "metric_comparison_report.html"

REQUIRED_IDS = {
    "section_area_km2",
    "section_length_km",
    "wet_area_km2",
    "wet_length_km",
    "wet_perimeter_km",
    "npools",
    "awmsi_legacy",
    "awre_legacy",
    "awmpa",
    "awmpl",
    "awmpw",
    "apsec_legacy",
    "lpsec_legacy",
    "pf",
    "plf",
    "pp_mean",
    "ra_area",
    "pixel_persistence",
    "nni",
    "pcf",
    "centrality",
    "occurrence",
    "refuge_area",
    "apsec",
    "number_of_pools",
    "lpi",
    "awre",
    "awmsi",
    "recurrence",
    "hydroperiod",
    "extent_contraction",
    "reconnection_timing",
    "refuge_spatial_stability",
    "lpsec",
    "inter_pool_gap",
    "mesh",
    "pool_width",
    "realised_connectivity",
    "tcf",
}


def _report() -> str:
    """Read the report as UTF-8 from the repository root."""

    return REPORT.read_text(encoding="utf-8")


def test_report_is_self_contained_and_has_metric_records() -> None:
    text = _report()

    assert "const metricRecords" in text
    assert "<script src=" not in text.lower()
    assert '<link rel="stylesheet"' not in text.lower()
    assert "fetch(" not in text


def test_all_metric_ids_and_required_fields_are_present() -> None:
    text = _report()

    for metric_id in REQUIRED_IDS:
        assert f'id: "{metric_id}"' in text

    assert text.count("equation:") >= len(REQUIRED_IDS)
    assert text.count("citation:") >= len(REQUIRED_IDS)
    assert text.count("source:") >= len(REQUIRED_IDS)
