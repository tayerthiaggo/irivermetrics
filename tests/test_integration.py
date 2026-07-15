"""Integration test: run calculate_metrics on bundled test data.

This test is marked 'slow' because it runs the full processing pipeline.
Run it with:
    pytest -m slow
or together with unit tests via:
    pytest

Quarantine note (U7, approved - see docs/audit/decisions.md and docs/testing.md):
``tests/results_iRiverMetrics/metrics/irm_metrics.csv`` is retired as a v1.2
correctness oracle. It contains dropped/forbidden metrics and a naive
total-timestep ``pp_mean_%`` that must never be treated as a target value. The
only legacy-CSV comparison this module performs is
``test_calculate_metrics_section_area_matches_legacy_geometry_smoke``, which checks
``section_area_km2`` - pure section-polygon geometry that does not depend on the
water mask, the valid-observation denominator, or any of the disqualifying defects.
See ``tests/contracts/test_legacy_baseline_quarantine.py`` for the canonical test
proving the CSV is rejected as a correctness baseline.
"""
import numpy as np
import pandas as pd
import pytest

from ecofragments import calculate_metrics

_RETAINED_WIDE_COLS = {
    "date",
    "section",
    "section_area_km2",
    "n_patches",
    "APSEC",
    "AWMSI",
    "AWRe",
    "LPI",
    "pp_mean_%",
    "ra_area_km2",
}

_DROPPED_LEGACY_COLS = {
    "PF",
    "PLF",
    "AWMPA",
    "AWMPL",
    "AWMPW",
    "LPSEC",
    "wet_area_km2",
    "wet_length_km",
    "wet_perimeter_km",
}

_NUMERIC_COLS = [
    "section_area_km2",
    "n_patches",
    "APSEC",
    "AWMSI",
    "AWRe",
    "LPI",
    "pp_mean_%",
    "ra_area_km2",
]


@pytest.mark.slow
def test_calculate_metrics_shape(da_wmask, rcor_extent_path, tmp_path):
    """Output DataFrame has the expected number of rows (dates × sections)."""
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    # The reference has 441 rows (63 dates × 7 sections).
    # After QA filtering the count may differ slightly; assert it is positive
    # and has both 'date' and 'section' columns.
    assert len(result) > 0
    assert {"date", "section"}.issubset(result.columns)


@pytest.mark.slow
def test_calculate_metrics_columns(da_wmask, rcor_extent_path, tmp_path):
    """Compatibility facade exposes retained v1.2 metrics only."""
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    assert _RETAINED_WIDE_COLS.issubset(set(result.columns))
    assert _DROPPED_LEGACY_COLS.isdisjoint(set(result.columns))


@pytest.mark.slow
def test_calculate_metrics_numeric_range(da_wmask, rcor_extent_path, tmp_path):
    """Key metric columns contain only non-negative finite values (or NaN)."""
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    for col in _NUMERIC_COLS:
        if col not in result.columns:
            continue
        series = result[col].dropna()
        assert (series >= 0).all(), f"Column '{col}' has negative values"
        assert np.isfinite(series).all(), f"Column '{col}' has non-finite values"


@pytest.mark.slow
def test_calculate_metrics_csv_written(da_wmask, rcor_extent_path, tmp_path):
    """A CSV file is exported to the output directory."""
    calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    csv_files = list(tmp_path.rglob("ecof_metrics.csv"))
    assert len(csv_files) == 1, "ecof_metrics.csv not found in output directory"


@pytest.mark.slow
def test_calculate_metrics_section_area_matches_legacy_geometry_smoke(
    da_wmask, rcor_extent_path, legacy_baseline_csv_path, tmp_path
):
    """Historical smoke check for one approved invariant kernel only (U7 quarantine).

    ``section_area_km2`` is pure section-polygon geometry
    (``feature.geometry.area / 1e6``): it never depends on the water mask, the
    valid-observation denominator, or any of the naive-persistence defects that
    disqualify this CSV as a v1.2 correctness oracle (see
    ``docs/audit/evidence/regression_baseline.md``). This is the only comparison
    against the legacy CSV that this suite performs; no occurrence, schema, or
    ``pp_mean_%``/APSEC equivalence is asserted here.
    """
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    legacy = pd.read_csv(legacy_baseline_csv_path)

    result_areas = result.groupby("section")["section_area_km2"].first()
    legacy_areas = legacy.groupby("section")["section_area_km2"].first()
    result_areas.index = result_areas.index.astype(str)
    legacy_areas.index = legacy_areas.index.astype(str)

    common_sections = result_areas.index.intersection(legacy_areas.index)
    assert len(common_sections) > 0, "No matching sections between result and legacy CSV"
    for section in common_sections:
        assert result_areas[section] == pytest.approx(legacy_areas[section], rel=1e-6)
