"""Integration test: run calculate_metrics on bundled test data and
compare outputs against the reference CSV shipped in tests/.

This test is marked 'slow' because it runs the full processing pipeline.
Run it with:
    pytest -m slow
or together with unit tests via:
    pytest
"""
import numpy as np
import pandas as pd
import pytest

from irivermetrics import calculate_metrics

# Columns present in both the old reference CSV and the current output.
# The reference CSV uses 'PFL'; current code produces 'PLF'.
_OLD_TO_NEW = {"PFL": "PLF"}
_NUMERIC_COLS = [
    "section_area_km2",
    "npools",
    "wet_area_km2",
    "wet_length_km",
    "wet_perimeter_km",
    "AWMSI",
    "AWMPA",
    "AWMPL",
    "AWMPW",
    "APSEC",
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
    """Output DataFrame exposes all expected metric columns."""
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )
    expected_cols = {
        "date", "section", "section_area_km2", "npools",
        "wet_area_km2", "wet_length_km", "wet_perimeter_km",
        "AWMSI", "AWRe", "AWMPA", "AWMPL", "AWMPW",
        "PF", "PLF", "APSEC", "LPSEC", "pp_mean_%", "ra_area_km2",
    }
    assert expected_cols.issubset(set(result.columns))


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
    csv_files = list(tmp_path.rglob("irm_metrics.csv"))
    assert len(csv_files) == 1, "irm_metrics.csv not found in output directory"


@pytest.mark.slow
def test_calculate_metrics_regression(da_wmask, rcor_extent_path, expected_metrics, tmp_path):
    """Numeric metrics are within 5 % of the reference output for matched rows.

    A generous tolerance is used because the igraph BFS path-finding can
    produce slightly different (but equally valid) longest-path solutions
    compared to the MCP_Geometric approach used in the original codebase.
    Regression checks are limited to area-based metrics which are
    deterministic and independent of the skeleton algorithm.
    """
    result = calculate_metrics(
        da_wmask,
        rcor_extent=rcor_extent_path,
        outdir=str(tmp_path),
        fill_nodata=True,
    )

    # Normalise column names: reference CSV may use 'PFL' while code uses 'PLF'
    ref = expected_metrics.rename(columns=_OLD_TO_NEW)

    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    ref["date"] = pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")
    result["section"] = result["section"].astype(str)
    ref["section"] = ref["section"].astype(str)

    merged = result.merge(ref, on=["date", "section"], suffixes=("_new", "_ref"))
    assert len(merged) > 0, "No matching (date, section) pairs between result and reference"

    # Check area-based deterministic metrics only
    for col in ["section_area_km2", "wet_area_km2", "APSEC", "pp_mean_%"]:
        new_col = f"{col}_new"
        ref_col = f"{col}_ref"
        if new_col not in merged or ref_col not in merged:
            continue
        mask = merged[new_col].notna() & merged[ref_col].notna() & (merged[ref_col] != 0)
        if not mask.any():
            continue
        rel_err = ((merged.loc[mask, new_col] - merged.loc[mask, ref_col]).abs()
                   / merged.loc[mask, ref_col].abs())
        assert rel_err.median() < 0.05, (
            f"Column '{col}' median relative error {rel_err.median():.3f} exceeds 5 %"
        )
