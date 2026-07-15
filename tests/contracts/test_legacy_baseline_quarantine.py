"""Canonical quarantine test for the legacy iRiverMetrics regression baseline (U7).

Milestone 1 explicit deliverable: prove the legacy CSV is rejected/fails when someone
tries to use it as a v1.2 correctness baseline, rather than only asserting a metadata
flag about it (see ``tests/contracts/test_fixture_characterisation.py::
test_baseline_csv_is_legacy_not_v12_oracle`` for that lighter-weight check).

Decision: docs/audit/decisions.md U7 — approved, retire as v1.2 correctness oracle.
Evidence: docs/audit/evidence/regression_baseline.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.contracts.legacy_quarantine import (
    assert_rejected_as_v12_correctness_baseline,
    find_v12_correctness_baseline_defects,
)

TEST_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = TEST_DIR / "results_iRiverMetrics" / "metrics" / "irm_metrics.csv"


@pytest.fixture(scope="module")
def legacy_baseline_df() -> pd.DataFrame:
    return pd.read_csv(BASELINE_PATH)


def test_legacy_csv_is_rejected_as_v12_correctness_baseline(legacy_baseline_df):
    reasons = assert_rejected_as_v12_correctness_baseline(legacy_baseline_df)
    assert any("forbidden dropped metric" in r for r in reasons)
    assert any("pp_mean_%" in r for r in reasons)
    assert any("valid_fraction_month" in r for r in reasons)


def test_rejection_reasons_name_the_forbidden_columns_actually_present(legacy_baseline_df):
    reasons = find_v12_correctness_baseline_defects(legacy_baseline_df)
    forbidden_reason = next(r for r in reasons if "forbidden dropped metric" in r)
    assert "PF" in forbidden_reason
    assert "AWMPA" in forbidden_reason


def test_clean_v12_shaped_dataframe_is_not_rejected():
    """Sanity check: the guard is specific to known U7 defects, not a blanket rejection."""
    clean = pd.DataFrame(
        {
            "section": ["A", "A", "B", "B"],
            "date": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "valid_fraction_month": [0.9, 0.95, 0.6, 0.8],
            "occurrence": [0.5, 0.6, 0.2, 0.3],
        }
    )
    assert find_v12_correctness_baseline_defects(clean) == []
