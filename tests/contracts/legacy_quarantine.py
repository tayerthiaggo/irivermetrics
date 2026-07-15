"""Quarantine guard for the legacy iRiverMetrics regression baseline (U7).

Enforces that ``tests/results_iRiverMetrics/metrics/irm_metrics.csv`` (or any dataframe
shaped like it) can never silently pass as a v1.2 correctness oracle. See
``docs/audit/decisions.md`` (U7) and ``docs/audit/evidence/regression_baseline.md`` for
the approved disposition this module encodes.
"""
from __future__ import annotations

import pandas as pd

# Metrics the v1.2 schema forbids outright (Q5, approved) — circular/dropped/renamed.
FORBIDDEN_V12_METRIC_COLUMNS = frozenset({"PF", "PFL", "PLF", "AWMPA", "AWMPL", "AWMPW"})

# v1.2 canonical schema requires a valid-observation-based monthly support field
# (implementation_plan.md §5); the legacy CSV has no equivalent column at all.
REQUIRED_V12_SUPPORT_COLUMNS = frozenset({"valid_fraction_month"})


def find_v12_correctness_baseline_defects(df: pd.DataFrame) -> list[str]:
    """Return the reasons ``df`` cannot serve as a v1.2 correctness oracle.

    An empty list means no known U7 defect was detected in this dataframe's shape.
    """
    reasons: list[str] = []

    forbidden_present = sorted(FORBIDDEN_V12_METRIC_COLUMNS & set(df.columns))
    if forbidden_present:
        reasons.append(
            "contains v1.2-forbidden dropped metric columns: " + ", ".join(forbidden_present)
        )

    missing_support = sorted(REQUIRED_V12_SUPPORT_COLUMNS - set(df.columns))
    if missing_support:
        reasons.append(
            "missing required v1.2 support columns: " + ", ".join(missing_support)
        )

    if "pp_mean_%" in df.columns and "section" in df.columns:
        static_per_section = bool((df.groupby("section")["pp_mean_%"].nunique() == 1).all())
        if static_per_section:
            reasons.append(
                "pp_mean_% is constant per section across all dates, evidence it is a "
                "total-timestep mean rather than a valid_obs-based occurrence ratio"
            )

    return reasons


def assert_rejected_as_v12_correctness_baseline(df: pd.DataFrame) -> list[str]:
    """Assert ``df`` fails v1.2 correctness-baseline validation and return why.

    Raises ``AssertionError`` if it would unexpectedly pass, meaning the quarantine has
    silently regressed and must be re-reviewed before any test may treat it as an oracle.
    """
    reasons = find_v12_correctness_baseline_defects(df)
    assert reasons, (
        "expected the legacy metrics dataframe to fail v1.2 correctness-baseline "
        "validation, but no defects were found - re-review U7 before treating it as "
        "an oracle"
    )
    return reasons
