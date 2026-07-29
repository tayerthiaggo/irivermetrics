"""B1 safety net: pin analyze() row output before compute-path restructuring.

Golden snapshot of hydrofragments.api.analyze() applied to a small
deterministic synthetic cube (see tests/parity/conftest.py::synthetic_cube).
This test's only job is to catch unintended changes to the (metric,
statistic, value) triples analyze() emits; it is not asserting the values
are "correct" in any scientific sense.

analyze() is called with an explicit ``contracts_core`` metric profile
because that is the smallest working configuration -- it needs no
drainage/channel context, matching the first ``analyze()`` call in
tests/compat/test_hydrofragments_public_api.py
(``test_analyze_returns_tidy_core_metrics_without_forbidden_ids``), which is
the canonical example of how production constructs an analyze() call.

Note: since W4.1, the *default* ``metric_profiles`` (no key supplied) is
``all_available``, which additionally computes recurrence/hydroperiod for
this same synthetic cube -- see
tests/gating/test_all_available_profile.py::test_default_run_selects_core_recurrence_and_hydroperiod_only
for that behaviour pinned explicitly. This snapshot stays scoped to
``contracts_core`` alone so it keeps pinning only the narrow (metric,
statistic, value) row shape it was designed for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hydrofragments import HydroConfig, analyze

SNAPSHOT_PATH = Path(__file__).parent / "analyze_snapshot.json"


def _serialize(metrics_table: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Reduce metrics_table to a stable, sortable (metric, statistic, value) view.

    Rows whose value is null (pandas NA, e.g. the diagnostic-only
    ``occurrence`` metric) are dropped -- they carry no numeric signal to
    pin. ``statistic`` is also nullable (pd.NA) for many core metrics, so
    it is normalised to "" rather than compared via hasattr as the brief
    sketch assumed (metrics_table rows are pandas Series, not records).
    """
    rows = []
    for _, row in metrics_table.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        statistic = row["statistic"]
        statistic = "" if pd.isna(statistic) else str(statistic)
        rows.append((str(row["metric"]), statistic, round(float(value), 6)))
    return sorted(rows)


def test_analyze_row_snapshot(synthetic_cube, tmp_path):
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )
    result = analyze(
        synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0
    )
    got = _serialize(result.metrics_table)

    # Freeze procedure (Step 3): uncomment the next two lines, run once,
    # verify the file content, then re-comment before committing.
    # with open(SNAPSHOT_PATH, "w") as fh:
    #     json.dump(got, fh, indent=2)

    with open(SNAPSHOT_PATH) as fh:
        expected = json.load(fh)
    assert got == [tuple(row) for row in expected]
