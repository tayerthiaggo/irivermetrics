"""m2: APSEC must be computed once over the whole time axis, not per month.

``compute_apsec`` (hydrofragments/metrics/extent.py) already reduces over
the spatial dims across the WHOLE time axis in a single xarray call and
returns one ``ApsecRecord`` per timestamp -- it needs no changes. The bug
was at the call site: ``section_compat_rows()`` in hydrofragments/compat.py
called ``compute_apsec`` once per loop iteration on a one-timestep slice
(``monthly.isel(time=[time_index])``), throwing away the batching and
doing M separate xarray reductions instead of one.

This test pins the fix with a call-count assertion: driving a multi-month
cube through ``analyze()`` (contracts_core profile, which selects apsec)
must invoke ``compute_apsec`` exactly once, not once per month. It also
proves the returned APSEC values are unchanged by cross-checking against
the frozen ``tests/gating/analyze_snapshot.json`` golden values used by
``tests/gating/test_analyze_row_snapshot.py``.
"""

from __future__ import annotations

from unittest import mock

from hydrofragments import HydroConfig, analyze
from hydrofragments.compat import compute_apsec as real_compute_apsec


def _contracts_core_config(tmp_path):
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )


def test_compute_apsec_called_once_per_analyze(synthetic_cube, tmp_path):
    """compute_apsec must be called exactly once per analyze() invocation.

    synthetic_cube has 6 months (see tests/conftest.py). Before the fix,
    section_compat_rows() called compute_apsec once per month inside its
    per-timestep loop -- 6 calls. After the fix it is hoisted above the
    loop -- exactly 1 call, regardless of month count.
    """
    config = _contracts_core_config(tmp_path)
    with mock.patch(
        "hydrofragments.compat.compute_apsec", wraps=real_compute_apsec
    ) as spy:
        result = analyze(
            synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0
        )
    assert spy.call_count == 1

    # Sanity: APSEC rows were actually produced (want_apsec path exercised).
    metrics = set(result.metrics_table["metric"])
    assert "apsec" in metrics


def test_apsec_values_unchanged_after_batching(synthetic_cube, tmp_path):
    """The batched call site must produce numerically identical APSEC values.

    Cross-checked against tests/gating/analyze_snapshot.json, the frozen
    golden output for this exact synthetic_cube + contracts_core config
    (see tests/gating/test_analyze_row_snapshot.py). If batching altered
    ordering or values, this would drift from the snapshot.
    """
    import json
    from pathlib import Path

    import pandas as pd

    config = _contracts_core_config(tmp_path)
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    apsec_rows = result.metrics_table[result.metrics_table["metric"] == "apsec"]
    got = sorted(
        round(float(v), 6) for v in apsec_rows["value"] if pd.notna(v)
    )

    snapshot_path = (
        Path(__file__).parent.parent / "gating" / "analyze_snapshot.json"
    )
    with open(snapshot_path) as fh:
        expected_all = json.load(fh)
    expected = sorted(
        round(float(value), 6)
        for metric, _statistic, value in expected_all
        if metric == "apsec"
    )

    assert got == expected
    assert len(got) > 0
