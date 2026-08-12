"""APSEC call-count contract across two competing optimisations (m2, W3.3).

History: an earlier task (m2) hoisted ``compute_apsec`` above the per-month
loop in ``section_compat_rows()`` so it ran once over the WHOLE time axis in
a single xarray reduction, instead of once per month -- a real speedup when
the whole ``(time, y, x)`` payload was already resident in memory.

W3.3 (``docs/superpowers/plans/2026-07-27-dea-zones-and-catchment-speed.md``
section 3.3) requires the opposite memory shape: ``section_compat_rows()``
must never materialise more than one month's ``water``/``valid_obs`` at
once, so no whole-time-axis payload exists for ``compute_apsec`` to reduce
over in one call any more. Since bounded memory is the harder, catchment-
scale-correctness requirement (an OOM is a hard failure; a few extra small
xarray reductions per month is not), ``compute_apsec`` is now called once
PER MONTH again -- each call fed that month's already-materialised 2-D
slice wrapped back into a length-1 ``time`` dim (pure in-memory metadata,
not a new source read).

This test now pins THAT call-count contract instead: driving a multi-month
cube through ``analyze()`` (contracts_core profile, which selects apsec)
must invoke ``compute_apsec`` exactly once per month, not once for the
whole cube. It also proves the returned APSEC values are unchanged by
cross-checking against the frozen ``tests/gating/analyze_snapshot.json``
golden values used by ``tests/gating/test_analyze_row_snapshot.py`` --
value-correctness is preserved even though the call-count shape reversed.
"""

from __future__ import annotations

from unittest import mock

from hydrofragments import HydroConfig, analyze
from hydrofragments.section_analysis import compute_apsec as real_compute_apsec


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


def test_compute_apsec_called_once_per_month(synthetic_cube, tmp_path):
    """compute_apsec must be called exactly once PER MONTH, not once for
    the whole cube (W3.3 memory-bounding requirement; see module docstring).

    synthetic_cube has 6 months (see tests/conftest.py). Each call must
    receive an already-materialised single-month payload -- calling it
    once for the whole cube would require the whole (time, y, x) array to
    be resident in memory at once, exactly the OOM risk W3.3 eliminates.
    """
    config = _contracts_core_config(tmp_path)
    with mock.patch(
        "hydrofragments.section_analysis.compute_apsec", wraps=real_compute_apsec
    ) as spy:
        result = analyze(
            synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0
        )
    assert spy.call_count == 6, (
        "expected compute_apsec to be called once per month (6 months in "
        f"synthetic_cube); got {spy.call_count} calls -- a call count of 1 "
        "would mean the whole cube was batched into one reduction again, "
        "reintroducing the whole-array materialization W3.3 removed"
    )
    for call in spy.call_args_list:
        monthly_arg = call.args[0] if call.args else call.kwargs["monthly"]
        assert monthly_arg.sizes["time"] == 1, (
            "expected every compute_apsec call to be scoped to exactly one "
            f"month; got time size {monthly_arg.sizes['time']}"
        )

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
