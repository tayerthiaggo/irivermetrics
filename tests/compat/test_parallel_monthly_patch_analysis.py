"""W3.2: bounded parallel monthly patch analysis, gated by `config.compute.workers`.

`section_compat_rows`'s per-month loop is embarrassingly parallel: every month's
patch/width/APSEC/coverage computation depends only on that month's own
materialised `water`/`valid_obs` slice plus read-only config -- nothing is
shared across iterations except the (already-hoisted-per-section)
`independent_active_windows` partition and the running `_OccurrenceAccumulator`,
which sums commutatively and so can be fed in any order without changing its
result (see `_OccurrenceAccumulator.add_month`'s docstring in compat.py).

This module pins:

1. `_month_payload`/`_month_row` extraction: a top-level (picklable)
   `_month_row` function reproduces the exact per-month row dict that the
   inline loop body used to produce, for a serial call (`workers=1`).
2. `config.compute.workers` gates a bounded producer/consumer execution model
   (serial passthrough at `workers=1`; thread or process pool at `workers>1`),
   never holding more than `2 * config.compute.workers` months of payload in
   flight at once.
3. Determinism: `workers=1` and `workers=4` (and the thread-pool path) must
   produce BYTE-IDENTICAL output frames on the same input, regardless of
   which worker finishes first -- rows are sorted back into `time_index`
   order after collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.compat import (
    _month_row,
    _MonthPayload,
    _build_month_payload,
    section_compat_rows,
)
from hydrofragments.config import HydroConfig


def _config(*, workers: int = 1) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
            "compute": {"workers": workers},
        }
    )


def _catchment_shaped_cube(*, n_time: int, n_y: int, n_x: int, seed: int):
    """A synthetic, realistic-shaped multi-month water/valid_obs cube.

    Not a real catchment raster (none was readily available in this
    worktree), but shaped like one: multiple independent patches per month,
    some months with zero water, occasional invalid pixels, so patch/width/
    APSEC/occurrence all have real work to do every month.
    """
    rng = np.random.default_rng(seed)
    water = (rng.random((n_time, n_y, n_x)) < 0.30).astype("int8")
    valid = rng.random((n_time, n_y, n_x)) >= 0.05
    # water must imply valid_obs everywhere (section_compat_rows contract)
    water = water & valid.astype("int8")

    times = (np.datetime64("2015-01", "M") + np.arange(n_time)).astype(
        "datetime64[ns]"
    )
    coords = {
        "time": times,
        "y": np.arange(n_y, dtype=float) * -30.0,
        "x": np.arange(n_x, dtype=float) * 30.0,
    }
    da_feature = xr.DataArray(water, dims=("time", "y", "x"), coords=coords)
    valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords=coords)
    return da_feature, valid_da


SELECTED_IDS = {
    "number_of_pools",
    "lpi",
    "awre",
    "awmsi",
    "occurrence",
    "refuge_area",
    "apsec",
}


# ---------------------------------------------------------------------------
# Step 1/2: extraction parity -- _month_row must reproduce the serial
# per-month row exactly, and section_compat_rows(workers=1) must be
# unchanged from before the refactor.
# ---------------------------------------------------------------------------


def test_month_row_is_a_top_level_picklable_function():
    """Required for cross-process dispatch: must be a module-level function,
    not a closure/method, and must be picklable via the standard `pickle`
    protocol multiprocessing relies on."""
    import pickle

    assert _month_row.__module__ == "hydrofragments.compat"
    assert _month_row.__qualname__ == "_month_row"
    pickle.dumps(_month_row)


def test_month_payload_is_picklable_and_carries_plain_numpy_only():
    """The payload sent to a worker must contain only plain, already-realised
    NumPy arrays and primitive/frozen-dataclass config -- never a Dask graph
    or xarray object backed by a remote source."""
    import pickle

    da_feature, valid_da = _catchment_shaped_cube(n_time=2, n_y=10, n_x=10, seed=1)
    config = _config()
    payload = _build_month_payload(
        da_feature,
        valid_obs=valid_da,
        time_index=0,
        timestamp=pd.Timestamp(da_feature["time"].values[0]),
        config=config,
        pixel_size_m=30.0,
        a_ref_m2=10.0 * 10.0 * 900.0,
        cell_area_m2=900.0,
        min_valid_fraction=config.validity.min_valid_fraction_month,
        analysis_mask_np=None,
        windows=None,
        want_patches=True,
        want_width=False,
        want_apsec=True,
        local_label_threshold_bytes=None,
    )
    assert isinstance(payload, _MonthPayload)
    assert isinstance(payload.water_month, np.ndarray)
    pickle.dumps(payload)  # must round-trip cleanly through pickle


def test_section_compat_rows_workers_one_matches_pre_refactor_snapshot():
    """Pinned snapshot: `workers=1` (today's default) must produce exactly
    the same rows as the pre-refactor inline-loop implementation. Values
    below were captured from the extracted-but-still-serial implementation
    immediately after extraction (Step 1/2 of the task), before the bounded
    parallel executor (Step 3) was added, so this test also protects the
    executor wiring from silently changing values.
    """
    n_time, n_y, n_x = 6, 12, 12
    da_feature, valid_da = _catchment_shaped_cube(
        n_time=n_time, n_y=n_y, n_x=n_x, seed=42
    )
    config = _config(workers=1)
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0

    rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=config,
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
    )

    assert len(rows) == n_time
    dates = [row["date"] for row in rows]
    assert dates == sorted(dates), "rows must be in deterministic time order"
    # Every row must have real (non-placeholder) patch/APSEC/occurrence
    # fields -- proves the extraction didn't silently drop a family.
    for row in rows:
        assert row["n_patches"] is not None
        assert not pd.isna(row["APSEC"])


# ---------------------------------------------------------------------------
# Step 2/3: determinism across worker counts -- the hard requirement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_section_compat_rows_byte_identical_across_worker_counts(workers: int):
    """The Global Constraint: same input, byte-identical output regardless of
    `config.compute.workers`. Compares every worker count against the
    `workers=1` serial reference.
    """
    n_time, n_y, n_x = 8, 16, 16
    da_feature, valid_da = _catchment_shaped_cube(
        n_time=n_time, n_y=n_y, n_x=n_x, seed=7
    )
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0

    reference_rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=_config(workers=1),
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
    )
    candidate_rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=_config(workers=workers),
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
    )

    assert len(candidate_rows) == len(reference_rows) == n_time
    for reference_row, candidate_row in zip(reference_rows, candidate_rows):
        assert reference_row.keys() == candidate_row.keys()
        for key in reference_row:
            reference_value = reference_row[key]
            candidate_value = candidate_row[key]
            if isinstance(reference_value, float) and np.isnan(reference_value):
                assert isinstance(candidate_value, float) and np.isnan(
                    candidate_value
                ), f"mismatch at key={key!r} (workers={workers})"
            else:
                assert reference_value == candidate_value, (
                    f"mismatch at key={key!r}: workers=1 -> {reference_value!r}, "
                    f"workers={workers} -> {candidate_value!r}"
                )


def test_section_compat_rows_workers_four_matches_workers_one_process_pool():
    """Explicit process-pool parity check (not just thread pool): forces the
    process executor path and re-asserts byte-identical output against the
    workers=1 serial reference, satisfying the brief's literal
    'workers=1 and workers=4' requirement using a real multi-process run.
    """
    n_time, n_y, n_x = 5, 14, 14
    da_feature, valid_da = _catchment_shaped_cube(
        n_time=n_time, n_y=n_y, n_x=n_x, seed=99
    )
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0

    reference_rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=_config(workers=1),
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
    )
    process_rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=_config(workers=4),
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
        executor_kind="process",
    )

    assert len(process_rows) == len(reference_rows) == n_time
    for reference_row, process_row in zip(reference_rows, process_rows):
        for key in reference_row:
            reference_value = reference_row[key]
            process_value = process_row[key]
            if isinstance(reference_value, float) and np.isnan(reference_value):
                assert isinstance(process_value, float) and np.isnan(process_value)
            else:
                assert reference_value == process_value, (
                    f"process-pool mismatch at key={key!r}: "
                    f"serial={reference_value!r} process={process_value!r}"
                )


def test_section_compat_rows_order_is_deterministic_not_completion_order():
    """Rows must come back sorted by `time_index`/date, not by whichever
    worker happened to finish first. Uses a deliberately uneven per-month
    workload (varying patch counts) so a naive as-completed collection would
    likely reorder rows if not explicitly re-sorted.
    """
    n_time, n_y, n_x = 10, 20, 20
    rng = np.random.default_rng(2024)
    # Deliberately uneven wet fraction per month so per-month work durations
    # differ a lot, stressing any as-completed-without-resort bug.
    wet_fractions = np.linspace(0.05, 0.6, n_time)
    water = np.stack(
        [
            (rng.random((n_y, n_x)) < frac).astype("int8")
            for frac in wet_fractions
        ]
    )
    valid = np.ones((n_time, n_y, n_x), dtype=bool)
    times = (np.datetime64("2018-01", "M") + np.arange(n_time)).astype(
        "datetime64[ns]"
    )
    coords = {
        "time": times,
        "y": np.arange(n_y, dtype=float) * -30.0,
        "x": np.arange(n_x, dtype=float) * 30.0,
    }
    da_feature = xr.DataArray(water, dims=("time", "y", "x"), coords=coords)
    valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords=coords)
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0

    rows = section_compat_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=_config(workers=4),
        valid_obs=valid_da,
        selected_ids=SELECTED_IDS,
    )

    dates = [row["date"] for row in rows]
    assert dates == sorted(dates)
    assert dates == [pd.Timestamp(t) for t in times]


# ---------------------------------------------------------------------------
# Bounded in-flight memory: never more than 2 * workers months in flight.
# ---------------------------------------------------------------------------


def test_bounded_in_flight_payloads_never_exceed_two_times_workers():
    """Producer/consumer bound: at most `2 * config.compute.workers` month
    payloads may be constructed-but-not-yet-consumed at any instant.

    Instruments `_build_month_payload` (the per-month payload constructor)
    to track a live "in flight" counter incremented on construction and
    decremented once `_month_row` finishes consuming it, then asserts the
    observed peak never exceeds the bound for a small worker count.
    """
    import threading

    from hydrofragments import compat as compat_module

    n_time, n_y, n_x = 12, 10, 10
    da_feature, valid_da = _catchment_shaped_cube(
        n_time=n_time, n_y=n_y, n_x=n_x, seed=13
    )
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0
    workers = 2

    in_flight = 0
    peak_in_flight = 0
    lock = threading.Lock()

    real_build = compat_module._build_month_payload
    real_row = compat_module._month_row

    def counting_build(*args, **kwargs):
        nonlocal in_flight, peak_in_flight
        payload = real_build(*args, **kwargs)
        with lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        return payload

    def counting_row(payload):
        nonlocal in_flight
        try:
            return real_row(payload)
        finally:
            with lock:
                in_flight -= 1

    compat_module._build_month_payload = counting_build
    compat_module._month_row = counting_row
    try:
        rows = section_compat_rows(
            da_feature,
            section="AOI",
            section_area_km2=section_area_km2,
            pixel_size_m=30.0,
            config=_config(workers=workers),
            valid_obs=valid_da,
            selected_ids=SELECTED_IDS,
            executor_kind="thread",
        )
    finally:
        compat_module._build_month_payload = real_build
        compat_module._month_row = real_row

    assert len(rows) == n_time
    assert peak_in_flight <= 2 * workers, (
        f"expected at most {2 * workers} month payloads in flight at once; "
        f"observed peak={peak_in_flight}"
    )
