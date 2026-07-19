"""m8: temporal AOI summaries must materialize in one batched compute.

``_temporal_profile_records`` (hydrofragments/api.py) previously called
``.item()`` once for the AOI-mean recurrence value and once more per
hydroperiod year -- each ``.item()`` on a dask-backed DataArray triggers its
own independent graph execution. For N hydroperiod years that's 1 + N
separate materializations of overlapping (and redundant) parts of the same
underlying graph instead of a single batched compute.

This test builds a small dask-backed WaterCube (so ``.item()``/``.compute()``
calls are observable) spanning multiple hydroyears, counts how many times
``dask.array.Array.compute`` actually runs while ``analyze()`` executes the
``pixel_temporal`` profile (recurrence + hydroperiod), and asserts the count
is small and constant rather than growing with the number of years.
"""

from __future__ import annotations

from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze
from hydrofragments.models import WaterCube


def _raw_arrays(n_years: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The raw water/valid_obs numpy arrays shared by the dask and eager cubes."""
    months = n_years * 12
    rng = np.random.default_rng(42)
    water_np = rng.random((months, 4, 4)) > 0.4
    valid_np = np.ones((months, 4, 4), dtype=bool)
    times = pd.date_range("2015-01-01", periods=months, freq="MS")
    return water_np, valid_np, times


def _dask_cube(n_years: int) -> WaterCube:
    """A dask-backed WaterCube spanning ``n_years`` of monthly data.

    12 months/year, 4x4 pixels, chunked so the underlying arrays are real
    ``dask.array.Array`` objects (not eagerly-evaluated numpy).
    """
    water_np, valid_np, times = _raw_arrays(n_years)

    water_data = da.from_array(water_np, chunks=(3, 4, 4))
    valid_data = da.from_array(valid_np, chunks=(3, 4, 4))

    water = xr.DataArray(water_data, dims=("time", "y", "x"), coords={"time": times})
    valid = xr.DataArray(valid_data, dims=("time", "y", "x"), coords={"time": times})

    return WaterCube(
        water=water,
        valid_obs=valid,
        source="synthetic_dask",
        cadence="monthly",
    )


def _eager_cube(n_years: int) -> WaterCube:
    """The identical cube as :func:`_dask_cube`, but plain numpy-backed (no dask).

    Same underlying ``water``/``valid_obs`` values (same RNG seed and shape),
    just without dask chunking. Used as a "known-good" eager reference to
    cross-check that batching the AOI-mean recurrence/hydroperiod
    materializations into one ``xr.Dataset().compute()`` call did not change
    any computed value relative to an unbatched, non-dask computation.
    """
    water_np, valid_np, times = _raw_arrays(n_years)

    water = xr.DataArray(water_np, dims=("time", "y", "x"), coords={"time": times})
    valid = xr.DataArray(valid_np, dims=("time", "y", "x"), coords={"time": times})

    return WaterCube(
        water=water,
        valid_obs=valid,
        source="synthetic_eager",
        cadence="monthly",
    )


def _pixel_temporal_config(tmp_path) -> HydroConfig:
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
            "metric_profiles": ["pixel_temporal"],
        }
    )


def _count_dask_computes(cube: WaterCube, config: HydroConfig) -> int:
    calls = {"n": 0}
    real_compute = da.Array.compute

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real_compute(self, *args, **kwargs)

    with mock.patch.object(da.Array, "compute", counting):
        analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    return calls["n"]


@pytest.mark.parametrize("n_years", [2, 5])
def test_temporal_summaries_materialize_in_bounded_calls(tmp_path, n_years):
    """Materialization count must not scale with the number of hydroperiod years.

    Before the m8 fix, each hydroperiod year triggered its own ``.item()``
    materialization on top of the one for recurrence, so the call count grew
    as ``1 + n_years``. After batching all summaries into a single
    ``xr.Dataset`` and calling ``.compute()`` once, the count must stay
    constant regardless of ``n_years``.
    """
    cube = _dask_cube(n_years)
    config = _pixel_temporal_config(tmp_path)
    calls = _count_dask_computes(cube, config)
    assert calls <= 2, f"expected a single batched materialization, got {calls} calls"


def test_temporal_summaries_materialization_does_not_scale_with_years(tmp_path):
    """Direct before/after comparator: 5-year run must not cost more computes than 2-year."""
    calls_2y = _count_dask_computes(_dask_cube(2), _pixel_temporal_config(tmp_path))
    calls_5y = _count_dask_computes(_dask_cube(5), _pixel_temporal_config(tmp_path))
    assert calls_2y == calls_5y, (
        f"materialization count scaled with year count ({calls_2y} vs {calls_5y}); "
        "temporal summaries are not batched into one compute"
    )


def _recurrence_and_hydroperiod_values(tmp_path, cube: WaterCube) -> dict[str, float]:
    """Run ``analyze()`` and pull out {"recurrence": v, "hydroperiod_<year>": v, ...}.

    Reads straight from ``HydroResult.metrics_table`` (the tidy output
    ``_temporal_profile_records`` feeds into via ``records_to_frame``), which
    is the same surface real callers and downstream tables consume -- not an
    internal shortcut into ``_temporal_profile_records`` itself.
    """
    config = _pixel_temporal_config(tmp_path)
    result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    table = result.metrics_table
    values: dict[str, float] = {}
    recurrence_rows = table[table["metric"] == "recurrence"]
    assert len(recurrence_rows) == 1, "expected exactly one recurrence row"
    values["recurrence"] = float(recurrence_rows["value"].iloc[0])

    hydroperiod_rows = table[table["metric"] == "hydroperiod"].sort_values("date")
    for _, row in hydroperiod_rows.iterrows():
        year = pd.Timestamp(row["date"]).year
        values[f"hydroperiod_{year}"] = float(row["value"])
    return values


def _hand_traceable_dask_cube() -> WaterCube:
    """A tiny, fully-hand-computable dask-backed cube: 2x2 pixels, 2 years (24 months).

    All months are valid (``valid_obs`` all ``True``), so every denominator in
    the recurrence/hydroperiod formulas below is simply the month count --
    there is no missingness to reason about. The water pattern is an explicit
    literal array (not random) chosen so each of the four pixels exercises a
    different case:

    - pixel (0, 0): wet Jan-Jun, dry Jul-Dec, in *both* 2015 and 2016.
    - pixel (0, 1): wet every month of both years.
    - pixel (1, 0): dry every month of both years.
    - pixel (1, 1): wet every month of 2015, dry every month of 2016.

    Expected values (see ``test_batched_temporal_summaries_match_hand_derived_values``
    for the derivation) are computed independently of
    ``_temporal_profile_records``/``compute_recurrence``/``compute_hydroperiod``,
    directly from the formulas documented in
    ``hydrofragments/metrics/persistence.py:165-208``.
    """
    n_years = 2
    months = n_years * 12
    times = pd.date_range("2015-01-01", periods=months, freq="MS")

    water = np.zeros((months, 2, 2), dtype=bool)
    for m in range(months):
        year_is_2015 = (m // 12) == 0
        month_in_year = m % 12
        water[m, 0, 0] = month_in_year < 6  # wet Jan-Jun, dry Jul-Dec, every year
        water[m, 0, 1] = True  # always wet
        water[m, 1, 0] = False  # always dry
        water[m, 1, 1] = year_is_2015  # wet all of 2015, dry all of 2016

    valid = np.ones((months, 2, 2), dtype=bool)

    water_data = da.from_array(water, chunks=(3, 2, 2))
    valid_data = da.from_array(valid, chunks=(3, 2, 2))

    water_da = xr.DataArray(water_data, dims=("time", "y", "x"), coords={"time": times})
    valid_da = xr.DataArray(valid_data, dims=("time", "y", "x"), coords={"time": times})

    return WaterCube(
        water=water_da,
        valid_obs=valid_da,
        source="hand_traceable_dask",
        cadence="monthly",
    )


def test_batched_temporal_summaries_match_hand_derived_values(tmp_path):
    """Pin batched recurrence/hydroperiod against independently hand-derived numbers.

    ``test_batched_temporal_summaries_match_eager_nondask_values`` (below) only
    proves the batched (dask) and unbatched (eager) runs agree with *each
    other* -- both go through the identical post-m8 batching/dict-assembly
    logic in ``_temporal_profile_records``, so a backend-agnostic bug (e.g. a
    value assigned to the wrong ``hydroperiod_{year}`` key, or an off-by-one
    in per-year windowing) would reproduce identically on both sides and pass
    that test anyway. This test instead pins ``analyze()``'s real output
    against values derived by hand from the formulas in
    ``compute_recurrence``/``compute_hydroperiod``
    (``hydrofragments/metrics/persistence.py:165-208``), independent of any
    code in ``_temporal_profile_records`` itself.

    Hand derivation (all months valid, so every denominator below is just a
    month count):

    Hydroperiod (``HP_{p,y} = wet valid months / valid months``, per pixel
    per year, then AOI-meaned over the 4 pixels):

    - 2015: pixel (0,0)=6/12=0.5, (0,1)=12/12=1.0, (1,0)=0/12=0.0, (1,1)=12/12=1.0
      -> AOI mean = (0.5 + 1.0 + 0.0 + 1.0) / 4 = 0.625
    - 2016: pixel (0,0)=6/12=0.5, (0,1)=12/12=1.0, (1,0)=0/12=0.0, (1,1)=0/12=0.0
      -> AOI mean = (0.5 + 1.0 + 0.0 + 0.0) / 4 = 0.375

    Recurrence (per pixel: group months by calendar month name across both
    years, take wet-valid/valid per month-group, average the 12 monthly
    ratios, then *100 for percent; then AOI-mean over the 4 pixels):

    - pixel (0,0): Jan-Jun wet in both years (2/2=1.0 each), Jul-Dec dry in
      both years (0/2=0.0 each) -> mean of twelve values = 6/12 = 0.5 -> 50%
    - pixel (0,1): every calendar month wet in both years (2/2=1.0) -> 100%
    - pixel (1,0): every calendar month dry in both years (0/2=0.0) -> 0%
    - pixel (1,1): every calendar month wet in 2015 only, dry in 2016
      (1/2=0.5 each) -> mean of twelve 0.5's = 0.5 -> 50%
    - AOI mean recurrence = (50 + 100 + 0 + 50) / 4 = 50.0%

    These numbers were cross-checked with a standalone script calling
    ``compute_recurrence``/``compute_hydroperiod`` directly (not through
    ``_temporal_profile_records``) and matched exactly; they are pinned here
    as literal expected values.
    """
    cube = _hand_traceable_dask_cube()
    values = _recurrence_and_hydroperiod_values(tmp_path, cube)

    assert values["recurrence"] == pytest.approx(50.0)
    assert values["hydroperiod_2015"] == pytest.approx(0.625)
    assert values["hydroperiod_2016"] == pytest.approx(0.375)
    assert set(values) == {"recurrence", "hydroperiod_2015", "hydroperiod_2016"}


@pytest.mark.parametrize("n_years", [2, 5])
def test_batched_temporal_summaries_match_eager_nondask_values(tmp_path, n_years):
    """The batched dask compute must produce the *same numbers* as an eager, non-dask run.

    m8 batched the AOI-mean recurrence scalar and every per-year AOI-mean
    hydroperiod scalar into a single ``xr.Dataset(...).compute()`` call
    instead of one independent ``.item()`` materialization each. That change
    is only safe if it is purely a performance/scheduling change -- the
    computed values themselves must be identical.

    This builds two ``WaterCube``s from the *exact same* underlying
    water/valid_obs numpy arrays (same RNG seed, same shape): one dask-backed
    (exercises the batched code path added in m8) and one plain numpy-backed
    (never touches dask, so it is a "known-good" eager reference unaffected
    by the batching change). Both go through the real ``analyze()`` ->
    ``_temporal_profile_records`` -> ``metrics_table`` path, and the
    resulting recurrence + per-year hydroperiod values must match exactly.
    """
    dask_values = _recurrence_and_hydroperiod_values(tmp_path, _dask_cube(n_years))
    eager_values = _recurrence_and_hydroperiod_values(tmp_path, _eager_cube(n_years))

    assert dask_values.keys() == eager_values.keys(), (
        "dask and eager runs emitted different sets of metric rows: "
        f"{sorted(dask_values)} vs {sorted(eager_values)}"
    )
    assert dask_values["recurrence"] == pytest.approx(eager_values["recurrence"]), (
        f"AOI-mean recurrence drifted between batched dask run "
        f"({dask_values['recurrence']}) and eager reference "
        f"({eager_values['recurrence']})"
    )
    for key in sorted(dask_values):
        if key == "recurrence":
            continue
        assert dask_values[key] == pytest.approx(eager_values[key]), (
            f"{key} drifted between batched dask run ({dask_values[key]}) "
            f"and eager reference ({eager_values[key]})"
        )

    # Sanity: this cube is designed to actually have variation, so the
    # comparison above is non-trivial (not just comparing two zeros/NaNs).
    assert 0.0 < dask_values["recurrence"] < 100.0
    assert n_years == sum(1 for key in dask_values if key.startswith("hydroperiod_"))
