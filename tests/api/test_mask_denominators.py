"""Final whole-branch review fix: ``aoi_mask``/``analysis_mask`` must actually
be read by the metric computation path through ``analyze()``, end to end.

W3.1 added ``WaterCube.aoi_mask``/``WaterCube.analysis_mask`` as optional 2-D
boolean masks, with two global constraints:

- "APSEC/LPI/reference-area denominators remain full catchment ``aoi_mask``."
- "Monthly coverage denominator is approved as conservative potential-water
  ``analysis_mask``, not full catchment."

``tests/metrics/test_analysis_mask_coverage.py`` pins the *kernel* behaviour
(``compute_apsec``/``compute_analysis_mask_coverage`` called directly with a
hand-passed denominator) but never proves the full ``analyze()`` pipeline
actually wires either mask in -- it never calls ``analyze()`` or
``open_water_cube()`` at all. Two bugs survived as a result:

- Bug A: ``hydrofragments/api.py``'s ``section_area_km2`` used
  ``cube.water.isel(time=0).size`` (the full bounding-box grid) instead of
  ``cube.aoi_mask``'s pixel count, so APSEC/LPI were silently wrong on any
  AOI that does not fill its own bounding box.
- Bug B: ``hydrofragments/compat.py``'s per-month coverage reduction averaged
  ``coverage_valid_obs_month`` over the WHOLE grid instead of restricting to
  ``payload.analysis_mask_np``, so a large unobserved margin outside a small
  analysis_mask could falsely suppress ``is_reportable`` via
  ``low_coverage_flag`` even when the analysis_mask's own footprint was 100%
  observed.

This file closes that end-to-end gap: both tests below go through
``open_water_cube()`` + ``analyze()``, the actual pipeline a caller uses.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube


def _config(tmp_path) -> HydroConfig:
    return HydroConfig.from_mapping(
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


def _make_cube(*, aoi_mask: np.ndarray | None = None, analysis_mask: np.ndarray | None = None):
    """20x20, 2-month synthetic cube with a fixed 40-pixel wet block.

    The wet block (rows 0:4, cols 0:10 -> 40 pixels) sits inside BOTH the
    "half grid" aoi_mask used by the Fix A test and the full grid, so its
    absolute wet-pixel count never changes between runs -- only the
    denominator (aoi_mask pixel count) differs.
    """
    t, y, x = 2, 20, 20
    water = np.zeros((t, y, x), dtype=bool)
    water[:, 0:4, 0:10] = True  # 40 wet pixels/month, well inside the half-mask
    valid = np.ones((t, y, x), dtype=bool)
    times = np.array(["2020-01", "2020-02"], dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0

    water_da = xr.DataArray(
        water, dims=("time", "y", "x"), coords={"time": times, "y": ys, "x": xs}
    )
    valid_da = xr.DataArray(
        valid, dims=("time", "y", "x"), coords={"time": times, "y": ys, "x": xs}
    )
    aoi_mask_da = (
        xr.DataArray(aoi_mask, dims=("y", "x"), coords={"y": ys, "x": xs})
        if aoi_mask is not None
        else None
    )
    analysis_mask_da = (
        xr.DataArray(analysis_mask, dims=("y", "x"), coords={"y": ys, "x": xs})
        if analysis_mask is not None
        else None
    )
    return open_water_cube(
        water_da,
        valid_obs=valid_da,
        input_kind="generic_binary",
        aoi_mask=aoi_mask_da,
        analysis_mask=analysis_mask_da,
    )


def _metric_values(result, metric: str) -> list[float]:
    table = result.metrics_table
    rows = table[table["metric"] == metric]
    return [float(v) for v in rows["value"].tolist() if v is not None and not np.isnan(v)]


def test_aoi_mask_denominates_apsec_and_lpi_not_full_grid(tmp_path):
    """Fix A regression test (end-to-end through analyze()).

    40 wet pixels out of a 400-pixel (20x20) full grid vs. 40 wet pixels out
    of a 200-pixel half-grid aoi_mask: APSEC (and consequently the wetted
    fraction feeding LPI's denominator) must roughly DOUBLE when the
    denominator halves, for the exact same absolute wet-pixel count. Before
    the fix, ``section_area_km2`` used ``.size`` (the full 400-pixel grid)
    regardless of ``aoi_mask``, so this assertion fails against the pre-fix
    code -- supplying aoi_mask changed nothing.
    """
    full_grid_cube = _make_cube()  # aoi_mask omitted -> defaults to all-true (400 px)

    half_mask = np.zeros((20, 20), dtype=bool)
    half_mask[:, 0:10] = True  # left half of the grid: 20*10 = 200 pixels
    half_mask_cube = _make_cube(aoi_mask=half_mask)

    full_grid_apsec = _metric_values(
        analyze(full_grid_cube, aoi_id="full", config=_config(tmp_path / "full"), pixel_size_m=30.0),
        "apsec",
    )
    half_mask_apsec = _metric_values(
        analyze(half_mask_cube, aoi_id="half", config=_config(tmp_path / "half"), pixel_size_m=30.0),
        "apsec",
    )

    assert full_grid_apsec, "expected at least one APSEC row"
    assert half_mask_apsec, "expected at least one APSEC row"

    # Full grid: 40 / 400 * 100 = 10%. Half mask: 40 / 200 * 100 = 20%.
    assert full_grid_apsec[0] == pytest.approx(10.0)
    assert half_mask_apsec[0] == pytest.approx(20.0)
    # The core regression assertion requested: halving the aoi_mask
    # denominator roughly doubles APSEC for the same absolute wet count.
    assert half_mask_apsec[0] == pytest.approx(2.0 * full_grid_apsec[0])

    full_grid_lpi = _metric_values(
        analyze(full_grid_cube, aoi_id="full2", config=_config(tmp_path / "full2"), pixel_size_m=30.0),
        "lpi",
    )
    half_mask_lpi = _metric_values(
        analyze(half_mask_cube, aoi_id="half2", config=_config(tmp_path / "half2"), pixel_size_m=30.0),
        "lpi",
    )
    assert full_grid_lpi and half_mask_lpi
    # LPI = largest patch area / a_total_m2 * 100 -- same doubling logic.
    assert half_mask_lpi[0] == pytest.approx(2.0 * full_grid_lpi[0])


def test_analysis_mask_denominates_monthly_coverage_not_full_grid(tmp_path):
    """Fix B regression test (end-to-end through analyze()).

    All invalid/unobserved pixels are placed ENTIRELY OUTSIDE a supplied
    analysis_mask that is itself 100% validly observed. Before the fix,
    ``valid_fraction_month``/``low_coverage_flag`` were computed over the
    WHOLE grid (diluted to ~50% by the outside-mask invalid pixels,
    incorrectly tripping ``low_coverage_flag``); after the fix they must
    reflect the analysis_mask's own real 100% coverage.
    """
    t, y, x = 1, 20, 20
    water = np.zeros((t, y, x), dtype=bool)
    water[:, 0:4, 0:10] = True  # 40 wet pixels, inside the mask below

    # analysis_mask: left half of the grid (rows 0:20, cols 0:10) = 200 px,
    # fully valid. Right half (cols 10:20) = 200 px, fully INVALID -- but
    # entirely outside the mask, so it must never enter the reduction.
    analysis_mask = np.zeros((y, x), dtype=bool)
    analysis_mask[:, 0:10] = True

    valid = np.zeros((t, y, x), dtype=bool)
    valid[:, :, 0:10] = True  # inside mask: fully valid
    valid[:, :, 10:20] = False  # outside mask: fully invalid

    times = np.array(["2020-01"], dtype="datetime64[M]").astype("datetime64[ns]")
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0

    water_da = xr.DataArray(
        water, dims=("time", "y", "x"), coords={"time": times, "y": ys, "x": xs}
    )
    valid_da = xr.DataArray(
        valid, dims=("time", "y", "x"), coords={"time": times, "y": ys, "x": xs}
    )
    analysis_mask_da = xr.DataArray(
        analysis_mask, dims=("y", "x"), coords={"y": ys, "x": xs}
    )

    cube = open_water_cube(
        water_da,
        valid_obs=valid_da,
        input_kind="generic_binary",
        analysis_mask=analysis_mask_da,
    )

    result = analyze(cube, aoi_id="cov", config=_config(tmp_path), pixel_size_m=30.0)
    table = result.metrics_table

    monthly_rows = table[table["valid_fraction_month"].notna()]
    assert not monthly_rows.empty, "expected at least one row with monthly coverage data"

    valid_fractions = monthly_rows["valid_fraction_month"].astype(float).tolist()
    low_coverage_flags = monthly_rows["low_coverage_flag"].tolist()

    # Full-grid average would be 200/400 = 0.5 (and likely trip
    # low_coverage, since default min_valid_fraction_month is 0.7). The
    # analysis_mask-restricted reduction must show ~1.0 / not-low-coverage.
    for frac in valid_fractions:
        assert frac == pytest.approx(1.0), (
            f"expected analysis_mask-restricted coverage ~1.0, got {frac} "
            "(looks like the full-grid average leaked back in)"
        )
    for flag in low_coverage_flags:
        assert flag is False
