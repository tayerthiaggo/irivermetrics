"""Tiny, hand-calculable, bundled fixture for the onboarding notebooks.

Not part of the ``hydrofragments`` public package -- this lives in
``examples/`` and is imported directly by ``01_quickstart.ipynb`` (and by
``tests/docs/test_example_fixtures.py`` / ``tests/docs/test_examples.py``, so
the notebook and its tests always run the exact same code, never a
copy-pasted variant).

Follows the same "tiny, documented, ground-truth" pattern as
``tests/fixtures/analytic_masks.py`` (see that module's docstring), but adds
a real ``time`` dimension and a projected CRS, since ``open_water_cube``
needs a genuine time series rather than a single 2-D mask, and HydroFragments
refuses undefined/geographic CRS input (spec Sec8 guard 8).

Ground truth: a 7x7 wet square, centred in an 11x11 dry canvas, that loses
one ring of pixels every month for 4 months -- a simple, visually obvious
"drying reach" story for a first-time user, with an exactly known wet-pixel
count per month (see ``QUICKSTART_WET_PIXEL_COUNTS``). This is plain
synthetic data, not gapfilled, not resampled, and not reprojected from
anything -- it is authored directly on one fixed 30 m/pixel EPSG:3577 grid,
so no CRS/grid-mismatch handling is exercised or implied here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

# One ring removed per month: 7x7 (49 px) -> 5x5 (25 px) -> 3x3 (9 px) -> 1x1 (1 px).
QUICKSTART_WET_PIXEL_COUNTS: list[int] = [49, 25, 9, 1]
QUICKSTART_CRS = "EPSG:3577"
QUICKSTART_PIXEL_SIZE_M = 30.0


def _square_mask(canvas: int, half_width: int) -> np.ndarray:
    """A solid wet square of side ``2*half_width + 1`` centred on ``canvas``x``canvas``."""
    mask = np.zeros((canvas, canvas), dtype=bool)
    centre = canvas // 2
    lo, hi = centre - half_width, centre + half_width + 1
    mask[lo:hi, lo:hi] = True
    return mask


def quickstart_water_timeseries(*, canvas: int = 11) -> xr.DataArray:
    """A 4-month synthetic water mask time series: a reach that dries down.

    Returns a boolean ``xr.DataArray`` with dims ``(time, y, x)``, a defined
    projected CRS (EPSG:3577, 30 m pixels -- matches DEA's native grid), and
    monthly timestamps. Ground truth wet-pixel counts per month are
    ``QUICKSTART_WET_PIXEL_COUNTS``: 49, 25, 9, 1.
    """
    half_widths = (3, 2, 1, 0)  # 7x7, 5x5, 3x3, 1x1
    frames = np.stack([_square_mask(canvas, hw) for hw in half_widths], axis=0)

    times = pd.date_range("2020-01-01", periods=len(half_widths), freq="MS")
    ys = np.arange(canvas, dtype=float) * -QUICKSTART_PIXEL_SIZE_M + 6_000_000.0
    xs = np.arange(canvas, dtype=float) * QUICKSTART_PIXEL_SIZE_M + 400_000.0

    water = xr.DataArray(
        frames,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
        name="water",
    )
    return water.rio.write_crs(QUICKSTART_CRS)


def quickstart_config(*, output_dir: str) -> dict[str, Any]:
    """Minimal ``HydroConfig.from_mapping``-ready dict for the quickstart notebook.

    ``generic_binary`` matches ``quickstart_water_timeseries``'s already-
    boolean values; ``monthly_composite: "supplied"`` / ``composite_owner:
    "caller"`` because the fixture is already one frame per month -- nothing
    for HydroFragments to composite.
    """
    return {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
        "output": {"output_dir": output_dir},
    }


def write_synthetic_tsfill_zarr(path) -> None:
    """Write a tiny local ``.zarr`` mimicking a WaterMask-TSFill export.

    Reproduces TSFill's canonical uint8 ``water_mask`` sentinel signature
    (see ``hydrofragments/io/adapters.py``'s ``_TSFILL_SENTINELS`` /
    ``_looks_like_tsfill``): ``0`` = dry, ``1`` = water, ``254`` = outside
    AOI, ``255`` = unobserved. This lets ``02_dea_via_tsfill.ipynb`` show
    the real ``watermask_tsfill`` adapter path end-to-end without a live
    DEA/TSFill output available at notebook-authoring time.

    In real use, point ``open_water_cube`` at your actual TSFill output
    path instead of this synthetic fixture -- see ``docs/input_format.md``
    for the full adapter contract (this function exists purely so the
    notebook has something runnable to demonstrate the handoff with).
    """
    canvas = 10
    frames = np.stack(
        [_square_mask(canvas, hw).astype(np.uint8) for hw in (3, 2, 1, 0)],
        axis=0,
    )
    # Mark a fixed 1-pixel corner strip "outside AOI" and one pixel per
    # frame "unobserved", so the notebook can show both sentinels being
    # decoded, not just the water/dry values.
    raw = frames.copy()
    raw[:, 0, 0] = 254  # outside AOI, every month
    raw[:, 0, 1] = 255  # unobserved, every month

    times = pd.date_range("2021-01-01", periods=frames.shape[0], freq="MS")
    ys = np.arange(canvas, dtype=float) * -QUICKSTART_PIXEL_SIZE_M + 6_000_000.0
    xs = np.arange(canvas, dtype=float) * QUICKSTART_PIXEL_SIZE_M + 400_000.0

    # Deliberately no rio.write_crs() here: writing a CRS onto a Dataset adds
    # a "spatial_ref" coordinate, which round-trips through zarr as a second
    # data variable and makes the Dataset ambiguous for auto-detection on
    # reopen. An *undefined* CRS is fine (only a *geographic*, degrees CRS
    # is refused -- see check_crs_defined); matches the same
    # no-CRS-on-disk pattern tests/conftest.py's tmp_zarr_path fixture uses.
    dataset = xr.Dataset(
        {"water_mask": (("time", "y", "x"), raw)},
        coords={"time": times, "y": ys, "x": xs},
    )
    dataset.to_zarr(path, mode="w")


def walkthrough_water_timeseries(
    *, n_years: int = 3, canvas_y: int = 7, canvas_x: int = 25
) -> xr.DataArray:
    """A multi-year two-pool reach for ``03_metrics_walkthrough.ipynb``.

    Two square pools sit at opposite ends of a wide, short canvas, connected
    only by a single row of "channel" pixels along the centre (``y ==
    canvas_y // 2``). Each pool's size breathes seasonally (bigger in the
    wet months, smaller in the dry months, following the same 12-month
    extent-percentage shape as ``tests/contracts/test_hydrofragments_public_api
    .py::test_analyze_calls_hydroseason_when_hydroyear_extent_is_supplied``'s
    ``WALKTHROUGH_HY_EXTENT_PATTERN`` -- reusing a pattern already proven to
    drive real hydrological-year anchor detection, rather than inventing a
    new one that might not). The centre channel row is only wet in the top
    ~40% wettest months, so the two pools sometimes merge into one
    connected component and sometimes fragment into two -- deliberately
    giving the Morphology/Fragmentation and Clustering/Connectivity
    sections something to show.

    Returns a boolean ``xr.DataArray`` with dims ``(time, y, x)`` and a
    defined projected CRS (EPSG:3577, 30 m pixels), ``n_years * 12`` months
    starting 2001-01-01 (matches the reused extent pattern's calendar
    alignment).
    """
    months_per_year = len(WALKTHROUGH_HY_EXTENT_PATTERN)
    n_months = n_years * months_per_year
    pct = np.tile(WALKTHROUGH_HY_EXTENT_PATTERN, n_years)  # 0-100 scale
    fraction = pct / 100.0

    pool_half_span_y = max(1, canvas_y // 2 - 1)
    max_half_width_x = 3
    centre_row = canvas_y // 2
    left_centre_x = max_half_width_x + 1
    right_centre_x = canvas_x - max_half_width_x - 2

    frames = np.zeros((n_months, canvas_y, canvas_x), dtype=bool)
    for month_index, frac in enumerate(fraction):
        half_width_x = max(1, round(frac * max_half_width_x))
        half_width_y = max(1, round(frac * pool_half_span_y))

        y_lo, y_hi = centre_row - half_width_y, centre_row + half_width_y + 1
        left_lo, left_hi = left_centre_x - half_width_x, left_centre_x + half_width_x + 1
        right_lo, right_hi = right_centre_x - half_width_x, right_centre_x + half_width_x + 1
        frames[month_index, y_lo:y_hi, left_lo:left_hi] = True
        frames[month_index, y_lo:y_hi, right_lo:right_hi] = True

        # Channel row only wets in the top ~40% wettest months -- gives the
        # reach a genuine connected/fragmented alternation through time.
        if frac >= 0.6:
            frames[month_index, centre_row, left_hi:right_lo] = True

    times = pd.date_range("2001-01-01", periods=n_months, freq="MS")
    ys = np.arange(canvas_y, dtype=float) * -QUICKSTART_PIXEL_SIZE_M + 6_000_000.0
    xs = np.arange(canvas_x, dtype=float) * QUICKSTART_PIXEL_SIZE_M + 400_000.0

    water = xr.DataArray(
        frames,
        dims=("time", "y", "x"),
        coords={"time": times, "y": ys, "x": xs},
        name="water",
    )
    return water.rio.write_crs(QUICKSTART_CRS)


# Reused verbatim from the already-passing
# test_analyze_calls_hydroseason_when_hydroyear_extent_is_supplied (see
# tests/contracts/test_hydrofragments_public_api.py) -- a 12-month
# extent-percentage shape already proven to drive real hydrological-year
# anchor detection via hydroseason.detect_hydrological_years, so the
# walkthrough notebook's dynamics section is not gambling on an untested
# synthetic seasonal shape.
WALKTHROUGH_HY_EXTENT_PATTERN: tuple[int, ...] = (
    70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55,
)


def walkthrough_hydroyear_extent(water: xr.DataArray) -> pd.Series:
    """The same extent-percentage series used to shape ``water``'s pool sizes.

    ``analyze``'s dynamics profile needs a wetted-extent-percentage series
    indexed by the cube's own timestamps to detect hydrological-year
    anchors (``AnalysisInputs.hydroyear_extent``) -- reusing the exact
    values that shaped the fixture keeps the two consistent by
    construction rather than by coincidence.
    """
    n_months = water.sizes["time"]
    n_years = n_months // len(WALKTHROUGH_HY_EXTENT_PATTERN)
    values = np.tile(WALKTHROUGH_HY_EXTENT_PATTERN, n_years)
    return pd.Series(values, index=pd.to_datetime(water["time"].values))


def walkthrough_channel_context(aoi_id: str = "walkthrough_reach"):
    """A straight-line drainage/AOI pair matching ``walkthrough_water_timeseries``.

    Builds a real :class:`hydrofragments.spatial.SpatialContext` (not a
    stub/mock) via :func:`hydrofragments.spatial.create_channel_context`, so
    the walkthrough notebook's Clustering & Connectivity section (LPSEC,
    inter-pool gap) runs the genuine channel-dependent code path, matching
    the pattern already exercised in
    ``tests/contracts/test_hydrofragments_public_api.py``. The line spans
    the full canvas width along the centre row used by
    ``walkthrough_water_timeseries``'s default ``canvas_y=7``.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, box

    from hydrofragments.spatial import create_channel_context

    canvas_x = 25
    x0 = 400_000.0
    x1 = 400_000.0 + (canvas_x - 1) * QUICKSTART_PIXEL_SIZE_M
    centre_y = 6_000_000.0 - 3 * QUICKSTART_PIXEL_SIZE_M  # matches centre_row=3

    aoi = gpd.GeoDataFrame(
        geometry=[box(x0 - 15.0, centre_y - 120.0, x1 + 15.0, centre_y + 120.0)],
        crs=QUICKSTART_CRS,
    )
    drainage = gpd.GeoDataFrame(
        {"HydroID": [1], "From_Node": [10], "To_Node": [11], "NextDownID": [-1]},
        geometry=[LineString([(x0, centre_y), (x1, centre_y)])],
        crs=QUICKSTART_CRS,
    )
    return create_channel_context(
        aoi_id, aoi, drainage, drainage_id="walkthrough-v1", target_crs=QUICKSTART_CRS
    )


def walkthrough_apsec_composites(
    water: xr.DataArray, extent: pd.Series
) -> tuple[list, list]:
    """Synthetic max-water/median APSEC composite pairs for the dynamics section.

    ``analyze``'s dynamics profile (``extent_contraction``) requires *both*
    ``max_water_apsec`` and ``median_apsec`` (see ``AnalysisInputs``'s
    docstring); absent either, dynamics metrics are skipped rather than
    computed. There is no real dual-composite pipeline wired into this
    walkthrough's synthetic fixture, so this reuses the same
    ``walkthrough_hydroyear_extent`` percentages directly as the max-water
    composite (median set to 90% of it, a plausible median-below-max
    relationship) purely to exercise the code path -- see the notebook's
    own markdown for that caveat; do not read the resulting
    ``extent_contraction`` values as scientifically meaningful.
    """
    from hydrofragments.metrics import ApsecRecord

    cell_area_m2 = QUICKSTART_PIXEL_SIZE_M**2
    a_ref_m2 = float(water.isel(time=0).size) * cell_area_m2
    times = [pd.Timestamp(t).to_pydatetime() for t in water["time"].values]
    max_records = [
        ApsecRecord(
            date=t, value=float(v), n_water_pixels=0,
            a_ref_m2=a_ref_m2, cell_area_m2=cell_area_m2,
        )
        for t, v in zip(times, extent.to_numpy(dtype=float))
    ]
    median_records = [
        ApsecRecord(
            date=t, value=float(v) * 0.9, n_water_pixels=0,
            a_ref_m2=a_ref_m2, cell_area_m2=cell_area_m2,
        )
        for t, v in zip(times, extent.to_numpy(dtype=float))
    ]
    return max_records, median_records


def walkthrough_config(*, output_dir: str) -> dict[str, Any]:
    """``HydroConfig.from_mapping``-ready dict selecting all 4 metric families.

    ``contracts_core`` -> extent/persistence/morphology/fragmentation,
    ``pixel_temporal`` -> persistence (recurrence/hydroperiod),
    ``channel`` -> extent (LPSEC)/clustering (inter-pool gap),
    ``dynamics`` -> dynamics (extent_contraction). ``monthly_composite:
    "max_water"`` (rather than quickstart's ``"supplied"``) because
    ``extent_contraction``'s ``monthly_composite`` grouping distinguishes
    max_water vs. median composites -- see
    ``walkthrough_apsec_composites``.
    """
    return {
        "config_schema_version": "1.0.0",
        "metric_profiles": [
            "contracts_core", "pixel_temporal", "channel", "dynamics",
        ],
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "max_water",
            "composite_owner": "caller",
        },
        "output": {"output_dir": output_dir},
    }


__all__ = [
    "QUICKSTART_CRS",
    "QUICKSTART_PIXEL_SIZE_M",
    "QUICKSTART_WET_PIXEL_COUNTS",
    "WALKTHROUGH_HY_EXTENT_PATTERN",
    "quickstart_config",
    "quickstart_water_timeseries",
    "walkthrough_apsec_composites",
    "walkthrough_channel_context",
    "walkthrough_config",
    "walkthrough_hydroyear_extent",
    "walkthrough_water_timeseries",
    "write_synthetic_tsfill_zarr",
]
