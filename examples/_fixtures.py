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


__all__ = [
    "QUICKSTART_CRS",
    "QUICKSTART_PIXEL_SIZE_M",
    "QUICKSTART_WET_PIXEL_COUNTS",
    "quickstart_config",
    "quickstart_water_timeseries",
]
