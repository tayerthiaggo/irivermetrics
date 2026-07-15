"""M8 — ecofragments compatibility facade routes to hydrofragments."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.compat import (
    DROPPED_LEGACY_METRICS,
    LegacyMetricMigrationError,
    request_legacy_metrics,
)


def test_ecofragments_import_emits_deprecation_warning() -> None:
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ecofragments

        importlib.reload(ecofragments)
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "hydrofragments" in str(item.message).lower()
        for item in caught
    )


def test_calculate_metrics_emits_deprecation_warning() -> None:
    from ecofragments import calculate_metrics

    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = calculate_metrics(water, rcor_extent=None, outdir=None)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("metric_id", sorted(DROPPED_LEGACY_METRICS))
def test_request_for_dropped_legacy_metric_raises_migration_error(
    metric_id: str,
) -> None:
    with pytest.raises(LegacyMetricMigrationError) as exc:
        request_legacy_metrics([metric_id])
    assert metric_id in str(exc.value)
    assert DROPPED_LEGACY_METRICS[metric_id] in str(exc.value)


def test_compat_wide_output_excludes_dropped_legacy_columns() -> None:
    from ecofragments import calculate_metrics

    times = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    water = xr.DataArray(
        np.ones((3, 4, 4), dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    result = calculate_metrics(water, rcor_extent=None, outdir=None)
    forbidden = {"PF", "PLF", "AWMPA", "AWMPL", "AWMPW", "LPSEC"}
    assert forbidden.isdisjoint(set(result.columns))


def test_metric_override_adding_dropped_metric_raises() -> None:
    from ecofragments import calculate_metrics

    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.ones((2, 3, 3), dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    with pytest.raises(LegacyMetricMigrationError):
        calculate_metrics(
            water,
            rcor_extent=None,
            outdir=None,
            legacy_metrics=["PF"],
        )
