"""Milestone 11 -- reconnection metric preference order, spec section 6.15.

Load-bearing contract (spec section 6.15, required output columns):

- Preference order: **RC** (fixed graph available) > **LPSEC** (real channel,
  no graph) > **LPI** (coarse proxy, last resort).
- ``reconnection_metric_used`` records exactly which metric decided
  ``t_reconnect``; ``proxy_reconnection_flag`` is ``True`` for every metric
  except RC (RC is the preferred, non-proxy metric per spec section 6.15).
- A metric only wins the preference slot if its series was actually
  supplied -- an unsupplied RC series must not silently short-circuit to
  "no reconnection," it must fall through to the next available metric.
"""
from __future__ import annotations

from datetime import datetime

from hydrofragments.metrics.dynamics import compute_reconnection_timing


def _series(monthly_values: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    return [(datetime.fromisoformat(date), value) for date, value in monthly_values]


def test_rc_series_present_takes_priority_over_lpi():
    end_dry_month = datetime(2001, 10, 1)
    rc_series = _series(
        [("2001-10-01", 5.0), ("2001-11-01", 70.0), ("2001-12-01", 90.0)]
    )
    lpi_series = _series(
        [("2001-10-01", 80.0), ("2001-11-01", 80.0), ("2001-12-01", 80.0)]
    )

    result = compute_reconnection_timing(
        rc_series=rc_series,
        rc_threshold=60.0,
        lpi_series=lpi_series,
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )

    assert result.reconnection_metric_used == "RC"
    assert result.proxy_reconnection_flag is False
    assert result.t_reconnect_months == 1


def test_rc_absent_falls_back_to_lpsec_when_supplied():
    end_dry_month = datetime(2001, 10, 1)
    lpsec_series = _series(
        [("2001-10-01", 5.0), ("2001-11-01", 55.0), ("2001-12-01", 90.0)]
    )
    lpi_series = _series(
        [("2001-10-01", 80.0), ("2001-11-01", 80.0), ("2001-12-01", 80.0)]
    )

    result = compute_reconnection_timing(
        lpsec_series=lpsec_series,
        lpsec_threshold=50.0,
        lpi_series=lpi_series,
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )

    assert result.reconnection_metric_used == "LPSEC"
    assert result.proxy_reconnection_flag is True
    assert result.t_reconnect_months == 1


def test_rc_and_lpsec_absent_falls_back_to_lpi():
    end_dry_month = datetime(2001, 10, 1)
    lpi_series = _series(
        [("2001-10-01", 5.0), ("2001-11-01", 20.0), ("2001-12-01", 65.0)]
    )

    result = compute_reconnection_timing(
        lpi_series=lpi_series,
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )

    assert result.reconnection_metric_used == "LPI"
    assert result.proxy_reconnection_flag is True
    assert result.t_reconnect_months == 2


def test_rc_supplied_but_never_crosses_threshold_does_not_fall_back():
    # RC is supplied and preferred -- if RC never reconnects, that is the
    # true answer for the preferred metric; must not silently fall back to
    # a proxy that might show reconnection RC does not confirm.
    end_dry_month = datetime(2001, 10, 1)
    rc_series = _series(
        [("2001-10-01", 5.0), ("2001-11-01", 10.0), ("2001-12-01", 15.0)]
    )
    lpi_series = _series(
        [("2001-10-01", 80.0), ("2001-11-01", 80.0), ("2001-12-01", 80.0)]
    )

    result = compute_reconnection_timing(
        rc_series=rc_series,
        rc_threshold=60.0,
        lpi_series=lpi_series,
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )

    assert result.reconnection_metric_used == "RC"
    assert result.proxy_reconnection_flag is False
    assert result.t_reconnect_months is None
