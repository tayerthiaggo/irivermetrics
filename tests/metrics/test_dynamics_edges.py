import numpy as np
import pytest
from datetime import datetime

from hydrofragments.metrics import dynamics
from hydrofragments.schema import EdgeFlag


def test_end_dry_value_degrades_to_nan():
    # no record matching the requested end-dry month
    from datetime import date
    val = dynamics._end_dry_value([], date(2015, 6, 1))
    assert np.isnan(val)


def test_first_crossing_rejects_unsorted():
    from datetime import date
    series = [(date(2015, 3, 1), 1.0), (date(2015, 1, 1), 9.0)]  # out of order
    with pytest.raises(AssertionError):
        dynamics._first_crossing(series, end_dry_month=date(2015, 1, 1), threshold=5.0)


def test_search_window_stops_before_next_hy_end_dry():
    from datetime import date

    window = dynamics._search_window_months(
        end_dry_month=date(2001, 10, 1),
        next_end_dry_month=date(2002, 10, 1),
        cube_month_keys=[
            (2001, m) for m in range(1, 13)
        ] + [(2002, m) for m in range(1, 13)],
    )
    assert window == [(2001, 11), (2001, 12)] + [(2002, m) for m in range(1, 10)]


def test_evaluate_refuge_empty_union_is_nan_not_zero():
    current = dynamics.EndDryState(
        hy=2002,
        date=datetime(2002, 10, 1),
        water=np.zeros((2, 2), dtype=bool),
        valid_obs=np.ones((2, 2), dtype=bool),
        hy_confidence="high",
    )
    previous = dynamics.EndDryState(
        hy=2001,
        date=datetime(2001, 10, 1),
        water=np.zeros((2, 2), dtype=bool),
        valid_obs=np.ones((2, 2), dtype=bool),
        hy_confidence="high",
    )
    result = dynamics.evaluate_refuge_spatial_stability(
        current=current,
        previous=previous,
        analysis_mask=None,
        min_valid_fraction=0.0,
    )
    assert result.edge_flag is EdgeFlag.EMPTY_REFUGE_UNION
    assert np.isnan(result.jaccard)
