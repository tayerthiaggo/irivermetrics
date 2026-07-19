import numpy as np
import pytest
from hydrofragments.metrics import dynamics


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
