import pytest
import xarray as xr
import pandas as pd
from hydrofragments.temporal.cadence import detect_cadence

def test_monthly_cadence_detected():
    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    da = xr.DataArray([1, 2, 3], coords={"time": times}, dims=["time"])
    
    cadence = detect_cadence(da)
    assert cadence == "monthly"

def test_submonthly_cadence_detected():
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    da = xr.DataArray([1, 2, 3], coords={"time": times}, dims=["time"])
    
    cadence = detect_cadence(da)
    assert cadence == "submonthly"
