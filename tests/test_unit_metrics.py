"""Unit tests for ecofragments.utils.calc_metrics helper functions."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecofragments.utils.calc_metrics import (
    batch_date_list,
    calculate_pixel_persistence,
    clip_data,
    coerce_water_mask_dataarray,
    process_metrics,
)


# ---------------------------------------------------------------------------
# coerce_water_mask_dataarray
# ---------------------------------------------------------------------------
class TestCoerceWaterMaskDataarray:
    def test_passthrough_dataarray(self, da_wmask):
        result = coerce_water_mask_dataarray(da_wmask)
        assert isinstance(result, xr.DataArray)
        assert result is da_wmask

    def test_dataset_with_water_var(self, da_wmask):
        ds = da_wmask.to_dataset(name="water")
        result = coerce_water_mask_dataarray(ds)
        assert isinstance(result, xr.DataArray)
        assert result.name == "water"

    def test_dataset_single_var(self, da_wmask):
        ds = da_wmask.to_dataset(name="mask")
        result = coerce_water_mask_dataarray(ds)
        assert isinstance(result, xr.DataArray)

    def test_dataset_ambiguous_raises(self, da_wmask):
        ds = da_wmask.to_dataset(name="a")
        ds["b"] = da_wmask
        with pytest.raises(AssertionError):
            coerce_water_mask_dataarray(ds)


# ---------------------------------------------------------------------------
# batch_date_list
# ---------------------------------------------------------------------------
class TestBatchDateList:
    def test_exact_multiple(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 13)]
        batches = batch_date_list(dates, batch_size=4)
        assert len(batches) == 3
        assert all(len(b) == 4 for b in batches)

    def test_remainder_batch(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        batches = batch_date_list(dates, batch_size=3)
        assert len(batches) == 4
        # last batch has the remainder
        assert len(batches[-1]) == 1

    def test_empty_list(self):
        assert batch_date_list([]) == []

    def test_preserves_order(self):
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        flat = [d for batch in batch_date_list(dates, batch_size=2) for d in batch]
        assert flat == dates


# ---------------------------------------------------------------------------
# clip_data
# ---------------------------------------------------------------------------
class TestClipData:
    def test_clips_spatial_extent(self, da_wmask):
        xmin = float(da_wmask.x.values[10])
        xmax = float(da_wmask.x.values[20])
        ymin = float(da_wmask.y.values[10])
        ymax = float(da_wmask.y.values[20])
        clipped = clip_data(da_wmask, xmin, xmax, ymin, ymax)
        assert clipped.x.min() >= xmin
        assert clipped.x.max() <= xmax


# ---------------------------------------------------------------------------
# calculate_pixel_persistence
# ---------------------------------------------------------------------------
class TestCalculatePixelPersistence:
    def test_all_water(self):
        arr = xr.DataArray(
            np.ones((5, 3, 3), dtype=np.int8),
            dims=["time", "y", "x"],
        )
        pp = calculate_pixel_persistence(arr)
        assert float(pp.max()) == pytest.approx(100.0)

    def test_no_water(self):
        arr = xr.DataArray(
            np.zeros((5, 3, 3), dtype=np.int8),
            dims=["time", "y", "x"],
        )
        pp = calculate_pixel_persistence(arr)
        assert float(pp.max()) == pytest.approx(0.0)

    def test_half_water(self):
        data = np.zeros((4, 2, 2), dtype=np.int8)
        data[:2, :, :] = 1  # half the time steps are wet
        arr = xr.DataArray(data, dims=["time", "y", "x"])
        pp = calculate_pixel_persistence(arr)
        assert float(pp.mean()) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# process_metrics
# ---------------------------------------------------------------------------
class TestProcessMetrics:
    def _make_group(self, areas, lengths, widths, perimeters,
                    section_area=1.0, section_length=1.0):
        return pd.DataFrame({
            "area_km2": areas,
            "length_km": lengths,
            "width_km": widths,
            "perimeter_km": perimeters,
            "section_area_km2": [section_area] * len(areas),
            "section_length_km": [section_length] * len(areas),
            "pp_mean_%": [50.0] * len(areas),
            "ra_area_km2": [0.1] * len(areas),
        })

    def test_single_pool(self):
        group = self._make_group([0.1], [1.0], [0.05], [0.5])
        result = process_metrics(group)
        assert result["n_patches"] == 1
        assert result["wet_area_km2"] == pytest.approx(0.1)
        assert result["wet_length_km"] == pytest.approx(1.0)

    def test_zero_area_returns_zeros(self):
        group = self._make_group([0.0], [0.0], [0.0], [0.0])
        result = process_metrics(group)
        assert result["n_patches"] == 0
        assert result["wet_area_km2"] == 0
        assert np.isnan(result["AWRe"])

    def test_multiple_pools_awmsi_positive(self):
        group = self._make_group(
            [0.1, 0.2], [1.0, 2.0], [0.05, 0.1], [0.5, 1.0]
        )
        result = process_metrics(group)
        assert result["n_patches"] == 2
        assert result["AWMSI"] > 0

    def test_nan_section_length_gives_nan_lpsec(self):
        group = self._make_group(
            [0.1], [1.0], [0.05], [0.5], section_length=float("nan")
        )
        result = process_metrics(group)
        assert np.isnan(result["LPSEC"])
