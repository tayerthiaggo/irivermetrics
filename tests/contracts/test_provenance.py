"""Git revision resolution and per-run provenance consistency."""

from __future__ import annotations

import importlib.metadata
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.output.tables import resolve_git_sha


def test_resolve_git_sha_prefers_ci_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDROFRAGMENTS_GIT_SHA", "ci-revision-abc")
    assert resolve_git_sha() == "ci-revision-abc"


def test_resolve_git_sha_uses_installed_package_revision_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYDROFRAGMENTS_GIT_SHA", raising=False)

    class FakeMetadata:
        def get(self, key: str, default: str = "") -> str:
            values = {
                "Source-Revision-Id": "wheel-revision-xyz",
            }
            return values.get(key, default)

    def fake_metadata(name: str) -> FakeMetadata:
        assert name == "hydrofragments"
        return FakeMetadata()

    monkeypatch.setattr(importlib.metadata, "metadata", fake_metadata)
    assert resolve_git_sha() == "wheel-revision-xyz"


def test_analyze_reuses_one_git_sha_for_metrics_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HYDROFRAGMENTS_GIT_SHA", "run-revision-shared")
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )

    result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    assert result.metrics_table["git_sha"].nunique() == 1
    assert result.metrics_table["git_sha"].iloc[0] == "run-revision-shared"
    assert result.manifest["git_sha"] == "run-revision-shared"
