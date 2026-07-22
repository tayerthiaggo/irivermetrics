"""TDD for hydrofragments/cli.py (Section 5).

The CLI is a thin argparse wrapper: parse args, call the same
``open_water_cube``/``analyze`` functions the notebooks use, write output --
it must not duplicate business logic. Covers both the argument-parsing
surface and an end-to-end smoke test against a tiny fixture zarr.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


def _write_fixture_zarr(path: Path) -> None:
    """A tiny already-boolean generic_binary source, written to .zarr."""
    times = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    water = np.array(
        [
            [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    ds = xr.Dataset(
        {"water": (("time", "y", "x"), water)},
        coords={"time": times},
    )
    ds.to_zarr(path, mode="w")


def _write_config_yaml(path: Path) -> None:
    config = {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


# ---- argument parsing -------------------------------------------------


def test_build_parser_requires_config_input_aoi_out() -> None:
    from hydrofragments.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze"])


def test_build_parser_accepts_full_argument_set() -> None:
    from hydrofragments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "--config",
            "cfg.yaml",
            "--input",
            "data.zarr",
            "--aoi",
            "my_reach",
            "--out",
            "results/",
        ]
    )
    assert args.config == "cfg.yaml"
    assert args.input == "data.zarr"
    assert args.aoi == "my_reach"
    assert args.out == "results/"
    assert args.pixel_size_m == 30.0  # documented default, matches api.analyze


def test_build_parser_accepts_optional_pixel_size_and_catchment() -> None:
    from hydrofragments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "--config",
            "cfg.yaml",
            "--input",
            "data.zarr",
            "--aoi",
            "my_reach",
            "--out",
            "results/",
            "--pixel-size-m",
            "10.0",
            "--catchment",
            "my_catchment",
        ]
    )
    assert args.pixel_size_m == 10.0
    assert args.catchment_id == "my_catchment"


# ---- end-to-end smoke test ---------------------------------------------


def test_main_runs_analyze_end_to_end_and_writes_output(tmp_path: Path) -> None:
    from hydrofragments.cli import main

    input_path = tmp_path / "fixture.zarr"
    config_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "results"
    _write_fixture_zarr(input_path)
    _write_config_yaml(config_path)

    exit_code = main(
        [
            "analyze",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--aoi",
            "my_reach",
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    from hydrofragments.output.tables import read_tidy_parquet

    metrics_path = out_dir / "metrics"
    assert metrics_path.is_dir()
    table = read_tidy_parquet(metrics_path)
    assert len(table) > 0
    assert (out_dir / "run_manifest.json").exists()


def test_main_accepts_json_config(tmp_path: Path) -> None:
    from hydrofragments.cli import main

    input_path = tmp_path / "fixture.zarr"
    config_path = tmp_path / "cfg.json"
    out_dir = tmp_path / "results"
    _write_fixture_zarr(input_path)
    config_path.write_text(
        json.dumps(
            {
                "config_schema_version": "1.0.0",
                "input": {"kind": "generic_binary"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "supplied",
                    "composite_owner": "caller",
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "analyze",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--aoi",
            "my_reach",
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "metrics").is_dir()


def test_main_missing_input_file_exits_nonzero_with_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hydrofragments.cli import main

    config_path = tmp_path / "cfg.yaml"
    _write_config_yaml(config_path)

    exit_code = main(
        [
            "analyze",
            "--config",
            str(config_path),
            "--input",
            str(tmp_path / "does_not_exist.zarr"),
            "--aoi",
            "my_reach",
            "--out",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "does_not_exist.zarr" in captured.err
