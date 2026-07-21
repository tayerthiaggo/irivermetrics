"""Thin command-line entry point over the public HydroFragments API.

"Notebook for learning, CLI for efficiency" -- this module parses arguments,
calls :func:`hydrofragments.api.open_water_cube` and
:func:`hydrofragments.api.analyze` (the exact same functions the onboarding
notebooks call), and writes output. It must never duplicate scientific or
I/O logic that already lives in ``hydrofragments.api``/``hydrofragments.output``.

Usage::

    hydrofragments analyze --config cfg.yaml --input data.zarr --aoi my_reach --out results/

``--config`` accepts YAML or JSON (detected by file extension; ``.json`` -> JSON,
anything else -> YAML, which is also valid for plain JSON since JSON is a YAML
subset). ``--input`` is passed straight to ``open_water_cube`` -- currently
that means a ``.zarr`` path (see ``open_water_cube``'s docstring for supported
source shapes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from hydrofragments._version import __version__


def _load_config_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydrofragments",
        description=(
            "HydroFragments: surface-water patch dynamics metrics for "
            "intermittent rivers."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"hydrofragments {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run open_water_cube -> analyze on one input and write output.",
    )
    analyze_parser.add_argument(
        "--config",
        required=True,
        help="Path to a HydroConfig YAML or JSON file (see HydroConfig.from_mapping).",
    )
    analyze_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input source passed to open_water_cube (e.g. a .zarr store).",
    )
    analyze_parser.add_argument(
        "--aoi",
        required=True,
        help="AOI identifier passed to analyze() as aoi_id.",
    )
    analyze_parser.add_argument(
        "--out",
        required=True,
        help="Output directory (also written into config.output.output_dir).",
    )
    analyze_parser.add_argument(
        "--pixel-size-m",
        dest="pixel_size_m",
        type=float,
        default=30.0,
        help="Pixel size in metres, forwarded to analyze() (default: 30.0).",
    )
    analyze_parser.add_argument(
        "--catchment",
        dest="catchment_id",
        default=None,
        help="Optional catchment identifier, forwarded to analyze() (default: aoi).",
    )
    return parser


def _run_analyze(args: argparse.Namespace) -> int:
    from hydrofragments.api import analyze, open_water_cube
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.tables import write_output_tables

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"hydrofragments analyze: input not found: {input_path}", file=sys.stderr)
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"hydrofragments analyze: config not found: {config_path}", file=sys.stderr)
        return 2

    raw_config = _load_config_mapping(config_path)
    raw_config = dict(raw_config)
    output_section = dict(raw_config.get("output") or {})
    output_section.setdefault("output_dir", args.out)
    raw_config["output"] = output_section

    config = HydroConfig.from_mapping(raw_config)
    cube = open_water_cube(input_path)
    result = analyze(
        cube,
        aoi_id=args.aoi,
        config=config,
        pixel_size_m=args.pixel_size_m,
        catchment_id=args.catchment_id,
    )
    # HydroResult.write() forwards a formats= kwarg that write_output_tables()
    # does not accept (pre-existing bug in hydrofragments.models, out of
    # Section 5's scope to fix) -- call write_output_tables() directly
    # instead, which is the same underlying writer analyze() itself uses,
    # not a reimplementation of output logic.
    write_output_tables(result.metrics_table, args.out)
    print(
        f"hydrofragments analyze: wrote {len(result.metrics_table)} metric "
        f"rows to {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            return _run_analyze(args)
        except (ValueError, OSError) as error:
            print(f"hydrofragments analyze: {error}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover -- parser.error() always raises SystemExit


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
