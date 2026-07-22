"""Thin adapter over the external `hydroseason` package (Decision Gate 0, Q7/V8).

HY detection and season mapping are owned entirely by the sibling package
`hydroseason` (pinned `==0.1.0`). This module contains **no detector logic** —
it only calls `hydroseason.detect_hydrological_years` /
`hydroseason.label_hydrological_months` and reshapes the result into the
HY-anchor vocabulary the rest of HydroFragments (``metrics/dynamics.py``, the
tidy output schema's ``hy``/``hy_anchor``/``hy_confidence`` columns) expects.

Reimplementing HY/season detection here would duplicate a published method
(V8: persistence-based HY vs Tayer 2025/2026 rainfall-based HY, verified in
agreement on Fitzroy/Gilbert) and break the single-source-of-truth contract
this adapter exists to preserve.
"""
from __future__ import annotations

from dataclasses import dataclass

import hydroseason
import pandas as pd


def hydroseason_config_from_hydroconfig(config):
    """Convert resolved local settings into pinned hydroseason config."""
    from hydrofragments.config import HydroConfig

    if not isinstance(config, HydroConfig):
        raise TypeError("config must be a HydroConfig")
    return hydroseason.HydroYearConfig(**dict(config.hydroyear.parameters))


def hydroseason_config_to_mapping(config: "hydroseason.HydroYearConfig") -> dict[str, object]:
    """Return stable JSON-compatible fields passed to hydroseason."""
    return {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }


class HydroYearAdapterError(ValueError):
    """Raised when the input series does not satisfy hydroseason's own contract."""


@dataclass(frozen=True)
class HyAnchorResult:
    """HY anchors and month labels, reshaped from `hydroseason`'s public API.

    ``anchors`` carries one row per detected hydrological year with
    ``peak_month``, ``end_dry_month``, and ``confidence`` (as assigned by
    `hydroseason.detect_hydrological_years` — never recomputed here).
    ``month_labels`` carries the per-month ``hy``/``season`` assignment from
    `hydroseason.label_hydrological_months`. ``hydroseason_version`` and
    ``config`` are recorded for run-manifest provenance.
    """

    anchors: pd.DataFrame
    month_labels: pd.DataFrame
    hydroseason_version: str
    config: "hydroseason.HydroYearConfig"


def detect_hy_anchors(
    extent: pd.Series,
    *,
    config: "hydroseason.HydroYearConfig | None" = None,
    hydrofragments_config=None,
) -> HyAnchorResult:
    """Detect HY anchors and season labels by delegating to `hydroseason`.

    ``extent`` is a monthly ``extent_pct`` series (e.g. APSEC). All detection,
    confidence assignment, and season-window logic is `hydroseason`'s; this
    function only calls its public API and renames columns to HydroFragments'
    HY-anchor vocabulary (``hy_year`` -> ``hy``, ``end_dry_month`` kept as-is).
    """
    if config is not None and hydrofragments_config is not None:
        raise TypeError("pass either config or hydrofragments_config, not both")
    resolved_config = config or (
        hydroseason_config_from_hydroconfig(hydrofragments_config)
        if hydrofragments_config is not None
        else hydroseason.HydroYearConfig()
    )
    try:
        hy_frame = hydroseason.detect_hydrological_years(
            extent, config=resolved_config, missing_month_policy="raise"
        )
        month_labels = hydroseason.label_hydrological_months(extent.index, hy_frame)
    except ValueError as error:
        raise HydroYearAdapterError(str(error)) from error

    anchors = hy_frame.rename(columns={"hy_year": "hy"})
    month_labels = month_labels.rename(columns={"hy_year": "hy"})

    return HyAnchorResult(
        anchors=anchors,
        month_labels=month_labels,
        hydroseason_version=hydroseason.__version__,
        config=resolved_config,
    )


__all__ = [
    "HyAnchorResult",
    "HydroYearAdapterError",
    "detect_hy_anchors",
    "hydroseason_config_from_hydroconfig",
    "hydroseason_config_to_mapping",
]
