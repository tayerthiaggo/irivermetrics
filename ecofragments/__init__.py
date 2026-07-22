"""Deprecated EcoFragments compatibility namespace."""

import warnings

warnings.warn(
    "ecofragments is deprecated; import hydrofragments instead. "
    "See docs/migration_v1_2.md.",
    DeprecationWarning,
    stacklevel=2,
)

from ecofragments.main import calculate_metrics

__all__ = ["calculate_metrics"]
