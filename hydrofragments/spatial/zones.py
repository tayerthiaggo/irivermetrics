"""Static occurrence-derived zones with optional real drainage Zone 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ZoneResult:
    mask: np.ndarray
    emitted_zones: tuple[int, ...]
    has_zone_1: bool


def build_zones(
    occurrence: np.ndarray,
    *,
    max_wet_mask: np.ndarray,
    valid_count: np.ndarray,
    drainage_mask: np.ndarray | None = None,
    t_persist: float = 0.50,
    t_season: float = 0.10,
    min_valid_obs: int = 20,
) -> ZoneResult:
    """Build mutually exclusive Zones 1-4; zero means outside zoned extent.

    Without drainage, Zone 1 is absent. Zone 2 then represents persistent
    water collectively and is never split using wet-mask morphology.
    """
    frequency = np.asarray(occurrence, dtype=float)
    max_wet = np.asarray(max_wet_mask, dtype=bool)
    support = np.asarray(valid_count)
    if frequency.ndim != 2:
        raise ValueError("occurrence must be a 2-D array")
    if max_wet.shape != frequency.shape:
        raise ValueError("occurrence and max_wet_mask must share shape")
    if support.shape != frequency.shape:
        raise ValueError("occurrence and valid_count must share shape")
    if min_valid_obs < 1:
        raise ValueError("min_valid_obs must be at least 1")
    if not 0.0 <= t_season < t_persist <= 1.0:
        raise ValueError("zone thresholds require 0 <= t_season < t_persist <= 1")

    valid = max_wet & np.isfinite(frequency) & (support >= min_valid_obs)
    mask = np.zeros(frequency.shape, dtype=np.uint8)
    mask[valid & (frequency < t_season)] = 4
    mask[valid & (frequency >= t_season) & (frequency <= t_persist)] = 3
    mask[valid & (frequency > t_persist)] = 2

    if drainage_mask is None:
        return ZoneResult(mask=mask, emitted_zones=(2, 3, 4), has_zone_1=False)

    drainage = np.asarray(drainage_mask, dtype=bool)
    if drainage.shape != frequency.shape:
        raise ValueError("occurrence and drainage_mask must share shape")
    adjacent = ndimage.binary_dilation(
        drainage, structure=np.ones((3, 3), dtype=bool)
    )
    zone_1 = drainage | (adjacent & valid & (frequency > t_persist))
    mask[zone_1] = 1
    return ZoneResult(mask=mask, emitted_zones=(1, 2, 3, 4), has_zone_1=True)


__all__ = ["ZoneResult", "build_zones"]
