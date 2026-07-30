"""Scientific guards for the persistence and extent core (Milestone 5).

Each guard encodes a spec invariant that must fail loudly rather than produce a
scientifically invalid number:

- :func:`guard_persistence_zone` — the occurrence/zone circularity guard
  (spec §3 circularity guard, §8 guard 16).
- :func:`guard_area_metric_crs` — the CRS/unit guard (spec §8 guard 8).
- :func:`guard_aoi_comparability` — refuse cross-AOI comparison by default
  (spec §8 guard 1; plan §1 processing step 10).
"""
from __future__ import annotations

import pyproj


class ScientificGuardError(ValueError):
    """Raised when a scientific invariant would be violated."""


# Metrics whose value is a persistence frequency derived from occurrence.
_PERSISTENCE_METRICS = frozenset({"occurrence", "refuge_area", "recurrence"})

# Zones legitimately usable for persistence: whole AOI or an externally defined
# channel mask. The occurrence-defined zones 1-4 are forbidden.
_ALLOWED_PERSISTENCE_ZONES = frozenset({None, "AOI", "channel"})
_OCCURRENCE_DEFINED_ZONES = frozenset({"1", "2", "3", "4"})


def guard_persistence_zone(metric_id: str, *, zone: str | None) -> None:
    """Refuse stratifying a persistence metric by an occurrence-defined zone.

    Zones 1-4 are defined from the RAW persistence surface, so reporting a
    persistence frequency within one is circular. Non-persistence metrics
    (fragmentation, morphology, clustering, connectivity) may stratify freely.

    Provenance-independence: this function's signature accepts only a zone
    LABEL (``"1"``-``"4"``/``"AOI"``/``"channel"``/``None``), never a
    ``ZoneResult`` or its ``source`` field, and by construction never
    inspects provenance. A zone labelled ``"2"`` built locally via
    ``hydrofragments.spatial.zones.build_zones`` (``ZoneResult.source ==
    "occurrence"``) and a zone labelled ``"2"`` built from a DEA product via
    ``zones_from_wo_statistics`` (``ZoneResult.source == stats.product``,
    e.g. ``"ga_ls_wo_fq_myear_3"``) are refused IDENTICALLY here, because
    both represent the same occurrence-defined stratification this guard
    exists to protect against -- a DEA Water Observation Statistics product
    is external to any given run, but measures the same wet-frequency
    phenomenon, so its provenance must never silently waive this guard. See
    ``tests/guards/test_scientific_guards.py::TestPersistenceZoneProvenanceIndependence``
    for the test proving this with both provenances.
    """
    if metric_id not in _PERSISTENCE_METRICS:
        return
    if zone in _ALLOWED_PERSISTENCE_ZONES:
        return
    if zone in _OCCURRENCE_DEFINED_ZONES:
        raise ScientificGuardError(
            f"persistence metric '{metric_id}' cannot be stratified by "
            f"occurrence-defined zone '{zone}': this is circular (spec §3, "
            "guard 16). Report AOI-wide or against an external channel mask."
        )
    raise ScientificGuardError(
        f"unsupported zone '{zone}' for persistence metric '{metric_id}'"
    )


def guard_area_metric_crs(crs: str, *, area_method: str) -> None:
    """Refuse area metrics on a geographic CRS without per-pixel correction.

    An equal-area projected CRS gives correct pixel areas directly. A
    geographic CRS (units of degrees) does not, so it is refused unless
    ``area_method`` is ``per_pixel`` (a validated per-pixel area correction is
    supplied).
    """
    if area_method == "per_pixel":
        return
    if pyproj.CRS.from_user_input(crs).is_geographic:
        raise ScientificGuardError(
            f"area metrics refused on geographic CRS '{crs}' (units are "
            "degrees). Reproject to an equal-area CRS or set "
            "spatial.area_method='per_pixel' with a per-pixel area correction "
            "(spec §8 guard 8)."
        )


def guard_aoi_comparability(
    left_aoi_id: str, right_aoi_id: str, *, override: bool = False
) -> None:
    """Refuse comparing results built on different AOI definitions by default.

    An AOI definition is a load-bearing scientific setting: metrics with fixed
    denominators (APSEC, LPI) and refuge areas are not comparable across
    different AOIs. Mismatches are refused unless an override is explicitly
    recorded.
    """
    if override:
        return
    if left_aoi_id != right_aoi_id:
        raise ScientificGuardError(
            "cannot compare results across different AOI definitions "
            f"('{left_aoi_id}' vs '{right_aoi_id}'): fixed-denominator "
            "metrics are not comparable. Pass override=True to record an "
            "explicit "
            "exception (spec §8 guard 1)."
        )


__all__ = [
    "ScientificGuardError",
    "guard_aoi_comparability",
    "guard_area_metric_crs",
    "guard_persistence_zone",
]
