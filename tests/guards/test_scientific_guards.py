"""Milestone 5 — scientific guards.

These enforce the non-negotiable scientific invariants around the persistence
and extent core:

- Circularity guard (spec §3, §8 guard 16): persistence metrics (occurrence, RA)
  may be summarised AOI-wide or against an external channel mask, but NEVER
  stratified by occurrence-defined zones (1/2/3/4). Reporting frequency within a
  zone defined by frequency is circular.
- CRS/unit guard (spec §8 guard 8): refuse area/length metrics when the CRS is
  geographic (degrees) unless a per-pixel area correction is supplied.
- AOI comparability guard (spec §8 guard 1; plan §1 step 10): comparisons across
  runs with different AOI definitions are refused by default.
"""
from __future__ import annotations

import pytest

from hydrofragments.guards.scientific import (
    ScientificGuardError,
    guard_aoi_comparability,
    guard_area_metric_crs,
    guard_persistence_zone,
)


class TestPersistenceZoneCircularity:
    def test_aoi_zone_is_allowed(self):
        guard_persistence_zone("occurrence", zone="AOI")  # no raise

    def test_channel_zone_is_allowed(self):
        guard_persistence_zone("refuge_area", zone="channel")  # no raise

    def test_occurrence_defined_zone_is_rejected(self):
        for zone in ("1", "2", "3", "4"):
            with pytest.raises(ScientificGuardError, match="circular"):
                guard_persistence_zone("occurrence", zone=zone)

    def test_refuge_area_by_zone_is_rejected(self):
        with pytest.raises(ScientificGuardError, match="circular"):
            guard_persistence_zone("refuge_area", zone="2")

    def test_non_persistence_metric_may_use_zones(self):
        # Fragmentation/morphology metrics legitimately stratify by zone.
        guard_persistence_zone("lpi", zone="3")  # no raise
        guard_persistence_zone("awmsi", zone="2")  # no raise


class TestAreaMetricCrsGuard:
    def test_projected_equal_area_crs_is_allowed(self):
        guard_area_metric_crs("EPSG:3577", area_method="projected")  # no raise

    def test_geographic_crs_refused_without_per_pixel_area(self):
        with pytest.raises(ScientificGuardError, match="degrees|geographic"):
            guard_area_metric_crs("EPSG:4326", area_method="projected")

    def test_geographic_crs_allowed_with_per_pixel_area(self):
        guard_area_metric_crs("EPSG:4326", area_method="per_pixel")  # no raise


class TestAoiComparability:
    def test_matching_aoi_definitions_pass(self):
        guard_aoi_comparability("gilbert_v1", "gilbert_v1")  # no raise

    def test_mismatched_aoi_refused_by_default(self):
        with pytest.raises(ScientificGuardError, match="AOI"):
            guard_aoi_comparability("gilbert_v1", "fitzroy_v1")

    def test_mismatched_aoi_allowed_with_explicit_override(self):
        guard_aoi_comparability(
            "gilbert_v1", "fitzroy_v1", override=True
        )  # no raise
