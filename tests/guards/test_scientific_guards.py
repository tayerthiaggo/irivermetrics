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

from types import SimpleNamespace

import numpy as np
import pytest

from hydrofragments.config import ValidityConfig, ZonesConfig
from hydrofragments.guards.scientific import (
    ScientificGuardError,
    guard_aoi_comparability,
    guard_area_metric_crs,
    guard_persistence_zone,
)
from hydrofragments.io.dea import WoStatistics
from hydrofragments.spatial.zones import build_zones, zones_from_wo_statistics


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


# --- Provenance independence (task W1.3) ------------------------------------
#
# guard_persistence_zone's signature takes only a metric_id and a plain zone
# LABEL string -- never a ZoneResult, never a `source` field -- so it cannot
# distinguish a locally-built zone (hydrofragments.spatial.zones.build_zones,
# ZoneResult.source == "occurrence" by default) from a DEA-derived zone
# (zones_from_wo_statistics, ZoneResult.source == stats.product, e.g.
# "ga_ls_wo_fq_myear_3"). The plan requires this explicitly: "A DEA product
# is external to this run, but it measures the same wet-frequency
# phenomenon; source provenance must not silently waive the scientific
# guard." These tests prove that invariant with real ZoneResult objects
# built via both provenance paths, not just by re-asserting the guard's
# label-only behaviour already covered above.
#
# A test in this class would FAIL if a future change added a provenance-based
# bypass such as `if zone_result.source != "occurrence": skip guard` inside
# guard_persistence_zone (or inside any future call site that wires this
# guard in) -- that is exactly the regression this class exists to catch.


def _config(*, t_persist: float = 0.50, t_season: float = 0.10, min_valid_obs: int = 20):
    return SimpleNamespace(
        zones=ZonesConfig(t_persist=t_persist, t_season=t_season),
        validity=ValidityConfig(min_valid_obs=min_valid_obs),
    )


def _wo_statistics(
    *,
    frequency: np.ndarray,
    count_wet: np.ndarray,
    count_clear: np.ndarray,
    product: str = "ga_ls_wo_fq_myear_3",
) -> WoStatistics:
    return WoStatistics(
        frequency=frequency,
        count_wet=count_wet,
        count_clear=count_clear,
        product=product,
        version="1.2.3",
        crs="EPSG:3577",
        time_span="2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        provenance={"product": product},
    )


class TestPersistenceZoneProvenanceIndependence:
    # Same occurrence/support inputs used to build both a local-cube
    # ZoneResult (build_zones) and a DEA-derived ZoneResult
    # (zones_from_wo_statistics), so both provenances emit the exact same
    # zone labels for the exact same underlying wet-frequency surface.
    _occurrence = np.array([[90.0, 45.0], [10.0, 1.0]])
    _max_wet = np.ones((2, 2), dtype=bool)
    _valid_count = np.full((2, 2), 20)

    def test_local_and_dea_zone_results_emit_the_same_label_set(self):
        local_result = build_zones(
            self._occurrence,
            max_wet_mask=self._max_wet,
            valid_count=self._valid_count,
        )
        stats = _wo_statistics(
            frequency=self._occurrence,
            count_wet=np.array([[5, 5], [5, 5]]),
            count_clear=self._valid_count,
        )
        dea_result = zones_from_wo_statistics(stats, config=_config())

        # Different provenance...
        assert local_result.source == "occurrence"
        assert dea_result.source == "ga_ls_wo_fq_myear_3"
        assert local_result.source != dea_result.source
        # ...but identical zone labels drawn from the same {2, 3, 4} set,
        # because both were built from the same occurrence surface.
        assert local_result.mask.tolist() == dea_result.mask.tolist()
        assert set(np.unique(local_result.mask)) == {2, 3, 4}
        assert set(np.unique(dea_result.mask)) == {2, 3, 4}

    def test_guard_refuses_local_cube_zone_labels_identically_to_dea_zone_labels(self):
        local_result = build_zones(
            self._occurrence,
            max_wet_mask=self._max_wet,
            valid_count=self._valid_count,
        )
        stats = _wo_statistics(
            frequency=self._occurrence,
            count_wet=np.array([[5, 5], [5, 5]]),
            count_clear=self._valid_count,
        )
        dea_result = zones_from_wo_statistics(stats, config=_config())

        for metric_id in ("occurrence", "refuge_area", "recurrence"):
            for zone_label in sorted(
                set(np.unique(local_result.mask)) | set(np.unique(dea_result.mask))
            ):
                if zone_label == 0:
                    continue  # unzoned background pixel value, not a zone label
                label = str(int(zone_label))
                with pytest.raises(ScientificGuardError, match="circular"):
                    guard_persistence_zone(metric_id, zone=label)

    def test_guard_does_not_special_case_zoneresult_source_field(self):
        # Explicitly prove the guard call itself never receives or branches
        # on `.source` -- passing only the zone LABEL (as every real caller
        # must, since the guard's signature has no ZoneResult parameter)
        # refuses identically whether that label came from a ZoneResult
        # stamped source="occurrence" or source="ga_ls_wo_fq_myear_3".
        local_result = build_zones(
            self._occurrence,
            max_wet_mask=self._max_wet,
            valid_count=self._valid_count,
        )
        stats = _wo_statistics(
            frequency=self._occurrence,
            count_wet=np.array([[5, 5], [5, 5]]),
            count_clear=self._valid_count,
        )
        dea_result = zones_from_wo_statistics(stats, config=_config())

        zone_label = str(int(local_result.mask[0, 1]))  # zone "3" pixel
        assert zone_label == str(int(dea_result.mask[0, 1]))

        for zone_result in (local_result, dea_result):
            with pytest.raises(ScientificGuardError, match="circular"):
                guard_persistence_zone(
                    "occurrence", zone=str(int(zone_result.mask[0, 1]))
                )

    def test_guard_allows_both_provenances_equally_for_aoi_and_channel(self):
        # Provenance-agnostic on the ALLOW path too: neither a local-cube
        # ZoneResult nor a DEA-derived one can smuggle a persistence metric
        # past the guard just by using the AOI/channel labels either
        # provenance's ZoneResult never actually emits, since those labels
        # aren't in _OCCURRENCE_DEFINED_ZONES.
        guard_persistence_zone("occurrence", zone="AOI")  # no raise
        guard_persistence_zone("refuge_area", zone="channel")  # no raise


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
