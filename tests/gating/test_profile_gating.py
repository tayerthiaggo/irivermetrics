"""B1: analyze() must resolve selected_ids before dispatching metric compute.

Today ``section_compat_rows()`` unconditionally computes occurrence, refuge,
APSEC, and patch morphology for every call, then ``analyze()`` filters the
resulting records down to the requested profile's ``selected_ids`` *after*
the fact. On a real satellite cube this forces the whole time-y-x cube
through every kernel even when the caller only asked for a narrow profile.

These tests pin the fix: narrow profiles that exclude patch-dependent
metrics must not invoke ``analyze_patch_bundle`` at all, and the
persistence-family kernels (``recurrence``/``hydroperiod``, emitted through
the separate ``_temporal_profile_records`` path) must still work correctly
when a narrow profile selects them.

``pixel_temporal`` is used as the narrow profile: per
``hydrofragments/metrics/registry.py`` PROFILES, it resolves to
``("recurrence", "hydroperiod")`` only -- neither has a PATCHES dependency,
so no patch-dependent metric (``number_of_pools``, ``lpi``, ``awre``,
``awmsi``, ``mesh``) is ever selected. Both ``recurrence`` and
``hydroperiod`` are ``MetricFamily.PERSISTENCE`` (see registry.py), so this
profile also proves persistence-family metrics keep working when patch
morphology is skipped -- matching the already-passing
``test_analyze_emits_pixel_temporal_profile_rows`` in
tests/compat/test_hydrofragments_public_api.py, which is the canonical
proof that ``pixel_temporal`` alone is a valid, self-sufficient profile.
"""

from __future__ import annotations

from unittest import mock

from hydrofragments import HydroConfig, analyze


def _pixel_temporal_config(tmp_path):
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["pixel_temporal"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )


def test_narrow_profile_skips_patch_morphology(synthetic_cube, tmp_path):
    """A profile with no patch-dependent metrics must not call analyze_patch_bundle."""
    config = _pixel_temporal_config(tmp_path)
    with mock.patch("hydrofragments.metrics.patches.analyze_patch_bundle") as patch_spy:
        analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    patch_spy.assert_not_called()


def test_pixel_temporal_profile_still_emits_persistence_metrics(synthetic_cube, tmp_path):
    """Persistence-family metrics selected by a narrow profile must still be emitted."""
    config = _pixel_temporal_config(tmp_path)
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    metrics = set(result.metrics_table["metric"])
    assert {"recurrence", "hydroperiod"} <= metrics
    assert metrics.isdisjoint({"number_of_pools", "lpi", "awre", "awmsi", "mesh"})
