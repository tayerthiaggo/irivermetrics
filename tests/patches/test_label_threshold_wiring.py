"""W4.3 (W3.4 follow-up): a configured ``target_chunk_bytes`` must actually
reach ``label_components``'s ``local_label_threshold_bytes`` kwarg.

W3.4 proved ``label_components`` itself routes correctly given an explicit
``local_label_threshold_bytes`` override, and that its *class* default comes
from ``ComputePolicy().target_chunk_bytes`` when no override is given. It
left one gap, tracked in ``.superpowers/sdd/progress.md`` (Task W3.4 entry):
nothing constructed a ``ComputePolicy`` from a run's resolved
``HydroConfig.compute.target_chunk_bytes`` and threaded it through, so a
user-configured override had zero effect on the labeling threshold. This
file proves that gap is closed by the ``measure_patch_properties`` call path
introduced in W4.3.
"""

from __future__ import annotations

from unittest.mock import patch

import dask.array as da
import numpy as np

from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.section_analysis import _resolve_local_label_threshold_bytes
from hydrofragments.config import HydroConfig
from hydrofragments.metrics import patches as patches_module
from hydrofragments.metrics.patches import measure_patch_properties
from hydrofragments.patches import labels as labels_module


def _config(*, target_chunk_bytes: int | None) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "compute": {"target_chunk_bytes": target_chunk_bytes},
        }
    )


def _mask_sized_between_configured_and_class_default() -> np.ndarray:
    """~3.8 MiB bool mask: below ComputePolicy's 128 MiB class default, but
    above the small configured override this file uses -- the exact "sized
    between the old default and the new configured value" fixture the brief
    asks for.
    """
    mask = np.zeros((2000, 2000), dtype=bool)
    mask[10:14, 10:14] = True
    return mask


def test_resolve_local_label_threshold_bytes_is_none_when_config_is_none():
    """None (config default) must preserve label_components's own
    ComputePolicy-class-default fallback exactly, per the brief's explicit
    backward-compatibility requirement.
    """
    config = _config(target_chunk_bytes=None)
    assert _resolve_local_label_threshold_bytes(config) is None


def test_resolve_local_label_threshold_bytes_uses_configured_value():
    config = _config(target_chunk_bytes=1_000_000)
    assert _resolve_local_label_threshold_bytes(config) == 1_000_000


def test_resolve_local_label_threshold_bytes_matches_compute_policy_construction():
    """Resolution must go through ComputePolicy(target_chunk_bytes=...), not
    just pass the raw int straight through untouched, per the brief's literal
    instruction -- this pins that ComputePolicy is genuinely constructed
    (and would surface e.g. a future ComputePolicy validation rule).
    """
    config = _config(target_chunk_bytes=2_000_000)
    resolved = _resolve_local_label_threshold_bytes(config)
    expected = ComputePolicy(target_chunk_bytes=2_000_000).target_chunk_bytes
    assert resolved == expected


def test_configured_target_chunk_bytes_changes_labeling_path_via_measure_patch_properties():
    """Central proof: a mask sized between a small configured
    target_chunk_bytes and ComputePolicy's 128 MiB class default takes
    DIFFERENT labeling paths depending on whether the configured value is
    actually threaded through measure_patch_properties -> label_components.
    """
    mask = _mask_sized_between_configured_and_class_default()
    dask_backed = da.from_array(mask, chunks=(500, 500))
    assert mask.nbytes < ComputePolicy().target_chunk_bytes

    small_config = _config(target_chunk_bytes=1_000_000)
    assert mask.nbytes > small_config.compute.target_chunk_bytes
    threshold = _resolve_local_label_threshold_bytes(small_config)

    # With the small configured threshold actually threaded through, the
    # mask exceeds it and must take the dask-image cross-chunk path. Note:
    # dask-image's own cross-chunk implementation calls ndimage.label
    # internally per block, so only dask_image_label's call is a valid
    # signal of which ROUTING branch was taken (see the identical caveat in
    # tests/patches/test_labels.py's above-threshold test).
    with patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        measure_patch_properties(
            dask_backed,
            pixel_size_m=30.0,
            local_label_threshold_bytes=threshold,
        )
    dask_image_label.assert_called_once()

    # Left unthreaded (None), label_components falls back to its own
    # 128 MiB class default, under which this same mask routes to SciPy
    # instead -- demonstrating the configured value is what changed the
    # routing above, not some property of the mask alone.
    with patch(
        "hydrofragments.patches.labels.ndimage.label",
        wraps=labels_module.ndimage.label,
    ) as scipy_label, patch(
        "hydrofragments.patches.labels.ndmeasure.label",
        wraps=labels_module.ndmeasure.label,
    ) as dask_image_label:
        measure_patch_properties(
            dask_backed,
            pixel_size_m=30.0,
            local_label_threshold_bytes=None,
        )
    scipy_label.assert_called_once()
    dask_image_label.assert_not_called()


def test_none_target_chunk_bytes_preserves_todays_fallback_behavior():
    """When config.compute.target_chunk_bytes is None (today's default),
    measure_patch_properties must behave exactly as if no threshold override
    were threaded through at all -- i.e. no behavior change for callers who
    never configure this value.
    """
    mask = _mask_sized_between_configured_and_class_default()
    dask_backed = da.from_array(mask, chunks=(500, 500))
    none_config = _config(target_chunk_bytes=None)
    resolved = _resolve_local_label_threshold_bytes(none_config)
    assert resolved is None

    baseline = measure_patch_properties(
        dask_backed, pixel_size_m=30.0, local_label_threshold_bytes=None
    )
    threaded = measure_patch_properties(
        dask_backed, pixel_size_m=30.0, local_label_threshold_bytes=resolved
    )
    assert len(baseline) == len(threaded)
    for left, right in zip(baseline, threaded):
        assert left.label == right.label
        assert left.area_pixels == right.area_pixels
        assert left.area_m2 == right.area_m2
        assert left.perimeter_m == right.perimeter_m


def test_analyze_section_rows_threads_configured_threshold_end_to_end(tmp_path):
    """End-to-end: HydroConfig.compute.target_chunk_bytes reaches
    label_components via section_compat_rows's real per-month patch call,
    not just via measure_patch_properties called directly.
    """
    import pandas as pd
    import xarray as xr

    from hydrofragments.section_analysis import analyze_section_rows

    water_np = np.zeros((1, 40, 40), dtype=bool)
    water_np[0, 5:9, 5:9] = True
    times = pd.to_datetime(["2020-01-01"])
    water = xr.DataArray(water_np, dims=("time", "y", "x"), coords={"time": times})

    config = _config(target_chunk_bytes=64)
    resolved = _resolve_local_label_threshold_bytes(config)
    assert resolved == 64

    with patch.object(
        patches_module,
        "label_components",
        wraps=patches_module.label_components,
    ) as spy:
        analyze_section_rows(
            water,
            section="demo",
            section_area_km2=0.36,
            pixel_size_m=30.0,
            config=config,
            selected_ids={"number_of_pools", "lpi", "awre", "awmsi"},
        )

    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["local_label_threshold_bytes"] == 64
