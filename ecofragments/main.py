"""Deprecated ecofragments entry point routed to HydroFragments v1.2."""

from __future__ import annotations

import warnings

from hydrofragments.compat import calculate_metrics_compat


def calculate_metrics(
    da_wmask,
    rcor_extent=None,
    outdir=None,
    section_length=None,
    section_name_col=None,
    min_patch_size=2,
    img_ext=".tif",
    export_shp=False,
    export_PP=False,
    fill_nodata=True,
    *,
    legacy_metrics=None,
):
    """Deprecated wide-table facade over retained HydroFragments v1.2 metrics."""
    warnings.warn(
        "ecofragments.calculate_metrics is deprecated; use hydrofragments.analyze "
        "for canonical tidy output. See docs/migration_v1_2.md.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_metrics_compat(
        da_wmask,
        rcor_extent=rcor_extent,
        outdir=outdir,
        section_length=section_length,
        section_name_col=section_name_col,
        min_patch_size=min_patch_size,
        img_ext=img_ext,
        export_shp=export_shp,
        export_PP=export_PP,
        fill_nodata=fill_nodata,
        legacy_metrics=legacy_metrics,
    )
