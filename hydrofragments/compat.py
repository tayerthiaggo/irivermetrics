"""Legacy compatibility helpers and dropped-metric migration errors."""

from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec
from hydrofragments.metrics.patches import analyze_patch_metrics
from hydrofragments.metrics.persistence import compute_occurrence, compute_refuge_area
from hydrofragments.metrics.registry import resolve_metrics
from hydrofragments.schema import MetricDependency

DROPPED_LEGACY_METRICS: dict[str, str] = {
    "PF": (
        "Removed in HydroFragments v1.2: naive patches-per-area fragmentation "
        "index. Use LPI (fixed AOI denominator) and number_of_pools instead."
    ),
    "PLF": (
        "Removed in HydroFragments v1.2: naive patches-per-length index. "
        "Use LPI with an explicit channel reference when drainage is available."
    ),
    "AWMPA": (
        "Removed in HydroFragments v1.2: area-weighted mean patch area is not "
        "part of the canonical register."
    ),
    "AWMPL": (
        "Removed in HydroFragments v1.2: area-weighted mean patch length is not "
        "part of the canonical register."
    ),
    "AWMPW": (
        "Removed in HydroFragments v1.2: area-weighted mean patch width is "
        "deferred pending resolution-floor validation."
    ),
}

FORBIDDEN_LEGACY_COLUMNS = frozenset(DROPPED_LEGACY_METRICS) | {"LPSEC"}

RETAINED_COMPAT_COLUMNS = (
    "date",
    "section",
    "section_area_km2",
    "n_patches",
    "APSEC",
    "AWMSI",
    "AWRe",
    "LPI",
    "pp_mean_%",
    "ra_area_km2",
)


class LegacyMetricMigrationError(ValueError):
    """Raised when a caller requests a metric removed from v1.2."""


def request_legacy_metrics(metric_ids: Iterable[str]) -> None:
    """Fail fast with migration guidance for dropped legacy metrics."""
    requested = [str(item).strip() for item in metric_ids if str(item).strip()]
    if not requested:
        return
    messages: list[str] = []
    for metric_id in requested:
        key = metric_id.upper()
        if key in DROPPED_LEGACY_METRICS:
            messages.append(f"{key}: {DROPPED_LEGACY_METRICS[key]}")
        elif key == "LPSEC":
            messages.append(
                "LPSEC: channel-dependent extent metric excluded from v1.2.0 "
                "core until a real drainage L_ref contract is active."
            )
    if messages:
        raise LegacyMetricMigrationError(
            "Requested legacy metrics are not available in HydroFragments v1.2. "
            "See docs/migration_v1_2.md. " + " ".join(messages)
        )
    unknown = [item for item in requested if item.upper() not in DROPPED_LEGACY_METRICS | {"LPSEC"}]
    if unknown:
        raise LegacyMetricMigrationError(
            "Unknown legacy metric request(s): "
            + ", ".join(unknown)
            + ". Canonical v1.2 metrics are documented in docs/migration_v1_2.md."
        )


def legacy_hydro_config(
    *,
    min_patch_size: int = 2,
    metric_profiles: tuple[str, ...] = ("contracts_core",),
) -> HydroConfig:
    """Build a v1.2 config for ecofragments-shaped calls."""
    min_patch_pixels = max(3, int(min_patch_size) + 1)
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": list(metric_profiles),
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {
                "min_patch_pixels": min_patch_pixels,
                "connectivity_rule": 8,
            },
        }
    )


def _monthly_dataset(da_feature: xr.DataArray) -> xr.Dataset:
    # Section clips are bounded; materialise once at the compat orchestration boundary.
    da_feature = da_feature.load()
    water = (da_feature == 1).astype(bool)
    valid_obs = xr.ones_like(water, dtype=bool)
    return xr.Dataset({"water": water, "valid_obs": valid_obs})


_PATCH_METRIC_IDS = frozenset({"number_of_pools", "lpi", "awre", "awmsi"})
_PERSISTENCE_METRIC_IDS = frozenset({"occurrence", "refuge_area"})


def section_compat_rows(
    da_feature: xr.DataArray,
    *,
    section: str,
    section_area_km2: float,
    pixel_size_m: float,
    config: HydroConfig,
    selected_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Compute retained v1.2 metrics in a legacy-compatible wide row shape.

    ``selected_ids`` is an optional B1 optimisation: when provided (only the
    canonical ``analyze()`` path does this), families whose metric ids are
    absent from ``selected_ids`` are skipped entirely rather than computed
    and discarded. When ``None`` (the default -- used by the legacy
    ``calculate_metrics_compat`` shim, which has no concept of "selected
    metrics" and always wants the full fixed wide-row export), every family
    is computed exactly as before. Skipped families still populate their row
    keys with ``None``/``nan`` placeholders so ``compat_dataframe()`` and
    ``_records_from_compat_rows`` (which filters by metric id after
    construction) never see a missing key.
    """
    want_patches = selected_ids is None or bool(selected_ids & _PATCH_METRIC_IDS)
    want_persistence = selected_ids is None or bool(selected_ids & _PERSISTENCE_METRIC_IDS)
    want_apsec = selected_ids is None or "apsec" in selected_ids

    monthly = _monthly_dataset(da_feature)
    cell_area_m2 = float(pixel_size_m) ** 2
    a_ref_m2 = float(section_area_km2) * 1_000_000.0

    pp_mean = float("nan")
    refuge = None
    if want_persistence:
        occurrence = compute_occurrence(monthly, config=config)
        refuge = compute_refuge_area(
            occurrence, cell_area_m2=cell_area_m2, config=config
        )
        pp_mean = float(occurrence.occurrence.mean(skipna=True).item())
        if np.isnan(pp_mean):
            pp_mean = float("nan")

    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(pd.to_datetime(monthly["time"].values)):
        n_patches: object = None
        awmsi = float("nan")
        awre = float("nan")
        lpi = float("nan")
        if want_patches:
            mask = np.asarray(
                monthly["water"].isel(time=time_index).values, dtype=bool
            )
            patch_metrics = analyze_patch_metrics(
                mask,
                pixel_size_m=pixel_size_m,
                a_total_m2=a_ref_m2,
                connectivity=config.patches.connectivity_rule,
                min_patch_pixels=config.patches.min_patch_pixels,
            )
            n_patches = patch_metrics.number_of_pools
            awmsi = patch_metrics.awmsi
            awre = patch_metrics.awre
            lpi = patch_metrics.lpi

        apsec_value = float("nan")
        if want_apsec:
            apsec_value = compute_apsec(
                monthly.isel(time=[time_index]),
                a_ref_m2=a_ref_m2,
                cell_area_m2=cell_area_m2,
                config=config,
            )[0].value

        rows.append(
            {
                "date": pd.Timestamp(timestamp),
                "section": section,
                "section_area_km2": section_area_km2,
                "n_patches": n_patches,
                "APSEC": apsec_value,
                "AWMSI": awmsi,
                "AWRe": awre,
                "LPI": lpi,
                "pp_mean_%": pp_mean,
                "ra_area_km2": refuge.value if refuge is not None else float("nan"),
            }
        )
    return rows


def compat_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    forbidden = FORBIDDEN_LEGACY_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise LegacyMetricMigrationError(
            "Compatibility output must not include dropped metrics: "
            + ", ".join(sorted(forbidden))
        )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["n_patches"] = frame["n_patches"].astype("int32")
    return frame.sort_values(["section", "date"]).reset_index(drop=True)


def calculate_metrics_compat(
    da_wmask: xr.DataArray | xr.Dataset | str,
    *,
    rcor_extent: gpd.GeoDataFrame | str | None = None,
    outdir: str | None = None,
    section_length: float | None = None,
    section_name_col: str | None = None,
    min_patch_size: int = 2,
    img_ext: str = ".tif",
    export_shp: bool = False,
    export_PP: bool = False,
    fill_nodata: bool = True,
    legacy_metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run retained v1.2 metrics and return a non-canonical wide pivot."""
    import os
    import tempfile

    from ecofragments.utils import calc_metrics

    if legacy_metrics is not None:
        request_legacy_metrics(legacy_metrics)

    if export_shp:
        raise LegacyMetricMigrationError(
            "export_shp is not supported on the v1.2 compatibility facade. "
            "Use hydrofragments analyze() with output.include_vectors instead."
        )
    if export_PP:
        raise LegacyMetricMigrationError(
            "export_PP via ecofragments.calculate_metrics is replaced by tidy "
            "occurrence/refuge rasters from hydrofragments.analyze()."
        )

    config = legacy_hydro_config(min_patch_size=min_patch_size)
    resolve_metrics(
        config.metric_profiles,
        available_dependencies={
            MetricDependency.VALIDITY,
            MetricDependency.PATCHES,
        },
    )

    if rcor_extent is None:
        outdir = outdir or tempfile.mkdtemp(prefix="hydrofragments_")
        array = calc_metrics.coerce_water_mask_dataarray(da_wmask)
        if array.sizes.get("time", 0) < 2:
            raise ValueError("at least two timesteps are required to calculate metrics")
        pixel_size = 30.0
        if hasattr(array, "rio") and "x" in array.coords and "y" in array.coords:
            try:
                pixel_size = float(abs(array.rio.resolution()[0]))
            except Exception:
                pixel_size = 30.0
        section_area_km2 = float(array.isel(time=0).size) * pixel_size**2 / 1_000_000.0
        rows = section_compat_rows(
            array,
            section="AOI",
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size,
            config=config,
        )
        metrics_df = compat_dataframe(rows)
        metrics_df.to_csv(os.path.join(outdir, "ecof_metrics.csv"), index=False)
        return metrics_df

    da_wmask, rcor_extent, section_length, crs, pixel_size, outdir = calc_metrics.validate(
        da_wmask,
        rcor_extent,
        outdir,
        section_length,
        img_ext,
        section_name_col,
    )
    da_wmask, rcor_extent = calc_metrics.preprocess(
        da_wmask, rcor_extent, fill_nodata
    )

    rows: list[dict[str, object]] = []
    for _, feature in rcor_extent.iterrows():
        prepared = calc_metrics.preprocess_feature_operations(
            da_wmask, feature, section_name_col
        )
        rows.extend(
            section_compat_rows(
                prepared["da_wmask_feature"],
                section=str(prepared["section"]),
                section_area_km2=float(prepared["section_area"]),
                pixel_size_m=float(pixel_size),
                config=config,
            )
        )

    metrics_df = compat_dataframe(rows)
    if outdir is not None:
        metrics_df.to_csv(f"{outdir}/ecof_metrics.csv", index=False)

    return metrics_df


__all__ = [
    "DROPPED_LEGACY_METRICS",
    "FORBIDDEN_LEGACY_COLUMNS",
    "LegacyMetricMigrationError",
    "RETAINED_COMPAT_COLUMNS",
    "calculate_metrics_compat",
    "compat_dataframe",
    "legacy_hydro_config",
    "request_legacy_metrics",
    "section_compat_rows",
]
