"""Read-only fixture inspection helpers for Milestone 0 evidence."""

from __future__ import annotations

import datetime
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_cadence(times: np.ndarray) -> str:
    if times.size < 2:
        return "unknown"
    deltas = np.diff(times.astype("datetime64[D]")).astype(int)
    unique_days = {int(d) for d in deltas}
    if unique_days == {28, 29, 30, 31} or unique_days <= {28, 29, 30, 31}:
        return "monthly"
    return "sub_monthly_irregular"


def inspect_water_mask_netcdf(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "legacy_netcdf_water_only",
    }
    if not path.exists():
        return report

    report["checksum_sha256"] = _sha256(path)
    ds = xr.open_dataset(path)
    try:
        var_name = "water" if "water" in ds else next(iter(ds.data_vars))
        da = ds[var_name]
        report["variable"] = var_name
        report["dtype"] = str(da.dtype)
        report["dimensions"] = {dim: int(da.sizes[dim]) for dim in da.dims}
        report["time_range"] = {
            "start": str(da.time.min().values)[:10],
            "end": str(da.time.max().values)[:10],
        }
        report["cadence"] = {"inferred": _infer_cadence(da.time.values)}

        try:
            report["crs"] = str(da.rio.crs)
        except Exception as exc:  # pragma: no cover - environment-specific
            report["crs"] = f"unavailable: {exc}"

        values = da.values
        uniq, counts = np.unique(values, return_counts=True)
        report["value_domain"] = {
            str(int(value)): int(count) for value, count in zip(uniq, counts)
        }

        wet_fracs = [float((frame == 1).mean()) for frame in values]
        report["wet_fraction_per_timestep"] = {
            "min": float(min(wet_fracs)),
            "max": float(max(wet_fracs)),
            "std": float(np.std(wet_fracs)),
            "unique_count": len({round(v, 8) for v in wet_fracs}),
        }
        report["sentinel_presence"] = {
            "uint8_254": 254 in uniq,
            "uint8_255": 255 in uniq,
            "legacy_nodata_-1": -1 in uniq,
        }
        report["has_valid_observation_layer"] = False
        report["suitable_uses"] = [
            "legacy_integration_smoke",
            "kernel_characterisation_with_synthetic_valid_layer",
        ]
        report["unsuitable_uses"] = [
            "v1.2_occurrence_denominator_contract_without_valid_obs",
            "dry_down_hy_stability_recurrence_without_monthly_composite_provenance",
            "watermask_tsfill_sentinel_contract_test",
        ]
    finally:
        ds.close()
    return report


def inspect_water_mask_zarr(path: Path) -> dict[str, Any]:
    """Read-only inspection of a real WaterMask-TSFill-style monthly zarr cube.

    Unlike ``inspect_water_mask_netcdf``, this targets the single-variable
    ``water_mask`` delivery observed for the Fitzroy validation catchment
    (values: -2 outside AOI, -1 unobserved, 0 dry, 1 wet), not the full
    four-variable canonical contract.
    """
    import zarr

    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "real_monthly_water_mask_zarr",
    }
    if not path.exists():
        return report

    metadata_path = path / ".zmetadata"
    if metadata_path.exists():
        report["zmetadata_checksum_sha256"] = _sha256(metadata_path)

    group = zarr.open(str(path), mode="r")
    wm = group["water_mask"]
    report["dimensions"] = {
        "time": wm.shape[0],
        "y": wm.shape[1],
        "x": wm.shape[2],
    }
    report["dtype"] = str(wm.dtype)

    wm_attrs = wm.attrs.asdict()
    report["cadence"] = {"declared_dims": wm_attrs.get("_ARRAY_DIMENSIONS")}
    report["n_time_steps_expected"] = wm_attrs.get("n_time_steps_expected")
    report["n_time_steps_source"] = wm_attrs.get("n_time_steps_source")
    report["n_inserted_timesteps"] = wm_attrs.get("n_inserted_timesteps")

    spatial_ref_attrs = group["spatial_ref"].attrs.asdict()
    crs_wkt = spatial_ref_attrs.get("crs_wkt", "")
    report["crs"] = "EPSG:3577" if "3577" in crs_wkt else f"unrecognised: {crs_wkt[:60]}"

    n_t = wm.shape[0]
    time_days = group["time"][:]
    time_units = group["time"].attrs.asdict().get("units", "days since 1987-01-01 00:00:00")
    epoch = datetime.date.fromisoformat(time_units.replace("days since ", "").split(" ")[0])
    calendar_months = [
        (epoch + timedelta(days=int(d))).month for d in time_days
    ]

    class_totals: dict[int, int] = {}
    wet_fracs: list[float] = []
    observed_fracs: list[float] = []
    per_calendar_month_wet: dict[int, int] = {m: 0 for m in range(1, 13)}
    per_calendar_month_observed: dict[int, int] = {m: 0 for m in range(1, 13)}
    for t in range(n_t):
        layer = wm[t]
        uniq, counts = np.unique(layer, return_counts=True)
        counts_by_value = dict(zip(uniq.tolist(), counts.tolist()))
        for value, count in counts_by_value.items():
            class_totals[value] = class_totals.get(value, 0) + count
        observed = counts_by_value.get(0, 0) + counts_by_value.get(1, 0)
        wet = counts_by_value.get(1, 0)
        aoi = observed + counts_by_value.get(-1, 0)
        wet_fracs.append(float(wet / observed) if observed > 0 else float("nan"))
        observed_fracs.append(float(observed / aoi) if aoi > 0 else float("nan"))
        month = calendar_months[t]
        per_calendar_month_wet[month] += wet
        per_calendar_month_observed[month] += observed

    per_month_wet_frac = {
        m: (per_calendar_month_wet[m] / per_calendar_month_observed[m])
        if per_calendar_month_observed[m] > 0
        else float("nan")
        for m in range(1, 13)
    }
    naive_pooled_wet_frac = sum(per_calendar_month_wet.values()) / sum(
        per_calendar_month_observed.values()
    )
    stratified_wet_frac = float(np.mean(list(per_month_wet_frac.values())))

    report["value_domain"] = {str(k): v for k, v in sorted(class_totals.items())}
    report["sentinel_presence"] = {
        "outside_aoi_-2": -2 in class_totals,
        "unobserved_-1": -1 in class_totals,
    }
    finite_wet = [v for v in wet_fracs if v == v]  # drop NaN
    report["wet_fraction_per_timestep"] = {
        "min": float(min(finite_wet)) if finite_wet else None,
        "max": float(max(finite_wet)) if finite_wet else None,
        "std": float(np.std(finite_wet)) if finite_wet else None,
        "unique_count": len({round(v, 8) for v in finite_wet}),
    }
    finite_obs = [v for v in observed_fracs if v == v]
    report["observed_frac_of_aoi"] = {
        "min": float(min(finite_obs)) if finite_obs else None,
        "median": float(np.median(finite_obs)) if finite_obs else None,
        "mean": float(np.mean(finite_obs)) if finite_obs else None,
        "n_zero_coverage_months": sum(1 for v in observed_fracs if v == 0.0),
    }
    report["has_per_pixel_confidence_or_method_flag"] = False
    report["seasonal_mnar"] = {
        "per_calendar_month_wet_frac": per_month_wet_frac,
        "naive_pooled_wet_frac": naive_pooled_wet_frac,
        "season_stratified_wet_frac": stratified_wet_frac,
        "stratified_minus_naive": stratified_wet_frac - naive_pooled_wet_frac,
    }
    report["suitable_uses"] = [
        "v1.2_P_native_denominator_sensitivity_evidence",
        "reliability_diagnostic_observed_frac_of_aoi_regression_fixture",
        "seasonal_mnar_stratified_estimator_regression_fixture",
    ]
    report["unsuitable_uses"] = [
        "P_provenance_policy_no_method_flag_band",
        "dual_composite_dry_down_single_product_only",
    ]
    return report


def inspect_drainage_geopackage(path: Path) -> dict[str, Any]:
    import geopandas as gpd

    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "real_drainage_centreline_gpkg",
    }
    if not path.exists():
        return report

    report["checksum_sha256"] = _sha256(path)
    gdf = gpd.read_file(path)
    report["feature_count"] = int(len(gdf))
    report["crs"] = str(gdf.crs)
    report["geometry_types"] = sorted(gdf.geometry.geom_type.unique().tolist())
    report["columns"] = list(gdf.columns)
    topology_cols = ["From_Node", "To_Node", "NextDownID"]
    report["topology_null_counts"] = {
        col: int(gdf[col].isna().sum()) for col in topology_cols if col in gdf
    }
    report["total_length_m"] = float(gdf["GeodesLen"].sum()) if "GeodesLen" in gdf else None
    report["hierarchy_counts"] = (
        gdf["Hierarchy"].value_counts().to_dict() if "Hierarchy" in gdf else None
    )
    report["has_drainage_centreline"] = True
    report["suitable_uses"] = [
        "L_ref_channel_reference",
        "zone_1_geomorphic_channel",
        "inter_pool_gap_ordered_traversal",
    ]
    return report


def inspect_shapefile(path: Path) -> dict[str, Any]:
    import geopandas as gpd

    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "legacy_corridor_polygons",
    }
    if not path.exists():
        return report

    report["checksum_sha256"] = _sha256(path)
    gdf = gpd.read_file(path)
    report["feature_count"] = int(len(gdf))
    report["crs"] = str(gdf.crs)
    report["columns"] = list(gdf.columns)
    report["geometry_types"] = sorted(gdf.geometry.geom_type.unique().tolist())
    report["has_drainage_centreline"] = False
    report["suitable_uses"] = ["legacy_section_aoi_smoke"]
    report["unsuitable_uses"] = [
        "L_ref_channel_reference",
        "zone_1_geomorphic_channel",
        "inter_pool_gap_fixed_graph",
    ]
    return report


def inspect_metrics_csv(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "legacy_wide_metrics_csv",
    }
    if not path.exists():
        return report

    report["checksum_sha256"] = _sha256(path)
    df = pd.read_csv(path)
    report["row_count"] = int(len(df))
    report["section_count"] = int(df["section"].nunique()) if "section" in df else None
    report["date_count"] = int(df["date"].nunique()) if "date" in df else None
    report["date_range"] = {
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
    }
    dropped = ["PF", "PLF", "AWMPA", "AWMPL", "AWMPW", "PFL"]
    report["dropped_metrics_present"] = {
        name: name in df.columns for name in dropped
    }
    if "pp_mean_%" in df.columns and "section" in df.columns:
        report["pp_mean_static_per_section"] = bool(
            (df.groupby("section")["pp_mean_%"].nunique() == 1).all()
        )
    else:
        report["pp_mean_static_per_section"] = None

    report["naive_denominator_evidence"] = (
        "pp_mean_% constant per section across all dates implies "
        "time-invariant persistence from total-series mean, not valid_obs denominator"
    )
    report["suitable_as_v12_correctness_oracle"] = False
    report["suitable_uses"] = ["historical_kernel_smoke_with_explicit_exclusions"]
    return report
