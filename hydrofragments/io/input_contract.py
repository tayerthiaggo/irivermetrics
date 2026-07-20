"""Inspect-then-act input contract for :func:`hydrofragments.api.open_water_cube`.

Two kinds of outcomes only, per the locked spec guard (docs/HydroFragments_v1.2_spec.md
§8 guard 8, §14): a mismatch is either safe to auto-fix (structural only --
renaming an unambiguous variable, coercing an already-binary dtype to bool,
reordering dims) or it must raise a specific, actionable
:class:`InputContractError` naming the exact field and expected-vs-actual
value. This module never silently reprojects or resamples to paper over a
grid or CRS mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import xarray as xr

from hydrofragments.io.alignment import validate_alignment

EXPECTED_DIM_ORDER: tuple[str, ...] = ("time", "y", "x")


class InputContractError(ValueError):
    """Raised when an input mismatch cannot be safely auto-fixed."""


@dataclass(frozen=True)
class ResolvedInput:
    """Result of running the input contract over a raw source.

    ``array`` is the normalized ``DataArray`` ready for adapter dispatch.
    ``fixes`` is the ordered, human-readable log of every safe auto-fix
    applied (empty tuple if none were needed) -- callers thread this into
    ``WaterCube.provenance`` for auditability.
    """

    array: xr.DataArray
    fixes: tuple[str, ...]


def check_crs_defined(water: xr.DataArray) -> None:
    """Raise if ``water`` carries an explicit, geographic (degrees) CRS.

    Only applies when ``water`` has a ``.rio`` accessor with a CRS actually
    set -- skips cleanly for arrays without a CRS concept at all (e.g. a
    pure in-memory test fixture with no spatial coordinates/CRS metadata)
    or with no CRS set, matching the existing geographic-CRS guard
    elsewhere in this codebase
    (:func:`hydrofragments.spatial.crs.normalize_spatial_inputs`). An
    *undefined* CRS is not itself refused here -- many valid in-memory/
    generic_binary inputs carry no georeferencing at all -- but a CRS that
    *is* set and in degrees is always refused, never silently reprojected
    (spec §8 guard 8).
    """
    if not hasattr(water, "rio"):
        return
    crs = water.rio.crs
    if crs is None:
        return
    if crs.is_geographic:
        raise InputContractError(
            f"water.crs is geographic (degrees): '{crs.to_string()}'. "
            "HydroFragments requires a projected CRS in metres and never "
            "silently reprojects (spec §8 guard 8) -- reproject to an "
            "equal-area/projected CRS (e.g. EPSG:3577) before calling "
            "open_water_cube"
        )


def check_grid_alignment(water: xr.DataArray, valid_obs: xr.DataArray) -> None:
    """Raise if ``water`` and ``valid_obs`` disagree on shape/transform/CRS.

    Wraps :func:`hydrofragments.io.alignment.validate_alignment`'s generic
    ``ValueError`` with a specific, actionable :class:`InputContractError`
    naming the mismatched field -- never silently resamples/reprojects to
    reconcile the two layers (spec §14).
    """
    if water.sizes != valid_obs.sizes:
        raise InputContractError(
            "grid mismatch between water and valid_obs: shape "
            f"{dict(water.sizes)} != {dict(valid_obs.sizes)}"
        )
    if water.dims != valid_obs.dims:
        raise InputContractError(
            "grid mismatch between water and valid_obs: dim order "
            f"{water.dims} != {valid_obs.dims}"
        )
    try:
        validate_alignment(water, valid_obs)
    except ValueError as error:
        message = str(error)
        if message == "CRS mismatch":
            water_crs = water.rio.crs if hasattr(water, "rio") else None
            valid_crs = valid_obs.rio.crs if hasattr(valid_obs, "rio") else None
            raise InputContractError(
                "CRS mismatch between water and valid_obs: "
                f"{water_crs} != {valid_crs}"
            ) from error
        # "Grid mismatch" -- validate_alignment does not name the specific
        # coordinate, so name the first disagreeing shared dim/coord here.
        for dim in water.dims:
            if dim in valid_obs.dims and not water.coords[dim].equals(
                valid_obs.coords[dim]
            ):
                raise InputContractError(
                    f"grid mismatch between water and valid_obs: coordinate "
                    f"'{dim}' does not align (differing transform/values)"
                ) from error
        raise InputContractError(f"grid mismatch between water and valid_obs: {message}") from error


def _pick_single_variable(source: xr.Dataset) -> tuple[xr.DataArray, str, str | None]:
    """Return ``(array, chosen_name, rename_fix)`` for a Dataset source."""
    if "water" in source:
        return source["water"], "water", None
    data_vars = list(source.data_vars)
    if len(data_vars) == 1:
        name = data_vars[0]
        return (
            source[name],
            name,
            f"renamed_variable:{name}->water",
        )
    raise InputContractError(
        "ambiguous Dataset with multiple variables "
        f"({sorted(data_vars)}) and no unambiguous water-like candidate "
        "(expected 'water' or a single data variable); pass variable_map "
        "to disambiguate or select the variable explicitly"
    )


def normalize_structure(
    source: xr.DataArray | xr.Dataset,
    *,
    variable_map: Mapping[str, str] | None = None,
) -> tuple[xr.DataArray, tuple[str, ...]]:
    """Auto-fix safe, structural-only mismatches; log every fix applied.

    Safe fixes (never silent):
      - Renaming a single unambiguous data variable to ``water`` (or
        applying an explicit ``variable_map`` rename).
      - Coercing an already-binary ``{0, 1}`` int/float array to ``bool``.
      - Reordering dims to ``EXPECTED_DIM_ORDER`` (``time, y, x``) when all
        three are present.

    Never resamples or reprojects -- grid/CRS mismatches are the caller's
    job to raise via :func:`check_grid_alignment`/:func:`check_crs_defined`.

    Returns ``(array, fixes)`` where ``fixes`` is an ordered tuple of
    human-readable descriptions of what was normalized (empty if nothing
    needed fixing).
    """
    fixes: list[str] = []

    if variable_map:
        if isinstance(source, xr.Dataset):
            rename = {k: v for k, v in variable_map.items() if k in source}
            if rename:
                source = source.rename(rename)
                for old, new in rename.items():
                    fixes.append(f"renamed_variable:{old}->{new}")

    if isinstance(source, xr.Dataset):
        array, _chosen_name, rename_fix = _pick_single_variable(source)
        if rename_fix is not None:
            fixes.append(rename_fix)
    else:
        array = source

    if array.dtype != bool:
        is_binary = bool(((array == 0) | (array == 1)).all())
        if is_binary:
            original_dtype = array.dtype
            array = array.astype(bool)
            fixes.append(f"coerced_dtype:{original_dtype}->bool")

    present_expected_dims = [dim for dim in EXPECTED_DIM_ORDER if dim in array.dims]
    if len(present_expected_dims) >= 2 and tuple(
        dim for dim in array.dims if dim in present_expected_dims
    ) != tuple(present_expected_dims):
        target_order = tuple(present_expected_dims) + tuple(
            dim for dim in array.dims if dim not in present_expected_dims
        )
        array = array.transpose(*target_order)
        fixes.append(f"reordered_dims:->{target_order}")

    return array, tuple(fixes)
