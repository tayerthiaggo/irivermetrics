"""Cross-run scientific compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hydrofragments.output.manifest import read_run_manifest


_MISSING = object()


class ComparisonGuardError(ValueError):
    """Raised when two results are not scientifically comparable."""


@dataclass(frozen=True)
class ComparisonMismatch:
    field: str
    left: object
    right: object
    override_reason: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "field": self.field,
            "left": "<missing>" if self.left is _MISSING else self.left,
            "right": "<missing>" if self.right is _MISSING else self.right,
            "override_reason": self.override_reason,
        }


@dataclass(frozen=True)
class ComparisonApproval:
    left_run_id: str
    right_run_id: str
    mismatches: tuple[ComparisonMismatch, ...]
    overrides: dict[str, str]
    approved: bool = True

    def to_mapping(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "left_run_id": self.left_run_id,
            "right_run_id": self.right_run_id,
            "overrides": dict(self.overrides),
            "mismatches": [item.to_mapping() for item in self.mismatches],
        }


@dataclass(frozen=True)
class _ComparisonField:
    paths: tuple[tuple[str, ...], ...]
    required: bool = False


COMPARISON_FIELDS: dict[str, _ComparisonField] = {
    "aoi_id": _ComparisonField(
        (("comparison", "aoi_id"), ("aoi_id",)), required=True
    ),
    "source": _ComparisonField(
        (("comparison", "source"), ("source",)), required=True
    ),
    "resolution_m": _ComparisonField(
        (("comparison", "resolution_m"), ("resolution_m",)), required=True
    ),
    "crs": _ComparisonField(
        (
            ("comparison", "crs"),
            ("crs",),
            ("resolved_config", "spatial", "target_crs"),
        ),
        required=True,
    ),
    "validity_policy": _ComparisonField(
        (
            ("comparison", "validity_policy"),
            ("resolved_config", "validity", "policy"),
        ),
        required=True,
    ),
    "monthly_composite": _ComparisonField(
        (
            ("comparison", "monthly_composite"),
            ("resolved_config", "temporal", "monthly_composite"),
        ),
        required=True,
    ),
    "water_threshold": _ComparisonField(
        (("resolved_config", "input", "water_threshold"),)
    ),
    "threshold_method": _ComparisonField(
        (("resolved_config", "input", "threshold_method"),)
    ),
    "min_valid_obs": _ComparisonField(
        (("resolved_config", "validity", "min_valid_obs"),)
    ),
    "min_valid_fraction_month": _ComparisonField(
        (("resolved_config", "validity", "min_valid_fraction_month"),)
    ),
    "area_method": _ComparisonField(
        (("resolved_config", "spatial", "area_method"),)
    ),
    "min_patch_pixels": _ComparisonField(
        (("resolved_config", "patches", "min_patch_pixels"),)
    ),
    "connectivity_rule": _ComparisonField(
        (("resolved_config", "patches", "connectivity_rule"),)
    ),
}


def _nested_value(source: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _field_value(
    manifest: Mapping[str, object], field: _ComparisonField
) -> object:
    for path in field.paths:
        value = _nested_value(manifest, path)
        if value is not _MISSING:
            return value
    return _MISSING


def _coerce_manifest(
    value: Mapping[str, object] | str | Path,
) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return read_run_manifest(value)


def _display(value: object) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def guard_comparison(
    left: Mapping[str, object] | str | Path,
    right: Mapping[str, object] | str | Path,
    *,
    overrides: Mapping[str, str] | None = None,
) -> ComparisonApproval:
    """Approve compatible manifests or refuse mismatches by default.

    Overrides are field-specific and require a non-empty reason so any unsafe
    comparison remains explicit in downstream provenance.
    """

    left_manifest = _coerce_manifest(left)
    right_manifest = _coerce_manifest(right)
    override_reasons = dict(overrides or {})
    unknown = sorted(set(override_reasons) - set(COMPARISON_FIELDS))
    if unknown:
        raise ComparisonGuardError(f"unknown comparison override: {unknown[0]}")
    for field, reason in override_reasons.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ComparisonGuardError(
                f"comparison override for {field} requires a non-empty reason"
            )

    mismatches: list[ComparisonMismatch] = []
    for name, contract in COMPARISON_FIELDS.items():
        left_value = _field_value(left_manifest, contract)
        right_value = _field_value(right_manifest, contract)
        both_missing = left_value is _MISSING and right_value is _MISSING
        if both_missing and not contract.required:
            continue
        if left_value != right_value or (both_missing and contract.required):
            mismatches.append(
                ComparisonMismatch(
                    name,
                    left_value,
                    right_value,
                    override_reasons.get(name),
                )
            )

    unapproved = [item for item in mismatches if item.override_reason is None]
    if unapproved:
        details = "; ".join(
            f"{item.field} mismatch ({_display(item.left)} vs {_display(item.right)})"
            for item in unapproved
        )
        raise ComparisonGuardError(f"comparison refused: {details}")

    mismatch_fields = {item.field for item in mismatches}
    unused = sorted(set(override_reasons) - mismatch_fields)
    if unused:
        raise ComparisonGuardError(
            f"comparison override has no matching mismatch: {unused[0]}"
        )

    return ComparisonApproval(
        left_run_id=str(left_manifest.get("run_id", "")),
        right_run_id=str(right_manifest.get("run_id", "")),
        mismatches=tuple(mismatches),
        overrides=override_reasons,
    )


check_comparison_compatibility = guard_comparison
compare_manifests = guard_comparison


__all__ = [
    "COMPARISON_FIELDS",
    "ComparisonApproval",
    "ComparisonGuardError",
    "ComparisonMismatch",
    "check_comparison_compatibility",
    "compare_manifests",
    "guard_comparison",
]
