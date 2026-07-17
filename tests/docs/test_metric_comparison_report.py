"""Completeness and offline contract for the metric comparison report."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "metric_comparison_report.html"

REQUIRED_IDS = {
    "section_area_km2",
    "section_length_km",
    "wet_area_km2",
    "wet_length_km",
    "wet_perimeter_km",
    "npools",
    "awmsi_legacy",
    "awre_legacy",
    "awmpa",
    "awmpl",
    "awmpw",
    "apsec_legacy",
    "lpsec_legacy",
    "pf",
    "plf",
    "pp_mean",
    "ra_area",
    "pixel_persistence",
    "nni",
    "pcf",
    "centrality",
    "occurrence",
    "refuge_area",
    "apsec",
    "number_of_pools",
    "lpi",
    "awre",
    "awmsi",
    "recurrence",
    "hydroperiod",
    "extent_contraction",
    "reconnection_timing",
    "refuge_spatial_stability",
    "lpsec",
    "inter_pool_gap",
    "mesh",
    "pool_width",
    "realised_connectivity",
    "tcf",
}


def _report() -> str:
    """Read the report as UTF-8 from the repository root."""

    return REPORT.read_text(encoding="utf-8")


def _skip_js_literal_or_comment(text: str, index: int) -> int | None:
    """Return the index after a JS string/comment starting at ``index``."""

    if text[index] in {'"', "'", "`"}:
        quote = text[index]
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
            elif text[index] == quote:
                return index + 1
            else:
                index += 1
        return len(text)
    if text.startswith("//", index):
        newline = text.find("\n", index + 2)
        return len(text) if newline == -1 else newline + 1
    if text.startswith("/*", index):
        end = text.find("*/", index + 2)
        return len(text) if end == -1 else end + 2
    return None


def _matching_delimiter(text: str, start: int) -> int:
    """Find a matching bracket while ignoring strings, templates, and comments."""

    delimiters = {"{": "}", "[": "]", "(": ")"}
    closing = set(delimiters.values())
    stack = [text[start]]
    index = start + 1
    while index < len(text) and stack:
        skipped = _skip_js_literal_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        char = text[index]
        if char in delimiters:
            stack.append(char)
        elif char in closing:
            if not stack or delimiters[stack[-1]] != char:
                raise ValueError(f"unbalanced delimiter at offset {index}")
            stack.pop()
            if not stack:
                return index + 1
        index += 1
    raise ValueError("unterminated metricRecords delimiter")


def _metric_record_spans(text: str) -> list[str]:
    """Extract each object in the ``metricRecords`` array.

    A balanced-delimiter scan keeps nested objects/arrays and braces in quoted
    values from accidentally merging adjacent metric records.
    """

    declaration = text.find("const metricRecords")
    if declaration == -1:
        return []
    array_start = text.find("[", declaration)
    if array_start == -1:
        return []
    array_end = _matching_delimiter(text, array_start)

    records: list[str] = []
    index = array_start + 1
    while index < array_end - 1:
        skipped = _skip_js_literal_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == "{":
            record_end = _matching_delimiter(text, index)
            records.append(text[index:record_end])
            index = record_end
        else:
            index += 1
    return records


def test_report_is_self_contained_and_has_metric_records() -> None:
    text = _report()

    assert "const metricRecords" in text
    assert "<script src=" not in text.lower()
    assert '<link rel="stylesheet"' not in text.lower()
    assert "fetch(" not in text


def test_all_metric_ids_and_required_fields_are_present() -> None:
    text = _report()
    records = _metric_record_spans(text)

    for metric_id in REQUIRED_IDS:
        matching_records = [
            record
            for record in records
            if re.search(rf'\bid\s*:\s*"{re.escape(metric_id)}"', record)
        ]
        assert matching_records, f'missing metric record for id "{metric_id}"'
        record = matching_records[0]
        for field in ("equation", "citation", "source"):
            assert re.search(rf"\b{field}\s*:", record), (
                f'metric "{metric_id}" is missing {field}: field'
            )
