"""M13 — every number in docs/for-managers.md must resolve to a validation row.

A manager-facing document is dangerous exactly when a reader cannot tell
whether a number is real evidence or an example placeholder. This test
parses docs/for-managers.md for numeric values presented as findings and
checks each is either (a) explicitly marked as a placeholder/illustrative
slot, or (b) traceable to a row in a validation/results/*.csv table via a
cited run_id.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FOR_MANAGERS = REPO_ROOT / "docs" / "for-managers.md"
VALIDATION_RESULTS = REPO_ROOT / "validation" / "results"

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")


def _known_run_ids() -> set[str]:
    if not VALIDATION_RESULTS.exists():
        return set()
    ids: set[str] = set()
    for csv_path in VALIDATION_RESULTS.glob("*.csv"):
        frame = pd.read_csv(csv_path)
        if "run_id" in frame.columns:
            ids.update(frame["run_id"].astype(str).unique())
    return ids


def test_for_managers_doc_exists() -> None:
    assert FOR_MANAGERS.is_file()


def test_for_managers_doc_states_negative_scope_before_any_metric() -> None:
    text = FOR_MANAGERS.read_text(encoding="utf-8")
    lower = text.lower()
    first_metric_mention = min(
        (
            lower.find(term)
            for term in ("occurrence", "refuge area", "apsec", "dry-down", "contraction")
            if lower.find(term) != -1
        ),
        default=len(lower),
    )
    negative_scope_idx = lower.find("does not measure")
    assert negative_scope_idx != -1, "must state what the tool does not measure"
    assert negative_scope_idx < first_metric_mention, (
        "negative scope ('does not measure') must appear before headline metrics"
    )


def test_for_managers_doc_never_uses_forbidden_precision_or_claims() -> None:
    text = FOR_MANAGERS.read_text(encoding="utf-8")
    lower = text.lower()
    forbidden_phrases = [
        "will be fully dry by",
        "predicts refuge risk",
        "confirms refuge risk",
        "permanently a refuge",
        "officially a refuge",
        "proves",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in lower, f"forbidden claim present: {phrase!r}"


def test_every_real_numeric_claim_in_for_managers_traces_to_a_run_id() -> None:
    """Numbers outside placeholder brackets must cite a run_id present in
    validation/results/*.csv. Numbers used only as illustrative placeholders
    must stay inside bracket-slot syntax, e.g. [VALUE]."""
    text = FOR_MANAGERS.read_text(encoding="utf-8")
    known_run_ids = _known_run_ids()

    # Strip placeholder bracket slots and markdown table separator rows first.
    stripped = PLACEHOLDER_PATTERN.sub("", text)

    # Any standalone percentage/decimal figure outside brackets is a "real" claim
    # and must appear on a line that also cites a known run_id.
    number_pattern = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?(?:%|pp|km2|km|m)\b")
    for line in stripped.splitlines():
        if "|---" in line or set(line.strip()) <= {"-", "|", " "}:
            continue
        for match in number_pattern.finditer(line):
            has_run_id_citation = any(run_id in line for run_id in known_run_ids)
            assert has_run_id_citation, (
                f"numeric claim {match.group()!r} in line {line!r} does not "
                "cite a known validation run_id"
            )
