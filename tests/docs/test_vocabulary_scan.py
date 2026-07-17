"""M13 — automated vocabulary scan for prohibited manager/paper-facing claims.

Per docs/audit/manager_interpretation_audit.md and scientific_metrics_audit.md,
certain claim shapes must never appear in manager- or publication-facing
documentation because the underlying evidence does not support them:

- depth/volume inference from width or extent
- flow/discharge/recession-as-hydrograph claims for dry-down/contraction
- "permanent" refuge designations
- unsupported novelty claims (HY detection duplicating Tayer 2025/2026)
- predictive drying-date claims

Scanned surface: docs/for-managers.md and docs/validation_status.md — the two
audience-facing documents this milestone produces. Internal engineering docs
(spec, audits) are allowed to discuss these concepts analytically and are not
scanned.

Disclaiming a misreading ("width tells you nothing about depth") must stay
legal; asserting it ("width means more water") must not. Each pattern below
is checked with a small window of surrounding text for a negating word
(not/no/never/nothing/n't) before it counts as a real violation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_DOCS = [
    REPO_ROOT / "docs" / "for-managers.md",
    REPO_ROOT / "docs" / "validation_status.md",
]

_NEGATOR_PATTERN = re.compile(
    r"\b(not|no|never|nothing|n't|does not|do not)\b", re.IGNORECASE
)
_WINDOW_CHARS = 60

# Each pattern is a regex (case-insensitive); a match only fails the test if
# no negating word appears in the surrounding _WINDOW_CHARS window.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "depth_inference": r"\b(wider|width)\b[^.]{0,40}\b(more water|deeper|depth|volume)\b",
    "volume_or_flow_claim": r"\b(measures?|indicates?|shows?)\b[^.]{0,40}\b(flow|discharge|streamflow)\b",
    "recession_as_flow": r"\brecession\b",
    "permanent_refuge": r"\bpermanent(?:ly)?\b[^.]{0,20}\brefuge\b",
    "predictive_drying_date": r"\b(will be|fully)\s+dry\s+by\b|\bdry(?:ing)?\s+date\b",
    "unsupported_novelty_hy": r"\bnovel\b[^.]{0,60}\bhydrological[- ]year\b",
}


def _iter_existing_docs() -> list[Path]:
    return [p for p in SCANNED_DOCS if p.is_file()]


def _is_negated_nearby(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - _WINDOW_CHARS) : end + _WINDOW_CHARS]
    return _NEGATOR_PATTERN.search(window) is not None


@pytest.mark.parametrize("doc_path", SCANNED_DOCS, ids=lambda p: p.name)
def test_scanned_doc_exists(doc_path: Path) -> None:
    assert doc_path.is_file(), f"required audience-facing doc missing: {doc_path}"


@pytest.mark.parametrize(
    "doc_path", _iter_existing_docs() or SCANNED_DOCS, ids=lambda p: p.name
)
@pytest.mark.parametrize(
    "label,pattern", FORBIDDEN_PATTERNS.items(), ids=list(FORBIDDEN_PATTERNS)
)
def test_doc_contains_no_forbidden_claim(
    doc_path: Path, label: str, pattern: str
) -> None:
    if not doc_path.is_file():
        pytest.skip(f"{doc_path} does not exist yet")
    text = doc_path.read_text(encoding="utf-8")
    for match in re.finditer(pattern, text, re.IGNORECASE):
        if not _is_negated_nearby(text, match.start(), match.end()):
            pytest.fail(
                f"forbidden claim category {label!r} matched in "
                f"{doc_path.name} with no disclaiming negation nearby: "
                f"{match.group()!r}"
            )


def test_width_not_depth_guard_present_wherever_width_appears() -> None:
    for_managers = REPO_ROOT / "docs" / "for-managers.md"
    if not for_managers.is_file():
        pytest.skip("docs/for-managers.md does not exist yet")
    text = for_managers.read_text(encoding="utf-8")
    lower = text.lower()
    width_mentions = [m.start() for m in re.finditer(r"\bwidth\b", lower)]
    assert width_mentions, "expected at least one width mention to guard"
    for pos in width_mentions:
        window = lower[max(0, pos - 400) : pos + 400]
        assert "depth" in window, (
            "every mention of pool width must carry a width-is-not-depth "
            "guard within the surrounding paragraph"
        )
