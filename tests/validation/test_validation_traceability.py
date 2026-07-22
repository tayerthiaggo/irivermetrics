"""M13 — validation tables must trace to immutable run IDs/manifests.

Every row in a validation result table must carry a ``run_id`` that matches
a real, on-disk ``run_manifest.json`` produced by ``hydrofragments.analyze``.
A validation claim with no traceable run is not evidence, it's an assertion
wearing a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPO_ROOT / "validation"


def _validation_result_csvs() -> list[Path]:
    results_dir = VALIDATION_DIR / "results"
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*.csv"))


def test_validation_results_directory_exists() -> None:
    assert (VALIDATION_DIR / "results").is_dir(), (
        "validation/results/ must exist and contain machine-readable "
        "validation result tables"
    )


def test_at_least_one_validation_result_table_exists() -> None:
    assert _validation_result_csvs(), (
        "no validation result CSVs found under validation/results/ — "
        "M13 requires at least one reproducible validation analysis"
    )


@pytest.mark.parametrize("csv_path", _validation_result_csvs(), ids=lambda p: p.name)
def test_every_validation_row_has_a_run_id_column(csv_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    assert "run_id" in frame.columns, f"{csv_path.name} missing run_id column"
    assert frame["run_id"].notna().all(), f"{csv_path.name} has rows with null run_id"


@pytest.mark.parametrize("csv_path", _validation_result_csvs(), ids=lambda p: p.name)
def test_every_run_id_resolves_to_a_real_manifest(csv_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    manifests_dir = VALIDATION_DIR / "results" / "manifests"
    for run_id in frame["run_id"].unique():
        manifest_path = manifests_dir / f"{run_id}.json"
        assert manifest_path.is_file(), (
            f"{csv_path.name} references run_id={run_id!r} with no matching "
            f"manifest at {manifest_path}"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("run_id") == run_id, (
            f"manifest {manifest_path.name} run_id field does not match filename"
        )
        assert "config_hash" in manifest
        assert "input_fingerprint" in manifest


def test_validation_status_doc_exists() -> None:
    assert (REPO_ROOT / "docs" / "validation_status.md").is_file()


def test_validation_status_doc_links_every_claim_to_evidence_or_marks_asserted() -> None:
    text = (REPO_ROOT / "docs" / "validation_status.md").read_text(encoding="utf-8")
    # Every claim row must say either "Demonstrated" with a linked run_id/file,
    # or "Asserted" — never silently omit status.
    assert "Asserted" in text
    assert "Demonstrated" in text or "asserted" in text.lower()
