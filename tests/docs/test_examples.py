"""M8/M13 — README and docs quickstart/validation examples must be runnable."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
README = Path("README.md").read_text(encoding="utf-8")
SPATIAL_EXPORTS_EXAMPLE = REPO_ROOT / "examples" / "spatial_exports.py"

# M13 audience-facing docs are optional-code: they may contain zero python
# blocks (prose/tables only), but any block present must execute cleanly.
M13_DOCS = [
    REPO_ROOT / "docs" / "validation_status.md",
    REPO_ROOT / "docs" / "for-managers.md",
]


def _extract_python_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("```python"):
            in_block = True
            current = []
            continue
        if in_block and line.strip() == "```":
            blocks.append("\n".join(current))
            in_block = False
            continue
        if in_block:
            current.append(line)
    return blocks


def test_readme_uses_hydrofragments_import() -> None:
    assert "from hydrofragments import" in README
    assert "waterdetect_batch" not in README


def test_readme_release_status_present() -> None:
    assert "HydroFragments" in README
    assert re.search(r"\b0\.1\.0\b", README)


def test_readme_documents_actual_output_paths() -> None:
    assert "metrics/" in README
    assert "run_manifest.json" in README
    assert "no single-file" in README or "no single `metrics.parquet`" in README
    assert "`manifest.json`" in README or "Not `manifest.json`" in README


@pytest.mark.timeout(120)
def test_spatial_exports_example_runs_offline(tmp_path: Path) -> None:
    assert SPATIAL_EXPORTS_EXAMPLE.is_file()
    output_dir = tmp_path / "o"
    completed = subprocess.run(
        [
            sys.executable,
            str(SPATIAL_EXPORTS_EXAMPLE),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "spatial_exports.py failed:\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "rasters" / "occurrence.tif").is_file()


@pytest.mark.parametrize("block", _extract_python_blocks(README))
def test_readme_python_blocks_execute(block: str) -> None:
    namespace = {
        "np": np,
        "pd": pd,
        "xr": xr,
        "__name__": "__main__",
    }
    exec(block, namespace)  # noqa: S102


def _m13_doc_python_blocks() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for doc_path in M13_DOCS:
        if not doc_path.is_file():
            continue
        for block in _extract_python_blocks(doc_path.read_text(encoding="utf-8")):
            pairs.append((doc_path.name, block))
    return pairs


@pytest.mark.parametrize(
    "doc_name,block",
    _m13_doc_python_blocks(),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_m13_doc_python_blocks_execute(doc_name: str, block: str) -> None:
    namespace = {
        "np": np,
        "pd": pd,
        "xr": xr,
        "__name__": "__main__",
    }
    exec(block, namespace)  # noqa: S102


@pytest.mark.parametrize("doc_path", M13_DOCS, ids=lambda p: p.name)
def test_m13_doc_exists_and_is_nonempty(doc_path: Path) -> None:
    assert doc_path.is_file(), f"required M13 doc missing: {doc_path}"
    text = doc_path.read_text(encoding="utf-8").strip()
    assert len(text) > 0
