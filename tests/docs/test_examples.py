"""M8/M13 — README and docs quickstart/validation examples must be runnable."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
README = Path("README.md").read_text(encoding="utf-8")

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
    assert "irivermetrics" not in README.lower()
    assert "waterdetect_batch" not in README


def test_readme_status_banner_present() -> None:
    assert "HydroFragments" in README
    assert re.search(r"v1\.2|release candidate|migration", README, re.I)


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
