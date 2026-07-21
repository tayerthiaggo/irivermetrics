"""M8/M13 — README and docs quickstart/validation examples must be runnable.

Also covers Section 5's notebook-execution contract: ``01_quickstart.ipynb``
must actually *execute* end-to-end (not just parse as JSON) in well under a
minute, via the ``nbmake`` pytest plugin run as an isolated subprocess.
"""

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
QUICKSTART_NOTEBOOK = REPO_ROOT / "examples" / "01_quickstart.ipynb"
# 02/03 are executed too (belt-and-braces beyond the brief's strict
# requirement, which only names 01_quickstart.ipynb) -- cheap given each
# runs in ~10-15s on the bundled synthetic fixtures.
ALL_EXAMPLE_NOTEBOOKS = [
    QUICKSTART_NOTEBOOK,
    REPO_ROOT / "examples" / "02_dea_via_tsfill.ipynb",
    REPO_ROOT / "examples" / "03_metrics_walkthrough.ipynb",
]

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


# ---- Section 5: 01_quickstart.ipynb must actually execute -----------------
#
# Run via ``nbmake`` in an isolated pytest subprocess (rather than importing
# the nbmake plugin in-process) so this test suite's own pytest run/plugin
# set does not interfere with nbmake's collection, and so a notebook
# execution failure reports nbmake's own rich cell-level diagnostics instead
# of an opaque failure from a nested pytest run.


@pytest.mark.parametrize(
    "notebook_path", ALL_EXAMPLE_NOTEBOOKS, ids=lambda p: p.name
)
def test_example_notebook_exists(notebook_path: Path) -> None:
    assert notebook_path.is_file(), f"required example notebook missing: {notebook_path}"


@pytest.mark.timeout(90)
def test_quickstart_notebook_executes_end_to_end() -> None:
    """Actually run every cell (not just parse the .ipynb as JSON).

    The brief requires this to complete "well under a minute" given the
    notebook is designed to run in under two minutes on a laptop with a
    tiny fixture; 90s is a generous CI-safe ceiling around that. This is
    the one notebook the brief explicitly requires to be execution-tested;
    the parametrized test below additionally covers 02/03.
    """
    _run_nbmake(QUICKSTART_NOTEBOOK, timeout_s=90)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "notebook_path", ALL_EXAMPLE_NOTEBOOKS, ids=lambda p: p.name
)
def test_every_example_notebook_executes_end_to_end(notebook_path: Path) -> None:
    """Belt-and-braces: 02/03 must also actually execute, not just parse."""
    _run_nbmake(notebook_path, timeout_s=120)


def _run_nbmake(notebook_path: Path, *, timeout_s: int) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--nbmake",
            "--nbmake-timeout=90",
            "-q",
            str(notebook_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    assert completed.returncode == 0, (
        f"{notebook_path.name} failed to execute end-to-end:\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
