"""M8 — README and docs quickstart examples must be runnable."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


README = Path("README.md").read_text(encoding="utf-8")


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
