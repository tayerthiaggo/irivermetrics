"""M8 — package metadata exposes hydrofragments as the public namespace."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest


def test_pyproject_names_hydrofragments() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "hydrofragments"' in text


def test_pyproject_includes_both_packages() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "hydrofragments*" in text
    assert "ecofragments*" in text


def test_pyproject_has_repository_urls() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[project.urls]" in text
    assert "Repository" in text


def test_installed_distribution_metadata_matches_namespace() -> None:
    metadata = importlib.metadata.metadata("hydrofragments")
    assert metadata["Name"] == "hydrofragments"
    assert metadata["Version"].startswith("1.2.")
    summary = metadata.get("Summary", "")
    assert "river" in summary.lower() or "surface water" in summary.lower()


def test_cpu_only_import_has_no_cupy_dependency() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("CuPy not installed; CPU-only path is the default here")
    requires = importlib.metadata.requires("hydrofragments") or []
    assert not any("cupy" in req.lower() for req in requires)
