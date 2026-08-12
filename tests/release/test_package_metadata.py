from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest


def test_pyproject_names_only_current_package() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "hydrofragments"' in text
    assert 'include = ["hydrofragments*"]' in text


def test_pyproject_version_is_first_release() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in text


def test_pyproject_has_repository_urls_and_license_file() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[project.urls]" in text
    assert "Repository" in text
    assert 'license = { file = "LICENSE" }' in text


def test_installed_distribution_metadata_matches_release() -> None:
    metadata = importlib.metadata.metadata("hydrofragments")
    assert metadata["Name"] == "hydrofragments"
    assert metadata["Version"] == "0.1.0"
    summary = metadata.get("Summary", "")
    assert "river" in summary.lower() or "surface water" in summary.lower()


def test_cpu_only_import_has_no_cupy_dependency() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("CuPy not installed; CPU-only path is the default here")
    requires = importlib.metadata.requires("hydrofragments") or []
    assert not any("cupy" in requirement.lower() for requirement in requires)
