from __future__ import annotations

import importlib.metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def _load_pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


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


def test_mandatory_dependencies_exclude_cupy() -> None:
    project = _load_pyproject()
    mandatory = [Requirement(req) for req in project["dependencies"]]
    assert not any(req.name.lower().startswith("cupy") for req in mandatory)


def test_cuda_extra_declares_cupy_optional_dependency() -> None:
    project = _load_pyproject()
    cuda = [Requirement(req) for req in project["optional-dependencies"]["cuda"]]
    assert any(req.name.lower().startswith("cupy") for req in cuda)


def test_requires_python_matches_pinned_native_stack() -> None:
    project = _load_pyproject()
    assert project["requires-python"] == ">=3.10,<3.14"

    hydroseason_requires = importlib.metadata.metadata("hydroseason").get(
        "Requires-Python", ""
    )
    assert hydroseason_requires.startswith(">=3.10")

    hydrofragments_spec = SpecifierSet(project["requires-python"])
    hydroseason_spec = SpecifierSet(hydroseason_requires)
    for minor in (10, 11, 12, 13):
        version = f"3.{minor}.0"
        if hydrofragments_spec.contains(version):
            assert hydroseason_spec.contains(version)
    assert not hydrofragments_spec.contains("3.14.0")
