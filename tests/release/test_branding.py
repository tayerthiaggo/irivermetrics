from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_PACKAGE = "eco" + "fragments"
PREDECESSOR = "i" + "river" + "metrics"
TEXT_SUFFIXES = {
    ".cpg", ".css", ".csv", ".html", ".ipynb", ".json", ".md",
    ".prj", ".py", ".toml", ".txt", ".xml", ".yaml", ".yml",
}


def _tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPO_ROOT / value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    ]


def test_tracked_paths_use_only_current_brand() -> None:
    lowered = [path.relative_to(REPO_ROOT).as_posix().lower() for path in _tracked_paths()]
    assert not any(RETIRED_PACKAGE in path for path in lowered)
    assert not any(PREDECESSOR in path for path in lowered)


def test_tracked_text_uses_only_readme_lineage_mention() -> None:
    matches: dict[str, list[str]] = {RETIRED_PACKAGE: [], PREDECESSOR: []}
    for path in _tracked_paths():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        relative = path.relative_to(REPO_ROOT).as_posix()
        for term in matches:
            matches[term].extend([relative] * text.count(term))
    assert matches[RETIRED_PACKAGE] == []
    assert matches[PREDECESSOR] == ["README.md"]
