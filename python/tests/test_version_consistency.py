from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _match(path: str, pattern: str) -> str:
    text = (REPO / path).read_text()
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"version not found in {path}"
    return match.group(1)


def test_release_versions_agree():
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    locked_version = next(
        package["version"] for package in lock["package"] if package["name"] == "dfine-cpp-tools"
    )
    versions = {
        "CMakeLists.txt": _match("CMakeLists.txt", r"^project\(dfine VERSION ([^ )]+)"),
        "pyproject.toml": _match("pyproject.toml", r'^version = "([^"]+)"'),
        "python/pyproject.toml": _match("python/pyproject.toml", r'^version = "([^"]+)"'),
        "python/dfine/__init__.py": _match("python/dfine/__init__.py", r'^__version__ = "([^"]+)"'),
        "include/dfine/version.hpp": _match("include/dfine/version.hpp", r'^\s*return "([^"]+)";'),
        "uv.lock": locked_version,
    }

    assert len(set(versions.values())) == 1, versions
