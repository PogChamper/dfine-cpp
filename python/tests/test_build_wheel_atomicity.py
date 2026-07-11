from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VERSION = tomllib.loads((REPO / "python/pyproject.toml").read_text())["project"]["version"]
WHEEL = f"dfine-{VERSION}-py3-none-linux_{platform.machine()}.whl"


@pytest.fixture
def wheel_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    python_dir = repo / "python"
    (python_dir / "dfine").mkdir(parents=True)
    (python_dir / "dist").mkdir()
    (repo / "build").mkdir()
    (repo / "trt-files/scripts").mkdir(parents=True)

    shutil.copy2(REPO / "python/build_wheel.sh", python_dir / "build_wheel.sh")
    (python_dir / "pyproject.toml").write_text(f'version = "{VERSION}"\n')
    (repo / "build/libdfine.so").write_bytes(b"native-library")
    (repo / "trt-files/scripts/build_engine.py").write_text("# builder\n")
    (repo / "LICENSE").write_text("license\n")
    (repo / "NOTICE").write_text("notice\n")

    tools = tmp_path / "tools"
    tools.mkdir()
    fake_python = tools / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-c" ]]; then
    exit 0
fi
if [[ "${FAIL_WHEEL_BUILD:-0}" == 1 ]]; then
    exit 23
fi
while [[ "$1" != "--outdir" ]]; do
    shift
done
outdir=$2
mkdir -p "$outdir"
printf 'new-wheel' > "$outdir/$EXPECTED_WHEEL"
"""
    )
    fake_cmake = tools / "cmake"
    fake_cmake.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_readelf = tools / "readelf"
    fake_readelf.write_text("#!/usr/bin/env bash\nexit 0\n")
    for executable in (fake_python, fake_cmake, fake_readelf):
        executable.chmod(0o755)

    return repo, tools


def run_build(wheel_repo: tuple[Path, Path], *, fail: bool) -> subprocess.CompletedProcess:
    repo, tools = wheel_repo
    env = os.environ.copy()
    env.update(
        {
            "CMAKE": str(tools / "cmake"),
            "EXPECTED_WHEEL": WHEEL,
            "FAIL_WHEEL_BUILD": "1" if fail else "0",
            "PATH": f"{tools}:{env['PATH']}",
            "PYTHON": str(tools / "python"),
            "SKIP_BUILD": "1",
        }
    )
    return subprocess.run(
        ["bash", str(repo / "python/build_wheel.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_failed_wheel_build_preserves_previous_artifact(wheel_repo):
    repo, _ = wheel_repo
    target = repo / "python/dist" / WHEEL
    target.write_bytes(b"previous-wheel")

    result = run_build(wheel_repo, fail=True)

    assert result.returncode == 23
    assert target.read_bytes() == b"previous-wheel"
    assert not list(target.parent.glob(".dfine-wheel.*"))


def test_successful_wheel_build_replaces_previous_artifact(wheel_repo):
    repo, _ = wheel_repo
    target = repo / "python/dist" / WHEEL
    target.write_bytes(b"previous-wheel")

    result = run_build(wheel_repo, fail=False)

    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == b"new-wheel"
    assert not list(target.parent.glob(".dfine-wheel.*"))
