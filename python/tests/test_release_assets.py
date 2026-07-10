"""release_assets ``assemble`` — the validation gate between the export tree and
the upload staging dir. Exercised with tiny fake files; the gh-dependent
``verify`` path runs against the live release per docs/RELEASE_CHECKLIST.md."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "release_assets", REPO / "trt-files/scripts/release_assets.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

WHEEL = "dfine-0.3.1-py3-none-linux_x86_64.whl"


@pytest.fixture
def release(tmp_path):
    """A grammar-complete input dir + wheel: 10 graph/sidecar pairs, the right
    precision per recipe suffix, opset 19 everywhere."""
    inp = tmp_path / "input"
    inp.mkdir()
    for size in ra.SIZES:
        for recipe, precision in ra.RECIPES.items():
            (inp / f"dfine_{size}_{recipe}.onnx").write_bytes(f"graph {size} {recipe}".encode())
            (inp / f"dfine_{size}_{recipe}.json").write_text(
                json.dumps({"precision": precision, "opset": 19}))
    (tmp_path / WHEEL).write_bytes(b"wheel bytes")
    return Namespace(input=str(inp), wheel=str(tmp_path / WHEEL), out=str(tmp_path / "out"))


def test_happy_path_stages_all_and_sums_includes_wheel(release):
    ra.assemble(release)
    out = Path(release.out)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 21  # 20 model files + the wheel (the v0.3.1 audit gap)
    names = [ln.split("  ", 1)[1] for ln in lines]
    assert names == sorted(names)
    assert WHEEL in names
    for ln in lines:
        digest, name = ln.split("  ", 1)
        assert digest == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_missing_sidecar_refused_before_staging(release):
    (Path(release.input) / "dfine_m_slim.json").unlink()
    with pytest.raises(SystemExit, match="dfine_m_slim.json"):
        ra.assemble(release)
    assert not Path(release.out).exists()  # validation runs before any copy


def test_missing_graph_refused(release):
    (Path(release.input) / "dfine_x_op19.onnx").unlink()
    with pytest.raises(SystemExit, match="dfine_x_op19.onnx"):
        ra.assemble(release)


def test_precision_suffix_mismatch_refused(release):
    sc = Path(release.input) / "dfine_s_slim.json"
    sc.write_text(json.dumps({"precision": "fp32", "opset": 19}))
    with pytest.raises(SystemExit, match="precision"):
        ra.assemble(release)


def test_wrong_opset_refused(release):
    sc = Path(release.input) / "dfine_n_op19.json"
    sc.write_text(json.dumps({"precision": "fp32", "opset": 16}))
    with pytest.raises(SystemExit, match="opset"):
        ra.assemble(release)


def test_extra_dfine_file_refused(release):
    (Path(release.input) / "dfine_m_spim.onnx").write_bytes(b"typo")
    with pytest.raises(SystemExit, match="dfine_m_spim.onnx"):
        ra.assemble(release)


def test_unparseable_sidecar_refused(release):
    (Path(release.input) / "dfine_l_slim.json").write_text("{not json")
    with pytest.raises(SystemExit, match="parsed"):
        ra.assemble(release)
