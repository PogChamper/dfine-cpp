"""_publish_pair — the one publish path every artifact producer uses (exporter
inlines the same shape): both files staged before either swap, two adjacent
atomic renames, and a stale output sidecar is REMOVED when the source carries
no contract (otherwise a new graph would pair with the previous metadata
forever). Each script embeds its own copy (the scripts are deliberately
self-contained), so every importable copy is exercised."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCRIPTS = {
    "convert_fp16_surgical": ["onnx"],
    "convert_fp16": ["onnx", "numpy", "onnxconverter_common"],
    "convert_int8": ["onnx", "numpy", "onnxruntime"],
    "build_engine": ["tensorrt"],
}


def load(name: str):
    for dep in SCRIPTS[name]:
        pytest.importorskip(dep)
    spec = importlib.util.spec_from_file_location(name, REPO / "trt-files/scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=sorted(SCRIPTS))
def publish(request):
    return load(request.param)._publish_pair


def test_pair_lands_and_tmps_are_gone(publish, tmp_path):
    tmp = tmp_path / "g.onnx.tmp"
    tmp.write_bytes(b"graph-v2")
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    publish(tmp, out, '{"v": 2}', sc, "test")
    assert out.read_bytes() == b"graph-v2" and sc.read_text() == '{"v": 2}'
    assert not tmp.exists() and not Path(str(sc) + ".tmp").exists()


def test_no_contract_removes_stale_sidecar(publish, tmp_path, capsys):
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    out.write_bytes(b"graph-v1")
    sc.write_text('{"v": 1}')  # metadata of the PREVIOUS graph
    tmp = tmp_path / "g.onnx.tmp"
    tmp.write_bytes(b"graph-v2")
    publish(tmp, out, None, sc, "test")
    assert out.read_bytes() == b"graph-v2"
    assert not sc.exists()  # v1 metadata must not describe the v2 graph
    assert "stale sidecar" in capsys.readouterr().out


def test_failed_staging_leaves_previous_pair_intact(publish, tmp_path):
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    out.write_bytes(b"graph-v1")
    sc.write_text('{"v": 1}')
    with pytest.raises(FileNotFoundError):
        publish(tmp_path / "missing.tmp", out, '{"v": 2}', sc, "test")
    assert out.read_bytes() == b"graph-v1" and sc.read_text() == '{"v": 1}'
