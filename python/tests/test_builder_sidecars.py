from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_engine(monkeypatch):
    trt = types.ModuleType("tensorrt")
    trt.LayerType = types.SimpleNamespace(
        CONSTANT=0,
        SHAPE=1,
        ASSERTION=2,
        IDENTITY=3,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)

    spec = importlib.util.spec_from_file_location(
        "dfine_test_build_engine",
        REPO / "trt-files/scripts/build_engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shared_stem_preserves_onnx_sidecar(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    source_sidecar = tmp_path / "model.json"
    source_sidecar.write_text("{}")

    chosen, stale = build_engine._engine_sidecar_plan(onnx, tmp_path / "model.engine")

    assert chosen == tmp_path / "model.engine.json"
    assert stale is None


def test_suffix_in_onnx_stem_does_not_mark_source_as_stale(build_engine, tmp_path):
    onnx = tmp_path / "model.engine.onnx"
    source_sidecar = tmp_path / "model.engine.json"
    source_sidecar.write_text("{}")

    chosen, stale = build_engine._engine_sidecar_plan(onnx, tmp_path / "model.engine")

    assert chosen == tmp_path / "model.json"
    assert stale is None


def test_suffixless_output_with_shared_sidecar_is_rejected(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    (tmp_path / "model.json").write_text("{}")

    with pytest.raises(SystemExit, match="use an output ending in .engine"):
        build_engine._engine_sidecar_plan(onnx, tmp_path / "model")


def test_json_engine_output_uses_appended_sidecar(build_engine, tmp_path):
    chosen, stale = build_engine._engine_sidecar_plan(
        tmp_path / "source.onnx",
        tmp_path / "engine.json",
    )

    assert chosen == tmp_path / "engine.json.json"
    assert stale is None
