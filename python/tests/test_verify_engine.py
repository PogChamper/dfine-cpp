from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def verify_engine(monkeypatch):
    trt = ModuleType("tensorrt")
    trt.DataType = SimpleNamespace(
        FLOAT="float", HALF="half", INT32="int32", INT64="int64", BOOL="bool"
    )
    trt.TensorIOMode = SimpleNamespace(INPUT="input", OUTPUT="output")
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.int32 = "int32"
    torch.int64 = "int64"
    torch.bool = "bool"
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "torch", torch)

    scripts = REPO / "trt-files/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "verify_engine_test", scripts / "verify_engine.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


class Engine:
    def get_tensor_mode(self, name):
        return "input" if name == "images" else "output"

    def get_tensor_dtype(self, _name):
        return "float"


def test_rejected_shape_cannot_report_requested_batch_as_valid(verify_engine):
    context = SimpleNamespace(set_input_shape=lambda *_args: False)
    stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: None)

    with pytest.raises(RuntimeError, match="set_input_shape failed.*N=8"):
        verify_engine.run_batch(Engine(), context, ["images", "logits", "boxes"], 8, 640, stream)


def test_stale_resolved_shape_is_rejected(verify_engine):
    context = SimpleNamespace(
        set_input_shape=lambda *_args: True,
        get_tensor_shape=lambda name: (1, 3, 640, 640) if name == "images" else (1, 300, 80),
    )
    stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: None)

    with pytest.raises(RuntimeError, match="resolved to batch 1 at requested N=8"):
        verify_engine.run_batch(Engine(), context, ["images", "logits", "boxes"], 8, 640, stream)
