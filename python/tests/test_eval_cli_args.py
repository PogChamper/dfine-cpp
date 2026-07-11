from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "trt-files/scripts"


@pytest.fixture(autouse=True)
def stub_runtime_dependencies(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            "cv2",
            "numpy",
            "tensorrt",
            "torch",
            "pycocotools",
            "pycocotools.coco",
            "pycocotools.cocoeval",
        )
    }
    modules["torch"].no_grad = lambda: (lambda function: function)
    modules["pycocotools.coco"].COCO = object
    modules["pycocotools.cocoeval"].COCOeval = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("coco_eval", "cpp_coco_eval", "profile"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_coco_eval_requires_dataset_and_selected_backend_inputs(monkeypatch):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("coco_eval")
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "engine", "--engine", "model.engine"])
    assert exc.value.code == 2

    args = module.parse_args(
        [
            "--backends",
            "engine",
            "--engine",
            "model.engine",
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
        ]
    )
    assert args.engine == "model.engine"


def test_engine_backend_rejects_failed_shape_binding():
    module = _load("coco_eval")
    backend = module.EngineBackend.__new__(module.EngineBackend)
    backend.ctx = SimpleNamespace(set_input_shape=lambda *args: False)

    class Input:
        shape = (1, 3, 640, 640)

        def cuda(self):
            return self

        def contiguous(self):
            return self

    with pytest.raises(RuntimeError, match="TensorRT rejected input shape"):
        backend(Input())


def test_engine_backend_requires_single_fp32_images_input():
    module = _load("coco_eval")
    module.trt.TensorIOMode = SimpleNamespace(INPUT="input", OUTPUT="output")
    module.trt.DataType = SimpleNamespace(FLOAT="float", HALF="half")

    class Engine:
        def __init__(self, inputs, input_dtype):
            self.names = [*inputs, "logits", "boxes"]
            self.num_io_tensors = len(self.names)
            self.inputs = set(inputs)
            self.input_dtype = input_dtype

        def get_tensor_name(self, index):
            return self.names[index]

        def get_tensor_mode(self, name):
            return "input" if name in self.inputs else "output"

        def get_tensor_dtype(self, name):
            return self.input_dtype if name == "images" else "float"

    with pytest.raises(RuntimeError, match="must be FP32"):
        module._validated_engine_names(Engine(["images"], "half"))
    with pytest.raises(RuntimeError, match="exactly one input"):
        module._validated_engine_names(Engine(["images", "scales"], "float"))


def test_cpp_eval_requires_engine_and_dataset(monkeypatch):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    with pytest.raises(SystemExit) as exc:
        module.parse_args([])
    assert exc.value.code == 2

    args = module.parse_args(
        [
            "--engine",
            "model.engine",
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
        ]
    )
    assert args.images == "val2017"

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                "model.engine",
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                "--batch",
                "0",
            ]
        )
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                "model.engine",
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                "--filter-res",
                "640x0",
            ]
        )
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                "model.engine",
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                "--full-graph",
            ]
        )
    assert exc.value.code == 2


def test_profile_dataset_is_required_only_for_accuracy(monkeypatch):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN", "DFINE_SAMPLE_IMAGE"):
        monkeypatch.delenv(name, raising=False)
    module = _load("profile")
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "trt", "--engine", "model.engine", "--no-accuracy"])
    assert exc.value.code == 2

    args = module.parse_args(
        [
            "--backends",
            "trt",
            "--engine",
            "model.engine",
            "--no-accuracy",
            "--sample-image",
            "frame.jpg",
        ]
    )
    assert not args.do_accuracy
    assert args.sample_image == "frame.jpg"

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--no-latency", "--no-accuracy"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "trt", "--engine", "model.engine", "--accuracy"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "cpp-graph", "--engine", "model.engine"])
    assert exc.value.code == 2


def test_profile_torch_inputs_are_conditional(monkeypatch):
    for name in ("DFINE_CHECKPOINT", "DFINE_SEG_DIR"):
        monkeypatch.delenv(name, raising=False)
    module = _load("profile")
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "torch", "--no-accuracy", "--sample-image", "frame.jpg"])
    assert exc.value.code == 2

    args = module.parse_args(
        [
            "--backends",
            "torch",
            "--no-accuracy",
            "--sample-image",
            "frame.jpg",
            "--checkpoint",
            "model.pt",
            "--dfine-src",
            "D-FINE-seg",
        ]
    )
    assert args.checkpoint == "model.pt"


@pytest.mark.parametrize("flag,value", [("--batches", "0"), ("--warmup", "-1"), ("--iters", "0")])
def test_profile_rejects_invalid_measurement_counts(monkeypatch, flag, value):
    monkeypatch.setenv("ENGINE", "model.engine")
    monkeypatch.setenv("DFINE_SAMPLE_IMAGE", "frame.jpg")
    module = _load("profile")
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "trt", "--no-accuracy", flag, value])
    assert exc.value.code == 2


def test_profile_rejects_duplicate_batches(monkeypatch):
    monkeypatch.setenv("ENGINE", "model.engine")
    monkeypatch.setenv("DFINE_SAMPLE_IMAGE", "frame.jpg")
    module = _load("profile")

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "trt", "--no-accuracy", "--batches", "1", "1"])
    assert exc.value.code == 2


def test_cpp_latency_requires_every_requested_batch(monkeypatch, tmp_path):
    module = _load("profile")

    def incomplete_benchmark(command, **_kwargs):
        output = Path(command[command.index("--json") + 1])
        output.write_text(json.dumps({"results": [{"batch": 1}]}))

    monkeypatch.setattr(module.subprocess, "run", incomplete_benchmark)

    with pytest.raises(RuntimeError, match="expected every requested batch"):
        module.latency_cpp("model.engine", [1, 8], 0, 1, {}, tmp_path, image="frame.jpg")


def test_cpp_graph_latency_requires_confirmed_replay(monkeypatch, tmp_path):
    module = _load("profile")
    commands = []

    def fallback_benchmark(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps(
                {
                    "cuda_graph_required": True,
                    "results": [{"batch": 1, "cuda_graph_replay": False}],
                }
            )
        )

    monkeypatch.setattr(module.subprocess, "run", fallback_benchmark)

    with pytest.raises(RuntimeError, match="did not confirm CUDA Graph replay"):
        module.latency_cpp(
            "model.engine", [1], 0, 1, {}, tmp_path, image="frame.jpg", cuda_graph=True
        )
    assert "--require-cuda-graph" in commands[0]
