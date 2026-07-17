from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
    for name in ("coco_eval", "coco_metrics", "cpp_coco_eval", "profile"):
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


def _model_sidecar(**overrides):
    payload = {
        "schema_version": 1,
        "artifact_kind": "onnx",
        "model": "d-fine",
        "variant": "s",
        "task": "detect",
        "input_h": 640,
        "input_w": 640,
        "num_classes": 3,
        "num_queries": 300,
        "eval_idx": 2,
        "input_names": ["images"],
        "output_names": ["logits", "boxes"],
        "color_order": "RGB",
        "channel_layout": "NCHW",
        "normalize": "div255",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": "stretch",
        "checkpoint_load": "strict",
        "checkpoint_sha256": "a" * 64,
        "precision_mode": "fp32",
    }
    payload.update(overrides)
    return payload


def _profile_engine_sidecar(**overrides):
    payload = _model_sidecar(
        artifact_kind="engine",
        variant="m",
        num_classes=80,
        eval_idx=3,
        precision="fp16",
        precision_mode="strongly_typed_onnx_fp16_surgical_slim",
        network_typing="strong",
        tf32=False,
        dynamic_batch=True,
        min_batch=1,
        opt_batch=1,
        max_batch=8,
        max_aux_streams=None,
        cuda_graph_compat=False,
        trt_version="10.13.3.9.post1",
        sm_arch="89",
        onnx_sha256="b" * 64,
    )
    payload.update(overrides)
    return payload


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


def test_coco_eval_scopes_sliders_to_torch(monkeypatch):
    for name in ("DFINE_CHECKPOINT", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("coco_eval")
    args = module.parse_args(
        [
            "--backends",
            "torch",
            "--checkpoint",
            "model.pt",
            "--images",
            "images",
            "--ann",
            "instances.json",
            "--num-queries",
            "200",
            "--cascade",
            "1:100",
        ]
    )
    assert args.num_queries == 200
    assert args.cascade == "1:100"

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--backends",
                "onnx",
                "--onnx",
                "model.onnx",
                "--images",
                "images",
                "--ann",
                "instances.json",
                "--num-queries",
                "200",
            ]
        )
    assert exc.value.code == 2


def test_onnx_backend_requires_cuda_unless_explicitly_allowed(monkeypatch):
    module = _load("coco_eval")

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CPUExecutionProvider": {}}

    runtime = SimpleNamespace(__version__="1.24.0", get_device=lambda: "CPU")
    cuda_env = ModuleType("cuda_env")
    cuda_env.bootstrap = lambda: runtime
    cuda_env.make_session = lambda _path, **_kwargs: (Session(), ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "cuda_env", cuda_env)

    with pytest.raises(RuntimeError, match="did not activate CUDAExecutionProvider"):
        module.OrtBackend("model.onnx")

    backend = module.OrtBackend("model.onnx", allow_cpu=True)
    assert backend.provenance["providers"] == ["CPUExecutionProvider"]
    assert backend.provenance["runtime_version"] == "1.24.0"
    assert backend.provenance["cuda_required"] is False
    assert backend.provenance["numeric_policy"] == {"tf32_requested": False}


def test_onnx_backend_rejects_cuda_tf32(monkeypatch):
    module = _load("coco_eval")

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"use_tf32": "1"}}

    runtime = SimpleNamespace(__version__="1.24.0", get_device=lambda: "GPU")
    cuda_env = ModuleType("cuda_env")
    cuda_env.bootstrap = lambda: runtime
    cuda_env.make_session = lambda _path, **_kwargs: (
        Session(),
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "cuda_env", cuda_env)

    with pytest.raises(RuntimeError, match="did not disable TF32"):
        module.OrtBackend("model.onnx")


def test_coco_eval_requires_artifact_sidecars(tmp_path):
    module = _load("coco_eval")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")

    with pytest.raises(SystemExit, match="requires an adjacent JSON sidecar"):
        module._artifact_contract(onnx, "onnx")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resize", "letterbox"),
        ("color_order", "BGR"),
        ("channel_layout", "NHWC"),
        ("normalize", "imagenet"),
        ("mean", [0.485, 0.456, 0.406]),
        ("std", [0.229, 0.224, 0.225]),
    ],
)
def test_coco_eval_rejects_unsupported_preprocessing(tmp_path, field, value):
    module = _load("coco_eval")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_suffix(".json").write_text(json.dumps(_model_sidecar(**{field: value})))

    with pytest.raises(SystemExit, match=field):
        module._artifact_contract(onnx, "onnx")


def test_coco_eval_normalizes_cascade_contract(tmp_path):
    module = _load("coco_eval")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_suffix(".json").write_text(
        json.dumps(
            _model_sidecar(
                num_queries=100,
                cascade="1:100",
                cascade_initial_queries=200,
            )
        )
    )

    contract, _meta, _sidecar = module._artifact_contract(onnx, "onnx")

    assert contract["initial_queries"] == 200
    assert contract["num_queries"] == 100
    assert contract["cascade"] == "1:100"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant", "m"),
        ("input_h", 320),
        ("input_w", 320),
        ("num_classes", 4),
        ("initial_queries", 200),
        ("num_queries", 200),
        ("eval_idx", 1),
        ("cascade", "1:100"),
        ("checkpoint_sha256", "b" * 64),
    ],
)
def test_coco_eval_rejects_model_contract_mismatch(field, value):
    module = _load("coco_eval")
    reference = module._normalized_model_contract(
        _model_sidecar(), Path("model.json")
    )
    candidate = {**reference, field: value}

    with pytest.raises(SystemExit, match=field):
        module._require_matching_model_contracts(
            {"reference": reference, "candidate": candidate}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant", "m"),
        ("input_h", 320),
        ("input_w", 320),
        ("num_classes", 4),
    ],
)
def test_coco_eval_rejects_sidecar_cli_mismatch(field, value):
    module = _load("coco_eval")
    contract = module._normalized_model_contract(_model_sidecar(), Path("model.json"))
    contract[field] = value
    args = SimpleNamespace(model_name="s", img_size=640, num_classes=3)

    with pytest.raises(SystemExit, match=field):
        module._require_evaluation_arguments(contract, args, "onnx")


def test_coco_eval_validates_engine_source_hash(tmp_path):
    module = _load("coco_eval")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    digest = module.sha256_file(onnx)

    module._require_engine_source({"onnx_sha256": digest}, onnx)
    with pytest.raises(SystemExit, match="does not match"):
        module._require_engine_source({"onnx_sha256": "0" * 64}, onnx)


def test_coco_eval_normalizes_artifact_lineage(tmp_path):
    module = _load("coco_eval")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    sidecar = onnx.with_suffix(".json")
    metadata = _model_sidecar(source_onnx_sha256="b" * 64)
    sidecar.write_text(json.dumps(metadata))
    contract = module._normalized_model_contract(metadata, sidecar)

    lineage = module._artifact_lineage(onnx, "onnx", metadata, sidecar, contract)

    assert lineage == {
        "artifact_kind": "onnx",
        "precision_mode": "fp32",
        "checkpoint_sha256": "a" * 64,
        "artifact_sha256": module.sha256_file(onnx),
        "source_onnx_sha256": "b" * 64,
    }


def test_coco_eval_engine_lineage_requires_onnx_hash(tmp_path):
    module = _load("coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    sidecar = Path(f"{engine}.json")
    metadata = _model_sidecar(artifact_kind="engine")
    contract = module._normalized_model_contract(metadata, sidecar)

    with pytest.raises(SystemExit, match="onnx_sha256"):
        module._artifact_lineage(engine, "engine", metadata, sidecar, contract)

    metadata["onnx_sha256"] = "c" * 64
    lineage = module._artifact_lineage(engine, "engine", metadata, sidecar, contract)
    assert lineage["onnx_sha256"] == "c" * 64


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("onnx", "checkpoint_sha256"),
        ("onnx", "source_onnx_sha256"),
        ("engine", "onnx_sha256"),
    ],
)
def test_coco_eval_rejects_noncanonical_lineage_hash(tmp_path, kind, field):
    module = _load("coco_eval")
    artifact = tmp_path / f"model.{kind}"
    artifact.write_bytes(kind.encode())
    sidecar = Path(f"{artifact}.json")
    metadata = _model_sidecar(artifact_kind=kind, **{field: "A" * 64})

    with pytest.raises(SystemExit, match="lowercase SHA-256"):
        contract = module._normalized_model_contract(metadata, sidecar)
        module._artifact_lineage(artifact, kind, metadata, sidecar, contract)


def test_coco_eval_rejects_runtime_query_mismatch():
    module = _load("coco_eval")
    logits = SimpleNamespace(shape=(1, 200, 3))

    module._require_query_count(logits, 200, "onnx")
    with pytest.raises(SystemExit, match="returned 200 queries"):
        module._require_query_count(logits, 300, "onnx")


def test_cuda_session_disables_tf32_by_default(monkeypatch):
    module = _load("cuda_env")
    captured = {}

    class SessionOptions:
        pass

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def inference_session(model_path, *, sess_options, providers):
        captured.update(
            {"model_path": model_path, "sess_options": sess_options, "providers": providers}
        )
        return Session()

    runtime = SimpleNamespace(
        SessionOptions=SessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=inference_session,
    )
    monkeypatch.setattr(module, "bootstrap", lambda: runtime)

    _session, providers = module.make_session("model.onnx")

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert captured["providers"][0] == (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "use_tf32": "0",
        },
    )


def test_torch_backend_preserves_strict_load_and_source_provenance(monkeypatch):
    module = _load("coco_eval")
    numeric_policy = {
        "float32_matmul_precision": "highest",
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
    }
    monkeypatch.setattr(module, "configure_torch_numeric_policy", lambda: numeric_policy)
    report = {
        "mode": "strict",
        "selected_state": "ema.module",
        "loaded": 4,
        "missing": 0,
        "shape_mismatch": 0,
        "unused": 1,
        "sha256": "a" * 64,
        "deserialization": "weights_only",
    }

    class Model:
        decoder = SimpleNamespace(num_queries=200, eval_idx=2)

        def cuda(self):
            return self

        def deploy(self):
            return self

        def eval(self):
            return self

    model_module = ModuleType("dfine_model")
    model_module.build_model = lambda *_args: Model()
    exporter = ModuleType("export_dfine_onnx")
    exporter.__file__ = str(SCRIPTS / "export_dfine_onnx.py")
    exporter.load_checkpoint_state = lambda *_args, **_kwargs: report
    applied = []
    exporter.apply_sliders = lambda model, args: applied.append((model, args))
    exporter._model_source_manifest = lambda _path: {"sha256": "b" * 64, "files": ["model.py"]}
    exporter._validated_source_revision = lambda: "c" * 40
    exporter._exporter_sha256 = lambda: "d" * 64
    monkeypatch.setitem(sys.modules, "dfine_model", model_module)
    monkeypatch.setitem(sys.modules, "export_dfine_onnx", exporter)

    backend = module.TorchBackend(
        SimpleNamespace(
            model_name="s",
            num_classes=3,
            img_size=640,
            checkpoint="model.pt",
            num_queries=200,
            eval_idx=None,
            cascade="1:100",
        )
    )

    assert backend.provenance["checkpoint_load"] == report
    assert backend.provenance["bundled_model"]["manifest"]["sha256"] == "b" * 64
    assert backend.provenance["bundled_model"]["validated_upstream_commit"] == "c" * 40
    assert backend.provenance["exporter"]["sha256"] == "d" * 64
    assert backend.provenance["numeric_policy"] == numeric_policy
    assert backend.provenance["sliders"] == {
        "num_queries": 200,
        "eval_idx": None,
        "cascade": "1:100",
    }
    assert backend.model_contract == {
        "model": "d-fine",
        "variant": "s",
        "task": "detect",
        "input_h": 640,
        "input_w": 640,
        "num_classes": 3,
        "initial_queries": 200,
        "num_queries": 100,
        "eval_idx": 2,
        "cascade": "1:100",
        "checkpoint_sha256": "a" * 64,
        "preprocess": {
            "color_order": "RGB",
            "channel_layout": "NCHW",
            "normalize": "div255",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "resize": "stretch",
        },
    }
    assert backend.lineage == {
        "artifact_kind": "checkpoint",
        "precision_mode": "fp32",
        "checkpoint_sha256": "a" * 64,
        "artifact_sha256": "a" * 64,
    }
    assert len(applied) == 1
    assert applied[0][0] is backend.m
    assert applied[0][1].num_queries == 200


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


def test_engine_backend_reuses_bound_buffers_for_same_shape(monkeypatch):
    module = _load("coco_eval")
    backend = module.EngineBackend.__new__(module.EngineBackend)
    prepared = 0

    class Call:
        host_input = SimpleNamespace(shape=(1, 3, 640, 640))

        def __init__(self):
            self.updated = 0
            self.bound = 0
            self.transferred = 0

        def set_host_input(self, _value):
            self.updated += 1

        def bind(self):
            self.bound += 1

        def transfer(self):
            self.transferred += 1

        def synchronize(self):
            pass

        def require_finite(self, **_kwargs):
            pass

        def numpy_outputs(self):
            return "logits", "boxes"

    call = Call()

    def prepare(_value):
        nonlocal prepared
        prepared += 1
        return call

    monkeypatch.setattr(backend, "prepare", prepare)
    value = SimpleNamespace(shape=(1, 3, 640, 640))

    assert backend(value) == ("logits", "boxes")
    assert backend(value) == ("logits", "boxes")
    assert prepared == 1
    assert call.updated == 1
    assert call.bound == 1
    assert call.transferred == 2


def test_coco_eval_keeps_legacy_ap_summary(monkeypatch, capsys):
    module = _load("coco_eval")
    metrics = {
        "AP": 0.55,
        "AP50": 0.74,
        "AP75": 0.60,
        "AR100": 0.70,
    }
    monkeypatch.setattr(module, "evaluate_bbox", lambda *_args, **_kwargs: metrics)

    assert module.evaluate(object(), [{}], "engine", [1]) == metrics
    assert "AP@[.50:.95]=0.5500  AP@.50=0.7400" in capsys.readouterr().out


def test_coco_eval_report_is_consumed_by_accuracy_chain(monkeypatch, tmp_path):
    module = _load("coco_eval")
    accuracy_chain = _load("accuracy_chain")
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"image")
    annotations = tmp_path / "instances_test.json"
    annotations.write_text("{}")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = module.sha256_file(checkpoint)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_suffix(".json").write_text(
        json.dumps(_model_sidecar(num_classes=1, checkpoint_sha256=checkpoint_sha256))
    )
    report_path = tmp_path / "report.json"

    class Coco:
        dataset = {
            "info": {
                "dataset": "fixture",
                "split": "test",
                "source_archive_sha256": "d" * 64,
            }
        }

        def getCatIds(self):
            return [1]

        def getImgIds(self):
            return [7]

        def loadImgs(self, _image_id):
            return [{"id": 7, "file_name": "one.jpg"}]

    class Backend:
        provenance = {"runtime": "fixture"}
        model_contract = {
            "model": "d-fine",
            "variant": "s",
            "task": "detect",
            "input_h": 640,
            "input_w": 640,
            "num_classes": 1,
            "initial_queries": 300,
            "num_queries": 300,
            "eval_idx": 2,
            "cascade": None,
            "checkpoint_sha256": checkpoint_sha256,
            "preprocess": {
                "color_order": "RGB",
                "channel_layout": "NCHW",
                "normalize": "div255",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "resize": "stretch",
            },
        }
        lineage = {
            "artifact_kind": "checkpoint",
            "precision_mode": "fp32",
            "checkpoint_sha256": checkpoint_sha256,
            "artifact_sha256": checkpoint_sha256,
        }

        def __call__(self, _value):
            return "logits", "boxes"

    scalar = {
        "AP": 0.5,
        "AP50": 0.5,
        "AP75": 0.5,
        "APs": 0.5,
        "APm": None,
        "APl": None,
        "AR1": 0.5,
        "AR10": 0.5,
        "AR100": 0.5,
        "ARs": 0.5,
        "ARm": None,
        "ARl": None,
    }
    metrics = {
        **scalar,
        "max_dets": [1, 10, 100],
        "AP_by_iou": {f"{value / 100:.2f}": 0.5 for value in range(50, 100, 5)},
        "per_class": [{"category_id": 1, "name": "object", "gt_instances": 1, "AP": 0.5}],
        "GT_by_area": {"small": 1, "medium": 0, "large": 0},
        "model_space_area": {
            "input_h": 640,
            "input_w": 640,
            "resize": "stretch",
            "APs": 0.5,
            "APm": None,
            "APl": None,
            "ARs": 0.5,
            "ARm": None,
            "ARl": None,
            "GT_by_area": {"small": 1, "medium": 0, "large": 0},
        },
    }
    histogram = [
        {"range": name, "images": int(name == "1")} for name in accuracy_chain.DENSITY_RANGES
    ]
    ground_truth = {
        "images": 1,
        "gt_instances": 1,
        "crowd_instances": 0,
        "per_image": {
            "min": 1,
            "mean": 1.0,
            "median": 1.0,
            "p90": 1.0,
            "p95": 1.0,
            "p99": 1.0,
            "max": 1,
            "over_100": 0,
            "histogram": histogram,
        },
    }
    monkeypatch.setattr(module, "COCO", lambda _path: Coco())
    monkeypatch.setattr(module, "TorchBackend", lambda _args: Backend())
    monkeypatch.setattr(module, "OrtBackend", lambda _path, **_kwargs: Backend())
    monkeypatch.setattr(
        module,
        "require_detection_outputs",
        lambda *_args: (
            SimpleNamespace(shape=(1, 300, 1)),
            SimpleNamespace(shape=(1, 300, 4)),
        ),
    )
    monkeypatch.setattr(module, "preprocess", lambda *_args: object())
    monkeypatch.setattr(module, "decode", lambda *_args: ([[0, 0, 1, 1]], [0], [0.9]))
    monkeypatch.setattr(module, "evaluate", lambda *_args, **_kwargs: metrics)
    monkeypatch.setattr(module, "ground_truth_summary", lambda *_args: ground_truth)
    monkeypatch.setattr(module, "environment_metadata", lambda: {})
    monkeypatch.setattr(module.cv2, "IMREAD_COLOR", 1, raising=False)
    monkeypatch.setattr(
        module.cv2,
        "imread",
        lambda *_args: SimpleNamespace(shape=(10, 20, 3)),
        raising=False,
    )

    args = SimpleNamespace(
        ann=str(annotations),
        images=str(images),
        model_name="s",
        num_classes=1,
        img_size=640,
        topk=300,
        score_thresh=0.001,
        limit=0,
        backends=["torch", "onnx"],
        checkpoint=str(checkpoint),
        onnx=str(onnx),
        engine="",
        report=str(report_path),
        protocol_manifest="",
        overwrite=False,
    )
    assert module.main(args) == 0
    source_report = json.loads(report_path.read_text())
    assert source_report["backends"]["torch"]["backend_provenance"] == {"runtime": "fixture"}
    assert source_report["backends"]["onnx"]["model_contract"] == Backend.model_contract
    assert source_report["backends"]["torch"]["lineage"] == Backend.lineage
    assert source_report["backends"]["onnx"]["lineage"] == {
        "artifact_kind": "onnx",
        "precision_mode": "fp32",
        "checkpoint_sha256": checkpoint_sha256,
        "artifact_sha256": module.sha256_file(onnx),
    }

    chain_path = tmp_path / "chain.json"
    result = accuracy_chain.run(
        SimpleNamespace(
            stage=[("pytorch", report_path, "torch"), ("onnx", report_path, "onnx")],
            transition_label=[],
            output=str(chain_path),
            overwrite=False,
            allow_missing_lineage=False,
        )
    )
    assert result["evaluation_contract"]["dataset"]["name"] == "fixture"
    assert result["stages"][0]["artifact"]["sha256"] != result["stages"][1]["artifact"]["sha256"]
    assert result["transitions"][0]["label"] == "export fidelity"
    assert json.loads(chain_path.read_text()) == result


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
            "--img-size",
            "640",
            "--num-classes",
            "80",
        ]
    )
    assert args.images == "val2017"
    assert args.model_hw == (640, 640)
    assert args.model_resize == "stretch"

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                "model.engine",
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                "--img-size",
                "640",
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
                "--img-size",
                "640",
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
                "--img-size",
                "640",
                "--full-graph",
            ]
        )
    assert exc.value.code == 2


def test_cpp_eval_report_requires_complete_engine_sidecar(monkeypatch, tmp_path, capsys):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                str(engine),
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                "--img-size",
                "640",
                "--num-classes",
                "3",
                "--report",
                str(tmp_path / "report.json"),
            ]
        )

    assert exc.value.code == 2
    assert "requires a complete engine sidecar" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("color_order", "BGR"),
        ("channel_layout", "NHWC"),
        ("normalize", "imagenet"),
        ("mean", [0.485, 0.456, 0.406]),
        ("std", [0.229, 0.224, 0.225]),
        ("resize", "letterbox"),
    ],
)
def test_cpp_eval_report_rejects_unsupported_preprocessing(field, value):
    module = _load("cpp_coco_eval")
    metadata = _model_sidecar(
        artifact_kind="engine",
        onnx_sha256="b" * 64,
        **{field: value},
    )

    with pytest.raises(ValueError, match=field):
        module._normalized_model_contract(metadata, Path("model.engine.json"))


def test_cpp_eval_backend_report_is_accepted_by_accuracy_chain(monkeypatch, tmp_path):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    accuracy_chain = _load("accuracy_chain")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    sidecar = Path(f"{engine}.json")
    sidecar.write_text(
        json.dumps(
            _model_sidecar(
                artifact_kind="engine",
                precision_mode="strongly_typed_onnx_fp16_surgical_slim",
                onnx_sha256="b" * 64,
                source_onnx_sha256="c" * 64,
            )
        )
    )
    args = module.parse_args(
        [
            "--engine",
            str(engine),
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    monkeypatch.setattr(module, "package_runtime", lambda *_args: "TensorRT fixture")

    backend = module._backend_report(args, {})
    selected, _metrics, artifact_record, model_contract, lineage = (
        accuracy_chain._select_metrics(
            {"backends": {"cpp": backend}},
            "native",
            "cpp",
        )
    )

    assert selected == "cpp"
    assert model_contract == args.model_contract
    assert lineage == {
        "artifact_kind": "engine",
        "precision_mode": "strongly_typed_onnx_fp16_surgical_slim",
        "checkpoint_sha256": "a" * 64,
        "artifact_sha256": module.sha256_file(engine),
        "onnx_sha256": "b" * 64,
        "source_onnx_sha256": "c" * 64,
    }
    assert artifact_record["sha256"] == lineage["artifact_sha256"]


def test_cpp_eval_remaps_categories_in_place():
    module = _load("cpp_coco_eval")
    detections = [
        {
            "image_id": 7,
            "category_contig": 0,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
        },
        {
            "image_id": 8,
            "category_contig": 1,
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "score": 0.8,
        },
    ]
    entries = [id(detection) for detection in detections]

    remapped = module._remap_categories(detections, {0: 1, 1: 90})

    assert remapped is detections
    assert [id(detection) for detection in remapped] == entries
    assert remapped == [
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
        },
        {
            "image_id": 8,
            "category_id": 90,
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "score": 0.8,
        },
    ]


def test_cpp_eval_resolves_engine_metadata_and_preprocessing(monkeypatch, tmp_path):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    same_stem = engine.with_suffix(".json")
    same_stem.write_text(
        json.dumps(
            {
                "input_h": 480,
                "input_w": 800,
                "num_classes": 80,
                "resize": "letterbox",
                "letterbox_anchor": "topleft",
                "letterbox_pad": 7,
                "letterbox_upscale": False,
            }
        )
    )

    args = module.parse_args(
        [
            "--engine",
            str(engine),
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
        ]
    )

    assert args.resolved_meta == str(same_stem)
    assert args.model_hw == (480, 800)
    assert args.model_num_classes == 80
    assert args.model_resize == "letterbox"
    assert args.model_letterbox_anchor == "topleft"
    assert args.model_letterbox_pad == 7
    assert not args.model_letterbox_upscale


def test_cpp_eval_prefers_appended_engine_sidecar(monkeypatch, tmp_path):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    engine.with_suffix(".json").write_text(
        json.dumps({"input_h": 320, "input_w": 320, "num_classes": 80})
    )
    appended = Path(f"{engine}.json")
    appended.write_text(json.dumps({"input_h": 640, "input_w": 640, "num_classes": 80}))

    args = module.parse_args(
        [
            "--engine",
            str(engine),
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
        ]
    )

    assert args.resolved_meta == str(appended)
    assert args.model_hw == (640, 640)


def test_cpp_eval_cli_letterbox_overrides_metadata(monkeypatch, tmp_path):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    Path(f"{engine}.json").write_text(
        json.dumps(
            {
                "input_h": 640,
                "input_w": 640,
                "num_classes": 80,
                "resize": "letterbox",
                "letterbox_anchor": "topleft",
                "letterbox_upscale": False,
            }
        )
    )

    args = module.parse_args(
        [
            "--engine",
            str(engine),
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
            "--letterbox",
        ]
    )

    assert args.model_resize == "letterbox"
    assert args.model_letterbox_anchor == "center"
    assert args.model_letterbox_upscale


def test_cpp_eval_explicit_default_pad_enables_letterbox(monkeypatch):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")

    args = module.parse_args(
        [
            "--engine",
            "model.engine",
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
            "--img-size",
            "640",
            "--num-classes",
            "80",
            "--letterbox-pad",
            "114",
        ]
    )

    assert args.letterbox_pad == 114
    assert args.model_resize == "letterbox"
    assert args.model_letterbox_pad == 114


def test_cpp_eval_records_native_execution_path(monkeypatch):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")

    args = module.parse_args(
        [
            "--engine",
            "model.engine",
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
            "--img-size",
            "640",
            "--num-classes",
            "80",
            "--batch",
            "8",
            "--cuda-graph",
            "--gpu-decode",
            "--own-device-memory",
            "--freeze",
            "--full-graph",
            "--filter-res",
            "640x640",
            "--letterbox-topleft",
            "--letterbox-pad",
            "0",
            "--no-upscale",
        ]
    )

    assert module._execution_provenance(args) == {
        "requested": {
            "batch": 8,
            "cuda_graph": True,
            "filter_resolution": "640x640",
            "freeze": True,
            "full_graph": True,
            "gpu_decode": True,
            "letterbox": False,
            "letterbox_pad": 0,
            "letterbox_topleft": True,
            "no_upscale": True,
            "own_device_memory": True,
        },
        "resolved": {
            "batch": 8,
            "freeze_invoked": True,
            "full_pipeline_graph_option": True,
            "gpu_decode_option": True,
            "ordinary_cuda_graph_option": True,
            "dispatch_precedence": "full_pipeline_graph",
            "own_device_memory_option": True,
        },
        "verification": {
            "full_pipeline_graph": {
                "active": True,
                "evidence": (
                    "successful evaluator exit requires active capture and one replay per inference call"
                ),
            },
            "gpu_decode": {
                "active": True,
                "evidence": "included in the verified full-pipeline graph",
            },
            "ordinary_cuda_graph": {
                "active": False,
                "evidence": (
                    "full-pipeline graph takes precedence for every successful evaluation call"
                ),
            },
        },
        "preprocess": {
            "resize": "letterbox",
            "letterbox_anchor": "topleft",
            "letterbox_pad": 0,
            "letterbox_upscale": False,
        },
    }

    fallback_capable = module.parse_args(
        [
            "--engine",
            "model.engine",
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
            "--img-size",
            "640",
            "--num-classes",
            "80",
            "--cuda-graph",
            "--gpu-decode",
        ]
    )
    verification = module._execution_provenance(fallback_capable)["verification"]
    assert verification["gpu_decode"]["active"] is None
    assert verification["ordinary_cuda_graph"]["active"] is None


def test_cpp_eval_explicit_metadata_wins_over_discovery(monkeypatch, tmp_path):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    Path(f"{engine}.json").write_text(
        json.dumps(
            {"input_h": 320, "input_w": 320, "num_classes": 80, "variant": "discovered"}
        )
    )
    explicit = tmp_path / "selected.json"
    explicit.write_text(
        json.dumps({"input_h": 640, "input_w": 640, "num_classes": 80, "variant": "selected"})
    )

    args = module.parse_args(
        [
            "--engine",
            str(engine),
            "--meta",
            str(explicit),
            "--images",
            "val2017",
            "--ann",
            "instances_val2017.json",
        ]
    )

    assert args.resolved_meta == str(explicit)
    assert args.model_hw == (640, 640)
    assert "variant=selected" in module._engine_recipe(args)
    assert "variant=discovered" not in module._engine_recipe(args)


@pytest.mark.parametrize(
    ("meta", "extra", "match"),
    [
        ({}, [], "input dimensions are unknown"),
        ({"input_h": 640}, [], "input_h and input_w"),
        ({"input_h": 640, "input_w": 640}, ["--img-size", "320"], "contradicts"),
        (
            {"input_h": 640, "input_w": 640, "resize": "letterbox", "letterbox_upscale": 1},
            [],
            "letterbox_upscale",
        ),
        (
            {"input_h": 640, "input_w": 640, "resize": "letterbox", "letterbox_pad": 256},
            [],
            "letterbox_pad",
        ),
    ],
)
def test_cpp_eval_rejects_ambiguous_or_inconsistent_metadata(
    monkeypatch, tmp_path, meta, extra, match
):
    for name in ("ENGINE", "COCO_IMAGES", "COCO_ANN"):
        monkeypatch.delenv(name, raising=False)
    module = _load("cpp_coco_eval")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    Path(f"{engine}.json").write_text(json.dumps(meta))

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--engine",
                str(engine),
                "--images",
                "val2017",
                "--ann",
                "instances_val2017.json",
                *extra,
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
    assert args.rounds == 3

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
    monkeypatch.delenv("DFINE_CHECKPOINT", raising=False)
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
        ]
    )
    assert args.checkpoint == "model.pt"


def test_profile_torch_backend_uses_base_graph_sliders(monkeypatch):
    module = _load("profile")
    captured = {}

    def backend(args):
        captured.update(vars(args))
        return object()

    monkeypatch.setattr(module, "TorchBackend", backend)
    args = SimpleNamespace(
        model_name="s",
        num_classes=3,
        img_size=640,
        checkpoint="model.pt",
    )

    instance, artifact_path = module.make_backend("torch", args, {})

    assert instance is not None
    assert artifact_path is None
    assert captured["num_queries"] is None
    assert captured["eval_idx"] is None
    assert captured["cascade"] is None


@pytest.mark.parametrize(
    "flag,value",
    [("--batches", "0"), ("--warmup", "-1"), ("--iters", "0"), ("--rounds", "0")],
)
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


def test_profile_cpp_accuracy_rejects_topk_beyond_native_decode_limit(monkeypatch):
    monkeypatch.setenv("ENGINE", "model.engine")
    monkeypatch.setenv("COCO_IMAGES", "images")
    monkeypatch.setenv("COCO_ANN", "instances.json")
    module = _load("profile")

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", "cpp", "--topk", "301"])
    assert exc.value.code == 2

    # topk above 300 stays valid for python backends and for latency-only cpp runs.
    args = module.parse_args(["--backends", "trt", "--topk", "301"])
    assert args.topk == 301
    monkeypatch.setenv("DFINE_SAMPLE_IMAGE", "frame.jpg")
    args = module.parse_args(["--backends", "cpp", "--no-accuracy", "--topk", "301"])
    assert args.topk == 301


def test_profile_cpp_accuracy_truncates_native_detections_to_topk(monkeypatch, tmp_path):
    module = _load("profile")
    dets = [
        {"image_id": 1, "category_contig": 0, "bbox": [0, 0, 1, 1], "score": 0.9},
        {"image_id": 1, "category_contig": 1, "bbox": [0, 0, 1, 1], "score": 0.8},
        {"image_id": 1, "category_contig": 0, "bbox": [0, 0, 1, 1], "score": 0.7},
        {"image_id": 2, "category_contig": 1, "bbox": [0, 0, 1, 1], "score": 0.6},
    ]

    def fake_run(cmd, check, env, stdout):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps(dets))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    class _Coco:
        def getCatIds(self):
            return [7, 11]

    kept = module.accuracy_cpp(
        "model.engine", _Coco(), [1, 2], "images", "filelist", {}, 0.001, 2, tmp_path
    )
    by_image = {}
    for d in kept:
        by_image.setdefault(d["image_id"], []).append(d["score"])
    assert by_image == {1: [0.9, 0.8], 2: [0.6]}
    assert all(d["category_id"] in (7, 11) for d in kept)


@pytest.mark.parametrize("script", ["coco_eval", "profile"])
def test_evaluation_tools_reject_duplicate_backends(monkeypatch, script):
    monkeypatch.setenv("ENGINE", "model.engine")
    monkeypatch.setenv("DFINE_SAMPLE_IMAGE", "frame.jpg")
    monkeypatch.setenv("COCO_IMAGES", "images")
    monkeypatch.setenv("COCO_ANN", "instances.json")
    module = _load(script)
    extra = ["--no-accuracy", "--sample-image", "frame.jpg"] if script == "profile" else []
    backend = "engine" if script == "coco_eval" else "trt"

    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--backends", backend, backend, *extra])
    assert exc.value.code == 2


def test_profile_map_keeps_release_gate_keys(monkeypatch):
    module = _load("profile")
    expected = {"AP": 0.55, "AP50": 0.74, "AR100": 0.80}
    monkeypatch.setattr(module, "evaluate_bbox", lambda *_args, **_kwargs: expected)

    result = module.score_map(object(), [{}], [1])

    assert result["AP"] == 0.55
    assert result["AP50"] == 0.74


def test_profile_report_records_round_protocol_and_sample_hash(monkeypatch, tmp_path):
    module = _load("profile")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    metadata = _profile_engine_sidecar()
    Path(f"{engine}.json").write_text(json.dumps(metadata))
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"sample")
    output = tmp_path / "profile.json"
    args = module.parse_args(
        [
            "--backends",
            "trt",
            "--engine",
            str(engine),
            "--no-accuracy",
            "--sample-image",
            str(sample),
            "--batches",
            "1",
            "--workdir",
            str(tmp_path),
            "--out",
            str(output),
        ]
    )

    class Backend:
        pass

    latency = {
        "scopes": {
            "end_to_end": {
                "p50": 2.0,
                "p90": 2.1,
                "p99": 2.2,
                "mean": 2.0,
                "img_per_s": 500.0,
                "aggregation": module.ROUND_AGGREGATION,
                "rounds": [{"p50": 2.0}],
            }
        }
    }
    monkeypatch.setattr(module, "make_backend", lambda *_args: (Backend(), None))
    monkeypatch.setattr(module, "latency_python", lambda *_args: latency)
    monkeypatch.setattr(module, "gpu_used_mib", lambda: 0.0)
    monkeypatch.setattr(
        module,
        "environment_metadata",
        lambda: {"nvidia_gpus": [{"uuid": "GPU-1234"}]},
    )
    monkeypatch.setattr(module, "active_gpu_identity", lambda _environment: {"uuid": "GPU-1234"})
    monkeypatch.setattr(module.cv2, "IMREAD_COLOR", 1, raising=False)
    monkeypatch.setattr(
        module.cv2,
        "imread",
        lambda *_args: SimpleNamespace(shape=(480, 640, 3)),
        raising=False,
    )
    module.torch.cuda = SimpleNamespace(empty_cache=lambda: None)

    assert module.main(args) == 0
    report = json.loads(output.read_text())
    assert report["schema"] == 2
    assert report["latency_protocol"] == {
        "batches": [1],
        "warmup": 20,
        "iters": 100,
        "rounds": 3,
        "aggregation": module.ROUND_AGGREGATION,
        "score_threshold": 0.001,
        "topk": 300,
        "sample": {
            "path": str(sample.resolve()),
            "sha256": module.sha256_file(sample),
            "width": 640,
            "height": 480,
        },
        "gpu_identity": {"uuid": "GPU-1234"},
    }
    assert report["backends"]["trt"]["engine_contract"] == {
        "sidecar": str(Path(f"{engine}.json").resolve()),
        "sidecar_sha256": module.sha256_file(f"{engine}.json"),
        **{field: metadata[field] for field in module.ENGINE_CONTRACT_FIELDS if field in metadata},
    }
    assert report["backends"]["trt"]["model_contract"]["checkpoint_sha256"] == "a" * 64
    assert report["backends"]["trt"]["lineage"]["onnx_sha256"] == "b" * 64


def _profile_contract_args(module, tmp_path, onnx, engine, backend="trt"):
    return module.parse_args(
        [
            "--backends",
            "onnx",
            backend,
            "--onnx",
            str(onnx),
            "--engine",
            str(engine),
            "--model-name",
            "s",
            "--num-classes",
            "3",
            "--no-accuracy",
            "--sample-image",
            str(tmp_path / "sample.jpg"),
            "--batches",
            "1",
        ]
    )


def test_profile_rejects_cross_backend_graph_mismatch(tmp_path):
    module = _load("profile")
    onnx = tmp_path / "model.onnx"
    engine = tmp_path / "model.engine"
    onnx.write_bytes(b"onnx")
    engine.write_bytes(b"engine")
    onnx.with_suffix(".json").write_text(json.dumps(_model_sidecar()))
    Path(f"{engine}.json").write_text(
        json.dumps(
            _profile_engine_sidecar(
                variant="s",
                num_classes=3,
                eval_idx=2,
                num_queries=200,
                precision="fp32",
                precision_mode="fp32",
                network_typing="weak",
                onnx_sha256=module.sha256_file(onnx),
            )
        )
    )

    with pytest.raises(RuntimeError, match="queries"):
        module.validate_backend_contracts(_profile_contract_args(module, tmp_path, onnx, engine))


@pytest.mark.parametrize("backend", ["trt", "cpp", "cpp-graph"])
def test_profile_rejects_engine_not_built_from_selected_onnx(tmp_path, backend):
    module = _load("profile")
    onnx = tmp_path / "model.onnx"
    engine = tmp_path / "model.engine"
    onnx.write_bytes(b"onnx")
    engine.write_bytes(b"engine")
    onnx.with_suffix(".json").write_text(json.dumps(_model_sidecar()))
    Path(f"{engine}.json").write_text(
        json.dumps(
            _profile_engine_sidecar(
                variant="s",
                num_classes=3,
                eval_idx=2,
                precision="fp32",
                precision_mode="fp32",
                network_typing="weak",
                onnx_sha256="c" * 64,
            )
        )
    )

    with pytest.raises(RuntimeError, match="does not match the selected ONNX"):
        module.validate_backend_contracts(
            _profile_contract_args(module, tmp_path, onnx, engine, backend)
        )


def test_profile_rejects_incomplete_engine_build_contract(tmp_path):
    module = _load("profile")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    metadata = _profile_engine_sidecar()
    del metadata["sm_arch"]
    Path(f"{engine}.json").write_text(json.dumps(metadata))
    args = module.parse_args(
        [
            "--backends",
            "trt",
            "--engine",
            str(engine),
            "--no-accuracy",
            "--sample-image",
            str(tmp_path / "sample.jpg"),
            "--batches",
            "1",
        ]
    )

    with pytest.raises(RuntimeError, match="missing: sm_arch"):
        module.validate_backend_contracts(args)


def test_profile_binds_active_cuda_device_to_gpu_uuid():
    module = _load("profile")
    properties = SimpleNamespace(
        name="NVIDIA RTX",
        uuid="1234",
        total_memory=1024,
        major=8,
        minor=9,
    )
    module.torch.cuda = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _index: properties,
    )
    physical = {
        "index": 2,
        "name": "NVIDIA RTX",
        "uuid": "GPU-1234",
        "memory_mib": 1,
        "driver_version": "1",
    }

    identity = module.active_gpu_identity({"nvidia_gpus": [physical]})

    assert identity["logical_cuda_index"] == 0
    assert identity["uuid"] == "GPU-1234"
    assert identity["nvidia_smi"] == physical


def test_native_gpu_identity_uses_environment_without_torch_cuda():
    module = _load("profile")
    physical = {
        "index": 2,
        "name": "NVIDIA RTX",
        "uuid": "GPU-1234",
        "memory_mib": 16376,
        "driver_version": "1",
    }

    identity = module.native_gpu_identity(
        {"cuda_visible_devices": "2", "nvidia_gpus": [physical]}
    )

    assert identity == {
        "logical_cuda_index": 0,
        "name": "NVIDIA RTX",
        "uuid": "GPU-1234",
        "memory_bytes": 16376 * 1024 * 1024,
        "nvidia_smi": physical,
    }


def test_native_only_profile_does_not_initialize_parent_cuda(monkeypatch, tmp_path):
    module = _load("profile")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    Path(f"{engine}.json").write_text(json.dumps(_profile_engine_sidecar()))
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"sample")
    args = module.parse_args(
        [
            "--backends",
            "cpp",
            "--engine",
            str(engine),
            "--no-accuracy",
            "--sample-image",
            str(sample),
            "--batches",
            "1",
        ]
    )
    physical = {
        "index": 0,
        "name": "NVIDIA RTX",
        "uuid": "GPU-1234",
        "memory_mib": 16376,
        "driver_version": "1",
    }
    monkeypatch.setattr(
        module,
        "environment_metadata",
        lambda: {"cuda_visible_devices": None, "nvidia_gpus": [physical]},
    )
    monkeypatch.setattr(
        module,
        "active_gpu_identity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("initialized parent CUDA")),
    )
    monkeypatch.setattr(
        module,
        "gpu_used_mib",
        lambda: (_ for _ in ()).throw(AssertionError("initialized parent CUDA")),
    )
    monkeypatch.setattr(module, "_native_runtime_artifacts", lambda *_args: {"verified": True})
    monkeypatch.setattr(
        module,
        "latency_cpp",
        lambda *_args, **_kwargs: {
            1: {
                "scopes": {"end_to_end": {"p50": 2.0, "img_per_s": 500.0}},
                "gpu_mem_mib": 128.0,
                "native_stages": {},
            }
        },
    )
    monkeypatch.setattr(module.cv2, "IMREAD_COLOR", 1, raising=False)
    monkeypatch.setattr(
        module.cv2,
        "imread",
        lambda *_args: SimpleNamespace(shape=(480, 640, 3)),
        raising=False,
    )

    assert module.main(args) == 0


def _native_report(engine="model.engine", *, graph=False, rows=None):
    return {
        "engine": engine,
        "input": [640, 640],
        "cuda_graph": graph,
        "cuda_graph_required": graph,
        "results": rows if rows is not None else [],
    }


def test_cpp_latency_requires_every_requested_batch(monkeypatch, tmp_path):
    module = _load("profile")

    def incomplete_benchmark(command, **_kwargs):
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps(_native_report(rows=[{"batch": 1}], engine="model.engine"))
        )

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
                _native_report(
                    graph=True,
                    rows=[{"batch": 1, "cuda_graph_replay": False}],
                )
            )
        )

    monkeypatch.setattr(module.subprocess, "run", fallback_benchmark)

    with pytest.raises(RuntimeError, match="did not confirm CUDA Graph replay"):
        module.latency_cpp(
            "model.engine", [1], 0, 1, {}, tmp_path, image="frame.jpg", cuda_graph=True
        )
    assert "--require-cuda-graph" in commands[0]
    assert commands[0][commands[0].index("--threshold") + 1] == "0.001"


def _native_latency_row(*, graph=False):
    return {
        "batch": 1,
        "cuda_graph_replay": graph,
        "total_p50": 2.0,
        "total_mean": 2.1,
        "total_p90": 2.2,
        "total_p99": 2.4,
        "preprocess_p50": 0.3,
        "infer_p50": 1.4,
        "d2h_p50": 0.2,
        "decode_p50": 0.1,
        "img_per_s": 500.0,
        "gpu_mem_mib": 128,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("engine", "other.engine", "requested engine"),
        ("input", [320, 320], "measured input"),
        ("cuda_graph", True, "CUDA Graph mode"),
        ("total_p90", 1.0, "latency percentiles"),
        ("img_per_s", 400.0, "inconsistent with median latency"),
    ],
)
def test_cpp_latency_rejects_inconsistent_native_report(
    monkeypatch, tmp_path, field, value, match
):
    module = _load("profile")
    row = _native_latency_row()
    report = _native_report(rows=[row])
    if field in report:
        report[field] = value
    else:
        row[field] = value

    def benchmark(command, **_kwargs):
        output = Path(command[command.index("--json") + 1])
        output.write_text(json.dumps(report))

    monkeypatch.setattr(module.subprocess, "run", benchmark)
    with pytest.raises(RuntimeError, match=match):
        module.latency_cpp(
            "model.engine",
            [1],
            0,
            1,
            {},
            tmp_path,
            image="frame.jpg",
            input_wh=(640, 640),
        )


def test_cpp_latency_contract_against_real_dfine_bench(tmp_path):
    """End-to-end guard for the dfine_bench JSON emitter <-> latency_cpp parser contract.

    The fabricated-report tests above validate the parser against synthetic JSON;
    this one validates it against the real native emitter, so a field rename or
    semantic drift in apps/dfine_bench.cpp fails here instead of only on a live run.
    """
    engine = os.environ.get("DFINE_TEST_ENGINE")
    if not engine:
        pytest.skip("set DFINE_TEST_ENGINE to a dynamic-batch engine (and LD_LIBRARY_PATH)")
    module = _load("profile")
    if not module.NATIVE_BENCH.exists():
        pytest.skip("build/dfine_bench not built")

    result = module.latency_cpp(
        engine,
        [1],
        5,
        20,
        os.environ.copy(),
        tmp_path,
        rounds=2,
    )

    batch = result[1]
    e2e = batch["scopes"]["end_to_end"]
    assert e2e["img_per_s"] > 0
    assert e2e["p50"] > 0
    assert len(e2e["rounds"]) == 2
    assert e2e["aggregation"] == "median_across_independent_rounds"
    assert batch["scopes"]["trt_engine_device"]["p50"] > 0
    assert batch["native_stages"]["trt_engine_device"]["p50"] > 0
    assert batch["gpu_mem_mib"] > 0


def test_dfine_bench_rejects_sidecar_batch_profile_conflict(tmp_path):
    """The native binary must refuse an engine whose sidecar asserts a different profile."""
    engine = os.environ.get("DFINE_TEST_ENGINE")
    if not engine:
        pytest.skip("set DFINE_TEST_ENGINE to a dynamic-batch engine (and LD_LIBRARY_PATH)")
    module = _load("profile")
    if not module.NATIVE_BENCH.exists():
        pytest.skip("build/dfine_bench not built")
    sidecar = Path(engine).with_suffix(".json")
    if not sidecar.exists():
        pytest.skip("test engine has no adjacent sidecar to corrupt")

    staged_engine = tmp_path / "model.engine"
    staged_engine.write_bytes(Path(engine).read_bytes())
    meta = json.loads(sidecar.read_text())
    meta["max_batch"] = int(meta["max_batch"]) + 1
    (tmp_path / "model.json").write_text(json.dumps(meta))

    proc = subprocess.run(
        [
            str(module.NATIVE_BENCH),
            "--engine",
            str(staged_engine),
            "--batches",
            "1",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--json",
            str(tmp_path / "out.json"),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "contradicts" in (proc.stderr + proc.stdout)


def test_cpp_latency_keeps_engine_and_transfer_stages_distinct(monkeypatch, tmp_path):
    module = _load("profile")

    def benchmark(command, **_kwargs):
        output = Path(command[command.index("--json") + 1])
        output.write_text(json.dumps(_native_report(rows=[_native_latency_row()])))

    monkeypatch.setattr(module.subprocess, "run", benchmark)
    result = module.latency_cpp("model.engine", [1], 0, 1, {}, tmp_path, image="frame.jpg")[1]

    assert set(result["scopes"]) == {"end_to_end", "trt_engine_device"}
    assert result["scopes"]["trt_engine_device"]["p50"] == 1.4
    assert result["scopes"]["trt_engine_device"]["rounds"] == [{"p50": 1.4}]
    assert result["native_stages"]["device_to_host"]["p50"] == 0.2
    assert "infer_p50" not in result


def test_cpp_latency_aggregates_three_process_rounds(monkeypatch, tmp_path):
    module = _load("profile")
    calls = 0

    def benchmark(command, **_kwargs):
        nonlocal calls
        value = (3.0, 1.0, 2.0)[calls]
        calls += 1
        row = _native_latency_row()
        row.update(
            total_p50=value,
            total_mean=value + 0.1,
            total_p90=value + 0.2,
            total_p99=value + 0.4,
            img_per_s=1000.0 / value,
        )
        output = Path(command[command.index("--json") + 1])
        output.write_text(json.dumps(_native_report(rows=[row])))

    monkeypatch.setattr(module.subprocess, "run", benchmark)
    result = module.latency_cpp(
        "model.engine", [1], 0, 1, {}, tmp_path, image="frame.jpg", rounds=3
    )[1]

    scope = result["scopes"]["end_to_end"]
    assert calls == 3
    assert scope["p50"] == 2.0
    assert [row["p50"] for row in scope["rounds"]] == [3.0, 1.0, 2.0]
    assert scope["aggregation"] == module.ROUND_AGGREGATION


def test_native_runtime_provenance_hashes_executable_and_loaded_library(monkeypatch, tmp_path):
    module = _load("profile")
    executable = tmp_path / "dfine_bench"
    library = tmp_path / "libdfine.so.1"
    executable.write_bytes(b"executable")
    library.write_bytes(b"library")
    result = SimpleNamespace(
        returncode=0,
        stdout=f"libdfine.so.1 => {library} (0x00000000)\n",
        stderr="",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: result)

    provenance = module._native_runtime_artifacts([executable], {})

    assert provenance["executables"][0]["sha256"] == module.sha256_file(executable)
    assert provenance["libdfine"]["sha256"] == module.sha256_file(library)


def test_cpp_graph_latency_labels_captured_d2h(monkeypatch, tmp_path):
    module = _load("profile")

    def benchmark(command, **_kwargs):
        output = Path(command[command.index("--json") + 1])
        output.write_text(
            json.dumps(_native_report(graph=True, rows=[_native_latency_row(graph=True)]))
        )

    monkeypatch.setattr(module.subprocess, "run", benchmark)
    result = module.latency_cpp(
        "model.engine", [1], 0, 1, {}, tmp_path, image="frame.jpg", cuda_graph=True
    )[1]

    assert set(result["scopes"]) == {"end_to_end", "cuda_graph_engine_d2h"}
    assert "trt_engine_device" not in result["native_stages"]
    assert "device_to_host" not in result["native_stages"]


def test_python_framework_latency_does_not_claim_device_engine_scope(monkeypatch):
    module = _load("profile")

    class Backend:
        def __call__(self, _input):
            return "logits", "boxes"

    monkeypatch.setattr(module, "preprocess", lambda *_args: "input")
    monkeypatch.setattr(module, "_validate_outputs", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_wall_latency",
        lambda *_args: {"p50": 1.0, "p90": 1.1, "p99": 1.2, "img_per_s": 1000.0},
    )

    result = module.latency_python(Backend(), object(), 1, 0, 1, 640, 480, 80, 300, 640)

    assert set(result["scopes"]) == {"end_to_end", "backend_call_host_to_host"}
    assert "trt_engine_device" not in result["scopes"]


def test_cuda_events_are_created_before_timed_enqueues(monkeypatch):
    module = _load("profile")
    allocated = 0
    calls = 0

    class Event:
        def __init__(self, **_kwargs):
            nonlocal allocated
            allocated += 1

        def record(self, _stream):
            pass

        def elapsed_time(self, _other):
            return 1.0

    class Stream:
        def synchronize(self):
            pass

    module.torch.cuda = SimpleNamespace(Event=Event)

    def enqueue():
        nonlocal calls
        calls += 1
        if calls > 1:
            assert allocated == 6

    result = module._cuda_latency(Stream(), enqueue, warmup=1, iters=3, batch=2)

    assert calls == 4
    assert result["p50"] == 1.0
    assert result["img_per_s"] == 2000.0


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [-0.1]])
def test_latency_summary_rejects_invalid_samples(monkeypatch, samples):
    module = _load("profile")

    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        module.percentiles(samples)


def test_overnight_report_reads_named_latency_scopes(tmp_path):
    def entry(name):
        scopes = {"end_to_end": {"p50": 9.0, "img_per_s": 111.1}}
        if name in ("torch", "onnx"):
            scopes["backend_call_host_to_host"] = {"p50": 6.0}
        elif name == "trt":
            scopes["trt_transfer_inclusive"] = {"p50": 4.0}
            scopes["trt_engine_device"] = {"p50": 3.5}
        else:
            scopes["trt_engine_device"] = {"p50": 3.4}
        row = {"scopes": scopes, "gpu_mem_mib": 512}
        return {"latency": {"1": row, "8": row}, "map": {"AP": 0.5}}

    report = {
        "schema": 2,
        "backends": {name: entry(name) for name in ("torch", "onnx", "trt", "cpp")},
    }
    (tmp_path / "cmp_n_fp32.json").write_text(json.dumps(report))
    (tmp_path / "cmp_n_fp16.json").write_text(json.dumps(report))
    shell = (SCRIPTS / "overnight_bench.sh").read_text()
    consolidator = shell.split("<<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
    env = {
        **os.environ,
        "OUTDIR": str(tmp_path),
        "SIZES": "n",
        "SUBSET_ARGS": "--subset 1",
        "CM_LIMIT": "0",
    }

    subprocess.run([sys.executable, "-c", consolidator], check=True, env=env)
    rendered = (tmp_path / "REPORT.md").read_text()

    assert "| forward scope | e2e b1 | forward b1 | TRT device b1 |" in rendered
    assert "| TensorRT FP16 (py) | H2D + engine + D2H | 9.00 | 4.00 | 3.50 |" in rendered
    assert "| PyTorch (FP32) | backend call, host→host | 9.00 | 6.00 | — |" in rendered
    assert "infer b1" not in rendered
