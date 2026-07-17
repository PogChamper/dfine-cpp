from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_module():
    return _load_module("evaluation_report_test", "trt-files/scripts/evaluation_report.py")


class TinyCoco:
    def __init__(self, images):
        self.images = {image["id"]: image for image in images}
        self.dataset = {"info": {"dataset": "tiny", "split": "test", "version": "1"}}

    def loadImgs(self, image_id):
        return [self.images[image_id]]


def test_image_manifest_hashes_file_content(report_module, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    path = images / "one.jpg"
    path.write_bytes(b"first")
    coco = TinyCoco([{"id": 1, "file_name": path.name}])

    first = report_module.selected_image_manifest(coco, [1], images)
    path.write_bytes(b"other")
    second = report_module.selected_image_manifest(coco, [1], images)

    assert first["image_count"] == 1
    assert first["image_manifest_sha256"] != second["image_manifest_sha256"]


def test_evaluation_contract_records_exact_geometry(report_module, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"image")
    annotations = tmp_path / "instances.json"
    annotations.write_text("{}")
    coco = TinyCoco([{"id": 1, "file_name": "one.jpg"}])

    contract = report_module.evaluation_contract(
        coco,
        [1],
        images,
        annotations,
        score_threshold=0.001,
        topk=300,
        inference_batch_size=4,
        model_hw=(480, 640),
        metrics_source=REPO / "trt-files/scripts/coco_metrics.py",
    )

    assert contract["geometry"]["model_space_area"] == {
        "input_h": 480,
        "input_w": 640,
        "resize": "stretch",
    }
    assert contract["inference_batch_size"] == 4


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_evaluation_contract_rejects_invalid_batch(report_module, tmp_path, batch):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"image")
    annotations = tmp_path / "instances.json"
    annotations.write_text("{}")
    coco = TinyCoco([{"id": 1, "file_name": "one.jpg"}])

    with pytest.raises(ValueError, match="inference batch size"):
        report_module.evaluation_contract(
            coco,
            [1],
            images,
            annotations,
            score_threshold=0.001,
            topk=300,
            inference_batch_size=batch,
            model_hw=(640, 640),
            metrics_source=REPO / "trt-files/scripts/coco_metrics.py",
        )


def test_model_space_geometry_is_stretch_only(report_module):
    assert report_module.model_space_geometry((640, 640)) == {
        "input_h": 640,
        "input_w": 640,
        "resize": "stretch",
    }
    with pytest.raises(ValueError, match="positive integers"):
        report_module.model_space_geometry((640, 0))


def test_atomic_json_rejects_alias_and_nonfinite(report_module, tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="aliases"):
        report_module.atomic_json(source, {"ok": True}, protected=[source])
    with pytest.raises(ValueError, match="Out of range"):
        report_module.atomic_json(tmp_path / "report.json", {"metric": float("nan")})


def test_atomic_json_does_not_clobber_without_opt_in(report_module, tmp_path):
    output = tmp_path / "report.json"
    report_module.atomic_json(output, {"value": 1})

    with pytest.raises(ValueError, match="already exists"):
        report_module.atomic_json(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}

    report_module.atomic_json(output, {"value": 2}, overwrite=True)
    assert json.loads(output.read_text()) == {"value": 2}


def test_artifact_records_graph_and_sidecar(report_module, tmp_path):
    graph = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.json"
    graph.write_bytes(b"graph")
    sidecar.write_text(json.dumps({"schema_version": 1}))

    record = report_module.artifact(
        "onnx",
        graph,
        recipe="explicit-fp32",
        runtime="ONNX Runtime 1.24",
        sidecar=sidecar,
    )

    assert record["kind"] == "onnx"
    assert record["path"] == str(graph.resolve())
    assert record["recipe"] == "explicit-fp32"
    assert len(record["sha256"]) == 64
    assert record["sidecar"]["path"] == str(sidecar.resolve())


def test_tensorrt_runtime_uses_installed_distribution_alias(report_module):
    if importlib.util.find_spec("tensorrt") is None:
        pytest.skip("TensorRT is not installed")

    assert "unknown" not in report_module.package_runtime("TensorRT", "tensorrt")
    assert report_module.environment_metadata()["tensorrt"] is not None


def test_environment_records_installed_cuda_runtime_package(report_module, monkeypatch):
    versions = {"nvidia-cuda-runtime-cu12": "12.8.90"}

    def version(package):
        if package not in versions:
            raise report_module.importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(report_module.importlib.metadata, "version", version)
    monkeypatch.setattr(report_module, "gpu_identity", lambda: None)

    assert report_module.environment_metadata()["cuda_runtime_package"] == "12.8.90"


def test_gpu_identity_records_uuid(report_module, monkeypatch):
    result = type(
        "Result",
        (),
        {
            "returncode": 0,
            "stdout": "0, NVIDIA RTX, GPU-1234, 16376, 580.82.09\n",
            "stderr": "",
        },
    )()
    monkeypatch.setattr(report_module.subprocess, "run", lambda *_args, **_kwargs: result)

    assert report_module.gpu_identity() == [
        {
            "index": 0,
            "name": "NVIDIA RTX",
            "uuid": "GPU-1234",
            "memory_mib": 16376,
            "driver_version": "580.82.09",
        }
    ]
