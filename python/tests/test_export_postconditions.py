"""Export postconditions: the dynamic-batch run check must catch a graph whose
batch axis is formally symbolic but internally baked to 1 (the trace-batch-1
defect), and the concrete output shapes must match the sidecar contract.
CPU-only; skips without onnx/onnxruntime."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")

from onnx import TensorProto, helper  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
Q, C, HW = 5, 7, 4  # tiny graph: 5 queries, 7 classes, 4x4 input


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location(
        "export_dfine_onnx", REPO / "trt-files/scripts/export_dfine_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_graph(tmp_path: Path, *, bake_batch: bool) -> Path:
    """images[N,3,4,4] -> flatten -> matmul -> reshape into logits/boxes.

    bake_batch=True reshapes through a literal batch of 1 (the way a batch-1
    trace folds the axis into constants): the graph still DECLARES symbolic
    output batches, runs at N=1, and fails at N=2 — same signature as the real
    defect, caught only by executing the graph.
    """
    feat = 3 * HW * HW
    w_l = helper.make_tensor("w_l", TensorProto.FLOAT, [feat, Q * C],
                             np.zeros(feat * Q * C, np.float32))
    w_b = helper.make_tensor("w_b", TensorProto.FLOAT, [feat, Q * 4],
                             np.zeros(feat * Q * 4, np.float32))
    batch_dim = 1 if bake_batch else -1
    shp_l = helper.make_tensor("shp_l", TensorProto.INT64, [3], [batch_dim, Q, C])
    shp_b = helper.make_tensor("shp_b", TensorProto.INT64, [3], [batch_dim, Q, 4])
    nodes = [
        helper.make_node("Flatten", ["images"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "w_l"], ["ml"]),
        helper.make_node("MatMul", ["flat", "w_b"], ["mb"]),
        helper.make_node("Reshape", ["ml", "shp_l"], ["logits"]),
        helper.make_node("Reshape", ["mb", "shp_b"], ["boxes"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, HW, HW])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", Q, C]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", Q, 4])],
        initializer=[w_l, w_b, shp_l, shp_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    p = tmp_path / ("baked.onnx" if bake_batch else "dynamic.onnx")
    onnx.save(model, p)
    return p


def meta(**over) -> dict:
    base = {"num_queries": Q, "num_classes": C, "input_h": HW, "input_w": HW}
    base.update(over)
    return base


def test_dynamic_graph_passes(exporter, tmp_path):
    exporter._verify_dynamic_batch_runs(tiny_graph(tmp_path, bake_batch=False), meta())


def test_baked_batch_is_caught(exporter, tmp_path):
    with pytest.raises(AssertionError, match="trace-batch"):
        exporter._verify_dynamic_batch_runs(tiny_graph(tmp_path, bake_batch=True), meta())


def test_sidecar_shape_drift_is_caught(exporter, tmp_path):
    good = tiny_graph(tmp_path, bake_batch=False)
    with pytest.raises(AssertionError, match="num-classes"):
        exporter._verify_dynamic_batch_runs(good, meta(num_classes=C + 1))
    with pytest.raises(AssertionError, match="num-classes"):
        exporter._verify_dynamic_batch_runs(good, meta(num_queries=Q - 1))


def test_zero_std_is_rejected(exporter, tmp_path):
    good = onnx.load(tiny_graph(tmp_path, bake_batch=False))
    g = onnx.shape_inference.infer_shapes(good).graph
    with pytest.raises(AssertionError, match="std"):
        exporter._verify_io(g, meta(std=[0.0, 1.0, 1.0]))
