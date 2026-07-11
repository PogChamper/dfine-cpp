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
    trt.__version__ = "10.test"
    trt.LayerType = types.SimpleNamespace(
        CONSTANT=0,
        SHAPE=1,
        ASSERTION=2,
        IDENTITY=3,
    )
    trt.DataType = types.SimpleNamespace(FLOAT="float")
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


def test_shared_stem_reserves_absent_onnx_sidecar(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"

    chosen, stale = build_engine._engine_sidecar_plan(onnx, tmp_path / "model.engine")

    assert chosen == tmp_path / "model.engine.json"
    assert stale is None


def test_sm_arch_uses_cuda_runtime_device(build_engine, monkeypatch):
    class CudaRuntime:
        @staticmethod
        def cudaGetDevice(device):
            device._obj.value = 0
            return 0

        @staticmethod
        def cudaDeviceGetAttribute(value, attribute, device):
            assert device == 0
            value._obj.value = {75: 12, 76: 0}[attribute]
            return 0

    monkeypatch.setattr(build_engine.ctypes, "CDLL", lambda _: CudaRuntime())
    assert build_engine._sm_arch() == "120"


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


@pytest.mark.parametrize(
    ("onnx_name", "engine_name", "message"),
    [
        ("model.engine.tmp", "model.engine", "--onnx must end in .onnx"),
        ("model.onnx", "model.json", "--output must end in .engine"),
    ],
)
def test_artifact_plan_rejects_unsafe_suffixes(
    build_engine,
    tmp_path,
    onnx_name,
    engine_name,
    message,
):
    with pytest.raises(SystemExit, match=message):
        build_engine._validated_artifact_plan(
            tmp_path / onnx_name,
            tmp_path / engine_name,
        )


def test_onnx_metadata_requires_model_rgb(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    sidecar = tmp_path / "model.json"

    selected, meta, found = build_engine._load_onnx_metadata(onnx)
    assert selected == sidecar
    assert meta == {}
    assert found is False

    sidecar.write_text('{"artifact_kind": "onnx", "color_order": "RGB"}')

    selected, meta, found = build_engine._load_onnx_metadata(onnx)

    assert selected == sidecar
    assert meta["color_order"] == "RGB"
    assert found is True

    sidecar.write_text('{"artifact_kind": "onnx", "color_order": "BGR"}')
    with pytest.raises(SystemExit, match="model input must be RGB"):
        build_engine._load_onnx_metadata(onnx)


def test_onnx_artifact_snapshot_loads_graph_and_sidecar(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"graph snapshot")
    (tmp_path / "model.json").write_text('{"artifact_kind": "onnx"}')

    sidecar, meta, found, graph = build_engine._load_onnx_artifact(onnx)

    assert sidecar == tmp_path / "model.json"
    assert meta["artifact_kind"] == "onnx"
    assert found is True
    assert graph == b"graph snapshot"


def test_build_rejects_model_bgr_before_tensorrt_setup(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    (tmp_path / "model.json").write_text('{"color_order": "BGR"}')
    args = types.SimpleNamespace(onnx=str(onnx), output=str(tmp_path / "model.engine"))

    with pytest.raises(SystemExit, match="mark BGR source images at runtime"):
        build_engine.build(args)


def test_onnx_metadata_rejects_engine_sidecar(build_engine, tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    (tmp_path / "model.json").write_text('{"artifact_kind": "engine"}')

    with pytest.raises(SystemExit, match="expected 'onnx'"):
        build_engine._load_onnx_metadata(onnx)


@pytest.mark.parametrize(
    ("max_aux_streams", "cuda_graph_alias", "expected"),
    [
        (None, False, {"max_aux_streams": None}),
        (0, False, {"max_aux_streams": 0}),
        (3, False, {"max_aux_streams": 3}),
        (None, True, {"max_aux_streams": 0}),
        (0, True, {"max_aux_streams": 0}),
    ],
)
def test_graph_policy(build_engine, max_aux_streams, cuda_graph_alias, expected):
    assert build_engine._graph_policy(max_aux_streams, cuda_graph_alias) == expected


def test_graph_policy_rejects_conflict(build_engine):
    with pytest.raises(SystemExit, match="conflicts"):
        build_engine._graph_policy(1, True)


@pytest.mark.parametrize("cuda_graph_alias", [False, True])
def test_graph_policy_rejects_negative_limit(build_engine, cuda_graph_alias):
    with pytest.raises(SystemExit, match="non-negative"):
        build_engine._graph_policy(-1, cuda_graph_alias)


@pytest.mark.parametrize(
    "profile",
    [
        (0, 1, 8),
        (1, 0, 8),
        (2, 1, 8),
        (1, 9, 8),
        (1, 1, 0),
    ],
)
def test_batch_profile_rejects_invalid_order(build_engine, profile):
    with pytest.raises(SystemExit, match="1 <= min <= opt <= max"):
        build_engine._validate_batch_profile(*profile)


def test_batch_profile_accepts_ordered_bounds(build_engine):
    build_engine._validate_batch_profile(1, 4, 8)


@pytest.mark.parametrize("limit", [None, 0, 3])
def test_graph_policy_applies_config_and_metadata(
    build_engine,
    monkeypatch,
    tmp_path,
    limit,
):
    policy = build_engine._graph_policy(limit, False)
    config = types.SimpleNamespace()

    assert build_engine._apply_graph_policy(config, policy) == limit
    if limit is None:
        assert not hasattr(config, "max_aux_streams")
    else:
        assert config.max_aux_streams == limit

    args = types.SimpleNamespace(
        strongly_typed=True,
        no_tf32=True,
        min_batch=1,
        opt_batch=4,
        max_batch=8,
    )
    monkeypatch.setattr(build_engine, "_sm_arch", lambda: "89")
    facts = build_engine._engine_build_facts(args, "a" * 64, policy, outputs_fp32=True)

    assert facts["artifact_kind"] == "engine"
    assert facts["max_aux_streams"] == limit
    assert facts["cuda_graph_compat"] is (limit == 0)
    assert facts["onnx_sha256"] == "a" * 64

    fp16_facts = build_engine._engine_build_facts(args, "a" * 64, policy, outputs_fp32=False)
    assert fp16_facts["cuda_graph_compat"] is False


def test_graph_compat_ignores_unconsumed_outputs(build_engine):
    class Network:
        def __init__(self, outputs):
            self.outputs = outputs
            self.num_outputs = len(outputs)

        def get_output(self, index):
            return self.outputs[index]

    def tensor(name, dtype, shape):
        return types.SimpleNamespace(name=name, dtype=dtype, shape=shape)

    network = Network(
        [
            tensor("logits", "float", (-1, 300, 80)),
            tensor("boxes", "float", (-1, 300, 4)),
            tensor("debug", "half", (-1, 1)),
        ]
    )

    assert build_engine._graph_outputs_are_fp32(network, {})
    assert not build_engine._graph_outputs_are_fp32(network, {"output_names": ["logits", "debug"]})


def test_failed_engine_publish_preserves_stale_sidecar(build_engine, tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"old-engine")
    sidecar = tmp_path / "model.json"
    sidecar.write_text('{"version": 1}')
    stale = tmp_path / "model.engine.json"
    stale.write_text('{"version": 1}')

    with pytest.raises(FileNotFoundError):
        build_engine._publish_engine_pair(
            tmp_path / "missing.engine.tmp",
            engine,
            '{"version": 2}',
            sidecar,
            stale,
        )

    assert engine.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"version": 1}'
    assert stale.read_text() == '{"version": 1}'


def test_successful_engine_publish_removes_stale_sidecar(build_engine, tmp_path):
    staged = tmp_path / "model.engine.tmp"
    staged.write_bytes(b"new-engine")
    engine = tmp_path / "model.engine"
    sidecar = tmp_path / "model.json"
    stale = tmp_path / "model.engine.json"
    stale.write_text('{"version": 1}')

    build_engine._publish_engine_pair(
        staged,
        engine,
        '{"version": 2}',
        sidecar,
        stale,
    )

    assert engine.read_bytes() == b"new-engine"
    assert sidecar.read_text() == '{"version": 2}'
    assert not stale.exists()


def test_stale_twin_failure_restores_previous_engine_pair(build_engine, monkeypatch, tmp_path):
    staged = tmp_path / "model.engine.tmp"
    staged.write_bytes(b"new-engine")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"old-engine")
    sidecar = tmp_path / "model.json"
    sidecar.write_text('{"version": 1}')
    stale = tmp_path / "model.engine.json"
    stale.write_text('{"version": 1}')
    real_unlink = Path.unlink
    failed = False

    def fail_stale_once(path, *args, **kwargs):
        nonlocal failed
        if path == stale and not failed:
            failed = True
            raise OSError("injected stale-sidecar removal failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stale_once)
    with pytest.raises(OSError, match="injected stale-sidecar"):
        build_engine._publish_engine_pair(
            staged,
            engine,
            '{"version": 2}',
            sidecar,
            stale,
        )

    assert engine.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"version": 1}'
    assert stale.read_text() == '{"version": 1}'
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
