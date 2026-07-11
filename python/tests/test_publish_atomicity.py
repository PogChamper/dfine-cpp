"""Exercise every self-contained artifact producer's publication helper."""

from __future__ import annotations

import hashlib
import importlib.util
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Module-level imports each script pulls in; the fixture importorskips them so
# an environment without the heavy deps SKIPS (CI runs the surgical/fp16 copies;
# int8/build_engine run wherever tensorrt/pycocotools live).
SCRIPTS = {
    "convert_fp16_surgical": ["onnx", "onnxconverter_common"],
    "convert_fp16": ["onnx", "numpy", "onnxconverter_common"],
    "convert_int8": ["onnx", "numpy", "onnxruntime", "cv2", "torch", "tensorrt", "pycocotools"],
    "build_engine": ["tensorrt"],
    "export_dfine_onnx": ["torch"],
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


def test_adjacent_temps_are_unique_and_preserve_mode(publish, tmp_path):
    adjacent_temp = publish.__globals__["_adjacent_temp"]
    target = tmp_path / "g.onnx"
    target.write_bytes(b"old")
    target.chmod(0o640)

    first = adjacent_temp(target)
    second = adjacent_temp(target)
    try:
        assert first != second
        assert first.parent == target.parent and second.parent == target.parent
        assert first.stat().st_mode & 0o777 == 0o640
        assert second.stat().st_mode & 0o777 == 0o640
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_read_only_outputs_can_be_replaced_without_changing_mode(publish, tmp_path):
    adjacent_temp = publish.__globals__["_adjacent_temp"]
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    out.write_bytes(b"graph-v1")
    sc.write_text('{"v": 1}')
    out.chmod(0o444)
    sc.chmod(0o400)
    staged = adjacent_temp(out)
    staged.write_bytes(b"graph-v2")

    publish(staged, out, '{"v": 2}', sc, "test")

    assert out.read_bytes() == b"graph-v2" and sc.read_text() == '{"v": 2}'
    assert out.stat().st_mode & 0o777 == 0o444
    assert sc.stat().st_mode & 0o777 == 0o400


def test_second_swap_failure_restores_previous_pair(publish, monkeypatch, tmp_path):
    module_os = publish.__globals__["os"]
    real_replace = module_os.replace
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    out.write_bytes(b"graph-v1")
    sc.write_text('{"v": 1}')
    staged = tmp_path / "new.onnx.tmp"
    staged.write_bytes(b"graph-v2")
    failed = False

    def fail_sidecar_once(source, destination):
        nonlocal failed
        if Path(destination) == sc and not failed:
            failed = True
            raise OSError("injected sidecar swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module_os, "replace", fail_sidecar_once)
    with pytest.raises(OSError, match="injected sidecar"):
        publish(staged, out, '{"v": 2}', sc, "test")

    assert out.read_bytes() == b"graph-v1"
    assert sc.read_text() == '{"v": 1}'
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_second_swap_failure_without_previous_pair_removes_output(publish, monkeypatch, tmp_path):
    module_os = publish.__globals__["os"]
    real_replace = module_os.replace
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    staged = tmp_path / "new.onnx.tmp"
    staged.write_bytes(b"graph-v2")
    failed = False

    def fail_sidecar_once(source, destination):
        nonlocal failed
        if Path(destination) == sc and not failed:
            failed = True
            raise OSError("injected sidecar swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module_os, "replace", fail_sidecar_once)
    with pytest.raises(OSError, match="injected sidecar"):
        publish(staged, out, '{"v": 2}', sc, "test")

    assert not out.exists() and not sc.exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_concurrent_publishers_cannot_mix_pairs(publish, monkeypatch, tmp_path):
    module_os = publish.__globals__["os"]
    module_fcntl = publish.__globals__["fcntl"]
    real_replace = module_os.replace
    real_flock = module_fcntl.flock
    out, sc = tmp_path / "g.onnx", tmp_path / "g.json"
    first_graph_swapped = threading.Event()
    second_lock_attempted = threading.Event()
    release_first = threading.Event()
    delayed = False

    def delay_first_publisher(source, destination):
        nonlocal delayed
        result = real_replace(source, destination)
        if (
            threading.current_thread().name == "publisher-a"
            and Path(destination) == out
            and not delayed
        ):
            delayed = True
            first_graph_swapped.set()
            assert release_first.wait(timeout=5)
        return result

    monkeypatch.setattr(module_os, "replace", delay_first_publisher)

    def observe_lock(fd, operation):
        if threading.current_thread().name == "publisher-b" and operation & module_fcntl.LOCK_EX:
            second_lock_attempted.set()
        return real_flock(fd, operation)

    monkeypatch.setattr(module_fcntl, "flock", observe_lock)
    errors = []

    def run(label):
        try:
            staged = tmp_path / f"{label}.onnx.tmp"
            staged.write_text(label)
            publish(staged, out, label, sc, "test")
        except BaseException as exc:  # pragma: no cover - reported by the parent thread
            errors.append(exc)

    first = threading.Thread(target=run, args=("a",), name="publisher-a")
    second = threading.Thread(target=run, args=("b",), name="publisher-b")
    first.start()
    assert first_graph_swapped.wait(timeout=5)
    second.start()
    assert second_lock_attempted.wait(timeout=5)
    assert second.is_alive()
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert (out.read_text(), sc.read_text()) == ("b", "b")
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


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


@pytest.mark.parametrize(
    "script_name",
    ["convert_fp16", "convert_fp16_surgical", "convert_int8"],
)
def test_converters_reject_artifact_path_collisions(script_name, tmp_path):
    converter = load(script_name)
    source = tmp_path / "model.onnx"

    with pytest.raises(SystemExit, match="must end in .onnx"):
        converter._validated_conversion_paths(source, tmp_path / "model.json")
    with pytest.raises(SystemExit, match="artifact path collision"):
        converter._validated_conversion_paths(source, source)


def test_surgical_overrides_are_not_labeled_as_release_recipe(monkeypatch):
    converter = load("convert_fp16_surgical")
    monkeypatch.setattr(
        converter.importlib.metadata,
        "version",
        lambda distribution: "1.14.0" if distribution == "onnxconverter-common" else None,
    )

    converted = converter._converted_sidecar(
        {"tool_versions": {}},
        b"source",
        slim=True,
        overrides={"fp16_only_scopes": ["cross_attn"]},
    )

    assert converted["precision_mode"] == "strongly_typed_onnx_fp16_surgical_experimental"
    assert converted["conversion_overrides"] == {"fp16_only_scopes": ["cross_attn"]}


@pytest.mark.parametrize("slim", [False, True])
def test_surgical_sidecar_records_conversion_provenance(monkeypatch, slim):
    converter = load("convert_fp16_surgical")
    monkeypatch.setattr(
        converter.importlib.metadata,
        "version",
        lambda distribution: "1.14.0" if distribution == "onnxconverter-common" else None,
    )
    source = b"source ONNX bytes"
    original = {"tool_versions": {"onnx": "1.20.0"}, "precision": "fp32"}

    converted = converter._converted_sidecar(original, source, slim)

    assert original == {"tool_versions": {"onnx": "1.20.0"}, "precision": "fp32"}
    assert converted["source_onnx_sha256"] == hashlib.sha256(source).hexdigest()
    assert (
        converted["converter_sha256"]
        == hashlib.sha256(Path(converter.__file__).read_bytes()).hexdigest()
    )
    assert converted["tool_versions"] == {
        "onnx": "1.20.0",
        "onnxconverter-common": "1.14.0",
    }
    assert converted["precision"] == "fp16"
    suffix = "slim" if slim else "decoder"
    assert converted["precision_mode"] == f"strongly_typed_onnx_fp16_surgical_{suffix}"


def test_surgical_sidecar_rejects_invalid_tool_versions():
    converter = load("convert_fp16_surgical")

    with pytest.raises(SystemExit, match="tool_versions"):
        converter._converted_sidecar({"tool_versions": []}, b"source", slim=True)


def test_surgical_loads_and_hashes_one_source_snapshot(monkeypatch):
    converter = load("convert_fp16_surgical")

    class Source:
        reads = 0

        def read_bytes(self):
            self.reads += 1
            return b"source ONNX bytes"

    source = Source()
    model = object()
    monkeypatch.setattr(
        converter.onnx,
        "load_model_from_string",
        lambda source_bytes: model if source_bytes == b"source ONNX bytes" else None,
    )

    loaded, source_bytes = converter._load_source_onnx(source)

    assert loaded is model
    assert source_bytes == b"source ONNX bytes"
    assert source.reads == 1
