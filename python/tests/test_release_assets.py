"""Release-asset assembly tests with small local fixtures."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
from onnx import TensorProto, checker, helper

REPO = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "release_assets", REPO / "trt-files/scripts/release_assets.py"
)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

VERSION = "0.4.0"
WHEEL = f"dfine-{VERSION}-py3-none-linux_x86_64.whl"
COMMIT = (REPO / "trt-files/DFINE_SEG_REVISION").read_text().strip()


def _write_wheel(path: Path, *, name: str = "dfine", version: str = VERSION) -> None:
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n"
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: test\n"
        "Root-Is-Purelib: false\n"
        "Tag: py3-none-linux_x86_64\n\n"
    )
    dist_info = f"dfine-{VERSION}.dist-info"
    record_path = f"{dist_info}/RECORD"
    members = {
        f"{dist_info}/METADATA": metadata.encode(),
        f"{dist_info}/WHEEL": wheel_metadata.encode(),
        f"{dist_info}/licenses/LICENSE": b"test license\n",
        f"{dist_info}/licenses/NOTICE": b"test notice\n",
        "dfine/__init__.py": b'__version__ = "0.4.0"\n',
        "dfine/libdfine.so": b"\x7fELFtest",
        "dfine/_scripts/build_engine.py": (REPO / "trt-files/scripts/build_engine.py").read_bytes(),
    }
    records = []
    for member, payload in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        records.append(f"{member},sha256={digest},{len(payload)}")
    records.append(f"{record_path},,")
    members[record_path] = ("\n".join(records) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def _sidecar(size: str, precision: str) -> dict:
    metadata = {
        "schema_version": 1,
        "artifact_kind": "onnx",
        "model": "d-fine",
        "variant": size,
        "task": "detect",
        "input_h": 640,
        "input_w": 640,
        "num_classes": 80,
        "num_queries": 300,
        **ra.MODEL_FACTS[size],
        "input_names": ["images"],
        "output_names": ["logits", "boxes"],
        "logits_shape": ["N", 300, 80],
        "boxes_shape": ["N", 300, 4],
        "box_format": "cxcywh_normalized",
        "score_activation": "sigmoid",
        "color_order": "RGB",
        "channel_layout": "NCHW",
        "normalize": "div255",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": "stretch",
        "nms": "none",
        "has_masks": False,
        "dynamic_batch": True,
        "max_batch": 8,
        "trace_batch": 2,
        "precision": precision,
        "precision_mode": (
            "strongly_typed_onnx_fp16_surgical_slim" if precision == "fp16" else "fp32"
        ),
        "opset": 19,
        "deform_core": "explicit",
        "gridsample_nodes": 0,
        "onnx_simplification": "applied",
        "class_names": list(ra.COCO80_NAMES),
        "checkpoint_load": "strict",
        "checkpoint_deserialization": "weights_only",
        "checkpoint_selected_state": "checkpoint root",
        "checkpoint_unused_tensors": 0,
        "checkpoint_loaded_tensors": ra.OFFICIAL_CHECKPOINT_TENSOR_COUNTS[size],
        "checkpoint_sha256": ra.OFFICIAL_CHECKPOINTS[size],
        "exporter_sha256": ra._script_sha256("export_dfine_onnx.py"),
        "model_source": {
            "name": "D-FINE-seg",
            "repository": "https://github.com/ArgoHA/D-FINE-seg",
            "upstream_commit": COMMIT,
            "bundled": True,
            "implementation_sha256": ra._model_source_manifest()["sha256"],
            "implementation_files": ra._model_source_manifest()["files"],
        },
        "tool_versions": {"python": "3.11.0"},
    }
    if precision == "fp16":
        metadata.update(
            source_onnx_sha256="d" * 64,
            converter_sha256=ra._script_sha256("convert_fp16_surgical.py"),
        )
        metadata["tool_versions"]["onnxconverter-common"] = "1.14.0"
    return metadata


def _write_onnx(path: Path, precision: str) -> None:
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 300, 80])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 300, 4])
    batch_index = helper.make_tensor("batch_index", TensorProto.INT64, [1], [0])
    logits_tail = helper.make_tensor("logits_tail", TensorProto.INT64, [2], [300, 80])
    boxes_tail = helper.make_tensor("boxes_tail", TensorProto.INT64, [2], [300, 4])
    internal_type = TensorProto.FLOAT16 if precision == "fp16" else TensorProto.FLOAT
    zero = helper.make_tensor("zero", internal_type, [1], [0.0])
    logits_internal = "logits_fp16" if precision == "fp16" else "logits"
    boxes_internal = "boxes_fp16" if precision == "fp16" else "boxes"
    output_nodes = [
        helper.make_node("ConstantOfShape", ["logits_shape"], [logits_internal], value=zero),
        helper.make_node("ConstantOfShape", ["boxes_shape"], [boxes_internal], value=zero),
    ]
    if precision == "fp16":
        output_nodes.extend(
            [
                helper.make_node("Cast", [logits_internal], ["logits"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [boxes_internal], ["boxes"], to=TensorProto.FLOAT),
            ]
        )
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["images"], ["image_shape"]),
            helper.make_node("Gather", ["image_shape", "batch_index"], ["batch"], axis=0),
            helper.make_node("Concat", ["batch", "logits_tail"], ["logits_shape"], axis=0),
            helper.make_node("Concat", ["batch", "boxes_tail"], ["boxes_shape"], axis=0),
            *output_nodes,
        ],
        "release-fixture",
        [images],
        [logits, boxes],
        initializer=[batch_index, logits_tail, boxes_tail],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    checker.check_model(model)
    path.write_bytes(model.SerializeToString())


@pytest.fixture
def release(tmp_path):
    """A grammar-complete input dir + wheel: 10 graph/sidecar pairs, the right
    precision per recipe suffix, opset 19 everywhere."""
    inp = tmp_path / "input"
    inp.mkdir()
    for size in ra.SIZES:
        for recipe in ra.RECIPES:
            _write_onnx(
                inp / f"dfine_{size}_{recipe}.onnx",
                ra.RECIPES[recipe],
            )
        source_hash = hashlib.sha256((inp / f"dfine_{size}_op19.onnx").read_bytes()).hexdigest()
        for recipe, precision in ra.RECIPES.items():
            metadata = _sidecar(size, precision)
            if precision == "fp16":
                metadata["source_onnx_sha256"] = source_hash
            (inp / f"dfine_{size}_{recipe}.json").write_text(json.dumps(metadata))
    _write_wheel(tmp_path / WHEEL)
    return Namespace(
        input=str(inp),
        wheel=str(tmp_path / WHEEL),
        out=str(tmp_path / "out"),
        version=VERSION,
        frozen_model_pack=False,
    )


def test_happy_path_stages_all_and_sums_includes_wheel(release):
    ra.assemble(release)
    out = Path(release.out)
    assert out.stat().st_mode & 0o777 == 0o755
    lines = (out / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 21  # 20 model files plus the wheel
    names = [ln.split("  ", 1)[1] for ln in lines]
    assert names == sorted(names)
    assert WHEEL in names
    for ln in lines:
        digest, name = ln.split("  ", 1)
        assert digest == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_existing_empty_output_is_allowed(release):
    out = Path(release.out)
    out.mkdir()
    out.chmod(0o770)

    ra.assemble(release)

    assert (out / "SHA256SUMS").is_file()
    assert out.stat().st_mode & 0o777 == 0o770


def test_nonempty_output_is_refused_without_modification(release):
    out = Path(release.out)
    out.mkdir()
    existing = out / "stale-asset.whl"
    existing.write_bytes(b"do not delete")

    with pytest.raises(SystemExit, match="new or empty"):
        ra.assemble(release)

    assert existing.read_bytes() == b"do not delete"
    assert sorted(path.name for path in out.iterdir()) == [existing.name]


def test_output_file_is_refused(release):
    out = Path(release.out)
    out.write_text("not a directory")

    with pytest.raises(SystemExit, match="not a directory"):
        ra.assemble(release)

    assert out.read_text() == "not a directory"


def test_model_file_cannot_be_used_as_wheel(release):
    graph = Path(release.input) / "dfine_n_op19.onnx"
    original = graph.read_bytes()
    release.wheel = str(graph)

    with pytest.raises(SystemExit, match="wheel name"):
        ra.assemble(release)

    assert graph.read_bytes() == original
    assert not Path(release.out).exists()


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


def test_wheel_filename_must_match_requested_version(release, tmp_path):
    wheel = tmp_path / "dfine-0.4.1-py3-none-linux_x86_64.whl"
    _write_wheel(wheel, version="0.4.1")
    release.wheel = str(wheel)

    with pytest.raises(SystemExit, match="expected.*0.4.0"):
        ra.assemble(release)


@pytest.mark.parametrize(
    ("name", "version"),
    [("other", VERSION), ("dfine", "0.3.3")],
)
def test_wheel_metadata_must_match_release(release, name, version):
    _write_wheel(Path(release.wheel), name=name, version=version)

    with pytest.raises(SystemExit, match="METADATA identifies"):
        ra.assemble(release)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda meta: meta.pop("artifact_kind"), "artifact_kind"),
        (lambda meta: meta.update(schema_version=2), "schema_version"),
        (lambda meta: meta.update(checkpoint_load="partial"), "checkpoint_load"),
        (
            lambda meta: meta.update(checkpoint_loaded_tensors=100),
            "checkpoint_loaded_tensors",
        ),
        (
            lambda meta: meta.update(checkpoint_deserialization="unsafe_pickle"),
            "checkpoint_deserialization",
        ),
        (
            lambda meta: meta.update(checkpoint_selected_state="model"),
            "checkpoint_selected_state",
        ),
        (
            lambda meta: meta.update(checkpoint_unused_tensors=1),
            "checkpoint_unused_tensors",
        ),
        (lambda meta: meta.update(checkpoint_sha256="A" * 64), "checkpoint_sha256"),
        (lambda meta: meta.update(exporter_sha256="short"), "exporter_sha256"),
        (lambda meta: meta["model_source"].update(name="other"), "model_source"),
        (
            lambda meta: meta["model_source"].update(repository="https://example.com/model"),
            "model_source",
        ),
        (
            lambda meta: meta["model_source"].update(upstream_commit="d" * 40),
            "model_source",
        ),
        (
            lambda meta: meta["model_source"].update(implementation_sha256="d" * 64),
            "model_source",
        ),
        (lambda meta: meta.update(tool_versions={}), "tool_versions"),
    ],
)
def test_incomplete_release_provenance_is_refused(release, mutation, error):
    sidecar = Path(release.input) / "dfine_n_op19.json"
    meta = json.loads(sidecar.read_text())
    mutation(meta)
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match=error):
        ra.assemble(release)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda meta: meta.pop("source_onnx_sha256"), "source_onnx_sha256"),
        (lambda meta: meta.update(converter_sha256="E" * 64), "converter_sha256"),
        (
            lambda meta: meta["tool_versions"].pop("onnxconverter-common"),
            "onnxconverter-common",
        ),
    ],
)
def test_slim_release_requires_conversion_provenance(release, mutation, error):
    sidecar = Path(release.input) / "dfine_n_slim.json"
    meta = json.loads(sidecar.read_text())
    mutation(meta)
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match=error):
        ra.assemble(release)


def test_slim_release_must_derive_from_paired_fp32_graph(release):
    sidecar = Path(release.input) / "dfine_n_slim.json"
    meta = json.loads(sidecar.read_text())
    meta["source_onnx_sha256"] = "f" * 64
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="does not match dfine_n_op19.onnx"):
        ra.assemble(release)


def test_slim_release_must_preserve_source_identity(release):
    sidecar = Path(release.input) / "dfine_n_slim.json"
    meta = json.loads(sidecar.read_text())
    meta["checkpoint_sha256"] = "f" * 64
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="checkpoint_sha256"):
        ra.assemble(release)


def test_release_sidecar_must_be_json_object(release):
    (Path(release.input) / "dfine_n_op19.json").write_text("[]")

    with pytest.raises(SystemExit, match="JSON object"):
        ra.assemble(release)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("variant", "m", "variant"),
        ("precision_mode", "strongly_typed_onnx_fp16_surgical_decoder", "precision_mode"),
        ("resize", "letterbox", "resize"),
        ("onnx_simplification", "disabled", "onnx_simplification"),
    ],
)
def test_canonical_name_requires_canonical_recipe(release, field, value, error):
    sidecar = Path(release.input) / "dfine_n_slim.json"
    meta = json.loads(sidecar.read_text())
    meta[field] = value
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match=error):
        ra.assemble(release)


def test_fp32_and_slim_model_contracts_must_match(release):
    sidecar = Path(release.input) / "dfine_n_slim.json"
    meta = json.loads(sidecar.read_text())
    meta["hidden_dim"] = 999
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="hidden_dim"):
        ra.assemble(release)


def test_release_variant_requires_full_canonical_decoder(release):
    for recipe in ra.RECIPES:
        sidecar = Path(release.input) / f"dfine_m_{recipe}.json"
        meta = json.loads(sidecar.read_text())
        meta["num_decoder_layers"] = 3
        meta["eval_idx"] = 2
        sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="num_decoder_layers"):
        ra.assemble(release)


def test_release_requires_canonical_coco_labels(release):
    sidecar = Path(release.input) / "dfine_n_op19.json"
    meta = json.loads(sidecar.read_text())
    meta["class_names"][0] = "class_0"
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="COCO-80"):
        ra.assemble(release)


def test_invalid_onnx_payload_is_refused(release):
    (Path(release.input) / "dfine_n_op19.onnx").write_bytes(b"not an ONNX graph")

    with pytest.raises(SystemExit, match="invalid ONNX graph"):
        ra.assemble(release)


def test_slim_graph_must_contain_fp16_compute(release):
    input_dir = Path(release.input)
    (input_dir / "dfine_n_slim.onnx").write_bytes((input_dir / "dfine_n_op19.onnx").read_bytes())

    with pytest.raises(SystemExit, match="no FP16 internal tensors"):
        ra.assemble(release)


def test_incomplete_wheel_is_refused(release):
    wheel = Path(release.wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("dfine-0.4.0.dist-info/METADATA", "Name: dfine\nVersion: 0.4.0\n")

    with pytest.raises(SystemExit, match="dist-info/WHEEL"):
        ra.assemble(release)


def test_downloaded_release_requires_complete_asset_set(tmp_path):
    (tmp_path / "dfine_n_op19.onnx").write_bytes(b"only one asset")
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  dfine_n_op19.onnx\n")

    with pytest.raises(SystemExit, match="missing"):
        ra._validate_downloaded_release(tmp_path, VERSION)


def test_downloaded_release_manifest_covers_exact_set(tmp_path):
    names = ra._expected_payload_names(VERSION)
    for name in names:
        (tmp_path / name).write_bytes(b"payload")
    (tmp_path / "SHA256SUMS").write_text("".join(f"{'0' * 64}  {name}\n" for name in sorted(names)))

    entries = ra._validate_downloaded_release(tmp_path, VERSION)

    assert set(entries) == names


def test_extra_dfine_file_refused(release):
    (Path(release.input) / "dfine_m_spim.onnx").write_bytes(b"typo")
    with pytest.raises(SystemExit, match="dfine_m_spim.onnx"):
        ra.assemble(release)


def test_unparseable_sidecar_refused(release):
    (Path(release.input) / "dfine_l_slim.json").write_text("{not json")
    with pytest.raises(SystemExit, match="parsed"):
        ra.assemble(release)


def test_frozen_model_pack_manifest_is_complete():
    # Exactly the 20-file release grammar, and every digest is a sha256.
    assert set(ra.FROZEN_MODEL_PACK) == ra._model_asset_names()
    assert all(ra.SHA256.fullmatch(d) for d in ra.FROZEN_MODEL_PACK.values())


def test_frozen_model_pack_rejects_files_that_are_not_the_published_pack(release):
    # The fixture builds freshly-exported sidecars, which by construction do not
    # match the pinned published digests — a frozen republish must be byte-exact.
    release.frozen_model_pack = True
    with pytest.raises(SystemExit, match="do not match the published pack"):
        ra.assemble(release)


def test_verify_frozen_model_pack_reports_the_drifted_file(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in ra.FROZEN_MODEL_PACK:
        # Content that cannot hash to the pinned value: names the drifted file.
        (staging / name).write_bytes(name.encode())
    with pytest.raises(SystemExit, match="dfine_"):
        ra._verify_frozen_model_pack(staging)
