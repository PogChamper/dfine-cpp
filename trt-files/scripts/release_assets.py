#!/usr/bin/env python3
"""Validate, stage, and verify GitHub release assets.

``assemble`` requires the complete N/S/M/L/X opset-19 FP32/slim model pack and
the native wheel, then writes a manifest over every payload. ``verify`` downloads
a published release, checks the manifest, and rejects uncovered assets.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

SIZES = ("n", "s", "m", "l", "x")
RECIPES = {"op19": "fp32", "slim": "fp16"}  # recipe suffix -> precision its sidecar must carry

# The frozen model pack first published in v0.3.1: the ten ONNX graphs and their ten
# sidecars, byte-identical through v0.3.3 (verified by every release's download-back
# check). These artifacts predate the current sidecar provenance schema, so they cannot
# — and must not — be validated against it: their exporter/converter hashes belong to the
# tooling that actually produced them, which no longer exists in the tree. `assemble
# --frozen-model-pack` republishes exactly this set, admitting it by its cryptographic
# identity rather than by a schema it precedes. Any single-byte drift fails the build.
FROZEN_MODEL_PACK = {
    "dfine_l_op19.json": "9f982f3cb4b73364d92d48cc76bc861d6a4f015c281822ca7a645ed0d48f257a",
    "dfine_l_op19.onnx": "c3d8f4b3fd71b4b54ae3edd3f80d6841b28c98125c9b848f2c648cd8e4b922d4",
    "dfine_l_slim.json": "b9834387fdd3257fe1e555a4b9505301114b963052a46144d88a8931f81d4ff5",
    "dfine_l_slim.onnx": "6d66358f2dff1ae313dfe31cae61cf9de0a0c0670d355d6cd2e963bd12b7b19a",
    "dfine_m_op19.json": "83c60a44c5e6c9fe14d5c45149f5f43e4db5650c66b2204c9fbfebf3c0946b17",
    "dfine_m_op19.onnx": "902bfdd825e804912d094f78a583753eea89caa14aa4b397775e2136538a87bb",
    "dfine_m_slim.json": "be08d7ac7be7f4733ae87ae0f6de3f2d303ab294ac987f5880a8cb80561453c5",
    "dfine_m_slim.onnx": "0f0b8e9ecafa3112d3f7d983e52809c92514836ee1328b519fe81fe25abc7419",
    "dfine_n_op19.json": "72565d06749fca3c6b811d75372edbb573e98a6f00858708013f77d9b5672c76",
    "dfine_n_op19.onnx": "464bd787ef55ad59a84a948b986152cf75e410e2cd463f2dd03f737a1540ae78",
    "dfine_n_slim.json": "5d7e477e24b392863972961afa1220705a3fa5ed5a611af9228e22c15f2bbc7f",
    "dfine_n_slim.onnx": "dec936afa120cbce3d9d3bd14eb06cdf6fccc3b92f2b44093e459ebbf2a41cbe",
    "dfine_s_op19.json": "29ea4ac444094fed84baf7e2e636d02cf8fe99965dfe27d57065d443647d15df",
    "dfine_s_op19.onnx": "300db0c870f3e7824317d090f4d9447cf619230ac1c3e6fce1b94b9ad25a4d2c",
    "dfine_s_slim.json": "f1e53551596b5ae883cef4093ad2f765818fee82c65a92294a23e7fb9438fcee",
    "dfine_s_slim.onnx": "66c28cb3c2fd700bd97680f4e58f02850fa81112bb85c353bf348806f2e09dc6",
    "dfine_x_op19.json": "d42c28284432f5239904a0d578f0d0738d1148f210d1a07cad441108241d1def",
    "dfine_x_op19.onnx": "2abac2d64e05e90f3d5225e1bf3762e0d1df0f36c059f5fc28f917ad6be405d6",
    "dfine_x_slim.json": "8819a448fe4531fa6d066c65853ffe2f7eae587e73b7690c007cc72e28f9d35d",
    "dfine_x_slim.onnx": "7f5e5d6648914139f54f6d8615fabc396274c402c7d8f2142266496bf8d3446e",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"[0-9]+[.][0-9]+[.][0-9]+(?:[A-Za-z0-9.+-]*)")
CONVERSION_FIELDS = {
    "precision",
    "precision_mode",
    "source_onnx_sha256",
    "converter_sha256",
    "tool_versions",
}
MODEL_FACTS = {
    "n": {
        "reg_max": 32,
        "reg_scale": 4.0,
        "num_decoder_layers": 3,
        "eval_idx": 2,
        "num_levels": 2,
        "hidden_dim": 128,
        "feat_strides": [16, 32],
    },
    "s": {
        "reg_max": 32,
        "reg_scale": 4.0,
        "num_decoder_layers": 3,
        "eval_idx": 2,
        "num_levels": 3,
        "hidden_dim": 256,
        "feat_strides": [8, 16, 32],
    },
    "m": {
        "reg_max": 32,
        "reg_scale": 4.0,
        "num_decoder_layers": 4,
        "eval_idx": 3,
        "num_levels": 3,
        "hidden_dim": 256,
        "feat_strides": [8, 16, 32],
    },
    "l": {
        "reg_max": 32,
        "reg_scale": 4.0,
        "num_decoder_layers": 6,
        "eval_idx": 5,
        "num_levels": 3,
        "hidden_dim": 256,
        "feat_strides": [8, 16, 32],
    },
    "x": {
        "reg_max": 32,
        "reg_scale": 8.0,
        "num_decoder_layers": 6,
        "eval_idx": 5,
        "num_levels": 3,
        "hidden_dim": 256,
        "feat_strides": [8, 16, 32],
    },
}
OFFICIAL_CHECKPOINT_TENSOR_COUNTS = {
    "n": 674,
    "s": 794,
    "m": 1053,
    "l": 1173,
    "x": 1441,
}
OFFICIAL_CHECKPOINTS = {
    "n": "a6b913de83520d48bfc0d58d4645b3648662419ca0503b9386fef38548891ff6",
    "s": "4d878bf9be3e07bb0092f03ad45366d900bd0fa765c65c2d67ced1b08856182d",
    "m": "c8a5b79d4ed5718dacf5507e366a79c8103e463adf1899cd73f0187cc3d1d253",
    "l": "2323fabe153f30f1dc51089cbfd0a88ffe1d1fa0b71baec692b5334a9cabe726",
    "x": "8f46e0a96c51053951a5155b77d280c4a8d003b8e25fe7503127e666b0d152f9",
}
COCO80_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _model_asset_names() -> set[str]:
    return {
        f"dfine_{size}_{recipe}.{extension}"
        for size in SIZES
        for recipe in RECIPES
        for extension in ("onnx", "json")
    }


def _expected_payload_names(version: str) -> set[str]:
    return _model_asset_names() | {f"dfine-{version}-py3-none-linux_x86_64.whl"}


def _require_empty_output(out: Path) -> None:
    if not out.exists():
        return
    if not out.is_dir():
        raise SystemExit(f"--out is not a directory: {out}")
    try:
        first_entry = next(out.iterdir(), None)
    except OSError as e:
        raise SystemExit(f"cannot inspect --out {out}: {e}") from e
    if first_entry is not None:
        raise SystemExit(
            f"--out must be new or empty: {out} contains {first_entry.name}; "
            "refusing to mix staged and existing files"
        )


def _validated_model_revision() -> str:
    revision_file = Path(__file__).resolve().parents[1] / "DFINE_SEG_REVISION"
    try:
        revision = revision_file.read_text().strip()
    except OSError as exc:
        raise SystemExit(f"cannot read validated model revision {revision_file}: {exc}") from exc
    if not GIT_COMMIT.fullmatch(revision):
        raise SystemExit(f"invalid validated model revision in {revision_file}")
    return revision


def _script_sha256(name: str) -> str:
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()


def _model_source_manifest() -> dict:
    source = Path(__file__).resolve().parents[1] / "dfine_model"
    files = sorted(path for path in source.rglob("*.py") if path.is_file())
    if not files:
        raise SystemExit(f"bundled D-FINE model sources are missing: {source}")
    digest = hashlib.sha256()
    relative_files = []
    for path in files:
        relative = path.relative_to(source).as_posix()
        relative_files.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": relative_files}


def _validate_wheel(wheel: Path, version: str) -> None:
    expected = f"dfine-{version}-py3-none-linux_x86_64.whl"
    if wheel.name != expected:
        raise SystemExit(f"wheel name is {wheel.name}; expected {expected}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise SystemExit(f"{wheel.name}: wheel contains duplicate members")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise SystemExit(f"{wheel.name}: wheel member failed CRC validation: {corrupt}")
            dist_info = f"dfine-{version}.dist-info"
            metadata_path = f"{dist_info}/METADATA"
            wheel_path = f"{dist_info}/WHEEL"
            record_path = f"{dist_info}/RECORD"
            foreign_dist_info = sorted(
                name for name in names if ".dist-info/" in name and not name.startswith(dist_info)
            )
            if foreign_dist_info:
                raise SystemExit(
                    f"{wheel.name}: foreign dist-info members: {', '.join(foreign_dist_info)}"
                )
            for required in (metadata_path, wheel_path, record_path, "dfine/__init__.py"):
                if required not in names:
                    raise SystemExit(f"{wheel.name}: missing required wheel member: {required}")
            metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_path))
            wheel_metadata = BytesParser(policy=policy.default).parsebytes(archive.read(wheel_path))
            required_members = {
                "dfine/libdfine.so",
                "dfine/_scripts/build_engine.py",
            }
            missing = sorted(required_members - set(names))
            for license_name in ("LICENSE", "NOTICE"):
                if not any(name.endswith(f".dist-info/licenses/{license_name}") for name in names):
                    missing.append(license_name)
            if missing:
                raise SystemExit(
                    f"{wheel.name}: missing required wheel members: {', '.join(missing)}"
                )
            library = archive.read("dfine/libdfine.so")
            if not library.startswith(b"\x7fELF"):
                raise SystemExit(f"{wheel.name}: dfine/libdfine.so is not an ELF library")
            bundled_builder = archive.read("dfine/_scripts/build_engine.py")
            expected_builder = Path(__file__).with_name("build_engine.py").read_bytes()
            if bundled_builder != expected_builder:
                raise SystemExit(
                    f"{wheel.name}: bundled build_engine.py differs from the release source"
                )
            record_rows = list(csv.reader(io.StringIO(archive.read(record_path).decode("utf-8"))))
            if any(len(row) != 3 for row in record_rows):
                raise SystemExit(f"{wheel.name}: RECORD contains a malformed row")
            records = {row[0]: row[1:] for row in record_rows}
            if len(records) != len(record_rows):
                raise SystemExit(f"{wheel.name}: RECORD contains a duplicate member")
            if set(records) != set(names):
                raise SystemExit(f"{wheel.name}: RECORD does not cover every wheel member exactly")
            for name in names:
                digest_field, size_field = records[name]
                if name == record_path:
                    if digest_field or size_field:
                        raise SystemExit(f"{wheel.name}: RECORD must not hash itself")
                    continue
                payload = archive.read(name)
                expected_digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(
                    b"="
                )
                if digest_field != f"sha256={expected_digest.decode()}" or size_field != str(
                    len(payload)
                ):
                    raise SystemExit(f"{wheel.name}: invalid RECORD entry for {name}")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise SystemExit(f"{wheel.name}: cannot read wheel metadata ({exc})") from exc
    if metadata.get("Name") != "dfine" or metadata.get("Version") != version:
        raise SystemExit(
            f"{wheel.name}: METADATA identifies "
            f"{metadata.get('Name')!r} {metadata.get('Version')!r}; "
            f"expected 'dfine' {version!r}"
        )
    if (
        wheel_metadata.get("Root-Is-Purelib", "").lower() != "false"
        or wheel_metadata.get("Tag") != "py3-none-linux_x86_64"
    ):
        raise SystemExit(
            f"{wheel.name}: WHEEL must declare Root-Is-Purelib: false and "
            "Tag: py3-none-linux_x86_64"
        )


def _validate_sidecar(
    sidecar: Path,
    size: str,
    precision: str,
    validated_revision: str,
) -> dict:
    """Validate one complete canonical release sidecar."""
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"{sidecar.name} cannot be parsed ({e}); re-export before staging")
    if not isinstance(meta, dict):
        raise SystemExit(f"{sidecar.name}: sidecar must contain a JSON object")
    actual = meta.get("precision", "fp32")
    if actual != precision:
        raise SystemExit(
            f"{sidecar.name}: sidecar precision is {actual} but the name's "
            f"recipe suffix requires {precision}"
        )
    if meta.get("opset") != 19:
        raise SystemExit(
            f"{sidecar.name}: opset is {meta.get('opset')} but release assets are "
            "opset 19 (the surgical fp16 recipe hard-requires an opset-19 base)"
        )
    required = {
        "artifact_kind": "onnx",
        "schema_version": 1,
        "checkpoint_load": "strict",
        "checkpoint_deserialization": "weights_only",
        "checkpoint_selected_state": "checkpoint root",
        "checkpoint_unused_tensors": 0,
        "precision": precision,
        "model": "d-fine",
        "variant": size,
        "task": "detect",
        "input_h": 640,
        "input_w": 640,
        "num_classes": 80,
        "num_queries": 300,
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
        "deform_core": "explicit",
        "gridsample_nodes": 0,
        "onnx_simplification": "applied",
        "checkpoint_loaded_tensors": OFFICIAL_CHECKPOINT_TENSOR_COUNTS[size],
        "checkpoint_sha256": OFFICIAL_CHECKPOINTS[size],
        "exporter_sha256": _script_sha256("export_dfine_onnx.py"),
        **MODEL_FACTS[size],
    }
    for field, expected in required.items():
        if meta.get(field) != expected or type(meta.get(field)) is not type(expected):
            raise SystemExit(f"{sidecar.name}: {field} must be {expected!r} for a release artifact")
    if "cascade" in meta or "cascade_initial_queries" in meta:
        raise SystemExit(f"{sidecar.name}: release artifacts must not use a query cascade")
    class_names = meta.get("class_names")
    if class_names != list(COCO80_NAMES):
        raise SystemExit(f"{sidecar.name}: class_names must match canonical COCO-80 order")
    for field in ("checkpoint_sha256", "exporter_sha256"):
        value = meta.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise SystemExit(f"{sidecar.name}: {field} must be a 64-character lowercase SHA-256")
    source = meta.get("model_source")
    if not isinstance(source, dict):
        raise SystemExit(f"{sidecar.name}: model_source must be an object")
    implementation = _model_source_manifest()
    if (
        source.get("name") != "D-FINE-seg"
        or source.get("repository") != "https://github.com/ArgoHA/D-FINE-seg"
        or source.get("upstream_commit") != validated_revision
        or source.get("bundled") is not True
        or source.get("implementation_sha256") != implementation["sha256"]
        or source.get("implementation_files") != implementation["files"]
    ):
        raise SystemExit(
            f"{sidecar.name}: model_source must identify the validated bundled implementation"
        )
    tool_versions = meta.get("tool_versions")
    if (
        not isinstance(tool_versions, dict)
        or not tool_versions
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in tool_versions.items()
        )
    ):
        raise SystemExit(f"{sidecar.name}: tool_versions must map names to non-empty versions")
    if precision == "fp16":
        if meta.get("precision_mode") != "strongly_typed_onnx_fp16_surgical_slim":
            raise SystemExit(
                f"{sidecar.name}: precision_mode must identify the surgical slim recipe"
            )
        for field in ("source_onnx_sha256", "converter_sha256"):
            value = meta.get(field)
            if not isinstance(value, str) or not SHA256.fullmatch(value):
                raise SystemExit(
                    f"{sidecar.name}: {field} must be a 64-character lowercase SHA-256"
                )
        if meta.get("converter_sha256") != _script_sha256("convert_fp16_surgical.py"):
            raise SystemExit(f"{sidecar.name}: converter_sha256 differs from the release source")
        if "onnxconverter-common" not in tool_versions:
            raise SystemExit(f"{sidecar.name}: tool_versions must identify onnxconverter-common")
    elif meta.get("precision_mode") != "fp32":
        raise SystemExit(f"{sidecar.name}: precision_mode must be 'fp32'")
    return meta


def _onnx_shape(value) -> list[str | int]:
    tensor_type = value.type.tensor_type
    dimensions: list[str | int] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append("")
    return dimensions


def _validate_onnx(graph_path: Path, meta: dict, precision: str) -> None:
    try:
        import onnx
        from onnx import TensorProto
    except ImportError as exc:
        raise SystemExit(
            "release assembly requires onnx; run it from the locked tools env"
        ) from exc
    try:
        model = onnx.load(str(graph_path))
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{graph_path.name}: invalid ONNX graph ({exc})") from exc
    opset = max(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        default=0,
    )
    if opset != 19:
        raise SystemExit(f"{graph_path.name}: graph opset is {opset}, expected 19")
    inputs = {value.name: value for value in inferred.graph.input}
    outputs = {value.name: value for value in inferred.graph.output}
    if set(inputs) != {"images"} or set(outputs) != {"logits", "boxes"}:
        raise SystemExit(f"{graph_path.name}: graph IO must be images -> logits, boxes")
    expected_shapes = {
        "images": ["N", 3, meta["input_h"], meta["input_w"]],
        "logits": meta["logits_shape"],
        "boxes": meta["boxes_shape"],
    }
    for name, value in {**inputs, **outputs}.items():
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise SystemExit(f"{graph_path.name}: {name} must have FP32 graph IO")
        if _onnx_shape(value) != expected_shapes[name]:
            raise SystemExit(
                f"{graph_path.name}: {name} shape {_onnx_shape(value)} "
                f"does not match {expected_shapes[name]}"
            )
    if any(node.op_type == "GridSample" for node in model.graph.node):
        raise SystemExit(f"{graph_path.name}: explicit release graph contains GridSample")
    typed_values = [
        value.type.tensor_type.elem_type
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.output)
            + list(inferred.graph.value_info)
        )
    ]
    typed_values.extend(initializer.data_type for initializer in inferred.graph.initializer)
    has_fp16 = TensorProto.FLOAT16 in typed_values
    if precision == "fp16" and not has_fp16:
        raise SystemExit(f"{graph_path.name}: slim graph contains no FP16 internal tensors")
    if precision == "fp32" and has_fp16:
        raise SystemExit(f"{graph_path.name}: FP32 base contains FP16 internal tensors")


def _compare_artifact_contracts(fp32: dict, slim: dict, slim_sidecar: Path) -> None:
    fp32_contract = {key: value for key, value in fp32.items() if key not in CONVERSION_FIELDS}
    slim_contract = {key: value for key, value in slim.items() if key not in CONVERSION_FIELDS}
    if slim_contract != fp32_contract:
        differing = sorted(
            key
            for key in fp32_contract.keys() | slim_contract.keys()
            if fp32_contract.get(key) != slim_contract.get(key)
        )
        raise SystemExit(
            f"{slim_sidecar.name}: model contract differs from its FP32 source "
            f"({', '.join(differing)})"
        )
    slim_tools = dict(slim["tool_versions"])
    slim_tools.pop("onnxconverter-common", None)
    if slim_tools != fp32["tool_versions"]:
        raise SystemExit(f"{slim_sidecar.name}: base tool_versions differ from its FP32 source")


def _verify_frozen_model_pack(staging: Path) -> None:
    """Admit the previously published model pack by its pinned hashes. The set is
    already complete (checked before staging); here every file must reproduce its
    FROZEN_MODEL_PACK digest exactly, so no re-exported or drifted artifact can slip
    into a frozen republish."""
    mismatched = []
    for name, expected in sorted(FROZEN_MODEL_PACK.items()):
        actual = hashlib.sha256((staging / name).read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(name)
    if mismatched:
        raise SystemExit(
            "--frozen-model-pack: these files do not match the published pack "
            f"({', '.join(mismatched)}); a frozen republish must be byte-identical. "
            "Omit --frozen-model-pack to assemble a freshly exported pack."
        )


def assemble(args: argparse.Namespace) -> None:
    if not VERSION.fullmatch(args.version):
        raise SystemExit(f"invalid --version {args.version!r}")
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")
    wheel = Path(args.wheel).resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")
    out = Path(args.out).resolve()
    _require_empty_output(out)

    expected = _model_asset_names()
    present = {p.name for p in input_dir.glob("dfine_*") if p.is_file()}
    extra = sorted(present - expected)
    if extra:
        raise SystemExit(
            f"unexpected dfine_* files in {input_dir}: {', '.join(extra)} "
            "(release grammar is dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json})"
        )
    missing = sorted(expected - present)
    if missing:
        raise SystemExit(
            f"missing from {input_dir}: {', '.join(missing)} "
            "(a graph never ships without its sidecar, nor a sidecar alone)"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for name in sorted(expected):
            shutil.copy2(input_dir / name, staging / name)
        staged_wheel = staging / wheel.name
        shutil.copy2(wheel, staged_wheel)
        _validate_wheel(staged_wheel, args.version)

        if args.frozen_model_pack:
            _verify_frozen_model_pack(staging)
        else:
            validated_revision = _validated_model_revision()
            for size in SIZES:
                fp32_sidecar = staging / f"dfine_{size}_op19.json"
                slim_sidecar = staging / f"dfine_{size}_slim.json"
                fp32_meta = _validate_sidecar(fp32_sidecar, size, "fp32", validated_revision)
                slim_meta = _validate_sidecar(slim_sidecar, size, "fp16", validated_revision)
                fp32_graph = staging / f"dfine_{size}_op19.onnx"
                slim_graph = staging / f"dfine_{size}_slim.onnx"
                _validate_onnx(fp32_graph, fp32_meta, "fp32")
                _validate_onnx(slim_graph, slim_meta, "fp16")
                source_hash = hashlib.sha256(fp32_graph.read_bytes()).hexdigest()
                if slim_meta["source_onnx_sha256"] != source_hash:
                    raise SystemExit(
                        f"{slim_sidecar.name}: source_onnx_sha256 does not match "
                        f"dfine_{size}_op19.onnx"
                    )
                _compare_artifact_contracts(fp32_meta, slim_meta, slim_sidecar)

        names = sorted(expected | {wheel.name})
        lines = [
            f"{hashlib.sha256((staging / name).read_bytes()).hexdigest()}  {name}" for name in names
        ]
        (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        _require_empty_output(out)
        output_mode = stat.S_IMODE(out.stat().st_mode) if out.exists() else 0o755
        staging.chmod(output_mode)
        # rename(2) can replace an empty directory atomically.
        os.replace(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[assemble] staged {len(names)} payload files + SHA256SUMS -> {out}")


def _default_repo() -> str:
    # The same OWNER/NAME gh itself would infer from the cwd's git remote, made
    # explicit so the summary names the repo that was actually verified.
    out = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        capture_output=True,
        text=True,
    )
    repo = out.stdout.strip()
    if out.returncode != 0 or not repo:
        raise SystemExit("cannot resolve the repo via `gh repo view`; pass --repo OWNER/NAME")
    return repo


def _version_from_tag(tag: str) -> str:
    version = tag[1:] if tag.startswith("v") else ""
    if not VERSION.fullmatch(version):
        raise SystemExit(f"release tag must be v<version>, got {tag!r}")
    return version


def _manifest_entries(sums: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if not match or match.group(2) in entries:
            raise SystemExit(f"{sums.name}: invalid or duplicate manifest entry: {line!r}")
        entries[match.group(2)] = match.group(1)
    return entries


def _validate_downloaded_release(directory: Path, version: str) -> dict[str, str]:
    expected = _expected_payload_names(version)
    actual = {path.name for path in directory.iterdir() if path.name != "SHA256SUMS"}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise SystemExit("release asset set is incomplete or invalid (" + "; ".join(details) + ")")
    entries = _manifest_entries(directory / "SHA256SUMS")
    if set(entries) != expected:
        missing_manifest = sorted(expected - set(entries))
        extra_manifest = sorted(set(entries) - expected)
        details = []
        if missing_manifest:
            details.append(f"missing from SHA256SUMS: {', '.join(missing_manifest)}")
        if extra_manifest:
            details.append(f"unexpected in SHA256SUMS: {', '.join(extra_manifest)}")
        raise SystemExit("; ".join(details))
    return entries


def verify(args: argparse.Namespace) -> None:
    repo = args.repo or _default_repo()
    version = _version_from_tag(args.tag)
    with tempfile.TemporaryDirectory(prefix="dfine-release-verify-") as td:
        print(f"[verify] downloading {repo} {args.tag} -> {td}")
        dl = subprocess.run(["gh", "release", "download", args.tag, "--repo", repo, "--dir", td])
        if dl.returncode != 0:
            raise SystemExit(f"gh release download failed for {repo} {args.tag}")
        sums = Path(td) / "SHA256SUMS"
        if not sums.is_file():
            raise SystemExit(f"release {args.tag} has no SHA256SUMS asset")
        entries = _validate_downloaded_release(Path(td), version)
        chk = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"], cwd=td, capture_output=True, text=True
        )
        for line in chk.stdout.splitlines():
            print(f"[verify] {line}")
        if chk.stderr.strip():
            print(f"[verify] {chk.stderr.strip()}")
        ok = sum(1 for ln in chk.stdout.splitlines() if ln.endswith(": OK"))
        bad = sum(1 for ln in chk.stdout.splitlines() if ": FAILED" in ln)
        print(f"[verify] {ok} OK, {bad} FAILED, {len(entries)} required assets covered")
        if chk.returncode != 0:
            raise SystemExit(1)
        print(f"[verify] {repo} {args.tag}: all assets match SHA256SUMS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage (assemble) or check (verify) release assets")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser(
        "assemble", help="validate the 20 model files, stage them + the wheel, write SHA256SUMS"
    )
    a.add_argument(
        "--input", required=True, help="directory holding dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json}"
    )
    a.add_argument(
        "--wheel",
        required=True,
        help="the gated wheel (staged and hashed into SHA256SUMS with the models)",
    )
    a.add_argument(
        "--version", required=True, help="release version without the v prefix, e.g. 0.4.0"
    )
    a.add_argument("--out", required=True, help="new or empty staging directory for the upload")
    a.add_argument(
        "--frozen-model-pack",
        action="store_true",
        help="republish the pinned v0.3.1 model pack: admit the 20 files by their "
        "published SHA-256 instead of the current sidecar provenance schema (which they "
        "predate). The wheel is still fully validated.",
    )
    v = sub.add_parser("verify", help="download a published release and run sha256sum -c on it")
    v.add_argument("--tag", required=True, help="release tag, e.g. v0.3.1")
    v.add_argument("--repo", default=None, help="OWNER/NAME (default: `gh repo view` in the cwd)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"assemble": assemble, "verify": verify}[args.cmd](args)
