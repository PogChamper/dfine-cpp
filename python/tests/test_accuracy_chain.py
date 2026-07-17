from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "trt-files/scripts/accuracy_chain.py"
SPEC = importlib.util.spec_from_file_location("accuracy_chain", SCRIPT)
ACCURACY_CHAIN = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(ACCURACY_CHAIN)
finally:
    sys.path.remove(str(SCRIPT.parent))


def _contract() -> dict:
    return {
        "schema_version": 1,
        "dataset": {"name": "fixture", "split": "test", "version": "1"},
        "evaluator": {
            "name": "pycocotools.COCOeval:bbox",
            "version": "fixture",
            "metrics_source_sha256": "c" * 64,
        },
        "annotations_sha256": "a" * 64,
        "selection": {
            "image_count": 2,
            "image_manifest_sha256": "b" * 64,
            "image_manifest_scheme": "image-id/path/size/content-sha256 records",
        },
        "score_threshold": 0.001,
        "topk": 300,
        "inference_batch_size": 1,
        "geometry": {
            "canonical": "coco_original_pixels",
            "model_space_area": {
                "input_h": 640,
                "input_w": 640,
                "resize": "stretch",
            },
        },
    }


def _ground_truth() -> dict:
    return {
        "images": 2,
        "gt_instances": 3,
        "crowd_instances": 0,
        "per_image": {
            "min": 1,
            "mean": 1.5,
            "median": 1.5,
            "p90": 1.9,
            "p95": 1.95,
            "p99": 1.99,
            "max": 2,
            "over_100": 0,
            "histogram": [
                {"range": name, "images": 1 if name in {"1", "2-5"} else 0}
                for name in ACCURACY_CHAIN.DENSITY_RANGES
            ],
        },
    }


def _metrics(offset: float = 0.0) -> dict:
    scalars = {
        "AP": 0.50 + offset,
        "AP50": 0.70 + offset,
        "AP75": 0.55 + offset,
        "APs": 0.40 + offset,
        "APm": 0.60 + offset,
        "APl": None,
        "AR1": 0.25 + offset,
        "AR10": 0.60 + offset,
        "AR100": 0.72 + offset,
        "ARs": 0.62 + offset,
        "ARm": 0.75 + offset,
        "ARl": None,
    }
    return {
        **scalars,
        "max_dets": [1, 10, 100],
        "AP_by_iou": {
            key: 0.70 - index * 0.03 + offset
            for index, key in enumerate(ACCURACY_CHAIN.IOU_KEYS)
        },
        "per_class": [
            {"category_id": 3, "name": "beta", "gt_instances": 1, "AP": 0.40 + offset},
            {"category_id": 1, "name": "alpha", "gt_instances": 2, "AP": 0.55 + offset},
        ],
        "GT_by_area": {"small": 2, "medium": 1, "large": 0},
        "model_space_area": {
            "input_h": 640,
            "input_w": 640,
            "resize": "stretch",
            "APs": 0.45 + offset,
            "APm": 0.58 + offset,
            "APl": None,
            "ARs": 0.65 + offset,
            "ARm": 0.73 + offset,
            "ARl": None,
            "GT_by_area": {"small": 1, "medium": 2, "large": 0},
        },
    }


def _model_contract(**overrides) -> dict:
    contract = {
        "model": "d-fine",
        "variant": "s",
        "task": "detect",
        "input_h": 640,
        "input_w": 640,
        "num_classes": 3,
        "initial_queries": 300,
        "num_queries": 300,
        "eval_idx": 2,
        "cascade": None,
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
    contract.update(overrides)
    return contract


def _strict_backend(
    kind: str,
    *,
    artifact_sha256: str,
    precision_mode: str = "fp32",
    onnx_sha256: str | None = None,
    source_onnx_sha256: str | None = None,
    model_overrides: dict | None = None,
    offset: float = 0.0,
) -> dict:
    artifact_kind = {
        "checkpoint": "checkpoint",
        "onnx": "onnx",
        "engine": "tensorrt_engine",
    }[kind]
    lineage = {
        "artifact_kind": kind,
        "precision_mode": precision_mode,
        "checkpoint_sha256": "a" * 64,
        "artifact_sha256": artifact_sha256,
    }
    if onnx_sha256 is not None:
        lineage["onnx_sha256"] = onnx_sha256
    if source_onnx_sha256 is not None:
        lineage["source_onnx_sha256"] = source_onnx_sha256
    return {
        "artifact": {
            "kind": artifact_kind,
            "sha256": artifact_sha256,
            "recipe": precision_mode,
            "runtime": "fixture",
        },
        "lineage": lineage,
        "model_contract": _model_contract(**(model_overrides or {})),
        "map": _metrics(offset),
    }


def _report(backends: dict[str, dict] | None = None) -> dict:
    if backends is None:
        backends = {"result": {"map": _metrics()}}
    backends = {
        name: {
            **entry,
            "artifact": entry.get(
                "artifact",
                {
                    "kind": "fixture",
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "recipe": "fixture",
                    "runtime": "fixture",
                },
            ),
        }
        for name, entry in backends.items()
    }
    return {
        "schema": 1,
        "images": 2,
        "ground_truth": _ground_truth(),
        "evaluation_contract": _contract(),
        "backends": backends,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _set_letterbox(report: dict, pad: int) -> None:
    geometry = report["evaluation_contract"]["geometry"]["model_space_area"]
    geometry.update(
        {
            "resize": "letterbox",
            "letterbox_anchor": "center",
            "letterbox_pad": pad,
            "letterbox_upscale": True,
        }
    )
    model_area = report["backends"]["result"]["map"]["model_space_area"]
    model_area.update(
        {
            "resize": "letterbox",
            "letterbox_anchor": "center",
            "letterbox_pad": pad,
            "letterbox_upscale": True,
        }
    )


def _set_protocol_manifest(report: dict, digest: str, path: str = "/protocol.json") -> None:
    report["provenance"] = {"protocol_manifest": {"path": path, "sha256": digest}}


def _args(stages, output: Path, *, labels=None, kinds=None, overwrite=False, allow_missing=False):
    return Namespace(
        stage=stages,
        transition_label=labels or [],
        transition_kind=kinds or [],
        output=str(output),
        overwrite=overwrite,
        allow_missing_lineage=allow_missing,
    )


def test_ordered_chain_emits_absolute_metrics_deltas_and_hashes(tmp_path):
    names = ("pytorch", "onnx", "trt-fp32", "slim", "Q200")
    stages = []
    for index, name in enumerate(names):
        report = _report(
            {
                "result": {
                    "map": _metrics(-index * 0.01),
                    "artifact": {
                        "kind": name,
                        "sha256": hashlib.sha256(name.encode()).hexdigest(),
                        "recipe": name,
                        "runtime": "fixture",
                    },
                }
            }
        )
        path = _write(tmp_path / f"{name}.json", report)
        stages.append((name, path, None))

    output = tmp_path / "chain.json"
    result = ACCURACY_CHAIN.run(_args(stages, output, allow_missing=True))

    assert [entry["name"] for entry in result["stages"]] == list(names)
    assert [entry["kind"] for entry in result["transitions"]] == ["comparison"] * 4
    assert [entry["label"] for entry in result["transitions"]] == ["comparison"] * 4
    assert not any(entry["lineage_verified"] for entry in result["transitions"])
    assert result["stages"][0]["metrics"]["bbox"]["AP"] == pytest.approx(0.5)
    assert result["stages"][0]["artifact"]["recipe"] == "pytorch"
    assert result["transitions"][0]["delta"]["bbox"]["AP"] == pytest.approx(-0.01)
    assert result["transitions"][0]["delta"]["bbox"]["APl"] is None
    assert result["transitions"][0]["delta"]["per_class"][0]["category_id"] == 1
    assert result["source"]["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert (
        result["stages"][0]["report_sha256"]
        == hashlib.sha256(stages[0][1].read_bytes()).hexdigest()
    )
    assert json.loads(output.read_text()) == result


def test_verified_export_runtime_and_precision_chain(tmp_path):
    raw_onnx_sha = "b" * 64
    slim_onnx_sha = "e" * 64
    backends = [
        _strict_backend("checkpoint", artifact_sha256="a" * 64),
        _strict_backend("onnx", artifact_sha256=raw_onnx_sha, offset=-0.001),
        _strict_backend(
            "engine",
            artifact_sha256="c" * 64,
            onnx_sha256=raw_onnx_sha,
            offset=-0.002,
        ),
        _strict_backend(
            "engine",
            artifact_sha256="d" * 64,
            precision_mode="strongly_typed_onnx_fp16_surgical_slim",
            onnx_sha256=slim_onnx_sha,
            source_onnx_sha256=raw_onnx_sha,
            offset=-0.003,
        ),
    ]
    names = ("pytorch", "ort-fp32", "trt-fp32", "trt-slim")
    stages = [
        (
            name,
            _write(tmp_path / f"{name}.json", _report({"result": backend})),
            None,
        )
        for name, backend in zip(names, backends)
    ]

    result = ACCURACY_CHAIN.build_report(
        stages,
        transition_kinds=["export", "runtime", "precision"],
    )

    assert result["schema_version"] == 2
    assert [transition["kind"] for transition in result["transitions"]] == [
        "export",
        "runtime",
        "precision",
    ]
    assert all(transition["lineage_verified"] for transition in result["transitions"])


def test_verified_onnx_precision_and_slim_runtime_chain(tmp_path):
    raw_sha = "b" * 64
    slim_sha = "e" * 64
    backends = [
        _strict_backend("onnx", artifact_sha256=raw_sha),
        _strict_backend(
            "onnx",
            artifact_sha256=slim_sha,
            precision_mode="strongly_typed_onnx_fp16_surgical_slim",
            source_onnx_sha256=raw_sha,
        ),
        _strict_backend(
            "engine",
            artifact_sha256="d" * 64,
            precision_mode="strongly_typed_onnx_fp16_surgical_slim",
            onnx_sha256=slim_sha,
            source_onnx_sha256=raw_sha,
        ),
    ]
    names = ("ort-fp32", "ort-slim", "trt-slim")
    stages = [
        (
            name,
            _write(tmp_path / f"{name}.json", _report({"result": backend})),
            None,
        )
        for name, backend in zip(names, backends)
    ]

    result = ACCURACY_CHAIN.build_report(
        stages,
        transition_kinds=["precision", "runtime"],
    )

    assert [transition["kind"] for transition in result["transitions"]] == [
        "precision",
        "runtime",
    ]


def test_verified_preset_comparison_allows_only_graph_contract_change(tmp_path):
    precision = "strongly_typed_onnx_fp16_surgical_slim"
    base = _strict_backend(
        "engine",
        artifact_sha256="d" * 64,
        precision_mode=precision,
        onnx_sha256="b" * 64,
        source_onnx_sha256="f" * 64,
    )
    q200 = _strict_backend(
        "engine",
        artifact_sha256="e" * 64,
        precision_mode=precision,
        onnx_sha256="c" * 64,
        source_onnx_sha256="1" * 64,
        model_overrides={"initial_queries": 200, "num_queries": 200},
        offset=-0.002,
    )
    stages = [
        ("base", _write(tmp_path / "base.json", _report({"result": base})), None),
        ("q200", _write(tmp_path / "q200.json", _report({"result": q200})), None),
    ]

    result = ACCURACY_CHAIN.build_report(stages, transition_kinds=["preset"])

    transition = result["transitions"][0]
    assert transition["kind"] == "preset"
    assert transition["lineage_verified"]
    assert transition["delta"]["bbox"]["AP"] == pytest.approx(-0.002)


def test_transition_kind_rejects_wrong_artifact_lineage(tmp_path):
    raw = _strict_backend("onnx", artifact_sha256="b" * 64)
    engine = _strict_backend(
        "engine",
        artifact_sha256="c" * 64,
        onnx_sha256="f" * 64,
    )
    stages = [
        ("onnx", _write(tmp_path / "onnx.json", _report({"result": raw})), None),
        ("engine", _write(tmp_path / "engine.json", _report({"result": engine})), None),
    ]

    with pytest.raises(ValueError, match="engine was not built from the ONNX stage"):
        ACCURACY_CHAIN.build_report(stages, transition_kinds=["runtime"])


def test_named_fidelity_requires_embedded_lineage(tmp_path):
    stages = [
        ("pytorch", _write(tmp_path / "first.json", _report()), None),
        ("onnx", _write(tmp_path / "second.json", _report()), None),
    ]

    with pytest.raises(ValueError, match="--allow-missing-lineage"):
        ACCURACY_CHAIN.build_report(stages, transition_kinds=["export"])

    with pytest.raises(ValueError, match="requires model_contract and lineage"):
        ACCURACY_CHAIN.build_report(
            stages, transition_kinds=["export"], allow_missing_lineage=True
        )


def test_missing_lineage_is_rejected_without_explicit_override(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second = _write(tmp_path / "second.json", _report())
    stages = [("pytorch", first, None), ("onnx", second, None)]

    with pytest.raises(ValueError, match="--allow-missing-lineage"):
        ACCURACY_CHAIN.build_report(stages)

    result = ACCURACY_CHAIN.build_report(stages, allow_missing_lineage=True)
    assert [entry["kind"] for entry in result["transitions"]] == ["comparison"]
    assert not result["transitions"][0]["lineage_verified"]


def test_legacy_extended_max_dets_is_normalized(tmp_path):
    first_report = _report()
    second_report = _report()
    for report in (first_report, second_report):
        metrics = report["backends"]["result"]["map"]
        metrics["max_dets"] = [1, 10, 100, 300]
        metrics["AR300"] = 0.73
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", second_report)

    result = ACCURACY_CHAIN.build_report(
        [("pytorch", first, None), ("onnx", second, None)], allow_missing_lineage=True
    )

    for stage in result["stages"]:
        assert stage["metrics"]["max_dets"] == [1, 10, 100]
        assert stage["metrics"]["source_max_dets"] == [1, 10, 100, 300]
        assert "AR300" not in stage["metrics"]["bbox"]


def test_mixed_max_dets_sources_are_rejected(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second_report = _report()
    second_report["backends"]["result"]["map"]["max_dets"] = [1, 10, 100, 300]
    second_report["backends"]["result"]["map"]["AR300"] = 0.73
    second = _write(tmp_path / "second.json", second_report)

    with pytest.raises(ValueError, match="source_max_dets differs"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_multi_backend_report_requires_and_honors_selector(tmp_path):
    report = _report(
        {
            "torch": {"map": _metrics()},
            "onnx": {"map": _metrics(-0.01)},
        }
    )
    source = _write(tmp_path / "combined.json", report)

    with pytest.raises(ValueError, match="select one with REPORT::BACKEND"):
        ACCURACY_CHAIN.build_report([("pytorch", source, None), ("onnx", source, "onnx")])

    result = ACCURACY_CHAIN.build_report(
        [("pytorch", source, "torch"), ("onnx", source, "onnx")], allow_missing_lineage=True
    )
    assert [stage["backend"] for stage in result["stages"]] == ["torch", "onnx"]
    assert result["transitions"][0]["delta"]["bbox"]["AP"] == pytest.approx(-0.01)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["evaluation_contract"].__setitem__(
                "annotations_sha256", "c" * 64
            ),
            "contract.annotations_sha256 differs",
        ),
        (
            lambda report: report["evaluation_contract"].__setitem__("score_threshold", 0.01),
            "contract.score_threshold differs",
        ),
        (
            lambda report: report["evaluation_contract"]["selection"].__setitem__(
                "image_manifest_sha256", "d" * 64
            ),
            "contract.selection.image_manifest_sha256 differs",
        ),
        (
            lambda report: report["evaluation_contract"].__setitem__("topk", 200),
            "topk must be 300",
        ),
        (
            lambda report: report["evaluation_contract"].__setitem__("inference_batch_size", 8),
            "contract.inference_batch_size differs",
        ),
        (
            lambda report: report["evaluation_contract"]["geometry"]["model_space_area"].update(
                {"input_h": 512, "input_w": 512}
            ),
            "model_space_area.input_h",
        ),
    ],
)
def test_protocol_mismatch_is_rejected(tmp_path, mutate, message):
    first = _write(tmp_path / "first.json", _report())
    candidate = _report()
    mutate(candidate)
    if candidate["evaluation_contract"]["geometry"]["model_space_area"]["input_h"] == 512:
        candidate["backends"]["result"]["map"]["model_space_area"].update(
            {"input_h": 512, "input_w": 512}
        )
    second = _write(tmp_path / "second.json", candidate)

    with pytest.raises(ValueError, match=message):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_letterbox_reports_are_rejected(tmp_path):
    first_report = _report()
    second_report = _report()
    _set_letterbox(first_report, 114)
    _set_letterbox(second_report, 114)
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", second_report)

    with pytest.raises(ValueError, match="resize must be 'stretch'"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_protocol_manifest_hash_must_match_across_stages(tmp_path):
    first_report = _report()
    second_report = _report()
    _set_protocol_manifest(first_report, "1" * 64, "/first/protocol.json")
    _set_protocol_manifest(second_report, "2" * 64, "/second/protocol.json")
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", second_report)

    with pytest.raises(ValueError, match="protocol manifest SHA-256 differs"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_protocol_manifest_is_required_from_every_stage_when_present(tmp_path):
    first_report = _report()
    _set_protocol_manifest(first_report, "1" * 64)
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(ValueError, match="missing from: onnx"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_protocol_manifest_allows_different_paths_for_identical_bytes(tmp_path):
    first_report = _report()
    second_report = _report()
    digest = "1" * 64
    _set_protocol_manifest(first_report, digest, "/first/protocol.json")
    _set_protocol_manifest(second_report, digest, "/second/protocol.json")
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", second_report)

    result = ACCURACY_CHAIN.build_report(
        [("pytorch", first, None), ("onnx", second, None)], allow_missing_lineage=True
    )

    assert result["protocol_manifest"] == {
        "sha256": digest,
        "stage_paths": {
            "pytorch": "/first/protocol.json",
            "onnx": "/second/protocol.json",
        },
    }


def test_missing_evaluation_contract_names_required_producer_data(tmp_path):
    first_report = _report()
    del first_report["evaluation_contract"]
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(
        ValueError,
        match=(
            "evaluator must record dataset hashes, selection, thresholds, Top-K, "
            "inference batch, and geometry"
        ),
    ):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_backend_artifact_identity_is_required(tmp_path):
    first_report = _report()
    del first_report["backends"]["result"]["artifact"]
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(ValueError, match="backends.result.artifact must be an object"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "", "artifact.kind must be a non-empty string"),
        ("sha256", "invalid", "artifact.sha256 must be a lowercase SHA-256 digest"),
        ("recipe", "", "artifact.recipe must be a non-empty string"),
        ("runtime", "", "artifact.runtime must be a non-empty string"),
    ],
)
def test_backend_artifact_identity_is_validated(tmp_path, field, value, message):
    first_report = _report()
    first_report["backends"]["result"]["artifact"][field] = value
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(ValueError, match=message):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_ground_truth_density_histogram_is_validated(tmp_path):
    first_report = _report()
    first_report["ground_truth"]["per_image"]["histogram"][1]["images"] = 0
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(ValueError, match="histogram does not sum to image count"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_metric_population_mismatch_is_rejected(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second_report = _report()
    second_report["backends"]["result"]["map"]["per_class"][0]["name"] = "renamed"
    second = _write(tmp_path / "second.json", second_report)

    with pytest.raises(ValueError, match="per-class identity or GT counts differ"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_missing_nullable_metric_is_not_accepted_as_null(tmp_path):
    first_report = _report()
    del first_report["backends"]["result"]["map"]["APl"]
    first = _write(tmp_path / "first.json", first_report)
    second = _write(tmp_path / "second.json", _report())

    with pytest.raises(ValueError, match="missing required fields: APl"):
        ACCURACY_CHAIN.build_report([("pytorch", first, None), ("onnx", second, None)])


def test_metric_availability_must_match_between_stages(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second_report = _report()
    second_report["backends"]["result"]["map"]["APl"] = 0.0
    second = _write(tmp_path / "second.json", second_report)

    with pytest.raises(ValueError, match="metric availability differs"):
        ACCURACY_CHAIN.build_report(
            [("pytorch", first, None), ("onnx", second, None)], allow_missing_lineage=True
        )


def test_output_alias_and_existing_output_are_rejected(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second = _write(tmp_path / "second.json", _report())
    stages = [("pytorch", first, None), ("onnx", second, None)]

    with pytest.raises(ValueError, match="output aliases stage 'pytorch'"):
        ACCURACY_CHAIN.run(_args(stages, first))

    output = _write(tmp_path / "chain.json", {"preserve": True})
    with pytest.raises(ValueError, match="output already exists"):
        ACCURACY_CHAIN.run(_args(stages, output))
    assert json.loads(output.read_text()) == {"preserve": True}


def test_explicit_transition_labels_must_cover_each_pair(tmp_path):
    first = _write(tmp_path / "first.json", _report())
    second = _write(tmp_path / "second.json", _report())
    third = _write(tmp_path / "third.json", _report())
    stages = [("a", first, None), ("b", second, None), ("c", third, None)]

    with pytest.raises(ValueError, match="once per adjacent stage pair"):
        ACCURACY_CHAIN.build_report(stages, ["only one"])
    with pytest.raises(ValueError, match="--transition-kind"):
        ACCURACY_CHAIN.build_report(stages, transition_kinds=["comparison"])

    result = ACCURACY_CHAIN.build_report(stages, ["first", "second"], allow_missing_lineage=True)
    assert [entry["label"] for entry in result["transitions"]] == ["first", "second"]


def test_atomic_no_clobber_preserves_concurrent_output(tmp_path, monkeypatch):
    output = tmp_path / "chain.json"
    writer_module = sys.modules[ACCURACY_CHAIN.atomic_json.__module__]
    real_link = writer_module.os.link

    def concurrent_link(source, target):
        Path(target).write_text('{"owner":"other"}\n', encoding="utf-8")
        return real_link(source, target)

    monkeypatch.setattr(writer_module.os, "link", concurrent_link)
    with pytest.raises(ValueError, match="output already exists"):
        ACCURACY_CHAIN.atomic_json(output, {"owner": "chain"}, sort_keys=True)

    assert json.loads(output.read_text()) == {"owner": "other"}
    assert list(tmp_path.glob(".chain.json.*.tmp")) == []


def test_stage_argument_supports_backend_selector():
    name, path, backend = ACCURACY_CHAIN._parse_stage("slim=/tmp/report.json::engine")

    assert name == "slim"
    assert path == Path("/tmp/report.json")
    assert backend == "engine"

    with pytest.raises(ACCURACY_CHAIN.argparse.ArgumentTypeError, match="NAME=REPORT"):
        ACCURACY_CHAIN._parse_stage("invalid")
