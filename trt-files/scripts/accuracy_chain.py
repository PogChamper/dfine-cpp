#!/usr/bin/env python3
"""Compare ordered COCO accuracy reports under one frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from evaluation_report import atomic_json, paths_alias, sha256_file

SCALAR_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)
MODEL_SPACE_METRICS = ("APs", "APm", "APl", "ARs", "ARm", "ARl")
AREA_NAMES = ("small", "medium", "large")
IOU_KEYS = tuple(f"{value / 100:.2f}" for value in range(50, 100, 5))
DENSITY_RANGES = (
    "0",
    "1",
    "2-5",
    "6-10",
    "11-25",
    "26-50",
    "51-100",
    "101-200",
    "201-300",
    "301+",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
STAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
STANDARD_MAX_DETS = [1, 10, 100]
LEGACY_MAX_DETS = [1, 10, 100, 300]
TRANSITION_KINDS = ("export", "runtime", "precision", "preset", "comparison")
TRANSITION_LABELS = {
    "export": "export fidelity",
    "runtime": "backend fidelity",
    "precision": "precision",
    "preset": "preset",
    "comparison": "comparison",
}
MODEL_CONTRACT_FIELDS = (
    "model",
    "variant",
    "task",
    "input_h",
    "input_w",
    "num_classes",
    "initial_queries",
    "num_queries",
    "eval_idx",
    "cascade",
    "checkpoint_sha256",
    "preprocess",
)
MODEL_GRAPH_FIELDS = ("initial_queries", "num_queries", "eval_idx", "cascade")
MODEL_IDENTITY_FIELDS = tuple(
    field for field in MODEL_CONTRACT_FIELDS if field not in MODEL_GRAPH_FIELDS
)
LINEAGE_BASE_FIELDS = {
    "artifact_kind",
    "precision_mode",
    "checkpoint_sha256",
    "artifact_sha256",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload, hashlib.sha256(encoded).hexdigest()


def _require_object(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_keys(value: Mapping, expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {', '.join(unexpected)}")
    if details:
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")


def _require_fields(value: Mapping, expected: tuple[str, ...], path: str) -> None:
    missing = [name for name in expected if name not in value]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")


def _require_int(value, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{path} must be an integer {qualifier}")
    return value


def _require_metric(value, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be in [0, 1] or null")
    return result


def _require_probability(value, path: str) -> float:
    result = _require_metric(value, path)
    if result is None:
        raise ValueError(f"{path} must not be null")
    return result


def _require_nonempty_string(value, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_nonnegative_number(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{path} must be a finite non-negative number")
    return result


def _require_sha256(value, path: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _require_area_counts(value, path: str) -> dict[str, int]:
    counts = _require_object(value, path)
    _require_exact_keys(counts, set(AREA_NAMES), path)
    return {name: _require_int(counts[name], f"{path}.{name}") for name in AREA_NAMES}


def _validate_geometry(value, path: str) -> dict:
    geometry = _require_object(value, path)
    _require_exact_keys(geometry, {"canonical", "model_space_area"}, path)
    if geometry["canonical"] != "coco_original_pixels":
        raise ValueError(f"{path}.canonical must be 'coco_original_pixels'")

    model = _require_object(geometry["model_space_area"], f"{path}.model_space_area")
    if model.get("resize") != "stretch":
        raise ValueError(f"{path}.model_space_area.resize must be 'stretch'")
    _require_exact_keys(model, {"input_h", "input_w", "resize"}, f"{path}.model_space_area")
    _require_int(model["input_h"], f"{path}.model_space_area.input_h", minimum=1)
    _require_int(model["input_w"], f"{path}.model_space_area.input_w", minimum=1)
    return geometry


def _validate_contract(report: dict, stage: str) -> dict:
    path = f"stage '{stage}'.evaluation_contract"
    if "evaluation_contract" not in report:
        raise ValueError(
            f"{path} is required; the evaluator must record dataset hashes, selection, "
            "thresholds, Top-K, inference batch, and geometry"
        )
    contract = _require_object(report["evaluation_contract"], path)
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "dataset",
            "evaluator",
            "annotations_sha256",
            "selection",
            "score_threshold",
            "topk",
            "inference_batch_size",
            "geometry",
        },
        path,
    )
    if _require_int(contract["schema_version"], f"{path}.schema_version", minimum=1) != 1:
        raise ValueError(f"{path}.schema_version must be 1")
    dataset = _require_object(contract["dataset"], f"{path}.dataset")
    _require_exact_keys(dataset, {"name", "split", "version"}, f"{path}.dataset")
    for key in ("name", "split", "version"):
        _require_nonempty_string(dataset[key], f"{path}.dataset.{key}")
    evaluator = _require_object(contract["evaluator"], f"{path}.evaluator")
    _require_exact_keys(
        evaluator,
        {"name", "version", "metrics_source_sha256"},
        f"{path}.evaluator",
    )
    _require_nonempty_string(evaluator["name"], f"{path}.evaluator.name")
    _require_nonempty_string(evaluator["version"], f"{path}.evaluator.version")
    _require_sha256(evaluator["metrics_source_sha256"], f"{path}.evaluator.metrics_source_sha256")
    _require_sha256(contract["annotations_sha256"], f"{path}.annotations_sha256")
    selection = _require_object(contract["selection"], f"{path}.selection")
    _require_exact_keys(
        selection,
        {"image_count", "image_manifest_sha256", "image_manifest_scheme"},
        f"{path}.selection",
    )
    image_count = _require_int(selection["image_count"], f"{path}.selection.image_count", minimum=1)
    _require_sha256(selection["image_manifest_sha256"], f"{path}.selection.image_manifest_sha256")
    _require_nonempty_string(
        selection["image_manifest_scheme"], f"{path}.selection.image_manifest_scheme"
    )
    _require_probability(contract["score_threshold"], f"{path}.score_threshold")
    if _require_int(contract["topk"], f"{path}.topk", minimum=1) != 300:
        raise ValueError(f"{path}.topk must be 300 for the frozen evaluation protocol")
    _require_int(
        contract["inference_batch_size"],
        f"{path}.inference_batch_size",
        minimum=1,
    )
    _validate_geometry(contract["geometry"], f"{path}.geometry")

    if "images" not in report:
        raise ValueError(f"stage '{stage}'.images is required")
    if _require_int(report["images"], f"stage '{stage}'.images", minimum=1) != image_count:
        raise ValueError(f"stage '{stage}'.images does not match {path}.selection.image_count")
    _validate_ground_truth(report.get("ground_truth"), stage, image_count)
    return contract


def _validate_ground_truth(value, stage: str, image_count: int) -> None:
    path = f"stage '{stage}'.ground_truth"
    ground_truth = _require_object(value, path)
    _require_fields(
        ground_truth,
        ("images", "gt_instances", "crowd_instances", "per_image"),
        path,
    )
    if _require_int(ground_truth["images"], f"{path}.images", minimum=1) != image_count:
        raise ValueError(f"{path}.images does not match evaluation_contract.selection.image_count")
    gt_instances = _require_int(ground_truth["gt_instances"], f"{path}.gt_instances")
    _require_int(ground_truth["crowd_instances"], f"{path}.crowd_instances")

    per_image = _require_object(ground_truth["per_image"], f"{path}.per_image")
    required = ("min", "mean", "median", "p90", "p95", "p99", "max", "over_100", "histogram")
    _require_fields(per_image, required, f"{path}.per_image")
    minimum = _require_int(per_image["min"], f"{path}.per_image.min")
    maximum = _require_int(per_image["max"], f"{path}.per_image.max")
    over_100 = _require_int(per_image["over_100"], f"{path}.per_image.over_100")
    if minimum > maximum or maximum > gt_instances:
        raise ValueError(f"{path}.per_image min/max are inconsistent with GT count")
    if over_100 > image_count:
        raise ValueError(f"{path}.per_image.over_100 exceeds image count")
    ordered = [
        float(minimum),
        _require_nonnegative_number(per_image["median"], f"{path}.per_image.median"),
        _require_nonnegative_number(per_image["p90"], f"{path}.per_image.p90"),
        _require_nonnegative_number(per_image["p95"], f"{path}.per_image.p95"),
        _require_nonnegative_number(per_image["p99"], f"{path}.per_image.p99"),
        float(maximum),
    ]
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{path}.per_image percentiles are not monotonic")
    mean = _require_nonnegative_number(per_image["mean"], f"{path}.per_image.mean")
    if not minimum <= mean <= maximum:
        raise ValueError(f"{path}.per_image.mean is outside min/max")

    histogram = per_image["histogram"]
    if not isinstance(histogram, list) or len(histogram) != len(DENSITY_RANGES):
        raise ValueError(f"{path}.per_image.histogram must cover the frozen density ranges")
    histogram_total = 0
    for index, (row, expected_range) in enumerate(zip(histogram, DENSITY_RANGES)):
        row_path = f"{path}.per_image.histogram[{index}]"
        row = _require_object(row, row_path)
        _require_exact_keys(row, {"range", "images"}, row_path)
        if row["range"] != expected_range:
            raise ValueError(f"{row_path}.range must be '{expected_range}'")
        histogram_total += _require_int(row["images"], f"{row_path}.images")
    if histogram_total != image_count:
        raise ValueError(f"{path}.per_image.histogram does not sum to image count")


def _validate_model_contract(value, path: str) -> dict:
    contract = _require_object(value, path)
    _require_exact_keys(contract, set(MODEL_CONTRACT_FIELDS), path)
    if contract["model"] != "d-fine" or contract["task"] != "detect":
        raise ValueError(f"{path} must describe D-FINE detection")
    if contract["variant"] not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"{path}.variant is unsupported")
    for field in ("input_h", "input_w", "num_classes", "initial_queries", "num_queries"):
        _require_int(contract[field], f"{path}.{field}", minimum=1)
    _require_int(contract["eval_idx"], f"{path}.eval_idx")
    _require_sha256(contract["checkpoint_sha256"], f"{path}.checkpoint_sha256")

    cascade = contract["cascade"]
    if cascade is None:
        if contract["initial_queries"] != contract["num_queries"]:
            raise ValueError(f"{path} changes query count without a Cascade declaration")
    else:
        if not isinstance(cascade, str):
            raise ValueError(f"{path}.cascade must be null or K:KEEP")
        try:
            layer, keep = (int(component) for component in cascade.split(":"))
        except ValueError:
            raise ValueError(f"{path}.cascade must be null or K:KEEP") from None
        if (
            layer < 0
            or keep != contract["num_queries"]
            or keep >= contract["initial_queries"]
        ):
            raise ValueError(f"{path}.cascade contradicts the query contract")

    preprocess = _require_object(contract["preprocess"], f"{path}.preprocess")
    expected_preprocess = {
        "color_order": "RGB",
        "channel_layout": "NCHW",
        "normalize": "div255",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": "stretch",
    }
    if preprocess != expected_preprocess:
        raise ValueError(f"{path}.preprocess is outside the maintained evaluation contract")
    return contract


def _validate_lineage(value, model_contract: dict, artifact: dict, path: str) -> dict:
    lineage = _require_object(value, path)
    kind = lineage.get("artifact_kind")
    if kind not in {"checkpoint", "onnx", "engine"}:
        raise ValueError(f"{path}.artifact_kind is unsupported")
    expected_fields = set(LINEAGE_BASE_FIELDS)
    if kind == "engine":
        expected_fields.add("onnx_sha256")
    if "source_onnx_sha256" in lineage:
        expected_fields.add("source_onnx_sha256")
    _require_exact_keys(lineage, expected_fields, path)
    _require_nonempty_string(lineage["precision_mode"], f"{path}.precision_mode")
    for field in ("checkpoint_sha256", "artifact_sha256"):
        _require_sha256(lineage[field], f"{path}.{field}")
    if kind == "engine":
        _require_sha256(lineage["onnx_sha256"], f"{path}.onnx_sha256")
    if "source_onnx_sha256" in lineage:
        _require_sha256(lineage["source_onnx_sha256"], f"{path}.source_onnx_sha256")
    if lineage["checkpoint_sha256"] != model_contract["checkpoint_sha256"]:
        raise ValueError(f"{path}.checkpoint_sha256 contradicts the model contract")
    if lineage["artifact_sha256"] != artifact["sha256"]:
        raise ValueError(f"{path}.artifact_sha256 contradicts the artifact record")
    expected_artifact_kind = {
        "checkpoint": "checkpoint",
        "onnx": "onnx",
        "engine": "tensorrt_engine",
    }[kind]
    if artifact["kind"] != expected_artifact_kind:
        raise ValueError(f"{path}.artifact_kind contradicts the artifact record")
    if kind == "checkpoint":
        if lineage["precision_mode"] != "fp32":
            raise ValueError(f"{path}.precision_mode must be fp32 for a checkpoint")
        if lineage["artifact_sha256"] != lineage["checkpoint_sha256"]:
            raise ValueError(f"{path} checkpoint hashes must match")
    return lineage


def _select_metrics(
    report: dict, stage: str, selector: str | None
) -> tuple[str, dict, dict, dict | None, dict | None]:
    backends = _require_object(report.get("backends"), f"stage '{stage}'.backends")
    if selector is None:
        if len(backends) != 1:
            names = ", ".join(sorted(backends)) or "none"
            raise ValueError(
                f"stage '{stage}' report has {len(backends)} backends ({names}); "
                "select one with REPORT::BACKEND"
            )
        selector = next(iter(backends))
    if selector not in backends:
        raise ValueError(f"stage '{stage}' backend not found: {selector}")
    backend = _require_object(backends[selector], f"stage '{stage}'.backends.{selector}")
    backend_path = f"stage '{stage}'.backends.{selector}"
    artifact = _require_object(backend.get("artifact"), f"{backend_path}.artifact")
    _require_fields(
        artifact,
        ("kind", "sha256", "recipe", "runtime"),
        f"{backend_path}.artifact",
    )
    _require_nonempty_string(artifact["kind"], f"{backend_path}.artifact.kind")
    _require_sha256(artifact["sha256"], f"{backend_path}.artifact.sha256")
    _require_nonempty_string(artifact["recipe"], f"{backend_path}.artifact.recipe")
    _require_nonempty_string(artifact["runtime"], f"{backend_path}.artifact.runtime")
    model_contract = backend.get("model_contract")
    lineage = backend.get("lineage")
    if (model_contract is None) != (lineage is None):
        raise ValueError(f"{backend_path} must record model_contract and lineage together")
    if model_contract is not None:
        model_contract = _validate_model_contract(
            model_contract, f"{backend_path}.model_contract"
        )
        lineage = _validate_lineage(
            lineage,
            model_contract,
            artifact,
            f"{backend_path}.lineage",
        )
    metrics = _require_object(backend.get("map"), f"{backend_path}.map")
    return selector, metrics, artifact, model_contract, lineage


def _protocol_manifest(report: dict, stage: str) -> dict | None:
    provenance = report.get("provenance")
    if provenance is None:
        return None
    provenance = _require_object(provenance, f"stage '{stage}'.provenance")
    manifest = provenance.get("protocol_manifest")
    if manifest is None:
        return None
    path = f"stage '{stage}'.provenance.protocol_manifest"
    manifest = _require_object(manifest, path)
    _require_exact_keys(manifest, {"path", "sha256"}, path)
    return {
        "path": _require_nonempty_string(manifest["path"], f"{path}.path"),
        "sha256": _require_sha256(manifest["sha256"], f"{path}.sha256"),
    }


def _extract_metrics(metrics: dict, contract: dict, ground_truth: dict, stage: str) -> dict:
    path = f"stage '{stage}'.map"
    _require_fields(metrics, SCALAR_METRICS, path)
    scalar = {name: _require_metric(metrics[name], f"{path}.{name}") for name in SCALAR_METRICS}
    max_dets = metrics.get("max_dets")
    if not isinstance(max_dets, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in max_dets
    ):
        raise ValueError(f"{path}.max_dets must be an integer array")
    if max_dets not in (STANDARD_MAX_DETS, LEGACY_MAX_DETS):
        raise ValueError(f"{path}.max_dets must be [1, 10, 100] or [1, 10, 100, 300]")

    by_iou = _require_object(metrics.get("AP_by_iou"), f"{path}.AP_by_iou")
    _require_exact_keys(by_iou, set(IOU_KEYS), f"{path}.AP_by_iou")
    ap_by_iou = {key: _require_metric(by_iou[key], f"{path}.AP_by_iou.{key}") for key in IOU_KEYS}

    rows = metrics.get("per_class")
    if not isinstance(rows, list):
        raise ValueError(f"{path}.per_class must be an array")
    per_class = []
    category_ids = set()
    for index, row in enumerate(rows):
        row_path = f"{path}.per_class[{index}]"
        row = _require_object(row, row_path)
        _require_exact_keys(row, {"category_id", "name", "gt_instances", "AP"}, row_path)
        category_id = _require_int(row["category_id"], f"{row_path}.category_id")
        if category_id in category_ids:
            raise ValueError(f"{path}.per_class has duplicate category_id {category_id}")
        category_ids.add(category_id)
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError(f"{row_path}.name must be a non-empty string")
        per_class.append(
            {
                "category_id": category_id,
                "name": row["name"],
                "gt_instances": _require_int(row["gt_instances"], f"{row_path}.gt_instances"),
                "AP": _require_metric(row["AP"], f"{row_path}.AP"),
            }
        )
    if not per_class:
        raise ValueError(f"{path}.per_class must not be empty")
    per_class.sort(key=lambda row: row["category_id"])

    gt_by_area = _require_area_counts(metrics.get("GT_by_area"), f"{path}.GT_by_area")
    gt_instances = ground_truth["gt_instances"]
    if sum(gt_by_area.values()) != gt_instances:
        raise ValueError(f"{path}.GT_by_area does not sum to ground_truth.gt_instances")
    if sum(row["gt_instances"] for row in per_class) != gt_instances:
        raise ValueError(f"{path}.per_class GT counts do not sum to ground_truth.gt_instances")

    model = _require_object(metrics.get("model_space_area"), f"{path}.model_space_area")
    model_geometry = contract["geometry"]["model_space_area"]
    for key, expected in model_geometry.items():
        if model.get(key) != expected:
            raise ValueError(
                f"{path}.model_space_area.{key} does not match "
                f"evaluation_contract.geometry.model_space_area.{key}"
            )
    _require_fields(model, MODEL_SPACE_METRICS, f"{path}.model_space_area")
    model_metrics = {
        name: _require_metric(model[name], f"{path}.model_space_area.{name}")
        for name in MODEL_SPACE_METRICS
    }
    model_gt = _require_area_counts(model.get("GT_by_area"), f"{path}.model_space_area.GT_by_area")
    if sum(model_gt.values()) != gt_instances:
        raise ValueError(
            f"{path}.model_space_area.GT_by_area does not sum to ground_truth.gt_instances"
        )

    return {
        "bbox": scalar,
        "AP_by_iou": ap_by_iou,
        "per_class": per_class,
        "max_dets": list(STANDARD_MAX_DETS),
        "source_max_dets": list(max_dets),
        "GT_by_area": gt_by_area,
        "model_space_area": {**model_metrics, "GT_by_area": model_gt},
    }


def _first_difference(left, right, path: str):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return path, sorted(left), sorted(right)
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(left_value, right_value, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right or type(left) is not type(right):
        return path, left, right
    return None


def _assert_comparable(reference: dict, candidate: dict, reference_name: str, name: str) -> None:
    for key in ("contract", "ground_truth"):
        difference = _first_difference(reference[key], candidate[key], key)
        if difference:
            path, expected, actual = difference
            raise ValueError(
                f"stage '{name}' is incomparable with stage '{reference_name}': "
                f"{path} differs ({actual!r} != {expected!r})"
            )

    reference_metrics = reference["metrics"]
    metrics = candidate["metrics"]
    for path in ("source_max_dets", "GT_by_area", "model_space_area.GT_by_area"):
        left = reference_metrics
        right = metrics
        for component in path.split("."):
            left = left[component]
            right = right[component]
        if left != right:
            raise ValueError(
                f"stage '{name}' is incomparable with stage '{reference_name}': "
                f"metrics.{path} differs"
            )
    left_classes = [
        (row["category_id"], row["name"], row["gt_instances"])
        for row in reference_metrics["per_class"]
    ]
    right_classes = [
        (row["category_id"], row["name"], row["gt_instances"]) for row in metrics["per_class"]
    ]
    if left_classes != right_classes:
        raise ValueError(
            f"stage '{name}' is incomparable with stage '{reference_name}': "
            "per-class identity or GT counts differ"
        )


def _metric_delta(left, right, path: str):
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError(f"metric availability differs at {path}")
    if left == right:
        return 0.0
    return right - left


def _deltas(left: dict, right: dict, left_name: str, right_name: str) -> dict:
    prefix = f"{left_name} -> {right_name}"
    per_class = []
    for left_row, right_row in zip(left["per_class"], right["per_class"]):
        per_class.append(
            {
                "category_id": left_row["category_id"],
                "name": left_row["name"],
                "gt_instances": left_row["gt_instances"],
                "AP": _metric_delta(left_row["AP"], right_row["AP"], f"{prefix}.per_class"),
            }
        )
    return {
        "bbox": {
            name: _metric_delta(left["bbox"][name], right["bbox"][name], f"{prefix}.{name}")
            for name in SCALAR_METRICS
        },
        "AP_by_iou": {
            key: _metric_delta(
                left["AP_by_iou"][key], right["AP_by_iou"][key], f"{prefix}.AP@{key}"
            )
            for key in IOU_KEYS
        },
        "per_class": per_class,
        "model_space_area": {
            name: _metric_delta(
                left["model_space_area"][name],
                right["model_space_area"][name],
                f"{prefix}.model_space_area.{name}",
            )
            for name in MODEL_SPACE_METRICS
        },
    }


def _require_model_fields_equal(left: dict, right: dict, fields, transition: str) -> None:
    for field in fields:
        if left[field] != right[field]:
            raise ValueError(
                f"{transition}: model_contract.{field} differs "
                f"({right[field]!r} != {left[field]!r})"
            )


def _graph_differs(left: dict, right: dict) -> bool:
    return any(left[field] != right[field] for field in MODEL_GRAPH_FIELDS)


def _inferred_transition_kind(left: dict, right: dict) -> str:
    left_lineage = left["lineage"]
    right_lineage = right["lineage"]
    if left_lineage is None or right_lineage is None:
        return "comparison"
    left_kind = left_lineage["artifact_kind"]
    right_kind = right_lineage["artifact_kind"]
    if left_kind == "checkpoint" and right_kind == "onnx":
        return "export"
    if left_kind == "onnx" and right_kind == "engine":
        return "runtime"
    if left_kind == right_kind and left_kind in {"onnx", "engine"}:
        if left_lineage["precision_mode"] != right_lineage["precision_mode"]:
            return "precision"
        if _graph_differs(left["model_contract"], right["model_contract"]):
            return "preset"
    return "comparison"


def _validate_transition(
    left: dict, right: dict, requested_kind: str | None, allow_missing_lineage: bool
) -> tuple[str, bool]:
    transition = f"{left['name']} -> {right['name']}"
    complete = all(
        value is not None
        for value in (
            left["model_contract"],
            left["lineage"],
            right["model_contract"],
            right["lineage"],
        )
    )
    inferred = _inferred_transition_kind(left, right)
    kind = requested_kind or inferred
    if kind not in TRANSITION_KINDS:
        raise ValueError(f"{transition}: unsupported transition kind {kind!r}")
    if not complete:
        missing = [
            stage["name"]
            for stage in (left, right)
            if stage["model_contract"] is None or stage["lineage"] is None
        ]
        if not allow_missing_lineage:
            raise ValueError(
                f"{transition}: stage(s) {', '.join(missing)} record no model_contract/lineage, "
                "so model identity cannot be verified; regenerate the reports with current "
                "tooling or pass --allow-missing-lineage to compare anyway"
            )
        if kind != "comparison":
            raise ValueError(
                f"{transition}: transition kind '{kind}' requires model_contract and lineage"
            )
        return kind, False

    left_model = left["model_contract"]
    right_model = right["model_contract"]
    left_lineage = left["lineage"]
    right_lineage = right["lineage"]
    _require_model_fields_equal(
        left_model,
        right_model,
        MODEL_IDENTITY_FIELDS,
        transition,
    )
    if left_lineage["checkpoint_sha256"] != right_lineage["checkpoint_sha256"]:
        raise ValueError(f"{transition}: checkpoint lineage differs")

    left_kind = left_lineage["artifact_kind"]
    right_kind = right_lineage["artifact_kind"]
    left_precision = left_lineage["precision_mode"]
    right_precision = right_lineage["precision_mode"]

    if kind == "comparison":
        return kind, True
    if requested_kind is not None and inferred != kind:
        raise ValueError(
            f"{transition}: artifacts describe '{inferred}', not requested '{kind}'"
        )

    if kind != "preset":
        _require_model_fields_equal(left_model, right_model, MODEL_GRAPH_FIELDS, transition)
    if kind == "export":
        if (left_kind, right_kind) != ("checkpoint", "onnx"):
            raise ValueError(f"{transition}: export requires checkpoint -> ONNX")
        if left_precision != "fp32" or right_precision != "fp32":
            raise ValueError(f"{transition}: export fidelity requires FP32 on both stages")
        if "source_onnx_sha256" in right_lineage:
            raise ValueError(f"{transition}: export target must be the raw ONNX graph")
    elif kind == "runtime":
        if (left_kind, right_kind) != ("onnx", "engine"):
            raise ValueError(f"{transition}: runtime fidelity requires ONNX -> engine")
        if left_precision != right_precision:
            raise ValueError(f"{transition}: runtime fidelity cannot change precision")
        if right_lineage["onnx_sha256"] != left_lineage["artifact_sha256"]:
            raise ValueError(f"{transition}: engine was not built from the ONNX stage")
    elif kind == "precision":
        if left_kind != right_kind or left_kind not in {"onnx", "engine"}:
            raise ValueError(f"{transition}: precision requires the same artifact kind")
        if left_precision == right_precision:
            raise ValueError(f"{transition}: precision mode did not change")
        source = right_lineage.get("source_onnx_sha256")
        expected_source = (
            left_lineage["artifact_sha256"]
            if left_kind == "onnx"
            else left_lineage["onnx_sha256"]
        )
        if source != expected_source:
            raise ValueError(f"{transition}: precision artifact has the wrong FP32 source")
    elif kind == "preset":
        if left_kind != right_kind or left_kind not in {"onnx", "engine"}:
            raise ValueError(f"{transition}: preset comparison requires one artifact kind")
        if left_precision != right_precision:
            raise ValueError(f"{transition}: preset comparison cannot change precision")
        if not _graph_differs(left_model, right_model):
            raise ValueError(f"{transition}: preset graph contract did not change")
    return kind, True


def _parse_stage(value: str) -> tuple[str, Path, str | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("stage must be NAME=REPORT or NAME=REPORT::BACKEND")
    name, source = value.split("=", 1)
    if not STAGE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "stage name must start with an alphanumeric character and use only ._-"
        )
    selector = None
    if "::" in source:
        source, selector = source.rsplit("::", 1)
        if not selector:
            raise argparse.ArgumentTypeError("backend selector must not be empty")
    if not source:
        raise argparse.ArgumentTypeError("report path must not be empty")
    return name, Path(source), selector


def build_report(
    stage_specs, transition_labels=None, transition_kinds=None, allow_missing_lineage=False
) -> dict:
    if len(stage_specs) < 2:
        raise ValueError("at least two --stage inputs are required")
    names = [name for name, _, _ in stage_specs]
    if len(names) != len(set(names)):
        raise ValueError("stage names must be unique")
    if transition_labels and len(transition_labels) != len(stage_specs) - 1:
        raise ValueError("--transition-label must be provided once per adjacent stage pair")
    if transition_kinds and len(transition_kinds) != len(stage_specs) - 1:
        raise ValueError("--transition-kind must be provided once per adjacent stage pair")

    stages = []
    comparable = []
    protocol_manifests = []
    for name, source, selector in stage_specs:
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"stage '{name}' report not found: {source}")
        report, report_sha256 = _read_json(source)
        contract = _validate_contract(report, name)
        backend, raw_metrics, artifact, model_contract, lineage = _select_metrics(
            report, name, selector
        )
        metrics = _extract_metrics(raw_metrics, contract, report["ground_truth"], name)
        protocol_manifest = _protocol_manifest(report, name)
        entry = {
            "name": name,
            "backend": backend,
            "report": str(source),
            "report_sha256": report_sha256,
            "artifact": artifact,
            "model_contract": model_contract,
            "lineage": lineage,
            "metrics": metrics,
            "protocol_manifest": protocol_manifest,
        }
        stages.append(entry)
        protocol_manifests.append(protocol_manifest)
        comparable.append(
            {"contract": contract, "ground_truth": report["ground_truth"], "metrics": metrics}
        )

    for index in range(1, len(stages)):
        _assert_comparable(comparable[0], comparable[index], names[0], names[index])

    present_manifests = [manifest for manifest in protocol_manifests if manifest is not None]
    if present_manifests and len(present_manifests) != len(protocol_manifests):
        missing = [name for name, manifest in zip(names, protocol_manifests) if manifest is None]
        raise ValueError(
            "protocol manifest must be present for every stage when any stage records one; "
            f"missing from: {', '.join(missing)}"
        )
    protocol_manifest = None
    if present_manifests:
        expected_hash = present_manifests[0]["sha256"]
        mismatched = [
            name
            for name, manifest in zip(names, protocol_manifests)
            if manifest["sha256"] != expected_hash
        ]
        if mismatched:
            raise ValueError(
                "protocol manifest SHA-256 differs across stages: " + ", ".join(mismatched)
            )
        protocol_manifest = {
            "sha256": expected_hash,
            "stage_paths": {
                name: manifest["path"] for name, manifest in zip(names, protocol_manifests)
            },
        }

    transitions = []
    for index, (left, right) in enumerate(zip(stages, stages[1:])):
        requested_kind = transition_kinds[index] if transition_kinds else None
        kind, lineage_verified = _validate_transition(
            left, right, requested_kind, allow_missing_lineage
        )
        label = (
            transition_labels[index]
            if transition_labels
            else TRANSITION_LABELS[kind]
        )
        if not isinstance(label, str) or not label.strip():
            raise ValueError("transition labels must be non-empty strings")
        transitions.append(
            {
                "kind": kind,
                "label": label,
                "lineage_verified": lineage_verified,
                "from_stage": left["name"],
                "to_stage": right["name"],
                "delta": _deltas(left["metrics"], right["metrics"], left["name"], right["name"]),
            }
        )

    return {
        "schema_version": 2,
        "tool": "accuracy-chain",
        "metric_unit": "fraction; deltas are to_stage - from_stage",
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "evaluation_contract": comparable[0]["contract"],
        "protocol_manifest": protocol_manifest,
        "ground_truth": comparable[0]["ground_truth"],
        "stages": stages,
        "transitions": transitions,
    }


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    stage_specs = [(name, path.resolve(), selector) for name, path, selector in args.stage]
    for name, source, _ in stage_specs:
        if paths_alias(output, source):
            raise ValueError(f"output aliases stage '{name}' report: {source}")
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {output}; pass --overwrite to replace it")
    report = build_report(
        stage_specs,
        args.transition_label,
        getattr(args, "transition_kind", None),
        allow_missing_lineage=args.allow_missing_lineage,
    )
    atomic_json(output, report, overwrite=args.overwrite, sort_keys=True)
    print(f"[accuracy-chain] wrote {output}")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ordered expanded COCO reports under one evaluation contract"
    )
    parser.add_argument(
        "--stage",
        action="append",
        type=_parse_stage,
        required=True,
        metavar="NAME=REPORT[::BACKEND]",
        help="ordered stage and optional backend selector",
    )
    parser.add_argument(
        "--transition-label",
        action="append",
        default=[],
        help="explicit adjacent transition label; repeat for every pair",
    )
    parser.add_argument(
        "--transition-kind",
        action="append",
        choices=TRANSITION_KINDS,
        default=[],
        help="validated adjacent axis; repeat for every pair",
    )
    parser.add_argument(
        "--allow-missing-lineage",
        action="store_true",
        help="compare stages whose reports predate model_contract/lineage recording; "
        "the affected transitions are marked lineage_verified=false",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except ValueError as exc:
        raise SystemExit(f"[accuracy-chain] {exc}") from exc
