#!/usr/bin/env python3
"""Shared dataset-evaluation preconditions."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sized

# The single preprocessing contract every report producer emits. The chain
# validator keeps its own literal on purpose, as an independent cross-check.
STRETCH_PREPROCESS = {
    "color_order": "RGB",
    "channel_layout": "NCHW",
    "normalize": "div255",
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "resize": "stretch",
}


MODEL_VARIANTS = {"n", "s", "m", "l", "x"}


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def positive_meta_int(meta: dict, field: str, fail, *, optional: bool = False) -> int | None:
    if optional and field not in meta:
        return None
    value = meta.get(field)
    if type(value) is not int or value <= 0:
        fail(f"{field} must be a positive integer")
    return value


def sha256_meta(meta: dict, field: str, fail) -> str:
    value = meta.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{field} must be a lowercase SHA-256")
    return value


def normalized_model_contract(meta: dict, fail, *, extra_expected: dict | None = None) -> dict:
    """Validate a D-FINE sidecar into the normalized model contract.

    `fail` must raise; the caller owns the error type and message prefix so the
    producers keep their established CLI wording.
    """
    expected = {
        **(extra_expected or {}),
        "model": "d-fine",
        "task": "detect",
        "input_names": ["images"],
        "output_names": ["logits", "boxes"],
        **STRETCH_PREPROCESS,
        "checkpoint_load": "strict",
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            fail(f"{field} must be {value!r}")

    variant = meta.get("variant")
    if variant not in MODEL_VARIANTS:
        fail(f"unsupported D-FINE variant {variant!r}")
    eval_idx = meta.get("eval_idx")
    if type(eval_idx) is not int or eval_idx < 0:
        fail("eval_idx must be a non-negative integer")

    output_queries = positive_meta_int(meta, "num_queries", fail)
    cascade = meta.get("cascade")
    if cascade is None:
        if "cascade_initial_queries" in meta:
            fail("cascade_initial_queries requires a cascade declaration")
        initial_queries = output_queries
    else:
        if not isinstance(cascade, str):
            fail("cascade must be a K:KEEP string")
        try:
            layer, keep = (int(value) for value in cascade.split(":"))
        except ValueError:
            fail("cascade must be a K:KEEP string")
        initial_queries = positive_meta_int(meta, "cascade_initial_queries", fail)
        if layer < 0 or keep != output_queries or keep >= initial_queries:
            fail("cascade fields contradict the query contract")

    return {
        "model": "d-fine",
        "variant": variant,
        "task": "detect",
        "input_h": positive_meta_int(meta, "input_h", fail),
        "input_w": positive_meta_int(meta, "input_w", fail),
        "num_classes": positive_meta_int(meta, "num_classes", fail),
        "initial_queries": initial_queries,
        "num_queries": output_queries,
        "eval_idx": eval_idx,
        "cascade": cascade,
        "checkpoint_sha256": sha256_meta(meta, "checkpoint_sha256", fail),
        "preprocess": dict(STRETCH_PREPROCESS),
    }


def artifact_lineage_from_meta(meta: dict, kind: str, contract: dict, fail) -> dict:
    precision_mode = meta.get("precision_mode")
    if not isinstance(precision_mode, str) or not precision_mode:
        fail("precision_mode must be a non-empty string")
    lineage = {
        "artifact_kind": kind,
        "precision_mode": precision_mode,
        "checkpoint_sha256": contract["checkpoint_sha256"],
    }
    if kind == "engine":
        lineage["onnx_sha256"] = sha256_meta(meta, "onnx_sha256", fail)
    if "source_onnx_sha256" in meta:
        lineage["source_onnx_sha256"] = sha256_meta(meta, "source_onnx_sha256", fail)
    return lineage


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return value


def probability(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("must be finite and within 0..1")
    return value


def byte_value(text: str) -> int:
    value = int(text)
    if value < 0 or value > 255:
        raise argparse.ArgumentTypeError("must be within 0..255")
    return value


def resolution(text: str) -> str:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must use WxH")
    try:
        width, height = (positive_int(part) for part in parts)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("must use positive WxH dimensions") from error
    return f"{width}x{height}"


def require_detection_outputs(logits, boxes, batch: int, classes: int, label: str):
    import numpy as np

    logits = np.asarray(logits)
    boxes = np.asarray(boxes)
    if logits.ndim != 3 or logits.shape[0] != batch or logits.shape[1] <= 0:
        raise SystemExit(f"{label}: logits must have shape [{batch},Q,C]")
    if logits.shape[2] != classes:
        raise SystemExit(f"{label}: logits expose {logits.shape[2]} classes; expected {classes}")
    if boxes.shape != (batch, logits.shape[1], 4):
        raise SystemExit(f"{label}: boxes must have shape [{batch},{logits.shape[1]},4]")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise SystemExit(f"{label}: model outputs contain NaN or Inf")
    return logits, boxes


def require_trt_success(success: bool, operation: str) -> None:
    if not success:
        raise RuntimeError(f"TensorRT rejected {operation}")


def require_detections(detections: Sized, label: str) -> None:
    if len(detections) != 0:
        return
    print(f"{label}: zero detections; evaluation aborted", file=sys.stderr)
    raise SystemExit(1)


def require_complete_images(expected: int, processed: int, label: str) -> None:
    if expected > 0 and processed == expected:
        return
    print(f"{label}: processed {processed}/{expected} images; evaluation aborted", file=sys.stderr)
    raise SystemExit(1)


def require_arguments(parser, args, requirements: list[tuple[str, str, str]]) -> None:
    missing = [
        f"{flag} (or {env_name})"
        for attr, flag, env_name in requirements
        if not getattr(args, attr)
    ]
    if missing:
        parser.error("required: " + ", ".join(missing))
