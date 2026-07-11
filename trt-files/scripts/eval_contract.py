#!/usr/bin/env python3
"""Shared dataset-evaluation preconditions."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sized


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
