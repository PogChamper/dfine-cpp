from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_eval_contract(name):
    path = REPO / "trt-files/scripts/eval_contract.py"
    spec = importlib.util.spec_from_file_location(f"eval_contract_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def require_detections():
    return _load_eval_contract("detections").require_detections


@pytest.fixture(scope="module")
def require_arguments():
    return _load_eval_contract("arguments").require_arguments


@pytest.fixture(scope="module")
def require_complete_images():
    return _load_eval_contract("images").require_complete_images


def test_nonempty_evaluation_continues(require_detections):
    require_detections([object()], "evaluator")


def test_empty_evaluation_exits_nonzero(require_detections, capsys):
    with pytest.raises(SystemExit) as exc:
        require_detections([], "evaluator")
    assert exc.value.code == 1
    assert capsys.readouterr().err == "evaluator: zero detections; evaluation aborted\n"


def test_incomplete_image_set_exits_nonzero(require_complete_images, capsys):
    with pytest.raises(SystemExit) as exc:
        require_complete_images(10, 9, "evaluator")
    assert exc.value.code == 1
    assert capsys.readouterr().err == "evaluator: processed 9/10 images; evaluation aborted\n"


def test_required_arguments_report_flags_and_environment(require_arguments, capsys):
    parser = argparse.ArgumentParser(prog="evaluator")
    args = argparse.Namespace(engine="", images="images")
    requirements = [
        ("engine", "--engine", "ENGINE"),
        ("images", "--images", "COCO_IMAGES"),
    ]
    with pytest.raises(SystemExit) as exc:
        require_arguments(parser, args, requirements)
    assert exc.value.code == 2
    assert "--engine (or ENGINE)" in capsys.readouterr().err


def test_required_arguments_accept_complete_namespace(require_arguments):
    parser = argparse.ArgumentParser(prog="evaluator")
    args = argparse.Namespace(engine="engine", images="images")
    require_arguments(
        parser,
        args,
        [("engine", "--engine", "ENGINE"), ("images", "--images", "COCO_IMAGES")],
    )


@pytest.mark.parametrize("value", ["1", "8"])
def test_positive_int_accepts_positive_values(value):
    module = _load_eval_contract("numeric")
    assert module.positive_int(value) == int(value)


@pytest.mark.parametrize("value", ["-1", "0"])
def test_positive_int_rejects_nonpositive_values(value):
    module = _load_eval_contract("positive")
    with pytest.raises(argparse.ArgumentTypeError):
        module.positive_int(value)


def test_nonnegative_int_accepts_zero_and_rejects_negative():
    module = _load_eval_contract("nonnegative")
    assert module.nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        module.nonnegative_int("-1")


@pytest.mark.parametrize("value", ["0", "0.5", "1"])
def test_probability_accepts_unit_interval(value):
    module = _load_eval_contract("probability")
    assert module.probability(value) == float(value)


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_probability_rejects_out_of_range_or_nonfinite(value):
    module = _load_eval_contract("bad_probability")
    with pytest.raises(argparse.ArgumentTypeError):
        module.probability(value)


def test_byte_value_enforces_range():
    module = _load_eval_contract("byte_value")
    assert module.byte_value("0") == 0
    assert module.byte_value("255") == 255
    with pytest.raises(argparse.ArgumentTypeError):
        module.byte_value("256")


def test_resolution_requires_positive_dimensions():
    module = _load_eval_contract("resolution")
    assert module.resolution("640X480") == "640x480"
    for value in ("640", "640x", "0x480", "640x480x3"):
        with pytest.raises(argparse.ArgumentTypeError):
            module.resolution(value)


def test_detection_output_contract_rejects_shape_class_and_nonfinite_values():
    module = _load_eval_contract("outputs")
    logits = np.zeros((1, 300, 80), dtype=np.float32)
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    valid_logits, valid_boxes = module.require_detection_outputs(logits, boxes, 1, 80, "evaluator")
    assert valid_logits is logits
    assert valid_boxes is boxes

    with pytest.raises(SystemExit, match="expose 80 classes; expected 3"):
        module.require_detection_outputs(logits, boxes, 1, 3, "evaluator")
    with pytest.raises(SystemExit, match="boxes must have shape"):
        module.require_detection_outputs(logits, boxes[:, :299], 1, 80, "evaluator")
    bad_logits = logits.copy()
    bad_logits[0, 0, 0] = np.nan
    with pytest.raises(SystemExit, match="NaN or Inf"):
        module.require_detection_outputs(bad_logits, boxes, 1, 80, "evaluator")


def test_trt_failure_is_not_ignored():
    module = _load_eval_contract("trt_success")
    module.require_trt_success(True, "enqueueV3")
    with pytest.raises(RuntimeError, match="TensorRT rejected enqueueV3"):
        module.require_trt_success(False, "enqueueV3")
