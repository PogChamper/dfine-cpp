"""coco_eval.py decode(): must handle topk >= Q*C — every 1-class model at the
default --topk 300 crashed np.argpartition before v0.3.1. Skips unless the
eval script's heavy deps (cv2/torch/tensorrt/pycocotools) are importable."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
for dep in ("cv2", "torch", "tensorrt", "pycocotools"):
    pytest.importorskip(dep)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def decode():
    spec = importlib.util.spec_from_file_location(
        "coco_eval", REPO / "trt-files/scripts/coco_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode


def run(decode, q, c, topk):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(1, q, c)).astype(np.float32)
    boxes = rng.random((1, q, 4), dtype=np.float32)
    return decode(logits, boxes, 640, 480, num_classes=c, topk=topk)


def test_one_class_model_at_default_topk(decode):
    xywh, labels, scores = run(decode, q=300, c=1, topk=300)  # Q*C == topk: crashed before
    assert len(scores) == 300 and labels.max() == 0
    assert (np.diff(scores) <= 0).all()  # descending


def test_topk_beyond_candidates_returns_all(decode):
    xywh, labels, scores = run(decode, q=5, c=2, topk=300)
    assert len(scores) == 10


def test_normal_case_unchanged(decode):
    xywh, labels, scores = run(decode, q=300, c=80, topk=300)
    assert len(scores) == 300
    assert (np.diff(scores) <= 0).all()
    assert xywh.shape == (300, 4)
