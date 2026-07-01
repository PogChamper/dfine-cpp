"""Shared pytest fixtures. Every fixture skips gracefully when its prerequisite
(the built library, a GPU/engine, a test image, PIL) is unavailable, so the
suite is a no-op on machines without the native stack rather than a failure."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]  # python/tests/ -> repo root


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def lib():
    """The loaded libdfine, or skip if it (or its TensorRT deps) can't load."""
    from dfine._ffi import get_lib

    try:
        return get_lib()
    except RuntimeError as e:
        pytest.skip(f"libdfine not loadable: {e}")


@pytest.fixture(scope="session")
def engine_path(lib) -> str:
    """Resolve a test engine: $DFINE_TEST_ENGINE or the dev-tree m/fp32 engine."""
    env = os.environ.get("DFINE_TEST_ENGINE")
    candidates = [env] if env else []
    candidates += [
        str(REPO / "trt-files/engines/dfine_m_fp32.engine"),
        str(REPO / "trt-files/engines/dfine_m_fp16_st.engine"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    pytest.skip("no test engine (set DFINE_TEST_ENGINE or build trt-files/engines/*.engine)")


@pytest.fixture()
def detector(engine_path):
    """A live Detector on the test engine (skips if construction fails, e.g. no GPU)."""
    from dfine import Detector

    try:
        det = Detector(engine_path, threshold=0.5)
    except RuntimeError as e:
        pytest.skip(f"cannot construct detector (no GPU?): {e}")
    try:
        yield det
    finally:
        det.close()


@pytest.fixture(scope="session")
def coco_image():
    """(numpy RGB HWC uint8, source path) for a real photo — or skip."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    env = os.environ.get("DFINE_TEST_IMAGE")
    candidates = [env] if env else []
    candidates += [
        "/mnt/d/datasets/coco/val2017/000000000139.jpg",
        "/mnt/d/datasets/coco/val2017/000000000285.jpg",
    ]
    for c in candidates:
        if c and Path(c).exists():
            arr = np.asarray(Image.open(c).convert("RGB"))
            return arr, c
    pytest.skip("no test image (set DFINE_TEST_IMAGE)")
