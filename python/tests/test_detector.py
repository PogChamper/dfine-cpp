"""Smoke + parity tests for the dfine Python bindings.

The headline test (`test_parity_with_cpp_dfine_detect`) proves the Python path
returns the SAME detections as the C++ `dfine_detect` binary on the same pixels:
it round-trips the image through a lossless PNG (so stb and PIL decode identical
bytes) and compares printf-formatted strings (so rounding matches exactly)."""

from __future__ import annotations

import os
import re
import subprocess

import pytest

import dfine

# ----- basic / no-GPU-needed ------------------------------------------------ #


def test_package_version_is_static():
    assert isinstance(dfine.__version__, str) and dfine.__version__


def test_library_version(lib):
    assert dfine.library_version()  # non-empty; matches the C dfine_version()


def test_bad_engine_raises(lib):
    with pytest.raises(RuntimeError):
        dfine.Detector("/nonexistent/does_not_exist.engine")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_invalid_threshold_before_library_load(monkeypatch, threshold):
    from dfine import detector as detector_module

    def unexpected_library_load():
        raise AssertionError("native library load must follow threshold validation")

    monkeypatch.setattr(detector_module, "get_lib", unexpected_library_load)
    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        detector_module.Detector("unused.engine", threshold=threshold)


def test_per_call_none_uses_constructor_threshold():
    detector = object.__new__(dfine.Detector)
    detector._threshold = 0.4

    assert detector._resolve_threshold(None) == pytest.approx(0.4)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_detect_rejects_invalid_threshold_before_native_call(threshold):
    detector = object.__new__(dfine.Detector)
    detector._threshold = 0.4

    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        detector.detect(None, threshold=threshold)


# ----- construction / introspection ---------------------------------------- #


def test_construct_and_introspect(detector):
    assert detector.input_width == 640
    assert detector.input_height == 640
    assert detector.num_classes == 80
    assert detector.num_queries == 300
    assert detector.max_batch >= 1
    assert detector.variant  # non-empty


def test_context_manager_closes(engine_path):
    with dfine.Detector(engine_path) as det:
        assert det.input_width == 640
    # After the block the handle is released; use-after-close raises.
    with pytest.raises(RuntimeError):
        det.input_width  # noqa: B018


def test_double_close_is_safe(engine_path):
    det = dfine.Detector(engine_path)
    det.close()
    det.close()  # idempotent, no crash


# ----- detection ------------------------------------------------------------ #


def test_detect_synthetic_gray(detector):
    np = pytest.importorskip("numpy")
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    dets = detector.detect(img)
    assert isinstance(dets, list)  # may be empty on a flat gray image


def test_detect_no_leak_stress(detector):
    """Repeated detect must free the C result every call (no growth / crash)."""
    np = pytest.importorskip("numpy")
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    for _ in range(50):
        detector.detect(img, threshold=0.3)


def test_detect_real_image(detector, coco_image):
    arr, _ = coco_image
    dets = detector.detect(arr, threshold=0.5)
    assert len(dets) > 0
    for d in dets:
        assert 0 <= d.class_id < detector.num_classes
        assert 0.5 <= d.score <= 1.0
        assert d.box.x2 > d.box.x1 and d.box.y2 > d.box.y1
        assert isinstance(d.class_name, str) and d.class_name
    # to_dict round-trips
    assert dets[0].as_dict()["class_id"] == dets[0].class_id


def test_threshold_monotonic(detector, coco_image):
    arr, _ = coco_image
    hi = detector.detect(arr, threshold=0.8)
    lo = detector.detect(arr, threshold=0.2)
    assert len(lo) >= len(hi)


def test_detect_flipped_view_is_safe(detector, coco_image):
    """A negative-row-stride view (np.flipud / [::-1]) must NOT read out of bounds and
    must give the same result as a contiguous copy of the same pixels (regression:
    the zero-copy fast path used to pass the negative stride straight to native code)."""
    np = pytest.importorskip("numpy")
    arr, _ = coco_image
    for view in (np.flipud(arr), arr[::-1], arr[:, ::-1]):
        assert view.strides[0] < 0 or view.strides[1] < 0  # genuinely non-forward
        got = detector.detect(view, threshold=0.4)
        ref = detector.detect(np.ascontiguousarray(view), threshold=0.4)
        assert [d.as_dict() for d in got] == [d.as_dict() for d in ref]


def test_construction_threshold_zero_is_honored(engine_path, coco_image):
    """Detector(threshold=0.0) must keep all detections, not silently snap to 0.5."""
    arr, _ = coco_image
    with dfine.Detector(engine_path, threshold=0.0) as d0:
        many = d0.detect(arr)  # no per-call override -> uses the 0.0 default
    with dfine.Detector(engine_path, threshold=0.5) as d5:
        few = d5.detect(arr)
    assert len(many) > len(few)


def test_detect_batch_matches_single(detector, coco_image):
    if detector.max_batch < 2:
        pytest.skip("engine max_batch < 2")
    arr, _ = coco_image
    single = detector.detect(arr, threshold=0.4)
    batch = detector.detect_batch([arr, arr], threshold=0.4)
    assert len(batch) == 2
    for b in batch:
        assert [d.as_dict() for d in b] == [d.as_dict() for d in single]


# ----- parity vs the C++ reference binary ----------------------------------- #

_LINE = re.compile(
    r"^\s+(\S.*?)\s+(\d+\.\d{3})\s+\[(-?\d+\.\d), (-?\d+\.\d), (-?\d+\.\d), (-?\d+\.\d)\]\s*$"
)


def _parse_cpp(output: str) -> set:
    rows = set()
    for line in output.splitlines():
        m = _LINE.match(line)
        if m:
            rows.add(tuple(m.groups()))  # (name, score3, x1, y1, x2, y2) as strings
    return rows


def _format_py(dets) -> set:
    # Format exactly like dfine_detect: score %.3f, box %.1f. Identical floats +
    # identical format spec => identical strings (IEEE round-to-even both sides).
    return {
        (
            d.class_name,
            f"{d.score:.3f}",
            f"{d.box.x1:.1f}",
            f"{d.box.y1:.1f}",
            f"{d.box.x2:.1f}",
            f"{d.box.y2:.1f}",
        )
        for d in dets
    }


def test_parity_with_cpp_dfine_detect(detector, engine_path, coco_image, repo_root, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    binary = repo_root / "build" / "dfine_detect"
    if not binary.exists():
        pytest.skip("build/dfine_detect not built")

    arr, _ = coco_image
    # Lossless PNG so the C++ side (stb) decodes byte-identical pixels to `arr`.
    png = tmp_path / "parity.png"
    Image.fromarray(arr).save(png)

    thr = 0.3
    proc = subprocess.run(
        [str(binary), "--engine", engine_path, "--image", str(png), "--threshold", str(thr)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        pytest.skip(f"dfine_detect failed (env/LD_LIBRARY_PATH?): {proc.stderr.strip()}")

    cpp_rows = _parse_cpp(proc.stdout)
    py_rows = _format_py(detector.detect(arr, threshold=thr))

    assert cpp_rows, "reference produced no parseable detections"
    assert (
        py_rows == cpp_rows
    ), f"\npython-only: {sorted(py_rows - cpp_rows)}\ncpp-only:    {sorted(cpp_rows - py_rows)}"
