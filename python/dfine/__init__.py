"""Python bindings for the D-FINE-cpp native runtime.

A thin ctypes wrapper over ``libdfine.so``. CUDA preprocessing, TensorRT
execution, and decode run in native code; Python marshals images and results.

    from dfine import Detector
    with Detector("model.engine") as det:
        detections = det.detect(rgb_hwc_uint8_array)

Requires a TensorRT 10.x + CUDA 12 runtime on the loader path (``pip install
tensorrt-cu12==10.13.*`` or LD_LIBRARY_PATH).
"""

from __future__ import annotations

from .detector import Box, Detection, Detector, set_log_callback

__all__ = ["Box", "Detection", "Detector", "set_log_callback", "__version__", "library_version"]

# Importing the package does not load the native library. library_version()
# performs the load and can be used as an installation smoke test.
__version__ = "0.5.0"


def library_version() -> str:
    """Return the version reported by the loaded native library."""
    from ._ffi import get_lib

    raw = get_lib().dfine_version()
    return raw.decode("utf-8", "replace") if raw else ""
