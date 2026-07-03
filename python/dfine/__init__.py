"""D-FINE-cpp — Python bindings for the C++/TensorRT D-FINE detector.

A thin, memory-safe ctypes wrapper over ``libdfine.so`` (the stable C ABI in
``dfine/c_api.h``). All CUDA/TensorRT work happens in native code; Python only
marshals image bytes in and detections out.

    from dfine import Detector
    with Detector("model.engine") as det:
        detections = det.detect(rgb_hwc_uint8_array)

Requires a TensorRT 10.x + CUDA 12 runtime on the loader path (``pip install
tensorrt`` or LD_LIBRARY_PATH). See docs/HANDOFF.md.
"""

from __future__ import annotations

from .detector import Box, Detection, Detector, set_log_callback

__all__ = ["Box", "Detection", "Detector", "set_log_callback", "__version__", "library_version"]

# Static package version (mirrors the C library's dfine_version()). Kept static
# so `import dfine` does not require libdfine to be loadable.
__version__ = "0.3.0"


def library_version() -> str:
    """The version reported by the loaded libdfine.so (requires the native lib)."""
    from ._ffi import get_lib

    raw = get_lib().dfine_version()
    return raw.decode("utf-8", "replace") if raw else ""
