"""Private ctypes layer for the D-FINE C ABI (dfine/c_api.h).

Nothing in here is part of the public Python API — use :class:`dfine.Detector`.
This module locates and loads ``libdfine.so``, best-effort preloads the TensorRT
runtime, mirrors the C structs, and pins argtypes/restypes exactly once.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int,
    c_uint8,
    c_void_p,
)
from pathlib import Path

# --------------------------------------------------------------------------- #
# C structs — layout must match dfine/c_api.h exactly.
# --------------------------------------------------------------------------- #


class _Box(Structure):
    _fields_ = [("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float)]


class _Detection(Structure):
    _fields_ = [("box", _Box), ("class_id", c_int), ("score", c_float)]


class _Detections(Structure):
    _fields_ = [("detections", POINTER(_Detection)), ("count", c_int)]


class _Image(Structure):
    _fields_ = [
        ("data", POINTER(c_uint8)),
        ("width", c_int),
        ("height", c_int),
        ("step", c_int),
        ("channels", c_int),
        ("is_bgr", c_int),
    ]


class _Options(Structure):
    _fields_ = [
        ("threshold", c_float),
        ("use_cuda_graph", c_int),
        ("graph_warmup_iters", c_int),
    ]


# (int severity, const char* message) -> void
LOG_FN = CFUNCTYPE(None, c_int, c_char_p)

# Platform library names. Windows lists both because CMake's default target name
# under MSVC is `dfine.dll` (no `lib` prefix) while MinGW emits `libdfine.dll`.
if sys.platform == "win32":
    _LIBNAMES = ("dfine.dll", "libdfine.dll")
elif sys.platform == "darwin":
    _LIBNAMES = ("libdfine.dylib",)
else:
    _LIBNAMES = ("libdfine.so",)


# --------------------------------------------------------------------------- #
# Library discovery + loading
# --------------------------------------------------------------------------- #


def _candidate_paths() -> list[Path]:
    """Ordered search locations for libdfine, most specific first."""
    here = Path(__file__).resolve()
    cands: list[Path] = []

    env = os.environ.get("DFINE_LIBRARY")
    if env:
        cands.append(Path(env))  # full path, any name

    repo = here.parents[2]  # python/dfine/_ffi.py -> repo root
    for name in _LIBNAMES:
        cands.append(here.parent / name)          # bundled next to the package (wheel)
        cands.append(here.parent / "lib" / name)  # or in a lib/ subdir
        cands.append(repo / "build" / name)       # developer tree
    return cands


def _preload_tensorrt() -> None:
    """Best-effort: dlopen the TensorRT runtime with RTLD_GLOBAL so libdfine's
    NVINFER symbols resolve without the user setting LD_LIBRARY_PATH. Silent on
    failure — the caller may already have the libs on the loader path."""
    if sys.platform == "win32":
        return
    try:
        import tensorrt_libs  # type: ignore

        libdir = Path(tensorrt_libs.__file__).parent
    except Exception:
        return
    for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        p = libdir / name
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load_library() -> ctypes.CDLL:
    _preload_tensorrt()
    tried: list[str] = []
    for cand in _candidate_paths():
        tried.append(str(cand))
        if cand.exists():
            try:
                return ctypes.CDLL(str(cand))
            except OSError as e:  # found it, but a dependency failed to load
                libpath = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
                raise RuntimeError(
                    f"Found {cand} but could not load it ({e}).\n"
                    "The TensorRT 10.x + CUDA 12 runtime must be importable. Either "
                    "`pip install tensorrt==10.13.*` into this environment, or add the "
                    f"TensorRT/CUDA lib dirs to {libpath} (see docs/HANDOFF.md)."
                ) from e

    # Last resort: let the loader search standard paths.
    for name in _LIBNAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    searched = "\n  ".join(tried)
    raise RuntimeError(
        f"Could not locate {_LIBNAMES[0]}. Searched:\n  {searched}\n"
        "Set DFINE_LIBRARY to the full path of the built library, or build it "
        "with ./build.sh (produces build/libdfine.so)."
    )


# --------------------------------------------------------------------------- #
# Signature binding
# --------------------------------------------------------------------------- #


def _configure(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.dfine_last_error.restype = c_char_p
    lib.dfine_last_error.argtypes = []

    lib.dfine_version.restype = c_char_p
    lib.dfine_version.argtypes = []

    lib.dfine_class_name.restype = c_char_p
    lib.dfine_class_name.argtypes = [c_int]

    lib.dfine_set_log_callback.restype = None
    lib.dfine_set_log_callback.argtypes = [LOG_FN]

    lib.dfine_detector_create.restype = c_void_p
    lib.dfine_detector_create.argtypes = [c_char_p, c_char_p]

    lib.dfine_detector_create_ex.restype = c_void_p
    lib.dfine_detector_create_ex.argtypes = [c_char_p, c_char_p, POINTER(_Options)]

    lib.dfine_detector_destroy.restype = None
    lib.dfine_detector_destroy.argtypes = [c_void_p]

    for fn in (
        "dfine_detector_input_width",
        "dfine_detector_input_height",
        "dfine_detector_num_queries",
        "dfine_detector_num_classes",
        "dfine_detector_max_batch",
    ):
        f = getattr(lib, fn)
        f.restype = c_int
        f.argtypes = [c_void_p]

    lib.dfine_detector_variant.restype = c_char_p
    lib.dfine_detector_variant.argtypes = [c_void_p]

    lib.dfine_detector_detect.restype = POINTER(_Detections)
    lib.dfine_detector_detect.argtypes = [
        c_void_p,
        POINTER(c_uint8),
        c_int,  # width
        c_int,  # height
        c_int,  # step
        c_int,  # channels
        c_int,  # is_bgr
        c_float,  # threshold
    ]

    lib.dfine_detector_detect_batch.restype = POINTER(POINTER(_Detections))
    lib.dfine_detector_detect_batch.argtypes = [c_void_p, POINTER(_Image), c_int, c_float]

    lib.dfine_detections_free.restype = None
    lib.dfine_detections_free.argtypes = [POINTER(_Detections)]

    lib.dfine_detections_free_batch.restype = None
    lib.dfine_detections_free_batch.argtypes = [POINTER(POINTER(_Detections)), c_int]
    return lib


_LIB: ctypes.CDLL | None = None


def get_lib() -> ctypes.CDLL:
    """Load + configure libdfine on first use (cached). Raises RuntimeError with
    an actionable message if the library or its TensorRT deps can't be loaded."""
    global _LIB
    if _LIB is None:
        _LIB = _configure(_load_library())
    return _LIB


def last_error() -> str:
    raw = get_lib().dfine_last_error()
    return raw.decode("utf-8", "replace") if raw else ""
