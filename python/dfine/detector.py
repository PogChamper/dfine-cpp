"""Pythonic wrapper over the D-FINE C ABI.

    import numpy as np
    from dfine import Detector

    with Detector("dfine_m_fp16_st.engine", threshold=0.4) as det:
        for d in det.detect(rgb_hwc_uint8):
            print(d.class_name, d.score, d.box)

The heavy lifting (CUDA preprocess, TensorRT inference, decode) happens in
``libdfine.so``; this module only marshals bytes across the C ABI and guarantees
the C-side result set is freed after every call and the detector on close.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from . import _ffi
from ._ffi import _Detections, _Image, _Options, get_lib, last_error

__all__ = ["Box", "Detection", "Detector", "set_log_callback"]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in original-image pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """One detection: dense class id (COCO-80 by default, no background slot)."""

    class_id: int
    score: float
    box: Box
    class_name: str

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "box": list(self.box.as_tuple()),
        }


class Detector:
    """A D-FINE detector backed by a TensorRT engine.

    Not thread-safe: use one instance per thread. Construct with a ``with``
    block, or call :meth:`close` when done (``__del__`` also releases the engine).

    Parameters
    ----------
    engine_path : str
        Path to the ``.engine`` file.
    meta_path : str, optional
        Path to the ``.json`` sidecar; defaults to ``<engine_path>.json``.
    threshold : float
        Default score threshold (overridable per :meth:`detect` call).
    use_cuda_graph : bool
        Opt-in CUDA-graph replay (helps batch-1 latency on 0-aux-stream engines;
        a safe no-op otherwise).
    is_bgr : bool
        Default channel order of images passed to :meth:`detect` (False = RGB).
    class_names : sequence of str, optional
        Override the class-name lookup (for models fine-tuned off COCO). When
        omitted, COCO-80 names are used.
    """

    def __init__(
        self,
        engine_path: str,
        meta_path: Optional[str] = None,
        *,
        threshold: float = 0.5,
        use_cuda_graph: bool = False,
        graph_warmup_iters: int = 3,
        is_bgr: bool = False,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._lib = get_lib()
        self._handle: Optional[int] = None  # set below; None once closed
        self._default_is_bgr = bool(is_bgr)
        self._threshold = float(threshold)  # per-call default (forwarded explicitly)
        self._class_names = list(class_names) if class_names is not None else None

        opts = _Options(
            threshold=float(threshold),
            use_cuda_graph=1 if use_cuda_graph else 0,
            graph_warmup_iters=int(graph_warmup_iters),
        )
        handle = self._lib.dfine_detector_create_ex(
            str(engine_path).encode("utf-8"),
            str(meta_path).encode("utf-8") if meta_path is not None else None,
            ctypes.byref(opts),
        )
        if not handle:
            raise RuntimeError(f"failed to create detector: {last_error()}")
        self._handle = handle

    # -- introspection ----------------------------------------------------- #

    @property
    def variant(self) -> str:
        raw = self._lib.dfine_detector_variant(self._require())
        return raw.decode("utf-8", "replace") if raw else ""

    @property
    def input_width(self) -> int:
        return int(self._lib.dfine_detector_input_width(self._require()))

    @property
    def input_height(self) -> int:
        return int(self._lib.dfine_detector_input_height(self._require()))

    @property
    def num_queries(self) -> int:
        return int(self._lib.dfine_detector_num_queries(self._require()))

    @property
    def num_classes(self) -> int:
        return int(self._lib.dfine_detector_num_classes(self._require()))

    @property
    def max_batch(self) -> int:
        return int(self._lib.dfine_detector_max_batch(self._require()))

    def class_name(self, class_id: int) -> str:
        # Explicit constructor override wins; then the model-aware C call (engine
        # sidecar class_names -> COCO-80 for 80-class engines -> "class_<id>");
        # then the static COCO table for libdfine builds predating that call.
        if self._class_names is not None:
            return self._class_names[class_id] if 0 <= class_id < len(self._class_names) else str(
                class_id
            )
        if self._handle is not None and hasattr(self._lib, "dfine_detector_class_name"):
            # Authoritative: "" means out of range for THIS model — do not fall
            # back to the COCO table (a 3-class model must not label id 5 "bus").
            raw = self._lib.dfine_detector_class_name(self._handle, int(class_id))
            return raw.decode("utf-8", "replace") if raw else str(class_id)
        raw = self._lib.dfine_class_name(int(class_id))
        return raw.decode("utf-8", "replace") if raw else "?"

    # -- inference --------------------------------------------------------- #

    def detect(self, image, *, threshold: Optional[float] = None, is_bgr: Optional[bool] = None):
        """Detect on one HWC uint8 image (numpy array, shape ``(H, W, 3)``).

        Returns a list of :class:`Detection`. ``threshold=None`` uses the
        detector default.
        """
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an HWC image with 3 channels, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"expected dtype uint8, got {arr.dtype}")

        h, w = int(arr.shape[0]), int(arr.shape[1])
        # Zero-copy only when rows are packed AND the row stride is a valid forward
        # stride (>= w*3). This rejects negative-stride views (np.flipud, img[::-1])
        # whose data pointer is the LAST row — passing their stride would be read
        # out of bounds by the native side. Anything else is materialized by copy.
        if arr.strides[1] == 3 and arr.strides[2] == 1 and arr.strides[0] >= w * 3:
            buf, step = arr, int(arr.strides[0])
        else:
            buf, step = np.ascontiguousarray(arr), w * 3

        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        res = self._lib.dfine_detector_detect(
            self._require(),
            ptr,
            w,
            h,
            step,
            3,
            self._resolve_bgr(is_bgr),
            self._resolve_threshold(threshold),
        )
        if not res:
            raise RuntimeError(f"detect failed: {last_error()}")
        try:
            return self._copy_out(res.contents)
        finally:
            self._lib.dfine_detections_free(res)
        # `buf` stays referenced until here, keeping the pixel pointer valid.

    def detect_batch(
        self, images: Sequence, *, threshold: Optional[float] = None, is_bgr: Optional[bool] = None
    ) -> list:
        """Detect on a list of HWC uint8 images. Requires an engine built with
        ``max_batch >= len(images)``. Returns a list-of-lists of :class:`Detection`."""
        import numpy as np

        if len(images) == 0:
            return []

        n = len(images)
        c_images = (_Image * n)()
        keep_alive = []  # hold numpy buffers so their pointers stay valid
        bgr = self._resolve_bgr(is_bgr)
        for i, image in enumerate(images):
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"image {i}: expected HWC 3-channel, got shape {arr.shape}")
            if arr.dtype != np.uint8:
                raise ValueError(f"image {i}: expected uint8, got {arr.dtype}")
            h, w = int(arr.shape[0]), int(arr.shape[1])
            # See detect(): reject negative/short row strides from the zero-copy path.
            if arr.strides[1] == 3 and arr.strides[2] == 1 and arr.strides[0] >= w * 3:
                buf, step = arr, int(arr.strides[0])
            else:
                buf, step = np.ascontiguousarray(arr), w * 3
            keep_alive.append(buf)
            c_images[i].data = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            c_images[i].width = w
            c_images[i].height = h
            c_images[i].step = step
            c_images[i].channels = 3
            c_images[i].is_bgr = bgr

        res = self._lib.dfine_detector_detect_batch(
            self._require(), c_images, n, self._resolve_threshold(threshold)
        )
        if not res:
            raise RuntimeError(f"detect_batch failed: {last_error()}")
        try:
            return [self._copy_out(res[i].contents) for i in range(n)]
        finally:
            self._lib.dfine_detections_free_batch(res, n)
        # keep_alive holds the buffers until the C call returns.

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Release the engine/context/CUDA resources. Idempotent."""
        if getattr(self, "_handle", None):
            self._lib.dfine_detector_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        # __del__ can run during interpreter shutdown; guard everything.
        try:
            self.close()
        except Exception:
            pass

    # -- internals --------------------------------------------------------- #

    def _require(self) -> int:
        if not self._handle:
            raise RuntimeError("detector is closed")
        return self._handle

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        # Forward the per-call threshold, or the detector's construction default.
        # The C ABI treats >= 0 literally (so a stored 0.0 keeps all detections)
        # and < 0 as "use the engine default", so passing our own stored default
        # avoids the C-side create-time "<=0 => 0.5" promotion.
        return self._threshold if threshold is None else float(threshold)

    def _resolve_bgr(self, is_bgr: Optional[bool]) -> int:
        return int(self._default_is_bgr if is_bgr is None else bool(is_bgr))

    def _copy_out(self, dets: "_Detections") -> list:
        out = []
        n = dets.count
        arr = dets.detections
        for i in range(n):
            d = arr[i]
            cid = int(d.class_id)
            out.append(
                Detection(
                    class_id=cid,
                    score=float(d.score),
                    box=Box(float(d.box.x1), float(d.box.y1), float(d.box.x2), float(d.box.y2)),
                    class_name=self.class_name(cid),
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Module-level log callback (process-wide). Keep a ref so it isn't GC'd.
# --------------------------------------------------------------------------- #

_LOG_HOLDER: Optional[ctypes._CFuncPtr] = None


def set_log_callback(callback: Optional[Callable[[int, str], None]]) -> None:
    """Route libdfine log messages to a Python callable ``(severity, message)``,
    where severity is 0=FATAL 1=ERROR 2=WARN 3=INFO 4=VERBOSE. Pass None to
    restore the default stderr logger."""
    global _LOG_HOLDER
    lib = get_lib()
    if callback is None:
        lib.dfine_set_log_callback(_ffi.LOG_FN())  # null callback
        _LOG_HOLDER = None
        return

    def _trampoline(severity: int, message: bytes) -> None:
        try:
            callback(int(severity), message.decode("utf-8", "replace") if message else "")
        except Exception:
            pass  # never let a Python exception unwind into C

    # Install the new callback C-side BEFORE dropping the previous holder, so the
    # old CFUNCTYPE can't be freed while the C side still points at it.
    holder = _ffi.LOG_FN(_trampoline)
    lib.dfine_set_log_callback(holder)
    _LOG_HOLDER = holder
