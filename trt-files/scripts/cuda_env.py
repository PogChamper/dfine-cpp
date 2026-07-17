"""Bootstrap onnxruntime-gpu so CUDAExecutionProvider actually loads on this machine.
Import and call bootstrap() BEFORE creating any InferenceSession.

WSL2: the driver libcuda.so.1 lives in /usr/lib/wsl/lib (often missing from
LD_LIBRARY_PATH). ORT >= 1.20 ships CUDA/cuDNN via pip nvidia-* packages and needs
onnxruntime.preload_dlls(cuda=True, cudnn=True) before the first session.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_WSL_DRIVER_DIR = "/usr/lib/wsl/lib"
_BOOTSTRAPPED = False


def _ensure_wsl_driver_on_path() -> None:
    if not os.path.isdir(_WSL_DRIVER_DIR):
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    if _WSL_DRIVER_DIR not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([_WSL_DRIVER_DIR, *parts])


def bootstrap(verbose: bool = False) -> Any:
    """Ensure onnxruntime-gpu sees the GPU + bundled CUDA/cuDNN. Idempotent."""
    global _BOOTSTRAPPED
    _ensure_wsl_driver_on_path()
    import onnxruntime as ort

    if _BOOTSTRAPPED:
        return ort
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload(cuda=True, cudnn=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[cuda_env] preload_dlls failed: {e}", file=sys.stderr)
    _BOOTSTRAPPED = True
    if verbose:
        print(f"[cuda_env] onnxruntime {ort.__version__} device={ort.get_device()}")
        print(f"[cuda_env] providers: {ort.get_available_providers()}")
    return ort


def make_session(
    model_path: str,
    *,
    device_id: int = 0,
    prefer_cuda: bool = True,
    use_tf32: bool = False,
    log_severity: int = 3,
    sess_options=None,
):
    """Create an InferenceSession on CUDA (fallback CPU). Returns (session, providers_used)."""
    ort = bootstrap()
    if sess_options is None:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = log_severity
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list = []
    if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "use_tf32": "1" if use_tf32 else "0",
                },
            )
        )
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    return sess, sess.get_providers()


if __name__ == "__main__":
    bootstrap(verbose=True)
