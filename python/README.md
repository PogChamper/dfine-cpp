# `dfine` — Python bindings

A thin, memory-safe [ctypes] wrapper over `libdfine.so` (the stable C ABI in
[`include/dfine/c_api.h`](../include/dfine/c_api.h)). All CUDA preprocessing,
TensorRT inference, and decode run in native code — Python only marshals image
bytes in and detections out. No compile step: it loads the prebuilt shared
library at import.

## From zero to detections (no repo checkout)

```sh
pip install "dfine[tensorrt,cli] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine-0.3.2-py3-none-linux_x86_64.whl"
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine_m_slim.onnx \
     -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine_m_slim.json
dfine build --model m --onnx dfine_m_slim.onnx --output dfine_m_slim.engine
```

The wheel bundles `libdfine.so` (sm_89 / linux_x86_64) and a snapshot of
`build_engine.py`, so `dfine build` compiles the engine on your GPU without the
repo. `dfine_m_slim` is the surgical-FP16 production ONNX — lossless full-val
mAP (0.5500, exactly the FP16 reference); its tier measures 526 img/s batch-8
on a 4070 Ti SUPER. Then:

```python
import numpy as np
from dfine import Detector

img = np.asarray(...)            # HWC uint8, RGB (or is_bgr=True for BGR)

with Detector("dfine_m_slim.engine", threshold=0.4) as det:
    print(det.variant, det.input_width, det.num_classes)     # 'm' 640 80
    for d in det.detect(img):
        print(d.class_name, round(d.score, 3), d.box.as_tuple())

    # Batch (engine must be built with max_batch >= len(images)):
    results = det.detect_batch([img, img])   # list[list[Detection]]
```

Notebook walkthrough: [../examples/python_quickstart.ipynb](../examples/python_quickstart.ipynb).

`Detector.detect()` returns a `list[Detection]`, each with `class_id` (dense
COCO-80 index — no background slot), `score`, `box` (xyxy pixel coords), and
`class_name`. The C result set is freed after every call; `__del__` /
`__enter__`/`__exit__` release the engine even on exceptions.

The frozen single-launch pipeline and letterbox preprocessing are reachable
directly from Python (C ABI v2):

```python
det = Detector("dfine_m_slim.engine",              # --max-aux-streams 0 build
               gpu_decode=True, own_device_memory=True, full_pipeline_graph=True)
det.freeze(1, src_w=1920, src_h=1080)              # warm + capture + lock
det.full_pipeline_graph_active                     # True -> one cudaGraphLaunch/frame
det.detect(frame)
det.last_timings()                                 # {'dispatch_ms': 0.08, ...}

Detector("...engine", letterbox=True)              # aspect-preserving preprocessing
Detector("...engine", letterbox=True,              # production smart_resize semantics
         letterbox_topleft=True, letterbox_pad=0, letterbox_upscale=False)
```

Stretch stays the default — it is D-FINE's training convention and measures
1.7–2.0 AP better than letterbox on the published weights.

## Requirements

- A **TensorRT 10.x + CUDA 12** runtime matching the engine's build. Either
  install the Releases wheel with the `[tensorrt]` extra —

  ```sh
  pip install "dfine[tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine-0.3.2-py3-none-linux_x86_64.whl"
  ```

  (the extra pulls the `tensorrt` pip wheel; the bindings best-effort preload
  it; the package is not on PyPI) **or** put the TRT/CUDA lib dirs on
  `LD_LIBRARY_PATH`:

  ```sh
  export LD_LIBRARY_PATH="$(python -c 'import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))'):/path/to/cuda/lib64"
  ```

- The prebuilt `libdfine.so`. It is located, in order, from: `$DFINE_LIBRARY`,
  next to the package (wheel), or the dev tree `<repo>/build/libdfine.so`
  (produced by `./build.sh`).

- A TensorRT `.engine` for a D-FINE model. Engines are GPU-arch- and
  TRT-version-specific, so compile one locally: download a prebuilt ONNX from the
  repo's [Releases](https://github.com/PogChamper/dfine-cpp/releases) and run
  [`trt-files/scripts/build_engine.py`](../trt-files/scripts/build_engine.py)
  (see the root [README Quickstart](../README.md#quickstart)), or let the
  `dfine` CLI build it for you.

## Custom (non-COCO) models

`class_name` defaults to the COCO-80 names. For a model fine-tuned on another
label set, pass your own:

```python
Detector("food.engine", class_names=["burger", "fries", "drink"])
```

## Tests

```sh
cd python
pip install -e ".[dev]"
export LD_LIBRARY_PATH=.../tensorrt_libs:.../cuda/lib
pytest tests -v          # skips cleanly with no GPU / engine / image
```

The suite includes a **parity test** that asserts the Python detections are
byte-for-byte identical to the C++ `dfine_detect` binary on the same pixels.

[ctypes]: https://docs.python.org/3/library/ctypes.html
