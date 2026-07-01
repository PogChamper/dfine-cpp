# `dfine` — Python bindings

A thin, memory-safe [ctypes] wrapper over `libdfine.so` (the stable C ABI in
[`include/dfine/c_api.h`](../include/dfine/c_api.h)). All CUDA preprocessing,
TensorRT inference, and decode run in native code — Python only marshals image
bytes in and detections out. No compile step: it loads the prebuilt shared
library at import.

```python
import numpy as np
from dfine import Detector

img = np.asarray(...)            # HWC uint8, RGB (or is_bgr=True for BGR)

with Detector("dfine_m_fp16_st.engine", threshold=0.4) as det:
    print(det.variant, det.input_width, det.num_classes)     # 'm' 640 80
    for d in det.detect(img):
        print(d.class_name, round(d.score, 3), d.box.as_tuple())

    # Batch (engine must be built with max_batch >= len(images)):
    results = det.detect_batch([img, img])   # list[list[Detection]]
```

`Detector.detect()` returns a `list[Detection]`, each with `class_id` (dense
COCO-80 index — no background slot), `score`, `box` (xyxy pixel coords), and
`class_name`. The C result set is freed after every call; `__del__` /
`__enter__`/`__exit__` release the engine even on exceptions.

## Requirements

- A **TensorRT 10.x + CUDA 12** runtime matching the engine's build. Either
  `pip install "dfine[tensorrt]"` (pulls the `tensorrt` wheel; the bindings
  best-effort preload it) **or** put the TRT/CUDA lib dirs on `LD_LIBRARY_PATH`:

  ```sh
  export LD_LIBRARY_PATH="$(python -c 'import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))'):/path/to/cuda/lib64"
  ```

- The prebuilt `libdfine.so`. It is located, in order, from: `$DFINE_LIBRARY`,
  next to the package (wheel), or the dev tree `<repo>/build/libdfine.so`
  (produced by `./build.sh`).

- A TensorRT `.engine` for a D-FINE model. Build one with the scripts in
  [`trt-files/scripts/`](../trt-files/scripts/) (see [docs/HANDOFF.md](../docs/HANDOFF.md)),
  or let the `dfine` CLI build it for you.

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
