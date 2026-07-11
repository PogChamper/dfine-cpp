# `dfine` Python package

`dfine` is a thin `ctypes` binding over the stable C ABI in [`include/dfine/c_api.h`](../include/dfine/c_api.h). CUDA preprocessing, TensorRT execution, and decode run in `libdfine.so`; Python validates arrays and materializes result objects.

## Install

The latest published wheel is v0.3.3. It targets Linux x86_64 and contains an `sm_89` native
library with forward PTX, validated on Ada and Blackwell:

```sh
python -m pip install "dfine[cli,tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine-0.3.3-py3-none-linux_x86_64.whl"
```

Turing, Ampere, Jetson, and development installations use a source build:

```sh
CUDA_ARCH=86 ./build.sh  # Ampere; select the target architecture
python -m pip install -e "python[cli]"
export DFINE_LIBRARY="$PWD/build/libdfine.so"
```

Build a target-local engine before inference. The canonical release-artifact workflow is in [Getting started](../docs/GETTING_STARTED.md).

The [quickstart notebook](../examples/python_quickstart.ipynb) covers the same installed-wheel path interactively.

## Detect

```python
import numpy as np
from dfine import Detector

image = np.asarray(...)  # HWC uint8, RGB

with Detector("dfine_m_slim.engine", threshold=0.5) as detector:
    for detection in detector.detect(image):
        print(
            detection.class_name,
            round(detection.score, 3),
            detection.box.as_tuple(),
        )
```

`Detector.detect()` accepts an HWC `uint8` array with three channels and returns `list[Detection]`.

| Object | Fields |
|---|---|
| `Detection` | `class_id`, `class_name`, `score`, `box` |
| `Box` | `x1`, `y1`, `x2`, `y2`, `width`, `height` |

Coordinates are floats in the original image. Class IDs are dense `0..C-1`; D-FINE has no background slot.

Use BGR input explicitly:

```python
detections = detector.detect(bgr_image, is_bgr=True)
```

The wrapper accepts positive padded row strides and copies layouts that cannot be passed safely, including negative-stride views. The input buffer remains owned by Python.

## Batch

```python
results = detector.detect_batch([first, second], threshold=0.4)
```

`results[i]` is the detection list for input `i`. Batch size must fit the engine profile and any frozen bound. Source dimensions may differ within a batch.

## Model facts and labels

```python
print(detector.variant)
print(detector.input_width, detector.input_height)
print(detector.num_queries, detector.num_classes)
print(detector.max_batch)
```

Class names come from the engine sidecar, then COCO-80 for an 80-class engine. Override them for a custom model:

```python
Detector("food.engine", class_names=["burger", "fries", "drink"])
```

The override length should match the engine class count.

## Runtime options

Python options map directly to the C ABI and native detector:

```python
detector = Detector(
    "dfine_m_slim_g0.engine",
    threshold=0.5,
    gpu_decode=True,
    own_device_memory=True,
    full_pipeline_graph=True,
)
detector.freeze(batch=1, src_w=1920, src_h=1080, src_is_bgr=False)

print(detector.full_pipeline_graph_active)
print(detector.last_timings())
```

| Option | Effect |
|---|---|
| `threshold` | Default score threshold; finite and within `[0, 1]` |
| `use_cuda_graph` | Capture engine enqueue/output copy by batch |
| `gpu_decode` | Decode device outputs on CUDA and transfer compact top-k records |
| `own_device_memory` | Use a detector-owned TensorRT activation block |
| `full_pipeline_graph` | Capture the frozen input-to-result path; implies GPU decode |
| `graph_warmup_iters` | Warmup iterations before capture; `<=0` uses 3, and 1 is raised to 2 |
| `letterbox` | Override stretch preprocessing with letterbox |

Explicit per-call thresholds use the same range; `None` keeps the detector default.

Graph modes require FP32 outputs and an engine built with `dfine build --cuda-graph` or the direct
builder's `--max-aux-streams 0`. Full-pipeline capture occurs inside `freeze()`. After a recoverable
capture failure, FP32 outputs use split GPU decode and FP16 outputs use CPU decode; execution
failures throw. Inspect `full_pipeline_graph_active` when capture is required.

Published weights use stretch resize. Letterbox is available with `letterbox_topleft`, `letterbox_pad`, and `letterbox_upscale`, but measures approximately 1.7–2.0 AP below stretch on those weights.

The complete execution and frozen-memory contracts are in [Runtime](../docs/RUNTIME.md).

## Lifetime and errors

`Detector` is not thread-safe. Use one instance per concurrently executing thread.

A context manager is preferred. `close()` releases the native detector explicitly; repeated `close()` calls are safe. The wrapper also releases native result sets after every call.

Construction errors include the native `dfine_last_error()` message. A supplied `meta_path` is
strict: it must exist, parse, and agree with the engine. Set `DFINE_LIBRARY` only to select an exact
native library; an invalid explicit path never falls through to another copy. Published v0.3.3
falls back from a missing explicit `meta_path`; strict failure begins in v0.4.0.

## CLI

The wheel installs `dfine` but does not download model files. [Getting started](../docs/GETTING_STARTED.md)
defines the commands and release-asset path; checkpoint export additionally requires the compatible
D-FINE source tree documented in [Conversion](../docs/CONVERSION.md).

## Library discovery

The package searches `DFINE_LIBRARY`, the package root, the package `lib/` directory, and an editable
checkout's `build/` directory, then delegates to the platform loader.

TensorRT and CUDA libraries must be visible to the dynamic loader. Diagnose discovery with:

```sh
dfine doctor
```

See [Troubleshooting](../docs/TROUBLESHOOTING.md) for loader, wheel-architecture, and engine-compatibility errors.

Development and GPU validation commands are maintained in [CONTRIBUTING.md](../CONTRIBUTING.md).
