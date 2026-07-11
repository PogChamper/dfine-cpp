# Runtime

`libdfine` is a synchronous in-process runtime for one D-FINE TensorRT engine. It accepts host image views and returns detections in source-image coordinates. Video decode, tracking, request scheduling, and multi-GPU routing stay outside the library.

## Data flow

```text
ImageU8 (host HWC uint8)
  → CUDA resize, channel conversion, /255, HWC→NCHW
  → TensorRT engine
  → sigmoid, global top-k, threshold, cxcywh→xyxy
  → detections in source-image pixels
```

The engine owns the network: backbone, encoder, decoder, deformable attention, and FDR/Integral/LQE box computation. The runtime owns image transfer, preprocessing, TensorRT orchestration, and final decode. It does not reimplement model layers.

## Engine contract

The detector validates the TensorRT interface when it loads an engine:

| Tensor | Required contract |
|---|---|
| Input | Device, linear FP32 `[B,3,H,W]`; static `B=1` or dynamic `B` |
| Logits | Device, linear FP32 or FP16 `[B,Q,C]` |
| Boxes | Device, linear FP32 or FP16 `[B,Q,4]` with the same `Q` |
| Profile | Exactly one profile; a dynamic engine may vary only input batch |

`H`, `W`, `Q`, and `C` are engine facts. When a sidecar is present, the runtime cross-checks its names, dimensions, query/class counts, and batch profile. Preprocessing fields are validated before they are applied. An explicit sidecar path is strict. Automatic discovery probes `<engine>.json` before `<stem>.json`.

The sidecar also provides variant and class names. Without custom labels, an 80-class model uses COCO-80 names; other models fall back to `class_<id>`.

## C++ interface

```cpp
#include <dfine/tasks/detector.hpp>

dfine::DetectorOptions options;
options.threshold = 0.5f;

dfine::DFineDetector detector("dfine_m_slim.engine", options);

dfine::ImageU8 image{
    data,             // non-owning host pointer
    height,
    width,
    3,
    row_stride_bytes,
    false             // false: RGB, true: BGR
};

dfine::Detections detections = detector.detect(image);
```

`ImageU8` is a non-owning view. The buffer must remain valid until `detect()` returns. The runtime accepts three-channel HWC `uint8`; a zero stride means `width * 3`. Rows may include positive padding.

`Detection` contains:

| Field | Meaning |
|---|---|
| `box` | `x1,y1,x2,y2` float coordinates in the original image |
| `class_id` | Dense `0..C-1` index; no background slot |
| `score` | Sigmoid confidence |

Batch inference uses one engine enqueue:

```cpp
std::vector<dfine::ImageU8> images = {first, second};
auto results = detector.detect_batch(images, 0.4f);
```

`results[i]` corresponds to `images[i]`. Batch size must be within the engine profile and the frozen bound, when set. Images in one batch may have different source dimensions; each is transformed to the fixed engine canvas.

The detector is move-only and not thread-safe. One instance owns one TensorRT execution context, one CUDA stream, and its working buffers. Use one detector per concurrently executing thread.

## Preprocessing

The published D-FINE weights use:

| Step | Default |
|---|---|
| Geometry | Stretch to engine `W×H` |
| Channel order | RGB; BGR inputs are swapped |
| Layout | HWC `uint8` → NCHW FP32 |
| Normalization | Divide by 255 |
| Mean/std | `[0,0,0]` / `[1,1,1]` |

Do not add ImageNet normalization. It changes the model input contract and collapses accuracy.

Letterbox is available for applications that require aspect-preserving geometry:

```cpp
dfine::DetectorOptions options;
options.preprocess.resize = dfine::PreprocessSpec::Resize::kLetterbox;
options.preprocess.anchor_topleft = false;
options.preprocess.pad_value = 114;
options.preprocess.allow_upscale = true;
```

The runtime reverses letterbox geometry and clips boxes to the source frame. Published weights were trained with stretch; measured letterbox preprocessing costs approximately 1.7–2.0 COCO AP. Treat it as an integration choice, not a quality improvement.

The sidecar selects stretch or letterbox when `Resize::kAuto` is used. An explicit option overrides it.

## Decode

The engine returns raw logits and normalized `cxcywh` boxes. Decode performs:

1. sigmoid over every query/class score;
2. global top `K` selection;
3. score-threshold filtering;
4. `cxcywh → xyxy` conversion;
5. reverse resize mapping; letterbox results are clipped to the source frame.

D-FINE does not use NMS in this runtime. The decode limit is fixed at
`min(300, Q×C)`; sidecars do not override it.

## Execution modes

Conversion/build recipes and runtime modes are independent. `slim`, `fast`, and `max` select graph and profile choices; the modes below change how one engine is executed.

| Mode | Device-to-host result | Capture | Use case |
|---|---|---|---|
| Default | Full logits and boxes | None | Simple, portable baseline |
| Engine graph | Full logits and boxes | TensorRT enqueue and output copy | Repeated batch size, lower launch cost |
| GPU decode | Compact top-k records | None | Lower CPU decode and transfer cost |
| Full pipeline graph | Compact top-k records | Input copy through result copy | Fixed-shape steady-state B1 latency |

A rejected TensorRT enqueue, failed tensor-address restoration, or CUDA stream execution error
makes the detector unusable; destroy and recreate it. A recoverable shape-transition failure is
cleared by the next successful transition.

### Default

The runtime preprocesses on CUDA, enqueues TensorRT, copies raw outputs to host memory, then decodes on the CPU. This path supports all maintained engines.

### Engine graph

```cpp
dfine::DetectorOptions options;
options.use_cuda_graph = true;
```

The detector warms and captures a graph per encountered batch size. Input packing and preprocessing remain outside the graph. Capture requires FP32 outputs and an engine built with `--max-aux-streams 0`; failure safely retains ordinary `enqueueV3` execution.

### GPU decode

```cpp
dfine::DetectorOptions options;
options.gpu_decode = true;
```

CUDA performs sigmoid, segmented ordering, thresholding, and box mapping against device outputs. The host receives a fixed compact top-k result buffer rather than full logits and boxes. Engine outputs must be FP32; unsupported outputs fall back to CPU decode.

GPU scores may differ from CPU scores by one ULP because CUDA and host sigmoid implementations differ. Dataset metrics are equivalent; bitwise equality is not the contract for this mode.

### Full pipeline graph

```cpp
dfine::DetectorOptions options;
options.full_pipeline_graph = true;  // implies GPU decode
options.own_device_memory = true;

dfine::DFineDetector detector("dfine_m_slim_g0.engine", options);
detector.freeze(dfine::FreezeSpec{1, 1920, 1080, false});

if (!detector.full_pipeline_graph_active()) {
    // The detector remains usable through split GPU decode.
}
```

`freeze()` captures input copy, preprocessing, `enqueueV3`, GPU decode, and compact result copy in one graph. Capture requires:

- FP32 engine outputs;
- an engine built with `--max-aux-streams 0`;
- a resolved freeze batch, source width/height, and channel order; zeros use engine defaults;
- stable device addresses provided by the frozen buffers.

Matching steady-state calls replay the graph. A supported shape transition uses split GPU decode while the TensorRT context is restored; later matching calls resume replay. A batch above the frozen bound, or a source frame above an explicit source bound, is rejected instead of allocating on the hot path. The score threshold remains a per-call value and is not baked into the graph.

`full_pipeline_graph_active()` reports whether capture succeeded. `full_graph_replays()` counts calls served by the captured path.

## Frozen memory

`freeze()` warms grow-only device buffers and locks their capacities:

```cpp
detector.freeze(8);  // engine batch bound; source staging remains unbounded
```

For a complete steady-state device-allocation contract, provide source bounds:

```cpp
detector.freeze(dfine::FreezeSpec{8, 3840, 2160, false});
```

After a successful freeze:

- the detector performs no steady-state device allocation within the declared bounds;
- device addresses remain stable;
- a larger batch or source bound throws;
- freezing the same resolved configuration is accepted; after `freeze(batch)`, explicit source
  bounds can lock source staging only when they equal the engine-default dimensions already
  resolved by the first call;
- freezing a different configuration throws; create another detector to reconfigure.

The contract covers device allocation. Returned C++ vectors and other host-side result objects still use normal host memory.

## Timings

`last_timings()` separates device work from host orchestration:

| Field | Scope |
|---|---|
| `infer_ms` | Call start through completion of device work and result transfer |
| `postprocess_ms` | Host decode or compact-result materialization after device completion |
| `total_ms` | In-memory image-to-detections call |
| `preprocess_cpu_ms` | Host packing and preprocess issue cost |
| `dispatch_ms` | Engine/decode issue cost or one graph launch |
| `wait_ms` | Final stream wait |
| `decode_host_ms` | CPU decode or compact result materialization |

These timings exclude JPEG/video decoding, camera capture, tracking, and application queues. Compare identical engine profiles, batch sizes, source shapes, and runtime modes. [Validation](VALIDATION.md) defines the published benchmark methodology.

## C ABI

[`include/dfine/c_api.h`](../include/dfine/c_api.h) is the stable FFI surface:

- opaque detector handles;
- no C++ exceptions or third-party types across the boundary;
- struct-size-versioned options and timing records;
- thread-local `dfine_last_error()`;
- result sets released with `dfine_detections_free()`;
- process-wide log callback with explicit lifetime rules.

```c
dfine_detector_t* detector = dfine_detector_create("model.engine", NULL);
if (!detector) {
    fprintf(stderr, "%s\n", dfine_last_error());
    return 1;
}

dfine_detections_t* result = dfine_detector_detect(
    detector, rgb, width, height, width * 3, 3, 0, 0.5f);

if (!result) {
    fprintf(stderr, "%s\n", dfine_last_error());
} else {
    /* consume result->detections[0..count) */
    dfine_detections_free(result);
}
dfine_detector_destroy(detector);
```

Initialize option structs to zero and set `struct_size = sizeof(...)` before calling extended entry points. New fields are appended; the library reads no more than the caller-provided size.

## Python

The Python package is a `ctypes` wrapper over the C ABI, not a separate inference implementation. It validates NumPy shape/dtype/stride, transfers the same host image view, and materializes immutable `Detection` objects.

```python
from dfine import Detector

with Detector("model.engine", threshold=0.5) as detector:
    detections = detector.detect(rgb_hwc_uint8)
    batch = detector.detect_batch([first, second])
```

Use a context manager or call `close()` explicitly. The native result set is freed after every call. Python options for GPU decode, letterbox, frozen memory, full-pipeline graph, and timings map directly to the C ABI; see the [Python reference](../python/README.md).

## Current boundaries

| Capability | Current contract |
|---|---|
| Spatial shapes | Fixed by the engine |
| Batch | Static B=1 or dynamic within one TensorRT profile |
| Input memory | Host pointer only |
| Output memory | Host `Detection` objects |
| Concurrency | One context and stream per detector |
| Device selection | Process/current CUDA device; no public selector |
| External CUDA stream | Not exposed |
| Video/NV12 | Application concern |
| Serving scheduler | Application or serving-system concern |

These are deliberate API boundaries, not hidden fallback behavior. Integration priorities are tracked in the [roadmap](ROADMAP.md).
