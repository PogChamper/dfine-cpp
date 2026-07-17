# D-FINE-cpp

**Correct D-FINE → TensorRT conversion, a lean native runtime, and reproducible evidence that the result is accurate.**

| Build correctly | Run natively | Prove it |
|---|---|---|
| Checkpoint → typed ONNX → local TensorRT engine | CUDA preprocess → TensorRT → detections | Parity, COCO mAP, provenance, and cross-GPU validation |

[![CI](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/PogChamper/dfine-cpp)](https://github.com/PogChamper/dfine-cpp/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

D-FINE can export and compile successfully while producing inaccurate TensorRT detections. On COCO val2017, the stock `grid_sample` deformable-attention path fell from 0.5509 PyTorch AP to 0.4455 TensorRT AP. D-FINE-cpp replaces that path with equivalent gather-bilinear operations, keeps the precision-sensitive box decoder in FP32 inside the FP16 engine, and runs the verified model through an OpenCV-free C++/CUDA library.

## Why this project exists

| D-FINE-M pipeline | COCO AP@[.50:.95] | Result |
|---|---:|---|
| PyTorch | 0.5509 | Reference |
| Stock TensorRT export | 0.4455 | Silent box drift |
| Explicit-gather TensorRT FP32 | 0.5507 | Parity restored |
| Release `slim` FP16 | 0.5500 | Production default |

The fix uses standard ONNX and TensorRT operations: no custom plugin and no Python at inference. The complete investigation and every rejected precision path are recorded in the [research matrix](docs/RESEARCH_MATRIX.md).

## Quickstart

This path uses the latest published release, v0.5.0: the Linux x86_64 wheel and the D-FINE-M ONNX artifact. It requires CUDA 12, TensorRT 10.13, and an Ada or Blackwell GPU; build from source on Turing or Ampere. TensorRT engines are compiled locally for the target GPU.

`pip` installs the released `dfine` wheel. Maintainer model tooling, dataset validation, and release
workflows use the root `uv.lock`.

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install "dfine[cli,tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/dfine-0.5.0-py3-none-linux_x86_64.whl"
curl -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/dfine_m_slim.onnx \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/dfine_m_slim.json
dfine doctor
dfine build --model m --onnx dfine_m_slim.onnx --output dfine_m_slim.engine
dfine predict --engine dfine_m_slim.engine --image image.jpg --out result.jpg
```

Use any JPEG or PNG as `image.jpg`. The build is a one-time operation and normally takes 1–3 minutes. For source builds, C++ integration, custom checkpoints, and artifact verification, see [Getting started](docs/GETTING_STARTED.md). What changed in this release is in the [v0.5.0 notes](docs/releases/v0.5.0.md).

## Build correctly

The artifact toolchain turns trained weights into a TensorRT plan without changing the model contract:

```text
checkpoint
  → strict model load
  → explicit-gather FP32 ONNX
  → surgical strongly typed FP16 ONNX
  → target-local TensorRT engine
```

The exporter verifies dynamic batch in ONNX Runtime. The surgical converter keeps only D-FINE's precision-sensitive FDR and deform-coordinate math in FP32. ONNX sidecars record model, preprocessing, precision, and an export-time batch recommendation; the production Python builder adds the compiled profile and source ONNX hash to the engine sidecar.

Released model packs contain FP32 and `slim` FP16 ONNX artifacts for D-FINE-N/S/M/L/X. A model pack is not an engine: engines remain local build products because TensorRT compatibility depends on the target stack. See [Conversion](docs/CONVERSION.md) and [artifact identity](docs/NAMING.md).

## Run natively

`libdfine` owns CUDA preprocessing, TensorRT execution, and decode. Its public C++ headers contain no TensorRT, CUDA, or OpenCV types.

```cpp
#include <dfine/tasks/detector.hpp>

dfine::DFineDetector detector("dfine_m_slim.engine");
dfine::ImageU8 image{data, height, width, 3, width * 3, false}; // RGB HWC uint8

for (const auto& detection : detector.detect(image, 0.5f)) {
    // detection.box: xyxy pixels; detection.class_id; detection.score
}
```

The default path is synchronous and uses CPU decode. Optional GPU decode reduces device-to-host traffic. `freeze()` locks warmed engine and decode buffers; explicit source bounds complete the steady-state device-allocation contract. With FP32 outputs, an engine built by `dfine build --cuda-graph` can capture the full frozen pipeline in one CUDA Graph launch. These modes are runtime choices, not model presets.

The library also exposes a struct-size-versioned C ABI and Python bindings over that ABI. See [Runtime](docs/RUNTIME.md), the [C header](include/dfine/c_api.h), and the [Python package](python/README.md).

## Prove it

Correctness is gated at each boundary:

| Boundary | Gate |
|---|---|
| Checkpoint → ONNX | Strict load; ONNX Runtime batch 1 and 2 |
| ONNX → engine | Source ONNX hash recorded and checked when available; release smoke at batches 1, 2, and 8 |
| Engine → detections | Box-aware parity and full COCO mAP |
| Runtime optimization | CPU/GPU decode parity; graph replay parity |
| Release | Asset grammar, SHA-256 manifest, clean-machine install |
| Hardware | Reproducible validation report |

The default `slim` recipe is full-val lossless on all five published sizes: N 0.4272, S 0.5060,
M 0.5500, L 0.5723, and X 0.5926 AP.

The v0.5 study measured five D-FINE-M operating points on RTX 4070 Ti SUPER. Throughput is native
C++ image-to-detections at batch 8; accuracy is full COCO `val2017` on the corresponding strongly
typed `slim` graph.

| Graph | b8 img/s | Throughput | COCO AP | ΔAP |
|---|---:|---:|---:|---:|
| `base` | 533 | — | 55.033 | — |
| `Q200` | 560 | +5.1% | 54.849 | −0.184 |
| `C300→150` | 564 | +5.9% | 54.804 | −0.230 |
| `C300→100` | 576 | +8.1% | 54.536 | −0.497 |
| `Q200→C100` | 585 | +9.7% | 54.518 | −0.515 |

`Q200` is the conservative measured point; `Q200→C100` is the fastest preset. On top of any
preset, the batch-8-optimized engine profile and full-pipeline CUDA Graph reach 662 img/s at
batch 8 and 1.99 ms at batch 1 on the same GPU (two runtime modes of the `max` graph,
−0.96 COCO AP). The complete
[benchmark tables](docs/BENCHMARKS.md) add runtime modes, object-size, recall, class, PyTorch
compiler, and cross-GPU results. The [preset report](docs/reports/v0.5.0-preset-evaluation.md)
records the cross-domain study and evidence boundary. `dfine sweep` builds the same decision
table for your own checkpoint and dataset in one command ([Validation](docs/VALIDATION.md)).

## Supported contract

| Area | Current support |
|---|---|
| Platform | Linux; x86_64 validated, Jetson/aarch64 not yet validated |
| GPU | NVIDIA Turing or newer build target; Ampere, Ada, and Blackwell validated |
| Stack | CUDA 12; TensorRT 10.x, validated on 10.13; Blackwell requires R570+ and CUDA ≥12.8 |
| Input | Host RGB/BGR, HWC `uint8`, 3 channels |
| Shapes | Dynamic batch 1–8 by default; fixed engine H/W |
| Preprocess | Stretch and `/255` by default; optional letterbox |
| Execution | Synchronous; one context and stream per detector; not thread-safe |
| Interfaces | C++17, stable C ABI, Python `ctypes`, CLI, CMake package |
| Outside the runtime | Video decode, tracking, request scheduling, multi-GPU routing |

The release wheel contains an `sm_89` native build with forward PTX and was validated on Ada and Blackwell. Turing, Ampere, Jetson, and other platforms use the source build. Current platform details and known installation failures are in [Getting started](docs/GETTING_STARTED.md) and [Troubleshooting](docs/TROUBLESHOOTING.md).

## Documentation

| Question | Document |
|---|---|
| Install and run the first image | [Getting started](docs/GETTING_STARTED.md) |
| Export a custom checkpoint or build an engine | [Conversion](docs/CONVERSION.md) |
| Embed and tune the native library | [Runtime](docs/RUNTIME.md) |
| Understand artifact names and sidecars | [Artifact identity](docs/NAMING.md) |
| Compare published accuracy and throughput | [Benchmarks](docs/BENCHMARKS.md) |
| Pick an operating point for your checkpoint and dataset | [Validation](docs/VALIDATION.md#sweep-your-checkpoint) |
| Reproduce accuracy and performance | [Validation](docs/VALIDATION.md) |
| Inspect precision research and rejected paths | [Research matrix](docs/RESEARCH_MATRIX.md) |
| See current priorities | [Roadmap](docs/ROADMAP.md) |
| Diagnose an installation | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Contribute a change | [Contributing](CONTRIBUTING.md) |

[Release notes](docs/releases/) describe versioned changes. [`docs/HANDOFF.md`](docs/HANDOFF.md)
and [`docs/impl/`](docs/impl/) preserve the engineering record; the documents above define the
current contract.

## Credits and license

D-FINE-cpp ports [D-FINE](https://github.com/Peterande/D-FINE) (Peng et al.) to TensorRT. Checkpoint export uses the bundled detection-only model definition; [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) remains the training implementation and pinned differential reference. Initial runtime scaffolding was derived from [rf-detr-cpp](https://github.com/infracv/rf-detr-cpp) and substantially adapted. The explicit-gather export, surgical FP16 conversion, GPU decode, frozen-memory contract, and full-pipeline CUDA Graph are original to this project.

The repository is licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Vendored `stb_image` is public domain; nlohmann/json is MIT-licensed and discovered or fetched at build time. NVIDIA TensorRT and CUDA are installed separately.
