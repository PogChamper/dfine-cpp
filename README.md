# D-FINE-cpp

**Production C++/TensorRT inference for the [D-FINE](https://github.com/Peterande/D-FINE) object detector.**
A zero-Python, OpenCV-free runtime that runs the full pipeline — CUDA preprocessing → TensorRT engine →
C++ decode — at up to **~460 FPS** on an RTX 4070 Ti SUPER, matching PyTorch mAP to **±0.001 AP**.

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#using-the-library">C++ API</a> ·
  <a href="#from-python">Python</a> ·
  <a href="#precision-guide">Precision</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

[![CI](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/PogChamper/dfine-cpp)](https://github.com/PogChamper/dfine-cpp/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

![PyTorch vs D-FINE-cpp throughput](assets/demo_video.gif)
*Both panels race through the same clip at 10× slow motion; frame counters advance at each backend's
measured e2e throughput (D-FINE-M, RTX 4070 Ti SUPER). Detections are identical — the C++ runtime
produced both. Reproduce: `trt-files/scripts/make_demo_gif.py` (clip: Mixkit free license).*

---

## Why

D-FINE is a state-of-the-art real-time detector, but getting it to run *fast and correct* on TensorRT is
non-trivial — a naive export silently loses ~10 AP, and naive FP16 loses another ~7. This library solves both
and ships a lean, dependency-light C++ runtime:

- **Correct on TensorRT.** The stock `grid_sample` deformable-attention export costs **−10.5 AP** on TRT
  (full-val 0.5507 → 0.4455) — TRT compiles it divergently in-context. We export an **explicit gather-bilinear**
  core (same math, `Gather`+arithmetic) that is TRT-exact with **no plugin and no latency cost**.
- **FP16 that actually holds mAP.** Naive `kFP16` costs a fixed **−6.8 AP** on D-FINE regardless of layer
  pinning — the *builder flag itself* leaks lossy reformats onto the FDR box-decode. The fix is **strong
  typing** (precision baked into ONNX types): **1.6–2.2× faster at −0.2% mAP**, validated on all five sizes.
- **Lean runtime.** `libdfine.so` takes a raw `ImageU8` (HWC uint8) — **no OpenCV** in the core — and hides
  all TensorRT/CUDA behind a PIMPL. RAII everywhere, sanitizer-clean, `-Werror` clean.

## Quickstart

Prebuilt ONNX for all five sizes — FP32 and strongly-typed FP16, each with the `.json` sidecar the engine
build needs, plus `SHA256SUMS` — is attached to
[GitHub Releases](https://github.com/PogChamper/dfine-cpp/releases). TensorRT engines are
GPU-arch- and TRT-version-specific, so you compile the engine locally from the ONNX — that is why we ship
ONNX, not engines.

```sh
# 1. Build the C++
./build.sh                     # CUDA arch defaults to 'native' (probes the local GPU; CMake >= 3.24)
CUDA_ARCH=86 ./build.sh        # explicit SM for headless/CI builds: RTX 30xx = 86, RTX 40xx = 89
# — or plain CMake:
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=native   # [-DTENSORRT_DIR=/path/to/tensorrt]
cmake --build build -j

# 2. Get the prebuilt ONNX + sidecar, build an engine on your GPU
#    (fp16_st is the recommended production build — strongly-typed, decoder kept FP32)
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.1.0/dfine_m_fp16_st.onnx
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.1.0/dfine_m_fp16_st.json
python trt-files/scripts/build_engine.py --strongly-typed --no-tf32 --max-batch 8 \
    --onnx dfine_m_fp16_st.onnx --output trt-files/engines/dfine_m_fp16_st.engine

# 3. Run — libnvinfer/libcudart must be on LD_LIBRARY_PATH; any TensorRT 10.x libs work,
#    e.g. `python -m pip install "tensorrt==10.13.*"` then the venv's .../site-packages/tensorrt_libs
export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH
./build/dfine_detect --engine trt-files/engines/dfine_m_fp16_st.engine --image dog.jpg --threshold 0.5
```

For an FP32 engine, drop `--strongly-typed` and point `--onnx` at the fp32 file (`dfine_m.onnx`).

### Docker

```sh
docker build --build-arg CUDA_ARCH=89 -t dfine-cpp .    # 86 = RTX 30xx, 89 = RTX 40xx
docker run --rm --gpus all -v /path/to/engines:/engines dfine-cpp \
    dfine_detect --engine /engines/dfine_m_fp16_st.engine --image dog.jpg
```

The image builds and runs the C++ consumer of an already-built `.engine`; ONNX export is not included
(it needs the D-FINE-seg source — see the Dockerfile header for the exact limitations).

### Export from a checkpoint

To build the ONNX yourself (fine-tuned weights, non-COCO class counts), you need the training/export source
repo [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) — for this step only; `build_engine` /
`convert_fp16` / `profile` / `coco_eval` and the C++ runtime are self-contained.

```sh
git clone https://github.com/ArgoHA/D-FINE-seg                       # export-time dependency only
python trt-files/scripts/export_dfine_onnx.py --model-name m \
    --checkpoint dfine_m_obj2coco.pt --dfine-src ./D-FINE-seg        # -> trt-files/onnx/dfine_m.onnx + .json
python trt-files/scripts/convert_fp16.py \
    --output trt-files/onnx/dfine_m_fp16_st.onnx                     # FP32 -> strongly-typed FP16 ONNX
```

Standard checkpoints (`dfine_<size>_<dataset>.pt`) can be fetched from Hugging Face with D-FINE-seg's
`ensure_pretrained` helper (`src/d_fine/utils.py` in that repo); nano has no obj2coco checkpoint — use
`dfine_n_coco.pt`. Any other checkpoint: pass its path to `--checkpoint`. The `dfine export` CLI (below)
wraps the same script.

## Benchmarks

![Throughput over COCO stills](assets/demo.gif)
*The same comparison over COCO val2017 stills — one wall-clock window, per-backend frame counters.*

RTX 4070 Ti SUPER · COCO val2017 (5000 imgs) · D-FINE-M · latency = e2e p50 ms (preprocess+infer+decode).

| backend | e2e (b1) | **FPS b1** | **FPS b8** | GPU MiB | mAP |
|---|---|---|---|---|---|
| PyTorch (FP32) | 32.0 | 31 | 66 | — | 0.5509 |
| ONNXRuntime-GPU (FP32) | 25.0 | 40 | 89 | — | 0.5509 |
| TensorRT FP32 (Python) | 8.0 | 125 | 160 | 640 | 0.5507 |
| **C++ FP32** | 5.7 | 176 | 227 | 642 | 0.5506 |
| **C++ FP16** | **3.7** | **272** | **459** | **488** | 0.5500 |

C++ FP16 is **~8.7× the PyTorch throughput** at batch 1. FP16 is essentially lossless **across every size**:

| size | FP32 AP | FP16 AP | ΔAP | infer speedup (b1 / b8) |
|---|---|---|---|---|
| nano | 0.4280 | 0.4280 | +0.0000 | 1.32× / 1.93× |
| small | 0.5074 | 0.5069 | −0.0005 | 1.45× / 1.99× |
| medium | 0.5506 | 0.5500 | −0.0006 | 1.66× / 2.19× |
| large | 0.5725 | 0.5723 | −0.0002 | 1.67× / 2.19× |
| xlarge | 0.5931 | 0.5927 | −0.0004 | 2.21× / 2.81× |

**CUDA-graph (opt-in):** on a single-stream engine (`--max-aux-streams 0`) the graph cuts **batch-1 latency
−34.5%** (3.90 → 2.55 ms) — D-FINE is *dispatch-bound* at small batch (the CPU spends ~3.9 ms in `enqueueV3`
launching hundreds of kernels; `cudaGraphLaunch` is ~0.05 ms). See [docs/HANDOFF.md](docs/HANDOFF.md) §M2.2.

**Frozen pipeline (opt-in):** three composable `DetectorOptions` knobs for steady-state streaming.
`gpu_decode` runs sigmoid/top-k/box-decode as CUDA kernels so only the surviving detections cross PCIe
(Zero-D2H). `freeze()` / `FreezeSpec` warms every grow-only buffer to peak and locks it — zero steady-state
device allocation (VRAM Δ = +0 B over full runs). `full_pipeline_graph` captures H2D → preprocess → infer →
decode → D2H into **one `cudaGraphLaunch` per frame**; it needs a `--max-aux-streams 0` engine and a fixed
source resolution via `freeze(FreezeSpec{batch, src_w, src_h})`, and is byte-identical to the split path
(validated on 1061 real 640×480 COCO images). Measured on m FP16: batch-1 e2e wall **−34.3%**, CPU per frame
4.30 → 0.195 ms (dispatch 4.18 → 0.12 ms); batch-8 frees 19.2 ms CPU per call (wall stays GPU-bound, ±2%).

```cpp
dfine::DetectorOptions o;
o.full_pipeline_graph = true;                     // implies gpu_decode
dfine::DFineDetector det("dfine_m_fp16_st.engine", o);
det.freeze(dfine::FreezeSpec{1, 1920, 1080});     // batch, source WxH — captures + locks
det.detect(frame);                                // one cudaGraphLaunch per call
```

Measure with `dfine_bench --pipeline-compare`. `last_timings()` reports per-stage CPU cost
(`preprocess_cpu_ms` / `dispatch_ms` / `wait_ms` / `decode_host_ms`) — the dispatch column is what the graph
collapses. See [include/dfine/tasks/detector.hpp](include/dfine/tasks/detector.hpp) for the exact contract.

### Benchmark & compare backends yourself

The repo ships a **cross-backend profiler** — measure **PyTorch, ONNXRuntime-GPU, and TensorRT (FP32/FP16)
+ the C++ runtime** side by side on the same images, reporting latency, **FPS**, **GPU memory**, and **mAP**
in one table:

```sh
# PyTorch vs ONNXRuntime-GPU vs TensorRT vs C++ (+ CUDA-graph), full COCO val:
python trt-files/scripts/profile.py --backends torch onnx trt cpp cpp-graph \
       --engine trt-files/engines/dfine_m_fp16_st.engine --full --batches 1 8
```

`--backends` picks any subset of `torch · onnx · trt · trt-baseline · cpp · cpp-graph`. Per-stage C++ latency
(preprocess / infer / D2H / decode, percentiles) is `build/dfine_bench`; the honest CUDA-graph delta is
`dfine_bench --graph-compare` (on a `--max-aux-streams 0` engine); a full **n→x** sweep across all backends is
`bash trt-files/scripts/overnight_bench.sh`. (The `torch` backend and `.pt`→ONNX export need the
[D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) source on `PYTHONPATH`; the ONNX/TRT/C++ paths are
self-contained.)

## Using the library

```cpp
#include "dfine/tasks/detector.hpp"

dfine::DFineDetector det("dfine_m_fp16_st.engine");          // loads engine + .json sidecar
dfine::ImageU8 img{data, height, width, 3, width * 3, /*bgr=*/false};   // your HWC uint8 buffer
for (const dfine::Detection& d : det.detect(img, /*threshold=*/0.5f)) {
    // d.box (xyxy pixel space), d.class_id (0..79 COCO), d.score
}
```

`detect_batch(std::vector<ImageU8>)` runs dynamic batch (N=1..8). `DetectorOptions.use_cuda_graph` enables
CUDA-graph replay (needs a `--max-aux-streams 0` engine). Link `dfine::dfine`; no OpenCV required.

### From C (stable ABI)

`libdfine.so` exposes a pure-C ABI ([`include/dfine/c_api.h`](include/dfine/c_api.h), built by default;
`-DDFINE_BUILD_C_API=OFF` to skip) — opaque handle, no exceptions cross the boundary, thread-local
`dfine_last_error()`, heap result sets freed by `dfine_detections_free()`. It's the foundation for FFI from
any language:

```c
dfine_detector_t* det = dfine_detector_create("dfine_m_fp16_st.engine", NULL);
dfine_detections_t* r = dfine_detector_detect(det, rgb, w, h, w*3, 3, /*is_bgr=*/0, 0.5f);
for (int i = 0; i < r->count; ++i)
    printf("%s %.2f\n", dfine_class_name(r->detections[i].class_id), r->detections[i].score);
dfine_detections_free(r);
dfine_detector_destroy(det);
```

### From Python

A dependency-light [`dfine`](python/) package wraps the C ABI via `ctypes` (no compile step; loads the
prebuilt `.so`). See [`python/README.md`](python/README.md).

```sh
pip install "dfine[tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.1.0/dfine-0.1.0-py3-none-linux_x86_64.whl"
```

The wheel bundles `libdfine.so` built for sm_89 / linux_x86_64; other GPUs and platforms build from
source (`./build.sh`, then `pip install -e python/`).

```python
import numpy as np
from dfine import Detector

with Detector("dfine_m_fp16_st.engine", threshold=0.4) as det:
    for d in det.detect(rgb_hwc_uint8):        # numpy HWC uint8 (is_bgr=True for BGR)
        print(d.class_name, d.score, d.box.as_tuple())
```

The C result set is freed after every call; `with`/`__del__` release the engine. `detect_batch()` returns
per-image results. Detections are **byte-identical to the C++ `dfine_detect`** (verified by `pytest`).

### Zero-setup CLI

The package installs a `dfine` command that resolves an engine from `--engine`, an on-disk cache
(`~/.cache/dfine`, keyed by GPU arch + TRT version), the dev-tree, or builds one on demand:

```sh
dfine predict --model m --image dog.jpg --threshold 0.5 --out annotated.jpg   # detect + draw
dfine info    --model m                                                        # introspection
dfine build   --model m --precision fp16                                       # ONNX -> .engine (cached)
dfine export  --model m                                                        # .pt  -> ONNX (needs D-FINE-seg)
dfine bench   --model m --batches 1,2,4,8                                       # latency/throughput
```

## Apps

`dfine_detect` (single image) · `dfine_coco_eval` (mAP) · `dfine_bench` (per-stage latency / FPS / GPU-mem,
`--cuda-graph`, `--graph-compare`, `--pipeline-compare`) · `dfine_build` (pure-C++ engine build, FP32) ·
`dfine_inspect` / `dfine_smoke`.

## Architecture

The whole network is frozen into a `.engine`; C++ reimplements only the parts that must be fast or Python-free:

```
ImageU8 (HWC u8) ──► CUDA preprocess (stretch-resize + /255, BGR→RGB, →CHW)  [src/core/cuda_preprocess.cu]
                 ──► TensorRT engine (backbone+encoder+decoder+FDR/Integral/LQE)  [frozen; owns deform core]
                 ──► C++ decode (sigmoid → top-300 → cxcywh→xyxy → scale)  [src/core/postprocess.cpp]
```

Preprocessing is **`/255` only — no ImageNet mean/std** (a D-FINE quirk; copying a normal detector's kernel
collapses mAP). The FDR/Integral/LQE box math stays *inside* the engine and is never reimplemented in C++.

Resize is stretch by default (the training convention). Pipelines that standardize on aspect-preserving
inputs can opt into **letterbox** — `DetectorOptions.preprocess` or the sidecar `resize` field — with
configurable anchor (center/top-left), padding value, and upscale on/off; box coordinates are un-mapped
and clipped automatically, including under the GPU decode and the full-pipeline graph. Measured cost on
the stretch-trained weights: **−1.7…−2.0 AP** (`trt-files/scripts/letterbox_eval.py`); the CUDA path
reproduces the host reference to +0.0002 AP.

## Precision guide

| mode | mAP | speed | how |
|---|---|---|---|
| **FP32** | reference | 1× | `build_engine.py --no-tf32` |
| **FP16** ✅ | −0.2% | 1.6–2.2× | `convert_fp16.py` → `build_engine.py --strongly-typed` (**not** the `kFP16` flag) |
| BF16 | −27 AP | — | rejected: D-FINE's FDR needs mantissa precision |
| INT8 | −44 AP | — | rejected: 8-bit too coarse for the FDR (`convert_int8.py` kept for reference) |

The through-line: **D-FINE's FDR box-decode is acutely FP-precision-sensitive** — it amplifies tiny
upstream rounding into box error. That single fact explains the grid_sample trap, the kFP16-flag trap, and why
BF16/INT8 fail. Full forensics in [docs/impl/M0_STATUS.md](docs/impl/M0_STATUS.md) and
[docs/HANDOFF.md](docs/HANDOFF.md).

## Status & roadmap

**Done & validated:** M0 (export→engine→validate, all 5 sizes) · M1 (C++ detector, AP 0.5506 == reference) ·
M2 (FP16 + CUDA-graph; INT8 investigated & rejected) · **M4 bindings** (stable C ABI + Python `ctypes`
package + zero-setup `dfine` CLI — detections byte-identical to the C++ path) · **intensive-core P1–P3**
(GPU decode, `freeze()`, full-pipeline graph; P4 pending) · optional letterbox preprocessing.
M3 instance segmentation is shelved. Next: a WASM/WebGPU browser demo and real-time video apps — see
**[docs/ROADMAP.md](docs/ROADMAP.md)** and [docs/HANDOFF.md](docs/HANDOFF.md) for the single source of
truth on the current state.

## Requirements & environment

- **Runtime:** NVIDIA GPU + driver, CUDA 12.x, TensorRT 10.x (validated on 10.13). `libnvinfer`/`libcudart`
  must be on `LD_LIBRARY_PATH` — a system TensorRT install or a `pip install "tensorrt==10.13.*"` venv's
  `tensorrt_libs` dir both work.
- **C++ build:** CMake ≥ 3.24 (the default `CUDA_ARCHITECTURES=native` needs 3.24; the project floor in
  `CMakeLists.txt` is 3.20 if you pass an explicit arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=89`), a CUDA 12.x
  toolkit (`nvcc`), TensorRT headers + libs. A system TensorRT is found automatically
  ([cmake/FindTensorRT.cmake](cmake/FindTensorRT.cmake) searches `$TENSORRT_DIR`, `/usr/local/TensorRT`,
  `/opt/tensorrt`, `/usr`); alternatively populate `third_party/tensorrt` — see
  [third_party/README.md](third_party/README.md). `./build.sh` auto-discovers `cmake`/`nvcc` from `PATH`
  and applies a conda-`ld` workaround only for conda toolchains. No OpenCV.
- **Python scripts** (engine-build / convert / eval / profile): see `pyproject.toml` (uv). The **ONNX
  export** alone additionally needs the [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) source on
  `PYTHONPATH` for model construction; everything else (build_engine / convert_fp16 / profile / coco_eval)
  is self-contained.

Engines are machine-specific build outputs (gitignored) — compile them locally from the prebuilt ONNX on
[Releases](https://github.com/PogChamper/dfine-cpp/releases) or from your own export.

## Troubleshooting

- **`libnvinfer.so.10: cannot open shared object file`** — TensorRT libs are not on `LD_LIBRARY_PATH`;
  any TRT 10.x works, e.g. a `pip install "tensorrt==10.13.*"` venv's `tensorrt_libs` dir (Quickstart
  step 3).
- **Engine fails to deserialize** — `.engine` files are GPU-arch- and TRT-version-specific; rebuild from
  the ONNX on the target machine (`build_engine.py`).
- **mAP collapses after "fixing" preprocessing** — D-FINE is `/255` only, no ImageNet mean/std.
- **CMake error on `CUDA_ARCHITECTURES=native`** — needs CMake ≥ 3.24; pass an explicit arch
  (`CUDA_ARCH=89 ./build.sh`) on older CMake.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) — the validation bar (warning-clean,
sanitizer-clean, mAP-neutral) is spelled out there.

## Credits & license

C++/TensorRT port of **D-FINE** (Peng et al.), Apache-2.0; export path built on
[D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg). Vendored: `stb_image` (public domain); nlohmann/json
(MIT) is found on the system or fetched at configure time. Links NVIDIA TensorRT/CUDA (install separately).
This project is **Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
