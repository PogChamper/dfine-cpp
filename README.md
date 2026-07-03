# D-FINE-cpp

**Production C++/TensorRT inference for the [D-FINE](https://github.com/Peterande/D-FINE) object detector.**
A zero-Python, OpenCV-free runtime that runs the full pipeline — CUDA preprocessing → TensorRT engine →
C++ decode — at up to **1550+ FPS** (D-FINE-N) and **2.47 ms** end-to-end batch-1 latency on an
RTX 4070 Ti SUPER, matching PyTorch mAP to within **0.002 AP** in the default configuration.

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#using-the-library">C++ API</a> ·
  <a href="#from-python">Python</a> ·
  <a href="#precision-guide">Precision</a> ·
  <a href="docs/RESEARCH_MATRIX.md">Research matrix</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

[![CI](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PogChamper/dfine-cpp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/PogChamper/dfine-cpp)](https://github.com/PogChamper/dfine-cpp/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

![PyTorch vs D-FINE-cpp throughput](assets/demo_video.gif)
*All three panels race through the same clip at 12× slow motion; frame counters advance at each
backend's measured batch-8 throughput (D-FINE-M, RTX 4070 Ti SUPER). Detections are identical —
the C++ surgical-FP16 runtime produced every panel. Reproduce:
`trt-files/scripts/make_demo_gif.py` (clip: Mixkit free license).*

All five model sizes, three precision/speed tiers, end-to-end (preprocess + inference + decode),
mAP on full COCO val2017, throughput = batch-8 img/s (medians of 3×500-iter rounds), RTX 4070 Ti SUPER.
**surgical** = FP16 including the decoder (`convert_fp16_surgical.py`); **fast** = slim + export
sliders (`--num-queries 200 --cascade 1:100`):

| model | PyTorch mAP | prod fp16 b8 · mAP | **surgical b8 · mAP** | fast b8 · mAP |
|---|---|---|---|---|
| D-FINE-N | 0.4279 | 1234 · 0.4280 | **1309** · 0.4276 | **1556** · 0.4231 |
| D-FINE-S | 0.5073 | 637 · 0.5069 | **758** · 0.5065 | **880** · 0.5021 |
| D-FINE-M | 0.5509 | 469 · 0.5500 | **526** · 0.5502 | **598** · 0.5448 |
| D-FINE-L | 0.5724 | 357 · 0.5723 | **390** · 0.5724 | **453** · 0.5647 |
| D-FINE-X | 0.5931 | 244 · 0.5927 | **264** · 0.5929 | **292** · 0.5855 |

The release ships the **`--slim`** surgical variant (an even smaller FP32 island): separately
gated lossless on all five sizes (full-val n 0.4272 / s 0.5060 / m 0.5500 / l 0.5723 / x 0.5926)
and +2-3% faster at batch 8 where benched — the surgical column above is the conservative
non-slim measurement.

The full m ladder: **PyTorch 66 → 686 img/s (10.4×)** — fp32 230 → fp16 469 → surgical 526/561 →
fast 598 → max preset 686 (fast + `--eval-idx 2` + `--opt-batch 8`, −0.89 AP). At the other end,
the full-pipeline CUDA graph holds **2.47 ms** end-to-end batch-1 (byte-identical detections).
Every cell — including b1/b2/b4, VRAM, and the failed experiments (FP8, INT8, plugins) — is in
**[docs/RESEARCH_MATRIX.md](docs/RESEARCH_MATRIX.md)** with one-command reproduce paths.

## Three commands to first detection (Python)

No compiler, no OpenCV, no repo checkout — the wheel bundles the C++ runtime, the release ships the
ONNX, and the engine compiles on your GPU in one command (a one-time 1-3 minutes):

```sh
pip install "dfine[tensorrt,cli] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine-0.3.0-py3-none-linux_x86_64.whl"
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine_m_slim.onnx \
     -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine_m_slim.json
dfine build --model m --onnx dfine_m_slim.onnx --output dfine_m_slim.engine
```

```python
from dfine import Detector

det = Detector("dfine_m_slim.engine", threshold=0.5)
detections = det.detect(frame)            # numpy HWC uint8 → [Detection(box, class_name, score)]
```

That engine is the lossless-FP16 production build (`--slim` surgical; full-val 0.5500, exactly the
FP16 reference). Its tier measures 288 img/s at batch 1 and 526 at batch 8 on a 4070 Ti SUPER
(PyTorch: 31 / 66; the slim build itself benched +2% over that b8 figure). The wheel's `libdfine.so` targets
sm_89/linux_x86_64 — other GPUs/platforms build from source (`./build.sh`, then
`pip install -e python/`), everything else stays identical. Notebook version:
[examples/python_quickstart.ipynb](examples/python_quickstart.ipynb).

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

Prebuilt ONNX for all five sizes — the FP32 opset-19 base (`dfine_<size>_op19`) and the
surgical-FP16 production build (`dfine_<size>_slim`), each with the `.json` sidecar the engine
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
#    (slim is the recommended production build — surgical FP16, lossless full-val on all 5 sizes)
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine_m_slim.onnx
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine_m_slim.json
python trt-files/scripts/build_engine.py --strongly-typed --no-tf32 --max-batch 8 \
    --onnx dfine_m_slim.onnx --output trt-files/engines/dfine_m_slim.engine
#   (add --opt-batch 8 for batch serving: +6-10% b8 throughput, costs some b1 latency)

# 3. Run — libnvinfer/libcudart must be on LD_LIBRARY_PATH; any TensorRT 10.x libs work,
#    e.g. `python -m pip install "tensorrt==10.13.*"` then the venv's .../site-packages/tensorrt_libs
export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH
./build/dfine_detect --engine trt-files/engines/dfine_m_slim.engine --image dog.jpg --threshold 0.5
```

For an FP32 engine, drop `--strongly-typed` and point `--onnx` at the base file
(`dfine_m_op19.onnx`). The v0.2.0 `fp16_st` assets (decoder kept FP32) remain on the
[v0.2.0 release](https://github.com/PogChamper/dfine-cpp/releases/tag/v0.2.0) and are still a
valid, slightly slower build.

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
python trt-files/scripts/export_dfine_onnx.py --model-name m --opset 19 \
    --checkpoint dfine_m_obj2coco.pt --dfine-src ./D-FINE-seg \
    --output trt-files/onnx/dfine_m_op19.onnx                        # FP32 base + .json sidecar
python trt-files/scripts/convert_fp16_surgical.py --slim \
    --onnx trt-files/onnx/dfine_m_op19.onnx \
    --output trt-files/onnx/dfine_m_slim.onnx                        # production surgical FP16
```

`--opset 19` matters: the surgical converter refuses opset-16 exports (their decomposed LayerNorm
is miscompiled by TensorRT in FP16 — see the [Precision guide](#precision-guide)). The export also
takes accuracy/speed sliders (`--num-queries`, `--eval-idx`, `--cascade K:KEEP`) that shrink the
graph itself — the `fast` column in the hero table is `--num-queries 200 --cascade 1:100`; the
cost/gain of every slider is tabulated in [docs/RESEARCH_MATRIX.md](docs/RESEARCH_MATRIX.md).
The older `convert_fp16.py` (decoder kept FP32) remains the fallback for opset-16 exports.

Standard checkpoints (`dfine_<size>_<dataset>.pt`) can be fetched from Hugging Face with D-FINE-seg's
`ensure_pretrained` helper (`src/d_fine/utils.py` in that repo); nano has no obj2coco checkpoint — use
`dfine_n_coco.pt`. Any other checkpoint: pass its path to `--checkpoint`. The `dfine export` CLI (below)
wraps the same script.

## Benchmarks

![Throughput over COCO stills](assets/demo.gif)
*D-FINE-N over COCO val2017 stills at 30× slow motion: the `fast`-preset engine (1556 img/s b8)
against PyTorch FP32 (191 img/s b8, measured with `profile.py --backends torch --model-name n`);
identical detections both panels.*

RTX 4070 Ti SUPER · COCO val2017 (5000 imgs) · D-FINE-M · latency = e2e p50 ms
(preprocess+infer+decode). Measured with `profile.py` in the v0.2.0 session — a couple of percent
below the hero table's newer 3-round `dfine_bench` medians for the same engines.

| backend | e2e (b1) | **FPS b1** | **FPS b8** | GPU MiB | mAP |
|---|---|---|---|---|---|
| PyTorch (FP32) | 32.0 | 31 | 66 | — | 0.5509 |
| ONNXRuntime-GPU (FP32) | 25.0 | 40 | 89 | — | 0.5509 |
| TensorRT FP32 (Python) | 8.0 | 125 | 160 | 640 | 0.5507 |
| **C++ FP32** | 5.7 | 176 | 227 | 642 | 0.5506 |
| **C++ FP16** | **3.7** | **272** | **459** | **488** | 0.5500 |

FP16 (strongly typed) is lossless to ≤0.0006 AP on every size at 1.3–2.8× the FP32 throughput; per-size
FP32 columns and batch scaling are in the [nightly report](trt-files/scripts/overnight_bench.sh) output.

**Frozen pipeline (opt-in), for steady-state streaming:** `gpu_decode` decodes on the GPU so only
surviving detections cross PCIe; `freeze()` / `FreezeSpec` locks the memory footprint (zero steady-state
allocation, VRAM Δ = +0 B over full runs); `full_pipeline_graph` captures H2D → preprocess → infer →
decode → D2H into **one `cudaGraphLaunch` per frame** (needs a `--max-aux-streams 0` engine and a fixed
source resolution; byte-identical to the split path, validated on 1061 real 640×480 COCO images).
Measured on m FP16: batch-1 e2e wall **−34.3%**, CPU per frame **4.30 → 0.195 ms** (dispatch
4.18 → 0.12 ms); batch-8 frees 19.2 ms of CPU per call. The score threshold stays a live per-call knob
inside the captured graph.

```cpp
dfine::DetectorOptions o;
o.full_pipeline_graph = true;                     // implies gpu_decode
dfine::DFineDetector det("dfine_m_fp16_st.engine", o);
det.freeze(dfine::FreezeSpec{1, 1920, 1080});     // batch, source WxH — captures + locks
det.detect(frame);                                // one cudaGraphLaunch per call
```

`last_timings()` reports per-stage CPU cost — the `dispatch_ms` column is what the graph collapses.
Exact contract: [include/dfine/tasks/detector.hpp](include/dfine/tasks/detector.hpp).

### Reproduce

```sh
# PyTorch vs ONNXRuntime-GPU vs TensorRT vs C++ side by side (latency, FPS, GPU mem, mAP):
python trt-files/scripts/profile.py --backends torch onnx trt cpp cpp-graph \
       --engine trt-files/engines/dfine_m_fp16_st.engine --full --batches 1 8
bash trt-files/scripts/overnight_bench.sh    # the full n→x sweep behind the tables above
./build/dfine_bench --pipeline-compare ...   # per-stage CPU cost, split vs full graph
```

(The `torch` backend and `.pt`→ONNX export need the
[D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) source on `PYTHONPATH`; everything else is
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
pip install "dfine[tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.0/dfine-0.3.0-py3-none-linux_x86_64.whl"
```

The wheel bundles `libdfine.so` built for sm_89 / linux_x86_64 plus a snapshot of
`build_engine.py`, so `dfine build` works without a repo checkout; other GPUs and platforms build
from source (`./build.sh`, then `pip install -e python/`).

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

| mode | mAP (full-val, m) | b8 vs FP32 | how |
|---|---|---|---|
| **FP32** | 0.5506 | 1× | `build_engine.py --no-tf32` |
| FP16 (decoder FP32) | 0.5500 | 2.0× | `convert_fp16.py` → `build_engine.py --strongly-typed` (**not** the `kFP16` flag) |
| **surgical FP16** ✅ | lossless, all 5 sizes (m: 0.5502 surgical / 0.5500 `--slim`) | 2.3× (2.4× opt8) | opset-19 export → `convert_fp16_surgical.py` (release ships `--slim`) |
| FP8 | −17.6 AP (subset) | 1.8× (7-9% *slower* than FP16) | rejected: E4M3 mantissa + GeForce Ada runs FP8 at FP16 rate |
| INT8 | −3.2 AP | 2.26× | rejected: slower than surgical FP16 with real accuracy cost (`convert_int8.py` kept for reference) |
| BF16 | −0.0012 (sim) | — | no win over FP16 paths; the old "−27 AP" was a weak-typing artifact |

**Opset 19 is mandatory for surgical FP16**: opset-16 exports decompose LayerNorm into primitive
ops, and TensorRT 10.13 miscompiles that decomposition in FP16 (mAP collapses to ~0.005 while
ONNXRuntime stays healthy — a TRT-side bug; minimal repro archived, NVIDIA report in preparation).
The converter hard-errors on opset < 19.

**Export sliders** — accuracy you can trade back for speed at export time (numbers: m, full-val,
b8; gains relative to the surgical b8 median of 526 img/s; all five sizes and every
hyperparameter point in [docs/RESEARCH_MATRIX.md](docs/RESEARCH_MATRIX.md)):

| slider | cost | gain | note |
|---|---|---|---|
| `--cascade 1:150` | −0.18 AP | +8% | top-150 queries after layer 1, ranked by the trained aux head — the best single slider |
| `--num-queries 200` | −0.13 AP | +7% | decode cost halves; near-free on few-class fine-tunes |
| `--eval-idx 2` | −0.57 AP | +4% | drops a decoder layer; best b1 latency without a CUDA graph |
| `fast` = Q200+cascade 1:100 | −0.44…−0.77 AP | +10…21% | the hero-table column |
| `max` = fast + E2 + `--opt-batch 8` | −0.89 AP | +46% vs prod (686 img/s) | m ladder top, 10.4× PyTorch |

The through-line: **D-FINE's FDR box-decode is acutely FP-precision-sensitive** — it amplifies tiny
upstream rounding into box error. That single fact explains the grid_sample trap, the kFP16-flag
trap, the surgical converter's FP32 island (FDR scopes + deform *index* math), and why FP8/INT8
fail here. Full forensics in [docs/impl/M0_STATUS.md](docs/impl/M0_STATUS.md),
[docs/RESEARCH_MATRIX.md](docs/RESEARCH_MATRIX.md), and [docs/HANDOFF.md](docs/HANDOFF.md).

## Status & roadmap

**Done & validated:** M0 (export→engine→validate, all 5 sizes) · M1 (C++ detector, AP 0.5506 == reference) ·
M2 (FP16 + CUDA-graph) · **M4 bindings** (stable C ABI + Python `ctypes` package + zero-setup `dfine`
CLI — detections byte-identical to the C++ path) · **intensive-core P1–P3** (GPU decode, `freeze()`,
full-pipeline graph) · optional letterbox preprocessing · **v0.3.0 precision campaign** (surgical FP16
lossless on all five sizes, export sliders, cascade pruning; FP8/INT8/deform-plugin closed with
measurements — [docs/RESEARCH_MATRIX.md](docs/RESEARCH_MATRIX.md)).
M3 instance segmentation is shelved. Next: a WASM/WebGPU browser demo and real-time video apps — see
**[docs/ROADMAP.md](docs/ROADMAP.md)** and [docs/HANDOFF.md](docs/HANDOFF.md) for the single source of
truth on the current state.

## Requirements & environment

| Dependency | Version | Notes |
|:---|:---|:---|
| OS | Linux x86_64 | validated; other platforms untested |
| NVIDIA GPU + driver | CUDA-12-capable | validated on RTX 4070 Ti SUPER (sm_89) |
| CUDA toolkit | 12.x | `nvcc` needed for the C++ build only |
| TensorRT | 10.x | validated on 10.13; `pip install "tensorrt==10.13.*"` works as a lib source |
| CMake | ≥ 3.24 (`native` arch) / ≥ 3.20 (explicit arch) | `build.sh` auto-discovers the toolchain |
| Compiler | C++17 | |
| OpenCV | — | not required |
| Python | engine build / export scripts only | never at inference time |

A system TensorRT is found automatically ([cmake/FindTensorRT.cmake](cmake/FindTensorRT.cmake) searches
`$TENSORRT_DIR`, `/usr/local/TensorRT`, `/opt/tensorrt`, `/usr`); alternatively populate
`third_party/tensorrt` ([third_party/README.md](third_party/README.md)). The ONNX **export** alone needs
the [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) source on `PYTHONPATH`; every other script is
self-contained (`pyproject.toml`, uv).

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
