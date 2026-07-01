# D-FINE-cpp

**Production C++/TensorRT inference for the [D-FINE](https://github.com/Peterande/D-FINE) object detector.**
A zero-Python, OpenCV-free runtime that runs the full pipeline — CUDA preprocessing → TensorRT engine →
C++ decode — at up to **~460 FPS** on an RTX 4070 Ti SUPER, matching PyTorch mAP to **±0.001 AP**.

<!-- badges: build/license/release — wire up once a remote + CI are set (see .github/workflows) -->
`C++17` · `TensorRT 10.13` · `CUDA 12.8` · `Apache-2.0`

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

## Benchmarks

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
`bash trt-files/scripts/overnight_bench.sh`. (The `torch` backend and `.pt`→ONNX export need the D-FINE-seg
source on `PYTHONPATH`; the ONNX/TRT/C++ paths are self-contained.)

## Quickstart

```sh
# 1. Build the C++ (toolchain gotchas baked into build.sh — see docs/HANDOFF.md "Environment")
./build.sh

# 2. Export ONNX + build a TensorRT engine (Python; needs the D-FINE-seg venv, see below)
PY=<D-FINE-seg>/.venv/bin/python
$PY trt-files/scripts/export_dfine_onnx.py --model-name m --checkpoint dfine_m_obj2coco.pt   # -> dfine_m.onnx
$PY trt-files/scripts/build_engine.py --no-tf32 --max-batch 8                                 # -> dfine_m_fp32.engine

# 3. Run
export LD_LIBRARY_PATH=<D-FINE-seg>/.venv/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
./build/dfine_detect --engine trt-files/engines/dfine_m_fp32.engine --image dog.jpg --threshold 0.5
```

**FP16 engine** (the recommended production build — strongly-typed, decoder kept FP32):
```sh
$PY trt-files/scripts/convert_fp16.py --output trt-files/onnx/dfine_m_fp16_st.onnx
$PY trt-files/scripts/build_engine.py --strongly-typed --no-tf32 --max-batch 8 \
    --onnx trt-files/onnx/dfine_m_fp16_st.onnx --output trt-files/engines/dfine_m_fp16_st.engine
```

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

## Apps

`dfine_detect` (single image) · `dfine_coco_eval` (mAP) · `dfine_bench` (per-stage latency / FPS / GPU-mem,
`--cuda-graph`, `--graph-compare`) · `dfine_build` (pure-C++ engine build, FP32) · `dfine_inspect` /
`dfine_smoke`.

## Architecture

The whole network is frozen into a `.engine`; C++ reimplements only the parts that must be fast or Python-free:

```
ImageU8 (HWC u8) ──► CUDA preprocess (stretch-resize + /255, BGR→RGB, →CHW)  [src/core/cuda_preprocess.cu]
                 ──► TensorRT engine (backbone+encoder+decoder+FDR/Integral/LQE)  [frozen; owns deform core]
                 ──► C++ decode (sigmoid → top-300 → cxcywh→xyxy → scale)  [src/core/postprocess.cpp]
```

Preprocessing is **`/255` only — no ImageNet mean/std** (a D-FINE quirk; copying a normal detector's kernel
collapses mAP). The FDR/Integral/LQE box math stays *inside* the engine and is never reimplemented in C++.

## Precision guide

| mode | mAP | speed | how |
|---|---|---|---|
| **FP32** | reference | 1× | `build_engine.py --no-tf32` |
| **FP16** ✅ | −0.2% | 1.6–2.2× | `convert_fp16.py` → `build_engine.py --strongly-typed` (**not** the `kFP16` flag) |
| BF16 | −27 AP | — | rejected: D-FINE's FDR needs mantissa precision |
| INT8 | −44 AP | — | rejected: 8-bit too coarse for the FDR (`convert_int8.py` kept for reference) |

The through-line: **D-FINE's FDR box-decode is exquisitely FP-precision-sensitive** — it amplifies tiny
upstream rounding into box error. That single fact explains the grid_sample trap, the kFP16-flag trap, and why
BF16/INT8 fail. Full forensics in [docs/impl/M0_STATUS.md](docs/impl/M0_STATUS.md) and
[docs/HANDOFF.md](docs/HANDOFF.md).

## Status & roadmap

**Done & validated:** M0 (export→engine→validate, all 5 sizes) · M1 (C++ detector, AP 0.5506 == reference) ·
M2 (FP16 + CUDA-graph; INT8 investigated & rejected). Next: **M3 instance segmentation**, a C ABI + Python
bindings, and demo apps. See **[docs/ROADMAP.md](docs/ROADMAP.md)** for the prioritized plan and
[docs/HANDOFF.md](docs/HANDOFF.md) for the single source of truth on the current state.

## Requirements & environment

- **Runtime:** NVIDIA GPU (built for Ada `sm_89`; override `CUDA_ARCH`), CUDA 12.x, TensorRT 10.13.
- **C++ build:** CMake ≥ 3.20, a CUDA toolkit (`nvcc`), TensorRT headers (`third_party/tensorrt`, vendored).
  `./build.sh` wraps the exact incantation (incl. the conda-`ld` workaround). No OpenCV.
- **Python scripts** (export / engine-build / eval / profile): see `pyproject.toml` (uv). The **ONNX export**
  additionally needs the [D-FINE-seg](https://github.com/Peterande/D-FINE) source package on `PYTHONPATH` for
  model construction; everything else (build_engine / convert_fp16 / profile / coco_eval) is self-contained.

Engines and ONNX are gitignored build outputs — regenerate them with the scripts.

## Credits & license

C++/TensorRT port of **D-FINE** (Peng et al.), Apache-2.0. Vendored: `stb_image` (public domain),
nlohmann/json (MIT). Links NVIDIA TensorRT/CUDA (install separately). This project is **Apache-2.0** — see
[LICENSE](LICENSE) and [NOTICE](NOTICE).
