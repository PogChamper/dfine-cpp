# D-FINE-cpp Roadmap

D-FINE-cpp is a validated C++/TensorRT inference library, not a research prototype: **M0** shipped the
raw explicit-gather ONNX export that fixes a −10.5 AP TensorRT `grid_sample` trap, **M1** shipped an
OpenCV-free C++ detector matching PyTorch to 0.5506 vs 0.5507 AP, and **M2** shipped strongly-typed FP16
(−0.2% mAP at 1.6–2.2×) plus CUDA-graph replay (−34.5% batch-1 latency on 0-aux-stream engines) — while
INT8 and BF16 were both **investigated and rejected** because D-FINE's FDR box-decode is exquisitely
mantissa-precision-sensitive.

That FDR sensitivity is the through-line constraint for everything below: it explains why the naive
`grid_sample` export lost AP, why the `kFP16` builder flag alone lost AP, why BF16/INT8 fail outright, and
why several Tier 4 ideas exist specifically to work *around* it. The tiers below are ordered by what makes
the library complete and adoptable first, then developer experience and hype, then production-serving
infrastructure, and finally research bets that may or may not pay off.

---

## Tier 1 — Complete the core (do next)

The library is currently detection-only, C++-only, and has no stable ABI. These three items close that
gap and are the direct prerequisites for almost everything else in this document.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| M3 instance segmentation | SHELVED (no ckpts/use-case) | M-L | High | Medium |
| `extern "C"` ABI wrapper (`c_api.h`/`.cpp`) | ✅ DONE (M4) | S-M | High | Low |
| Python bindings over the C ABI | ✅ DONE (M4) | M | Very high | Medium |
| pip **wheel** (bundle `.so`, extra `tensorrt`) | DO-NEXT | S-M | High | Medium |

- **M3 instance segmentation.** D-FINE-seg already has the mask head, so this is additive rather than
  exploratory: add a `masks` output to the ONNX export, write a GPU bilinear-upsample + threshold-0.5
  decode kernel (mask logits are pre-sigmoid, so thresholding is a plain `> 0` compare on the logit rather
  than a sigmoid + `>0.5`), and populate a `mask` field on `Detection`. The existing OpenCV-free `ImageU8`
  and raw-export patterns extend directly to this task, and `rf-detr`'s `mask_decode.cu` is a working model
  to follow (see `docs/research/D07_cpp_mask_decode.md`).
- **`extern "C"` ABI wrapper.** An opaque handle with no C++ types or exceptions crossing the boundary
  (catch internally, return error codes) unlocks every non-C++ consumer of `libdfine.so`. The plan is
  already specced in `docs/synthesis/01_PLAN_dfine_cpp.md`, and this is a hard prerequisite for bindings of
  any kind.
- **Python bindings + pip wheel.** Built over the C ABI (pybind11 or ctypes), this is the single highest
  -leverage move for adoption and stars — most detector users are Python-first. The caveat is that
  TensorRT is not redistributable, so the wheel must depend on the user's local TRT install; document this
  clearly and ship the `.so` plus a thin loader rather than attempting to vendor TensorRT.

## Tier 2 — DX & hype

Once the core is complete and has a stable ABI, these items are about making the project legible and
shareable — cheap to build, high visibility, low technical risk.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| WASM/WebGPU browser demo | HIGH-HYPE / standout | M | Very high | Low |
| Real-time demo apps (video/camera, FPS overlay) | WORTH-IT | M | High | Low |
| Zero-setup CLI (export → build → cache → run) | ✅ DONE (M4, `dfine` CLI) | M | Medium-high | Low-medium |
| Dockerfile, GPU-runner CI, engine cache/registry | NICE | S-M | Incremental | Low |

- **WASM/WebGPU browser demo.** The explicit-gather export (no `GridSample` plugin) is exactly what makes
  the graph portable to `onnxruntime-web`, since browser runtimes can't load custom TRT plugins anyway. A
  GitHub-Pages webcam demo running this ONNX graph client-side is a top star-magnet and costs nothing to
  host.
- **Real-time demo apps.** `dfine_video` / camera capture with a side-by-side FPS overlay brings OpenCV
  back, but scoped strictly to the *app* layer — the core library stays OpenCV-free. A "PyTorch 15 FPS vs
  TRT 272 FPS" gif is the single most effective marketing asset this repo could produce.
- **Zero-setup CLI.** Auto-downloading weights from Hugging Face, then exporting, building, and running in
  one command hides the conversion pain that currently requires the D-FINE-seg source on `PYTHONPATH`. The
  wrinkle is that the export step still needs that source package, so this CLI has to vendor or fetch it.
- **Dockerfile/build.sh, GPU-runner CI, engine cache.** `build.sh` already exists; a Dockerfile, a
  GPU-enabled CI runner, and an engine cache/registry (hash of ONNX + SM arch → cached `.engine`) are
  incremental trust- and DX-building work rather than new capability.

## Tier 3 — Serving & infra (high-load)

These items matter once the library needs to serve concurrent traffic rather than run a single offline
batch job. `TrtSession` today is single-context/single-stream, so the async worker pool is the gating item
for most of the rest of this tier.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| Async worker pool + dynamic in-flight batching | WORTH-IT for serving | L | High | Medium |
| NVIDIA Triton custom C++ backend | NICHE, high-value (enterprise) | L | High (enterprise) | Medium |
| DeepStream/NVDEC zero-copy | WORTH-IT for video | L | High | Medium |
| Prometheus telemetry, multi-GPU balancing, OOM fallback, thermal watchdog | INCREMENTAL | S-M each | Medium | Low |

- **Thread-safe async worker pool.** Add a pool of TensorRT contexts, dynamic in-flight batching (N=1..8,
  already validated), and a `std::future`-based request API. This requires moving `TrtSession` off its
  current single-context/single-stream design and is the prerequisite for the Triton backend below.
- **NVIDIA Triton custom C++ backend.** Wrapping `libdfine.so` as a Triton backend gives gRPC/HTTP and
  dynamic batching "for free" once the async worker-pool story exists — enterprise-valuable but a narrow
  audience, hence niche-high-value rather than broadly worth-it on its own.
- **DeepStream/NVDEC zero-copy.** Accept `CUdeviceptr` NV12 frames directly from the decoder and fuse
  NV12→RGB→stretch→`/255` into the existing preprocess kernel, skipping the host round-trip entirely. High
  impact for any video-analytics deployment, since H2D copies are otherwise a fixed tax on every frame.
  
- **Telemetry & resiliency.** Prometheus metrics (the existing `Timings` struct is the natural hook),
  multi-GPU load balancing, graceful OOM fallback (graph → `enqueueV3` → smaller batch, extending the
  fallback path that already exists), and a thermal watchdog are all incremental — build them as production
  need arises rather than speculatively.

## Tier 4 — Hardcore / research

These are speculative bets, mostly aimed at either pushing precision/throughput further or broadening the
library's positioning. Effort and risk are both higher, and several are explicitly research rather than
engineering.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| FP8 (E4M3) precision path | RESEARCH-WORTH-A-SHOT | M | High if it works | High |
| Custom fused FDR TensorRT plugin | HARDCORE-niche | XL | High if it works | High |
| INT8 QAT / partial quantization via layer profiling | RESEARCH | L | Uncertain | Medium-high |
| Object tracking (ByteTrack/BoT-SORT) + zone counting | WORTH-IT for video-analytics | M | High | Low |
| DLA offload (Jetson) | NICHE (edge/robotics) | M | Medium | Medium (needs HW) |
| DLPack zero-copy FFI | WORTH-IT-later | M | Medium | Low |
| Encrypted-engine/TEE, declarative pipeline API, ORT/OpenVINO/CoreML fallback | SKIP / later | — | — | — |

- **FP8 (E4M3).** INT8 failed because 8-bit *integer* quantization is too coarse for the FDR box-decode,
  but FP8 E4M3 is a *floating*-point format (4 exponent / 3 mantissa bits) with native hardware support on
  Ada. Its 3 mantissa bits are still below FP16's 10, so it will plausibly fail the same way BF16 did — but
  it is a cheap, strongly-typed experiment (same recipe as the FP16 win) that could deliver up to a 2×
  speedup over FP16 if the FDR turns out to tolerate it.
- **Custom fused FDR TensorRT plugin.** Fuse Integral + distance2bbox + LQE into a single FP32 kernel so
  the rest of the network can run INT8/FP16 while the FDR stays numerically exact — potentially *rescuing*
  INT8 entirely. This deliberately reintroduces a custom plugin, the exact thing the explicit-gather export
  was built to avoid, so it's a large, high-risk undertaking reserved for when the precision ceiling
  actually blocks a use case.
- **INT8 QAT / partial quantization.** Use polygraphy layer-profiling to separate INT8-tolerant backbone
  layers from FDR-feeding, precision-sensitive ones, then build a mixed INT8/FP16/FP32 engine. This needs
  retraining (QAT), which is beyond the scope of a pure inference library, and may not beat plain FP16 in
  the end — but a reproducible script + writeup would be valuable to the wider quantization community even
  if it doesn't ship as a default path.
- **Object tracking.** ByteTrack or BoT-SORT plus zone counting / line-crossing turns the detector into a
  video-analytics building block, which is a natural, low-risk extension once the demo apps and streaming
  input exist.
- **DLA offload (Jetson).** Running backbone/encoder on the DLA and decoder on the GPU is edge/robotics
  -specific and needs physical Jetson hardware to validate — worth doing if that market materializes, not
  before.
- **DLPack zero-copy FFI.** Natural follow-on to the Python bindings once those exist, letting frameworks
  hand D-FINE-cpp a GPU tensor without a copy.
- **Encrypted-engine/TEE, declarative pipeline API, ORT/OpenVINO/CoreML fallback.** Explicitly deprioritized
  — a multi-backend fallback in particular would dilute the project's core positioning as a TensorRT
  -optimized runtime.

---

## Recommended v1.0 sequence

1. **C ABI → Python bindings.** Nothing downstream (demos, serving, tracking) is easy to build without a
   stable, non-C++ entry point first.
2. **M3 instance segmentation.** Closes the last major capability gap versus upstream D-FINE-seg while the
   codebase and export patterns are freshest.
3. **Demo apps + WASM demo + gifs.** Convert the now-complete, bindable library into visible proof —
   this is the cheapest, highest-leverage adoption work in the whole roadmap.
4. **Async worker pool for serving.** Unlocks Triton and DeepStream integration and turns the library from
   an offline-batch tool into something that can sit behind a production endpoint.
5. **Pick a lane.** From there, choose **video-analytics** (tracking + DeepStream/NVDEC zero-copy) or
   **hardcore-precision** (FP8, QAT, fused-FDR plugin) based on where demand actually shows up — both are
   legitimate next chapters, but trying to do both at once will stall progress on either.

`docs/hardcore-ideas.md` holds the deeper backlog for anything in Tier 4 that needs more detailed design
notes before it's picked up.
