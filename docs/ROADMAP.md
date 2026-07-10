# D-FINE-cpp Roadmap

D-FINE-cpp is a validated C++/TensorRT inference library, not a research prototype: **M0** shipped the
raw explicit-gather ONNX export that fixes a −10.5 AP TensorRT `grid_sample` trap, **M1** shipped an
OpenCV-free C++ detector matching PyTorch to 0.5506 vs 0.5507 AP, and **M2** shipped strongly-typed FP16
(−0.2% mAP at 1.6–2.2×) plus CUDA-graph replay (−34.5% batch-1 latency on 0-aux-stream engines) — while
the INT8/BF16 PTQ recipes we tried in v0.3.0 were **closed as measured negatives on GeForce Ada**
(details and scope in docs/RESEARCH_MATRIX.md §6).

FDR sensitivity to reduced precision is the through-line constraint for everything below: it explains why
the naive `grid_sample` export lost AP and why the `kFP16` builder flag alone lost AP. For INT8 the matrix's
own conclusion is narrower than "impossible" — accuracy behaved like an activation-*calibration* problem,
and the v0.3.0 closure was ultimately a throughput verdict on Ada (surgical FP16 dominated every INT8
recipe we could build); later calibration research has re-opened the accuracy question, so treat the
closure as scoped to those recipes/date/hardware, not as physics. The tiers below are ordered by what
makes the library complete and adoptable first, then developer experience and hype, then
production-serving infrastructure, and finally research bets that may or may not pay off.

---

## Tier 1 — Complete the core

The library is detection-only by design for now; the stable C ABI, Python bindings, and pip wheel below
are DONE and shipped (v0.1.0–v0.3.0). Segmentation remains shelved until checkpoints and a use-case show up.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| M3 instance segmentation | SHELVED (no ckpts/use-case) | M-L | High | Medium |
| `extern "C"` ABI wrapper (`c_api.h`/`.cpp`) | ✅ DONE (M4) | S-M | High | Low |
| Python bindings over the C ABI | ✅ DONE (M4) | M | Very high | Medium |
| pip **wheel** (bundle `.so`, extra `tensorrt`) | ✅ DONE (v0.1.0; v0.3.0 bundles `build_engine.py` too) | S-M | High | Medium |

- **M3 instance segmentation.** D-FINE-seg already has the mask head, so this is additive rather than
  exploratory: add a `masks` output to the ONNX export, write a GPU bilinear-upsample + threshold-0.5
  decode kernel (mask logits are pre-sigmoid, so thresholding is a plain `> 0` compare on the logit rather
  than a sigmoid + `>0.5`), and populate a `mask` field on `Detection`. The existing OpenCV-free `ImageU8`
  and raw-export patterns extend directly to this task, and `rf-detr`'s `mask_decode.cu` is a working model
  to follow.
- **`extern "C"` ABI wrapper — DONE (M4).** An opaque handle with no C++ types or exceptions crossing
  the boundary (catch internally, return error codes); shipped as `include/dfine/c_api.h`, built into
  `libdfine.so` by default, and it is what the Python bindings stand on.
- **Python bindings + pip wheel — DONE (M4/v0.1–v0.3).** ctypes over the C ABI; the wheel ships the
  `.so` plus a thin loader. TensorRT is not redistributable, so the wheel depends on the user's local
  TRT install — documented in the README.

## Tier 2 — DX & hype

Once the core is complete and has a stable ABI, these items are about making the project legible and
shareable — cheap to build, high visibility, low technical risk.

| Item | Verdict | Effort | Impact | Risk |
|---|---|---|---|---|
| WASM/WebGPU browser demo | HIGH-HYPE — needs a timeboxed feasibility spike first | M | Very high | Medium (operator/memory coverage on browser runtimes unproven) |
| Real-time demo apps (video/camera, FPS overlay) | WORTH-IT | M | High | Low |
| Zero-setup CLI (export → build → cache → run) | ✅ DONE (M4, `dfine` CLI) | M | Medium-high | Low-medium |
| Dockerfile ✅ · engine cache ✅ (v0.3.1, content-fingerprinted) · GPU-runner CI | NICE (CI runner remains) | S-M | Incremental | Low |

- **WASM/WebGPU browser demo.** The explicit-gather export (no `GridSample` plugin) removes the known
  blocker for `onnxruntime-web`, since browser runtimes can't load custom TRT plugins anyway — but
  operator coverage and memory on a real browser runtime are unproven, so a 3–4 h feasibility spike
  (nano model, one image, box parity) comes first. If it passes, a GitHub-Pages webcam demo is a top
  star-magnet and costs nothing to host.
- **Real-time demo apps.** `dfine_video` / camera capture with a side-by-side FPS overlay brings OpenCV
  back, but scoped strictly to the *app* layer — the core library stays OpenCV-free. A "PyTorch 15 FPS vs
  TRT 272 FPS" gif is the single most effective marketing asset this repo could produce.
- **Zero-setup CLI.** Auto-downloading weights from Hugging Face, then exporting, building, and running in
  one command hides the conversion pain that currently requires the D-FINE-seg source on `PYTHONPATH`. The
  wrinkle is that the export step still needs that source package, so this CLI has to vendor or fetch it.
- **Dockerfile/build.sh, GPU-runner CI, engine cache.** `build.sh`, the Dockerfile, and the engine
  cache all exist (v0.3.1 keys the cache by ONNX+sidecar content fingerprint + SM arch + TRT version);
  the remaining item is a GPU-enabled CI runner — incremental trust-building work rather than new
  capability.

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
| FP8 (E4M3) precision path | ❌ CLOSED-NEGATIVE (v0.3.0: −17.6 AP *and* 7-9% slower on GeForce Ada — [RESEARCH_MATRIX](RESEARCH_MATRIX.md)) | M | — | — |
| Custom fused FDR TensorRT plugin | ❌ CLOSED (v0.3.0: Myelin already fuses deform to 16 kernels; ≤8% e2e ceiling) | XL | — | — |
| INT8 QAT / partial quantization via layer profiling | ❌ CLOSED-NEGATIVE for PTQ (v0.3.0: best real engine 519 img/s < surgical 528/561 at −3.2 AP); QAT would need training, out of scope | L | — | — |
| Object tracking (ByteTrack/BoT-SORT) + zone counting | WORTH-IT for video-analytics | M | High | Low |
| DLA offload (Jetson) | NICHE (edge/robotics) | M | Medium | Medium (needs HW) |
| DLPack zero-copy FFI | WORTH-IT-later | M | Medium | Low |
| Encrypted-engine/TEE, declarative pipeline API, ORT/OpenVINO/CoreML fallback | SKIP / later | — | — | — |

- **FP8 (E4M3) — measured and closed (v0.3.0).** Real TRT engines (modelopt QDQ, decoder excluded):
  subset mAP 0.3909 (−17.6) *and* 7-9% slower than FP16-ST — GeForce Ada runs FP8 tensor cores at FP16
  rate (FP8 mandates FP32 accumulation, halved on GeForce) so Q/DQ overhead buys nothing. A torch
  ablation proved the accuracy loss is E4M3-mantissa-limited and calibration-invariant: only QAT could
  help, out of scope. Numbers: [RESEARCH_MATRIX.md](RESEARCH_MATRIX.md).
- **Custom fused FDR / deform plugin — measured and closed (v0.3.0).** Engine profiling shows Myelin
  already fuses the explicit deform core into 16 kernels (~8-17.5% of GPU time); a perfect plugin caps
  below that, and the FDR tail is only ~5.6%. The surgical-FP16 converter captured the actual win
  (decoder FP16 with an FP32 FDR island) with zero plugins.
- **INT8 — PTQ measured and closed (v0.3.0).** Best real engine (torch-side percentile calibration,
  scale injection) reached 0.5190 full-val at 519 img/s — slower than surgical FP16 (528/561) with real
  accuracy cost; the int8+surgical combo was slower still (486). The int8 conv gain is ~1.23× real on
  Ada once Q/DQ overhead is paid. QAT needs training and is out of scope for an inference library.
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

The v0.1–v0.3 releases already delivered the original step 1 (C ABI → Python bindings → wheel). What
remains, in order:

1. **Hardening and packaging**: fail-safe error recovery, strict custom-checkpoint export,
   artifact-bound engine cache, one canonical wheel path (shipped in v0.3.1); CMake
   `install()`/`find_package` + an out-of-tree consumer CI job (v0.3.2).
2. **External validation.** Ampere/Turing reports, a TensorRT 11 pass, Jetson build docs — every
   "validated" claim backed by a reproducible report, contributed or rented.
3. **Demo apps + gifs; a browser demo only after a timeboxed feasibility spike** (the explicit-gather
   graph removes the GridSample blocker, but operator coverage and memory on a real browser runtime are
   unproven — see Tier 2).
4. **Async worker pool for serving.** Unlocks Triton and DeepStream integration and turns the library from
   an offline-batch tool into something that can sit behind a production endpoint.
5. **Pick a lane.** From there, choose **video-analytics** (tracking + DeepStream/NVDEC zero-copy) or
   **hardcore-precision** (INT8 calibration productization, QAT) based on where demand actually shows
   up — both are legitimate next chapters, but trying to do both at once will stall progress on either.
   (M3 segmentation re-enters here if checkpoints and a use-case materialize.)

