# D-FINE-cpp — HANDOFF (read this first)

Single source of truth for the current state. Any agent/dev should be able to continue from here.
Goal: a production C++/TensorRT inference library for **D-FINE** (detection now, optional instance-seg later),
modeled on `rf-detr-cpp`. Repo: `/home/dxdxxd/projects/custom-dfine/D-FINE-cpp`.

## Status (2026-07-01)

- **M0 = export → engine → validate: DONE and validated on all 5 sizes (n/s/m/l/x).** The Python
  export/build/eval pipeline is canonical and correct; a trustworthy TRT engine is produced.
- **M1 = C++ detector: DONE and mAP-validated.** `libdfine.so` (`DFineDetector`) runs the full
  pipeline (CUDA preprocess → TRT engine → C++ decode). **COCO val2017 (5000 imgs): C++ AP = 0.5506
  vs the Python engine reference 0.5507 (PyTorch 0.5509) — a 0.0001 match.** `detect_batch` (batch 8)
  gives the identical AP to batch 1; dynamic batch N=1/2/8 verified; `dfine_build` rebuilds the engine
  in pure C++ (mAP-equal). See "M1 — C++ detector (DONE)" below.
- The big M0 discovery (and its fix) is the most important thing to carry forward — see "Decisions" below.

## The validated pipeline (reproduce)

Run from the D-FINE-seg repo (its `.venv` has torch 2.9 + TensorRT 10.13 + onnxruntime-gpu):
```sh
SEG=/home/dxdxxd/projects/custom-dfine/D-FINE-seg ; PY=$SEG/.venv/bin/python
S=/home/dxdxxd/projects/custom-dfine/D-FINE-cpp/trt-files/scripts ; cd $SEG
$PY $S/export_dfine_onnx.py --model-name m --checkpoint pretrained/dfine_m_obj2coco.pt   # -> trt-files/onnx/dfine_m.onnx (+ .json sidecar)
$PY $S/build_engine.py --no-tf32 --max-batch 8                                            # -> trt-files/engines/dfine_m_fp32.engine
$PY $S/verify_engine.py --batches 1 2 8                                                   # smoke (dynamic batch)
$PY $S/coco_eval.py --limit 2000 --backends engine                                        # mAP (engine vs ORT-GPU/torch)
```
Result: engine AP == PyTorch/ONNXRuntime to ≤0.0001 (m full-val 0.5507 vs torch 0.5509).

Scripts (canonical, `trt-files/scripts/`): `export_dfine_onnx.py` (raw export + the deform fix),
`build_engine.py` (native TRT build, dynamic-batch profile, precision flags), `verify_engine.py`
(smoke), `coco_eval.py` (mAP, multi-backend), `parity_check.py` (per-image torch/ORT/TRT),
`cuda_env.py` (onnxruntime-gpu bootstrap), `seg_export_repro.py` (the D-FINE-seg bug-report repro).
`experiments/` holds superseded one-offs. (`compare_export_backends.py` + `EXPORT_BACKEND_COMPARISON_RRS.md`
are the user's RRS food-detector validation — leave them.)

## Decisions & gotchas (the canonical truths — full detail in `impl/M0_STATUS.md`)

1. **Architecture:** freeze the whole net into a `.engine`; reimplement in C++ only preprocess (CUDA),
   detection decode (C++), orchestration (PIMPL). Reuse rf-detr-cpp's `TrtSession`/RAII/`c_api`/CUDA-graph
   verbatim. FDR Integral/LQE box decode stays **inside** the engine — never reimplement in C++.
2. **★ The deformable-attention TRT trap (M0's headline finding).** D-FINE's `F.grid_sample` deformable
   core is bit-exact on TRT *in isolation* but compiled **divergently in-context** → ~10 AP loss
   (full-val 0.4455 vs 0.5509), because the FDR box accumulation amplifies it. **Fix: export with the
   explicit gather-bilinear deform core** (`export_dfine_onnx.py --deform explicit`, the **default**) —
   `Gather`+arithmetic instead of `GridSample`, TRT-exact, **no plugin, no latency cost**, recovers full mAP
   on all sizes. **Express the index clamp as `minimum(maximum(...))`, NOT `.clamp()`** (dynamo lowers
   `.clamp()` to a `Clip` TRT 10.13 rejects). Works under legacy opset-16 AND dynamo opset-19+onnxsim.
   D-FINE-seg's own `run_parity` checks scores-only so it misses this; PR-ready writeup: `impl/DFINE_SEG_TRT_BUG_REPORT.md`.
3. **Preprocessing = `/255` ONLY, no ImageNet mean/std** (unlike rf-detr-cpp — copying its kernel collapses
   mAP). RGB, CHW, float32, stretch-resize to 640² (not letterbox), `orig_target_sizes` not used (raw export).
4. **Export contract:** RAW two-output graph `images[N,3,640,640] → logits[N,300,80], boxes[N,300,4]`
   (normalized cxcywh), opset 16, FP32. Decode in C++: sigmoid → top-300 over query×class →
   `label=idx%80`, `query=idx//80` → cxcywh→xyxy → scale to original → threshold. No NMS. No background slot.
5. **Dynamic batch:** trace export with **batch ≥ 2** (`--trace-batch 2`, default) or the batch axis bakes to
   1 and the engine rejects N>1. Build sets a min/opt/max profile (default 1/1/8). Validated N=1/2/8 for both
   grid_sample and explicit engines. CUDA-graph (M2) needs static shape → keep it opt-in with fallback.
6. **Precision:** FP32 baseline (build `--no-tf32` for FP32-faithful; TF32 on costs ~2× box L1). FP16 must
   keep the decoder FP32 (FP16 corrupts it) — **not yet mAP-validated**. INT8 backbone/encoder only (GridSample
   FP32/FP16-only) — **not done**.
7. **Per-size invariants:** `num_queries=300`, `reg_max=32`, 640² fixed across sizes. Varies: backbone B0–B5,
   `hidden_dim` 128/256/384, `num_layers` 3/3/4/6/6, `reg_scale` 4 (X=8), `num_levels` 2 (nano) else 3.
8. **Eval gotcha:** `COCOeval.params.imgIds` must be set to the processed subset, else missing images count as misses.

## Environment

- **Python/eval:** use the D-FINE-seg `.venv` (`uv`-managed; `uv pip install <x>` to add). Has torch 2.9.1+cu128,
  TensorRT 10.13.3, `onnxruntime-gpu==1.24.4` (CUDA-12 wheel — NOT 1.27, which needs CUDA 13), `socksio`
  (SOCKS proxy for HF). ORT-GPU via `cuda_env.bootstrap()` (WSL `/usr/lib/wsl/lib` libcuda + `preload_dlls`).
- **C++ toolchain (no sudo):** cmake `/home/dxdxxd/miniconda3/envs/dfine/bin/cmake` (4.3.1);
  `TENSORRT_DIR=D-FINE-cpp/third_party/tensorrt` (10.13 public headers from GitHub + symlinked libs);
  runtime `LD_LIBRARY_PATH=$SEG/.venv/lib/python3.11/site-packages/tensorrt_libs`. Verified compile+link.
  cmake: `-DTENSORRT_DIR=$TP -DCUDAToolkit_ROOT=/home/dxdxxd/miniconda3 -DCMAKE_CXX_FLAGS=-B/usr/bin`.
  M0/M1 apps need no OpenCV.
- **Checkpoints:** `m` = `D-FINE-seg/pretrained/dfine_m_obj2coco.pt`; `s/l/x` = `D-FINE-seg/dfine_{s,l,x}_obj2coco.pt`;
  `n` = `D-FINE-seg/dfine_n_coco.pt` (nano has no obj2coco on HF). Auto-downloadable via `ensure_pretrained`.
- **Data:** COCO val2017 at `/mnt/d/datasets/coco`. Engines/onnx are gitignored build outputs (regenerate via scripts).

## Doc map (reading order)

| Doc | Role |
|---|---|
| **`HANDOFF.md`** (this) | Current state + how to continue. **Start here.** |
| `impl/M0_STATUS.md` | M0 findings log — the grid_sample investigation, fixes tried, per-size validation. The "why". |
| `impl/DFINE_SEG_TRT_BUG_REPORT.md` | PR-ready writeup of the bug + fix for the D-FINE-seg author. |
| `impl/cpp_skeleton_spec.md` | Copy-faithful spec of the rf-detr-cpp C++ skeleton to port (TrtSession, EngineMeta, build app, CMake). |
| `synthesis/01_PLAN_dfine_cpp.md` | The pre-M0 design plan (18 sections). Still the architecture reference, but where it
  conflicts with M0 reality, **M0_STATUS/this doc win** (esp. deform core, export mode). |
| `synthesis/00_INDEX.md` + repo summaries / comparisons / pitfalls | Distilled design-phase analysis. |
| `research/*` (90 notes + `V00`) | Forensic evidence base with file:line proofs. Reference only. |
| `hardcore-ideas.md` | Backlog of advanced optimizations beyond M2 (throughput, precision, kernel/graph tricks, frontier). A menu for future milestones — read after the M2 roadmap. |

## M1 — C++ detector (DONE)

**Result:** `libdfine.so` + apps. Full pipeline is CUDA stretch-resize+`/255` → TRT engine → C++ decode.
COCO val2017 (5000): **C++ 0.5506 vs Python engine 0.5507** (0.0001). `detect_batch` (B=8) == B=1 AP;
dynamic batch N=1/2/8 OK; `dfine_build` rebuilds the engine in pure C++ (mAP-equal). The engine owns the
deformable/FDR core — C++ does only preprocess + decode + orchestration (no plugins at runtime).

**Build (toolchain gotcha baked in).** nvcc's host-link otherwise grabs conda's glibc-incompatible `ld`
(`__nptl_change_stack_perm@GLIBC_PRIVATE`); the wrapper `cmake/cuda_host_ccbin.sh` forces system binutils.
```sh
SEG=/home/dxdxxd/projects/custom-dfine/D-FINE-seg
TP=$PWD/third_party/tensorrt ; CM=/home/dxdxxd/miniconda3/envs/dfine/bin/cmake
$CM -B build -S . -DTENSORRT_DIR=$TP -DCUDAToolkit_ROOT=/home/dxdxxd/miniconda3 \
   -DCMAKE_CUDA_COMPILER=/home/dxdxxd/miniconda3/bin/nvcc \
   -DCMAKE_CUDA_HOST_COMPILER=$PWD/cmake/cuda_host_ccbin.sh \
   -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CXX_FLAGS=-B/usr/bin
$CM --build build -j4
export LD_LIBRARY_PATH=$SEG/.venv/lib/python3.11/site-packages/tensorrt_libs:/home/dxdxxd/miniconda3/lib
./build/dfine_detect --engine trt-files/engines/dfine_m_fp32.engine --image <img.jpg> --threshold 0.5
$SEG/.venv/bin/python trt-files/scripts/cpp_coco_eval.py --limit 0   # full-val mAP == coco_eval.py
```

**What shipped.** `libdfine`: `src/internal/trt_session.*` (name-driven, dynamic batch, grow-only buffers),
`src/core/engine_meta.cpp` (reads the Python sidecar; tries `<engine>.json` and `<engine-stem>.json`),
`src/core/cuda_preprocess.cu` (bilinear stretch-resize + `/255` only, BGR→RGB, HWC→CHW; pinned-buffer reuse
guarded by a CUDA event for batch), `src/core/postprocess.cpp` (sigmoid→top-300→`idx%C`/`idx//C`→cxcywh→xyxy,
no clamp/NMS — matches `coco_eval.py decode()`), `src/tasks/detector.cpp` (`DFineDetector` PIMPL, `ImageU8`
input — **OpenCV-free**, dynamic-ness/dims/dtype read from the engine bindings). Apps: `dfine_inspect`,
`dfine_smoke`, `dfine_build` (FP32-only), `dfine_detect`, `dfine_coco_eval`. Image decode via vendored
`third_party/stb/stb_image.h` (OpenCV absent from the `dfine` conda env). Validation driver:
`trt-files/scripts/cpp_coco_eval.py`.

**Design notes.** OpenCV-free by choice (detector takes a raw HWC `ImageU8`; stb for JPEG in apps). The
detector trusts engine bindings over the sidecar for shape facts (a stale/missing sidecar can't misconfigure
it) and rejects a non-FP32 input binding up front.

**Quality bar.** Three adversarial multi-agent reviews (correctness, cpp-pro C++, CUDA+TensorRT) with
per-finding verification; all confirmed items fixed. Builds clean under `-Wall -Wextra -Wpedantic` (baked into
the `dfine_warnings` target; `-DDFINE_WARNINGS_AS_ERRORS=ON` to enforce). All CUDA handles are RAII
(`CudaStream`/`CudaEvent` + `DevPtr`/`HostPtr` in `cuda_raii.hpp`; no leaks on exception paths). Sanitizer
build types `-DCMAKE_BUILD_TYPE=UBSAN` (safe for the full pipeline) and `ASAN` (host-isolated): the decode is
ASan+UBSan-clean on adversarial inputs (NaN/Inf/oob) and the full pipeline is UBSan-clean (single + batch).
The preprocess kernel is `compute-sanitizer` clean (memcheck+initcheck+synccheck = 0) across sizes incl. 1×1
and padded strides, and full `dfine_detect` is memcheck-clean. `dfine_build` persists a TensorRT timing cache
(`<engine>.timing.cache`) so same-architecture rebuilds skip re-benchmarking tactics.

**Profiling.** Two tools: `dfine_bench` (C++) times the detector's own path per stage
(preprocess+H2D / infer / D2H / decode) with warm-up, p50/p90/p99, throughput, and GPU-mem, across batch
sizes — e.g. `dfine_bench --engine <e> --batches 1,2,4,8 --iters 300` → batch-1 ~8.7 ms (115 img/s), batch-8
~146 img/s, ~674 MiB (abs latency varies with GPU boost clocks — no sudo to lock them, so read percentiles).
`trt-files/scripts/profile.py` compares backends on one deterministic dataset (`--subset N` / `--full` /
`--images DIR --ann JSON`): **`trt`** (ours), **`onnx`** (ORT-GPU reference), **`trt-baseline`** (the repo's
grid_sample export), **`torch`**, and **`cpp`** (our detector via `dfine_bench`/`dfine_coco_eval`) — reporting
latency + GPU mem + mAP in one table. The table shows two latency columns: **`e2e`** (preprocess+infer+decode
— real deployment) and **`infer`** (engine only: H2D+infer+D2H). `infer` is ~identical for `trt` vs `cpp`
(5.5 vs 5.4 ms) — proving the engine is the same; the win is in `e2e`: **C++ ~1.3× faster at batch 1 and
~1.7× at batch 4** than Python-TRT because preprocessing is a CUDA kernel (~0.17 ms) vs Python's cv2 CPU
resize (~3.5 ms/img, serial → the Python batch bottleneck) plus no torch/numpy per-call overhead. (ORT-GPU is
slower even at `infer`: ~15 ms.) It reuses `coco_eval.py`'s validated decode.
Demonstrated side-by-side:
ours **0.6606 AP** ≈ onnx 0.6602 ≈ cpp 0.6596, vs grid_sample baseline **0.5633 (−9.7 AP)** at equal speed —
the payoff of the explicit-deform export. The baseline engine is built via
`export_dfine_onnx.py --deform gridsample` + `build_engine.py`.

## M2 — production hardening (next milestone)

Feature parity vs a production C++/TRT detector (rf-detr-cpp is the model). **Done in M1** unless noted:

| Capability | Status | Notes |
|---|---|---|
| Runtime with no Python | ✅ | `libdfine.so`, stb for image decode |
| CUDA preprocess (fused resize+`/255`) | ✅ | ~0.17 ms |
| Async pinned H2D / grow-only buffers | ✅ | event-guarded pinned reuse |
| Dynamic batch (N=1..8) | ✅ | validated |
| FP32 accuracy == reference | ✅ | 0.5506 == 0.5507 |
| Full RAII / sanitizer-clean / warning-clean | ✅ | see Quality bar |
| Profiling (latency/mem/mAP, cross-backend) | ✅ | `dfine_bench` + `profile.py` |
| **FP16 (decoder pinned FP32)** | ⛔ M2 | biggest latency win; naive global FP16 corrupts the decoder |
| **CUDA-graph replay** | ⛔ M2 | currently per-frame `enqueueV3`; saves dispatch overhead |
| **INT8 (QDQ, backbone/encoder)** | ⛔ M2 | not the deprecated implicit calibrator |
| **Instance segmentation** | ⛔ M3 | D-FINE-seg mask head, threshold 0.5 (masks pre-sigmoid'd) |

**M2.1 — FP16 with FP32-pinned decoder (do first; ~1.5–2× latency).** A global `kFP16` flag corrupts D-FINE's
decoder the same way grid_sample did (~10 AP loss — quantify it as the anti-example with `profile.py`). Keep
backbone+encoder FP16 but pin the decoder head (class head, `bbox_head`, FDR Integral/LQE, distance2bbox) to
FP32. In `build_engine.py`: either a strongly-typed network driven by ONNX dtypes, or per-layer
`layer->setPrecision(kFLOAT)` + `setOutputType` on the decoder subgraph with `kPREFER_PRECISION_CONSTRAINTS`.
Identify decoder layers by name from the ONNX. **The C++ runtime already supports FP16 outputs**
(`get_output_f32` kHALF branch) — no C++ change needed if outputs stay FP32; if any output goes FP16 it's
handled. **Validate:** `profile.py --backends trt torch --subset 2000` — mAP must hold ~0.55; expect
`infer` to drop ~1.5–2×. Update the sidecar `precision` field. Do NOT ship FP16 from `dfine_build` until
mAP-validated (it currently rejects non-fp32 on purpose).

**M2.2 — CUDA-graph replay (opt-in).** Add to the **task layer** (`detector.cpp`), NOT `TrtSession` (see
`impl/cpp_skeleton_spec.md` §2.4g). After warm-up, `cudaStreamBeginCapture(session.stream())` over
`enqueueV3` + the D2H copies (keep H2D/preprocess outside the graph — image bytes differ per frame), then
`cudaGraphLaunch` each call. Needs a **static shape**, so capture one graph per batch size seen, and **fall
back to `enqueueV3` if capture fails** (transformer decoders can have data-dependent internal shapes — rf-detr
hits this and falls back). Add `DetectorOptions.use_cuda_graph` + a `cuda_graph_compat` meta flag. RAII the
`cudaGraph_t`/`cudaGraphExec_t` (add wrappers to `cuda_raii.hpp`). **Validate:** `dfine_bench` `infer` p50/p99
should drop; mAP unchanged.

**M2.3 — INT8 (QDQ).** Insert QDQ nodes via a calibration pass over COCO (a `convert_int8.py`, ONNX-level —
NOT TRT's deprecated implicit `IInt8EntropyCalibrator2`), keep the decoder FP32, build with `kINT8`. Expect
some mAP loss — quantify with `profile.py`. Backbone/encoder only.

**M2.4 — instance segmentation (M3, optional).** D-FINE-seg mask head → extra `masks` output; add a GPU
bilinear-upsample+threshold decode (model on rf-detr `mask_decode.cu`), threshold at 0.5 (masks are
pre-sigmoid'd, so compare >0.5 not logit>0), populate a mask field on `Detection`.

**Known latent (harmless today, non-happy-path):** `max_batch()` returns 0 when a dynamic engine's sidecar is
absent (TRT then bounds the batch itself); moved-from `DFineDetector` accessors are UB by convention (as in
most C++ types). Neither is on any exercised path.

**Where to start:** read this file top-to-bottom, then `impl/M0_STATUS.md` (the deform gotcha — the FP16 work
will re-encounter decoder FP-sensitivity), then `impl/cpp_skeleton_spec.md` §2.4g for the graph hook. Build +
run the pipeline (commands above), then `profile.py` to see the current baseline before changing anything.
