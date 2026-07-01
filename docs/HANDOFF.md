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
- **M2 = production hardening: FP16 + CUDA-graph DONE and validated; INT8 rejected.** Strongly-typed FP16
  (backbone+encoder FP16, decoder FP32) is **1.6–2.2× faster at −0.2% mAP**; opt-in CUDA-graph replay cuts
  **batch-1 latency −34.5%** on a single-stream (`--max-aux-streams 0`) engine (D-FINE is dispatch-bound). The weakly-typed `kFP16` flag is a trap (fixed −6.8 AP),
  and BF16/INT8 are dead — all for the same reason: D-FINE's FDR needs mantissa precision. See "M2" below.
- **M4 = bindings & DX: DONE and validated.** A stable `extern "C"` ABI (`include/dfine/c_api.h` +
  `src/c_api/c_api.cpp`, compiled into `libdfine.so` by default; `-DDFINE_BUILD_C_API=OFF` to skip), a
  dependency-light Python `ctypes` package (`python/dfine/`), and a zero-setup `dfine` CLI
  (predict/info/build/export/bench). The C-ABI **and** Python detections are **byte-identical to
  `dfine_detect`** — proven by `dfine_capi_parity` and 14 pytests. `-Werror`- and UBSan-clean; hardened by an
  adversarial multi-agent review (16 findings, all fixed incl. a negative-numpy-stride OOB). See "M4" below.
  **M3 instance-seg stays deliberately shelved** (no seg checkpoints / use-case).
- **Intensive core P1+P2+P3: DONE and validated** (P4 pending). P1 Zero-D2H GPU decode
  (`DetectorOptions.gpu_decode`), P2 arena + `freeze()` (zero steady-state allocation, VRAM Δ = +0 B),
  P3 full-pipeline CUDA graph (`DetectorOptions.full_pipeline_graph` + `freeze(FreezeSpec)`): **one
  `cudaGraphLaunch` per frame** covering H2D→preprocess→enqueueV3→GPU-decode→survivor-D2H. B=1 CPU/frame
  4.30 → 0.195 ms, e2e wall −34.3% on m FP16 0-aux; byte-parity vs the split path on 1061 real 640×480
  val2017 images; mAP configs identical (0.5660 subset-2000). See "Intensive core (P1–P3)" below and
  `impl/INTENSIVE_CORE_PLAN.md` for the full spec/validation record.
- The big M0/M2 discovery is the through-line: **D-FINE's FDR box-decode is exquisitely FP-precision-sensitive**
  (grid_sample, the kFP16 flag, BF16 and INT8 all fail through it) — see "Decisions" and the M2 section.

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
6. **Precision:** FP32 baseline (build `--no-tf32` for FP32-faithful; TF32 on costs ~2× box L1). **FP16 works
   only via strong typing** (backbone+encoder FP16 baked into ONNX types, decoder FP32) — the weakly-typed
   `kFP16` builder flag costs a fixed −6.8 AP regardless of pinning. BF16 and INT8 are worse (mantissa too
   coarse for the FDR). Validated: FP16 mAP −0.2% at 1.6–2.2×. See the M2 section.
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

## Repo entry points (added in the hardening pass)

Top-level OSS/production files now exist: **`README.md`** (public landing page + benchmark tables),
**`docs/ROADMAP.md`** (prioritized 4-tier plan for M3+), `CONTRIBUTING.md` (the invariants + build/validate
flow), `LICENSE`/`NOTICE` (Apache-2.0), `build.sh` (one-command build wrapping the toolchain gotchas),
`pyproject.toml` (uv env for the scripts), `Dockerfile`/`.dockerignore`, `.clang-format`/`.editorconfig`,
`.pre-commit-config.yaml`, `.github/` (CI: GPU-less lint/format + a documented GPU-runner stub; issue/PR
templates), `docs/README.md` (navigation map of the research notes), `examples/`. This HANDOFF remains the
single source of truth for *state*; README is the front door; ROADMAP is what's next.

## Doc map (reading order)

| Doc | Role |
|---|---|
| **`HANDOFF.md`** (this) | Current state + how to continue. **Start here.** |
| `README.md` (root) | Public landing page — benchmarks, quickstart, precision guide. |
| `docs/ROADMAP.md` | Prioritized roadmap for M3+ (segmentation, C ABI/bindings, WASM demo, serving, FP8, …). |
| `docs/README.md` | Navigation map of the 90 research notes + reading order. |
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

## M2 — production hardening (DONE for detection: FP16 + CUDA-graph; INT8 rejected)

| Capability | Status | Notes |
|---|---|---|
| Runtime with no Python | ✅ | `libdfine.so`, stb for image decode |
| CUDA preprocess (fused resize+`/255`) | ✅ | ~0.17 ms |
| Async pinned H2D / grow-only buffers | ✅ | event-guarded pinned reuse |
| Dynamic batch (N=1..8) | ✅ | validated |
| FP32 accuracy == reference | ✅ | 0.5506 == 0.5507 |
| Full RAII / sanitizer-clean / warning-clean | ✅ | see Quality bar |
| Profiling (latency/mem/mAP, cross-backend) | ✅ | `dfine_bench` + `profile.py` |
| **FP16 (strongly-typed, backbone+encoder)** | ✅ M2.1 | **1.6–2.2× infer, mAP −0.2%**; the `kFP16` *flag* is the trap, not FP16 |
| **CUDA-graph replay (opt-in)** | ✅ M2.2 | byte-identical; **−34.5% batch-1** on a `--max-aux-streams 0` engine (D-FINE is dispatch-bound); default 2-aux engines can't capture (gated) |
| **INT8 (QDQ)** | ⛔ rejected | builds, but mAP collapses to ~0.13 — D-FINE's FDR needs ≥FP16 precision |
| **Instance segmentation** | ⛔ M3 | D-FINE-seg mask head, threshold 0.5 (masks pre-sigmoid'd) |

Numbers below are RTX 4070 Ti SUPER, COCO subset-2000, m variant, FP32 baseline **0.5669** (trt) /
0.5666 (cpp). GPU clocks not lockable and the box was sometimes shared — mAP is deterministic (trustworthy);
latency was measured **back-to-back per pair** so the *relative* speedups hold.

### M2.1 — FP16, DONE via **strong typing** (the `kFP16` flag is the real trap)

**★ The headline finding: the weakly-typed `config.set_flag(kFP16)` path is unusable for D-FINE.** It costs a
**fixed ~6.8 AP** (0.5669 → 0.4985) **regardless of per-layer FP32 pinning** — proven by a control that pinned
*every* `/model` compute layer FP32 under `kOBEY_PRECISION_CONSTRAINTS` with `kFP16` on and *still* got 0.4985.
TRT inserts uncontrolled FP16 reformats on the FDR's precision-critical data path that OBEY/`setPrecision`
don't cover. Same failure *class* as the M0 grid_sample trap (the FDR integral amplifies tiny FP deltas).
Every weakly-typed placement gives the same 0.498 (decoder-only / encoder+decoder / backbone-only / even
stem-only FP32-pinned), i.e. one FP16 reformat anywhere → the fixed loss.

**The fix: bake precision into ONNX types, build strongly-typed, NO `kFP16` flag.**
1. `convert_fp16.py` — `onnxconverter_common` casts backbone+encoder to FP16, block-lists the whole decoder
   (kept FP32, found by ONNX name prefix `/model/decoder`,`model.decoder`; all compute is cleanly scoped, the
   OTHER-region nodes are shape/constant only), and **retypes graph outputs back to FP32** (the converter
   otherwise leaves a trailing output→FP16 downcast).
2. `build_engine.py --strongly-typed --no-tf32 --onnx <fp16 onnx>` — precision is 100% from the ONNX types;
   TRT cannot leak FP16. (`convert_fp16.py` also runs a `harmonize_float_types` pass: onnxconverter_common
   leaves size-dependent mixed Half/Float nodes strongly-typed TRT rejects — a stray FP32 attention-scale
   constant in the FP16 encoder, a missing FP16→FP32 boundary cast into the decoder — so it duplicates the
   shared scale per FP16/FP32 consumer and inserts boundary casts. Without it, s/l/x/n fail to parse.)

**Validated on FULL COCO val2017 (5000 imgs), all sizes — FP16 is essentially lossless and the speedup scales
with model size** (C++ detector; latency e2e p50 ms and infer-only p50 ms on the RTX 4070 Ti SUPER):

| size | FP32 AP | FP16 AP | ΔAP | infer b1 F32→F16 | infer b8 F32→F16 | e2e b8 F32→F16 |
|---|---|---|---|---|---|---|
| nano   | 0.4280 | 0.4280 | +0.0000 | 2.36→1.78 (1.32×) | 8.63→4.48 (1.93×)  | 10.76→6.60 (1.63×) |
| small  | 0.5074 | 0.5069 | −0.0005 | 3.77→2.60 (1.45×) | 20.40→10.23 (1.99×)| 22.57→12.45 (1.81×) |
| medium | 0.5506 | 0.5500 | −0.0006 | 5.42→3.27 (1.66×) | 32.95→15.07 (2.19×)| 35.20→17.28 (2.04×) |
| large  | 0.5725 | 0.5723 | −0.0002 | 6.92→4.15 (1.67×) | 44.85→20.50 (2.19×)| 47.16→23.18 (2.03×) |
| xlarge | 0.5931 | 0.5927 | −0.0004 | 11.96→5.40 (2.21×)| 85.76→30.50 (2.81×)| 87.90→32.59 (2.70×) |

Worst-case −0.0006 AP (0.1%); the FP32 column reproduces the canonical D-FINE obj2coco numbers (m = 0.5506 =
the M0/M1 full-val figure) so the whole export→build→eval pipeline is validated at full-val on all sizes. GPU
mem (m) 674→**520 MiB**. Engine I/O stays FP32 → **no C++ change**, and CUDA-graph works on it. Build:
```sh
$PY $S/convert_fp16.py --output trt-files/onnx/dfine_m_fp16_st.onnx
$PY $S/build_engine.py --strongly-typed --no-tf32 --max-batch 8 --cuda-graph \
    --onnx trt-files/onnx/dfine_m_fp16_st.onnx --output trt-files/engines/dfine_m_fp16_st.engine
$PY $S/profile.py --backends cpp --engine trt-files/engines/dfine_m_fp16_st.engine --subset 2000 --no-latency
```
**Anti-examples (quantified):** weakly-typed `--fp16` (or `--fp16-decoder-fp32`) = 0.4985; `--bf16-decoder-fp32`
= **0.2968** — *worse* than FP16, because D-FINE's FDR needs mantissa precision and BF16 trades 2 mantissa bits
for range it doesn't use (`dump_activations.py`: activations peak ~5.4e3, nowhere near FP16's 65504 — **no
overflow**; the loss is pure mantissa). `dfine_build` (C++) stays FP32-only on purpose.

### M2.2 — CUDA-graph replay, DONE (task layer, opt-in)

`detector.cpp` captures `enqueueV3` + the two output D2H copies (into detector-owned pinned buffers) after
≥2 warm-up enqueues (`setEnqueueEmitsProfile(false)`, `cudaStreamCaptureModeThreadLocal`), and
`cudaGraphLaunch`es each call; preprocess/H2D stay outside. One graph per batch size (`graphs_` map), only
replayed when the shape is already flushed (`graph_ctx_batch_ == B`), re-captured if a grow-only realloc moves
a baked pointer (`graph_stale_`, 5-pointer check), and **no-throw fallback to `enqueueV3`** if capture fails /
outputs aren't FP32 / the engine uses aux streams. `DetectorOptions.use_cuda_graph` + `cuda_graph_compat`
sidecar flag; `CudaGraph`/`CudaGraphExec` RAII in `cuda_raii.hpp`; `--cuda-graph` on `dfine_bench`/
`dfine_detect`/`dfine_coco_eval`. Our RAW single-input/FP32-output export sidesteps rf-detr's two graph hazards
(int64 `labels`, `orig_target_sizes`) — see `research/P12_cuda_graph.md`.

**★ The graph requires a single-stream (0-aux) engine — this is the key gotcha.** TRT builds these D-FINE
engines with **2 auxiliary streams** by default (`getNbAuxStreams()==2`), and `cudaStreamCaptureModeThreadLocal`
records only the main stream → an *incomplete* graph that silently drops the aux-stream kernels (fast but
wrong). The detector/bench correctly **gate capture on `num_aux_streams()==0`**, so on a default engine the
graph is a safe no-op (falls back to enqueueV3). To actually use it, **build with
`build_engine.py --max-aux-streams 0`** (`config.max_aux_streams=0`), which makes the engine single-stream and
capturable. (An earlier "−1.3 ms / 2.36×" reading was a *pre-gate incomplete capture* running fewer kernels —
not a real speedup.)
- **mAP unchanged:** on a 0-aux engine, graph vs no-graph detections are **byte-identical** (`dfine_coco_eval`
  diff) — the graph is correct.
- **Latency (0-aux FP16 m, rigorous same-run `dfine_bench --graph-compare`):** D-FINE is **dispatch-bound at
  small batch** — enqueueV3 spends **3.87 ms of CPU** launching the hundreds of kernels vs **0.09 ms** for
  `cudaGraphLaunch`, and the GPU *starves* waiting (3.90 ms wall for 2.55 ms of real compute). The graph
  removes the starvation: **batch-1 full wall 3.90 → 2.55 ms = −34.5%**; batch-8 −4.9% (GPU-bound there).
- **Recommendation:** for **fixed-shape batch-1 streaming**, build `--max-aux-streams 0` + `use_cuda_graph` —
  **2.55 ms beats the default 2-aux engine's 3.30 ms** (the graph more than repays the lost stream
  parallelism). For **batch throughput**, keep the default 2-aux engine (parallelism wins, graph gives little).
  Measure with `dfine_bench --graph-compare` (needs a 0-aux engine) or `profile.py --backends cpp cpp-graph`.

### M2.3 — INT8 (QDQ), investigated, **rejected** (mAP 0.13)

`convert_int8.py` (ORT `quantize_static`, QDQ, **symmetric** + **no bias-quant** — both TRT requirements —
Conv/MatMul, decoder excluded) + `build_engine.py --int8`. Builds cleanly and is fast, but **mAP collapses to
0.1274 (weakly-typed) / 0.1314 (strongly-typed)** vs 0.5666 — a −0.44 AP loss either way. INT8's 8-bit
precision on the backbone/encoder features is far below what D-FINE's FDR tolerates (FP16's 10-bit mantissa is
already the floor). **Not recommended for D-FINE-M.** Script kept for future variants / less FP-sensitive heads.

### M2.4 — instance segmentation (M3, optional)

D-FINE-seg mask head → extra `masks` output; add a GPU bilinear-upsample+threshold decode (model on rf-detr
`mask_decode.cu`), threshold at 0.5 (masks are pre-sigmoid'd, so compare >0.5 not logit>0), populate a mask
field on `Detection`.

**M2 new files:** `convert_fp16.py` (strongly-typed FP16 ONNX), `convert_int8.py` (INT8 QDQ), `build_engine.py`
(`--fp16-decoder-fp32`/`--bf16-decoder-fp32`/`--int8`/`--strongly-typed`/`--decoder-prefixes`/`--constraints`/
`--cuda-graph`). `profile.py` gained a fix for its own filename shadowing stdlib `profile` (crashed the torch
backend via `cProfile`). C++: `CudaGraph`/`CudaGraphExec` RAII, `TrtSession::num_aux_streams()`,
`DetectorOptions.use_cuda_graph`, `EngineMeta.cuda_graph_compat`.

**Quality bar (M2):** adversarial multi-agent review (correctness / cpp-pro / CUDA-TRT dimensions, each finding
independently verified) — 1 real defect found and fixed (capture path now no-throw so the enqueueV3 fallback
actually runs). Builds `-Wall -Wextra -Wpedantic` clean; graph path re-verified byte-identical post-fix.

**Known latent (harmless today, non-happy-path):** `max_batch()` returns 0 when a dynamic engine's sidecar is
absent (TRT then bounds the batch itself); moved-from `DFineDetector` accessors are UB by convention. Neither
is on any exercised path (review confirmed both as non-triggering).

**Where to start (M3):** read this file, then `impl/M0_STATUS.md` (FDR FP-sensitivity — the through-line behind
grid_sample, kFP16-flag, BF16 and INT8 all failing the same way). The FP16 engine + CUDA-graph are the
production speed path; seg (M2.4) is the next feature.

**Cross-backend reference (D-FINE-M, full COCO val, `profile.py --backends torch onnx trt cpp cpp-graph`):**
FPS at batch 1 / batch 8 — PyTorch 31/66, ONNXRuntime-GPU 40/89, TensorRT-FP32(py) 125/160, C++ FP32
176/227, **C++ FP16 272/459** (≈8.7×/7× PyTorch); e2e batch-1 latency 32.0 / 25.0 / 8.0 / 5.7 / **3.7** ms;
GPU mem FP16 488 MiB vs FP32 642 (−24%); all backends mAP 0.5500–0.5509. (These engines are the default 2-aux
build, so `cpp-graph` there == `cpp`; the graph win needs a `--max-aux-streams 0` engine — see M2.2.)

## M4 — bindings & DX (DONE)

Make the fast engine callable from any language and runnable in one command. Detection-only (M3 seg shelved).

### C ABI — `include/dfine/c_api.h` + `src/c_api/c_api.cpp`
Modeled near-verbatim on `rf-detr-cpp`'s `c_api.h`: opaque `dfine_detector_t*`, `DFINE_API` visibility macro,
thread-local `dfine_last_error()` (returns `""`, never NULL), heap `dfine_detections_t {dfine_detection_t*
detections; int count;}` freed by `dfine_detections_free()`, every entry point wraps try/catch → last_error +
NULL/0 (no exception crosses the boundary). Compiled into `libdfine.so` (option `DFINE_BUILD_C_API`, default
ON; adds `DFINE_BUILDING_LIB`). **D-FINE deltas from the rf-detr template (deliberate):** (a) `class_id` is the
**dense COCO-80** index 0..79 — D-FINE has **no background slot**, so no "+1 for COCO-91"; `dfine_class_name()`
maps 0..79. (b) core is OpenCV-free & channel-agnostic, so `dfine_detector_detect(det, data, w, h, step,
channels, is_bgr, thr)` exposes `channels`+`is_bgr` (builds an `ImageU8` internally, no cv::Mat). (c) the log
callback is **severity-aware** (`void(*)(int severity, const char*)`, 0=FATAL..4=VERBOSE). **Extras beyond the
template:** `dfine_detector_create_ex(engine, meta, dfine_options_t{threshold, use_cuda_graph,
graph_warmup_iters})` surfaces the flagship CUDA-graph path; `dfine_detector_detect_batch` +
`dfine_detections_free_batch` surface batch throughput; introspection `variant/input_w/h/num_queries/
num_classes/max_batch`. Threshold: per-call `<0` ⇒ engine default; options `threshold<=0` keeps 0.5 (zero-init
safe). **Validation:** `apps/dfine_capi_parity.cpp` asserts C-ABI detections **byte-identical** to
`DFineDetector::detect` (20/44/231 dets, FP32+FP16, single+batch); `apps/dfine_capi_smoke.c` (pure C) proves
the header is valid C + exception-safe. Both `-Werror` (`-Wall -Wextra -Wpedantic`) and UBSan-clean.

### Python — `python/dfine/` (ctypes, no compile step)
`Detector(engine_path, meta=None, *, threshold=0.5, use_cuda_graph=False, is_bgr=False, class_names=None)`
hides all ctypes behind `_ffi.py` (private `_Box/_Detection/_Detections/_Image/_Options` Structures,
argtypes/restypes pinned once, best-effort TensorRT preload, lib discovery via `$DFINE_LIBRARY` → bundled →
`build/libdfine.so`). `.detect(np_hwc_uint8, threshold=None) -> list[Detection]` (each `class_id/score/box/
class_name`, `.as_dict()`); `.detect_batch([...])`; the C result is freed **every call** (try/finally) and
`__del__`/`__enter__`/`__exit__` destroy the engine. Memory-safe: numpy buffers kept alive across the C call;
zero-copy only when rows are packed **and** `strides[0] >= w*3` (rejects `np.flipud`/`[::-1]` negative-stride
views — the one real bug the review caught). `pyproject.toml` declares `tensorrt` as an **extra** (not
redistributable). **14 pytests** pass incl. a parity test that is byte-exact vs the C++ `dfine_detect` binary
(via a lossless-PNG round-trip so stb and PIL decode identical pixels) + negative-stride and threshold=0.0
regressions. Run: `cd python && PYTHONPATH=. LD_LIBRARY_PATH=<tensorrt_libs>:<conda>/lib pytest tests`.

### CLI — `dfine` console entry point (`python/dfine/cli.py`)
`dfine predict|info|build|export|bench`. Resolves an engine from `--engine` → cache
(`~/.cache/dfine/dfine_{model}_{prec}-sm{arch}-trt{ver}.engine`, arch/TRT-versioned since engines aren't
portable) → dev-tree `trt-files/engines/` → **builds on demand** (ONNX→engine via `build_engine.py`). `predict`
decodes+draws with PIL (`--out`, `--json`); `build` wraps `build_engine.py`; `export` wraps
`export_dfine_onnx.py` (needs D-FINE-seg src) and, for `--precision fp16`, chains `convert_fp16.py` so the
output name matches what `build`/`predict` look up; `bench` shells to the C++ `dfine_bench`. Validated: `info`/
`predict` on existing engines, and the full **build→cache→resolve** loop (built nano fp32 into the cache, then
`info --model n` resolved the cached engine). *No HF auto-download helper exists in this repo* — `export`
requires the D-FINE-seg checkpoints (mapped per model) or an explicit `--checkpoint`.

## Intensive core (P1–P3) — device-resident frozen pipeline (DONE; P4 pending)

Execution spec + full validation record: `impl/INTENSIVE_CORE_PLAN.md`. Summary of what shipped:

- **P1 — Zero-D2H GPU decode** (`DetectorOptions.gpu_decode`, FP32-output engines): sigmoid→top-k→
  threshold→xyxy as CUDA kernels (`decode_gpu.cu`: k_pack → CUB segmented radix sort by raw logit →
  k_decode_topk); D2H shrinks to the compact survivors. mAP == CPU decode; ~1 ULP score delta (GPU expf),
  ranking bit-identical.
- **P2 — arena + freeze** (`DetectorOptions.own_device_memory`, `DFineDetector::freeze`): one `DeviceArena`
  block for the 9 decode slabs, TRT activation in a detector-owned kUSER_MANAGED block, grow-guards that
  throw post-freeze. Zero steady-state allocation (VRAM Δ = +0 B over full runs).
- **P3 — full-pipeline graph** (`DetectorOptions.full_pipeline_graph`, implies gpu_decode): capture inside
  `freeze(FreezeSpec{batch, src_w, src_h, src_is_bgr})`, pre-lock. Steady state = pack frames into a pinned
  slab + **one `cudaGraphLaunch`** + sync. Gates: 0-aux engine (`build_engine.py --max-aux-streams 0`) +
  FP32 outputs + exact config match; otherwise no-throw fallback to the split path. Threshold stays a live
  per-call knob (mapped-pinned scalar read at kernel execution). `freeze(int)` = legacy unbounded-source
  behavior (staging may still grow for oversized frames — the one documented zero-alloc exception);
  explicit `src_w/src_h` locks the preprocessor too; re-freeze with a different config throws.
- **Timings** now splits host cost per stage: `preprocess_cpu_ms / dispatch_ms / wait_ms / decode_host_ms` —
  the dispatch column is what the graph collapses (m, B=1: 4.18 → 0.12 ms; B=8: 18.7 → 0.18 ms).
- **Measure**: `dfine_bench --pipeline-compare` (per-stage CPU table, in-run byte parity, live-threshold
  probe); `dfine_coco_eval --full-graph --filter-res WxH` (fixed-resolution regime over real COCO);
  `overnight_bench.sh` (config matrix + parity per size; `NSYS=1` for an Nsight trace).

### Not done (next): pre-compiled wheels + GitHub Actions, and the visual README gif.
No git remote here, so "GitHub release artifact" wheel hosting is deferred. A true PyPI manylinux wheel is
infeasible (TensorRT non-redistributable, `.so` is CUDA/arch-specific); the realistic MVP is a Linux wheel that
bundles `libdfine.so` (via `package-data`, already declared) and depends on the user's local TRT 10.13 + CUDA
12. The side-by-side "PyTorch ~31 FPS vs D-FINE-cpp ~272 FPS" gif still needs a manual screen-record.

**M4 new files:** `include/dfine/c_api.h`, `src/c_api/c_api.cpp`, `apps/dfine_capi_smoke.c`,
`apps/dfine_capi_parity.cpp`; `python/{pyproject.toml,README.md}`, `python/dfine/{__init__,_ffi,detector,cli}.py`,
`python/tests/{conftest,test_detector}.py`. CMake gained `DFINE_BUILD_C_API` + the two harness targets;
`build.sh` now also sets `CMAKE_C_FLAGS=-B/usr/bin` (system binutils for the C target).
