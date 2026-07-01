# M0 status — export → build → smoke → parity (variant m, dynamic batch)

**Done and verified on real artifacts** (RTX 4070 Ti SUPER, CUDA 12.8, TensorRT 10.13.3, torch 2.9.1).
Run from the D-FINE-seg venv; checkpoint `dfine_m_obj2coco.pt` (clean load: 0 missed, 0 unmatched).

## Pipeline (trt-files/scripts/)

| Step | Script | Artifact | Result |
|---|---|---|---|
| Export | `export_dfine_onnx.py` | `trt-files/onnx/dfine_m.onnx` + `.json` sidecar | raw `images`→`logits,boxes`, opset 16, dynamic batch |
| Build | `build_engine.py` | `trt-files/engines/dfine_m_fp32*.engine` + `.json` | TRT 10.13, dynamic profile min1/opt1/max8 |
| Smoke | `verify_engine.py` | — | binds + runs at N=1, 2, 8 |
| Parity | `parity_check.py` | — | PyTorch vs TRT/ONNX on surviving top-K |

Reproduce:
```sh
V=/home/dxdxxd/projects/custom-dfine/D-FINE-seg/.venv/bin/python
cd /home/dxdxxd/projects/custom-dfine/D-FINE-seg
CK=pretrained/dfine_m_obj2coco.pt
$V ../D-FINE-cpp/trt-files/scripts/export_dfine_onnx.py --model-name m --checkpoint $CK
$V ../D-FINE-cpp/trt-files/scripts/build_engine.py --no-tf32 --max-batch 8
$V ../D-FINE-cpp/trt-files/scripts/verify_engine.py --batches 1 2 8
$V ../D-FINE-cpp/trt-files/scripts/parity_check.py
```

## Confirmed on real artifacts (matches the plan)

- **Native deformable attention, zero plugins.** The raw ONNX has **12 GridSample nodes** (3 levels × 4 deploy layers), all **rank-4** (no 5D), and **no non-standard-domain ops**; the engine builds with the stock `OnnxParser` + `build_serialized_network` and no plugin registration. (Confirms V02/V06/P01.)
- **`model.deploy()` truncates to eval layers.** m's decoder folds to `eval_idx+1 = 4` layers; `num_decoder_layers=4`, not 6. Sidecar records it.
- **Per-size invariants hold.** `num_queries=300`, `reg_max=32`, `reg_scale=4.0`, `num_levels=3`, `hidden_dim=256`, `feat_strides=[8,16,32]` — read off the model, match V09.
- **Detection checkpoint is drop-in.** `dfine_m_obj2coco.pt` loads with 0 missed / 0 unmatched into the detection model (V07).
- **Export is numerically faithful.** torch ≈ ONNXRuntime; the strong top-5 detections match to ~4 decimals.

## New findings (not in the original plan — fold into §16/§18)

1. **Batch-1 trace bakes the batch axis → engine rejects N>1.** Tracing `torch.onnx.export` with a batch-1 dummy let the tracer constant-fold the query-selection anchor batch (anchors are `[1, ΣHW, 4]`; 1 is the broadcast identity) into a literal `1`, so a decoder `GatherElements` had `data extent 1 vs indices extent N` and TRT raised *"Profile kMAX values are not self-consistent"* at build and a shape error at N≥2.
   **Fix:** trace with a dummy batch **≥ 2** (`--trace-batch 2`, the default). After this the kMAX warning disappears and the engine runs at N=1/2/8. **This is a hard requirement for any dynamic-batch D-FINE export.**

2. **Full-tensor output cosine is a misleading parity metric.** Over all 300 queries it sits at ~0.96–0.99 even for torch-vs-ONNXRuntime, because ~290 background queries carry noise boxes. Compare the **surviving top-K** (sigmoid → top-k over query×class), as D-FINE-seg's own parity harness does. `parity_check.py` now does this.

3. **TF32 is on by default and measurably hurts box accuracy.** On surviving detections, disabling TF32 (`--no-tf32`) roughly **halves box L1 (0.0224 → 0.0108 normalized)** and lifts score cosine (0.924 → 0.942). This empirically supports keeping the decoder FP32-faithful (plan §10). Use `--no-tf32` for the accuracy/parity engine; a TF32-on build is faster but must be mAP-validated.

4. **Residual TRT-vs-PyTorch gap on detections** (score cos ≈ 0.94, box L1 ≈ 0.011 even with TF32 off) is larger than torch-vs-ONNXRuntime (0.988 / 0.009). Consistent with the P01 H5 open question (TRT GridSample FILL vs PyTorch `padding_mode=zeros` at sub-pixel borders, compounded over 12 samplers). **Not a blocker for M0; quantify via COCO mAP once the M1 C++ decode exists** before trusting FP16/production.

## Parity numbers (park_gen.jpg, top-30 surviving)

| Compare | score cosine | max\|Δscore\| | box L1 (norm cxcywh) | rank overlap |
|---|---|---|---|---|
| torch ~ ONNXRuntime | 0.988 | 0.43 | 0.0094 | 27/30 |
| torch ~ TRT (TF32 off) | 0.942 | 0.91 | 0.0108 | 26/30 |
| torch ~ TRT (TF32 on)  | 0.924 | 0.91 | 0.0224 | 23/30 |

(The top-K includes weak/background queries that are inherently unstable across *any* backend — the strong top-5 agree to ~4 decimals. The right production gate is COCO mAP, not this cosine.)

## CRITICAL FINDING — TensorRT GridSample costs ~10 AP on D-FINE (M1 blocker)

COCO val2017 mAP, variant m, same stretch preprocessing + decode for every backend:

| Backend | AP@[.50:.95] | AP@.50 | notes |
|---|---|---|---|
| PyTorch (reference) | **0.5509** | 0.7258 | matches the D-FINE-M paper |
| ONNXRuntime (CPU, same ONNX) | **0.6471*** | 0.8274 | *100-img subset; on the same 100, torch=0.6472 → ORT reproduces PyTorch to 4 decimals |
| **TensorRT FP32 (TF32 off)** | **0.4455** | 0.6155 | **−10.5 AP vs PyTorch** |

**The exported ONNX is perfect** (ORT == PyTorch). The entire loss is **TensorRT's execution of the GridSample-based deformable attention** — the P01 H5 open question, now quantified at a large, production-unacceptable cost.

Diagnosis (`parity_check.py`, per-image torch-vs-engine):
- **Scores stay faithful, boxes drift** — sorted-topK score cosine 0.982–0.997 (would *pass* D-FINE-seg's ≥0.99 `run_parity` gate), but top-20 box L1 ranges 0.005 → **0.14 normalized** (≈90 px at 640) image-to-image. Drift is image-dependent and concentrated in **foreground** queries (full-tensor stats are background-dominated and show TRT≈ORT, hiding it).

**GridSample is NOT the culprit (isolation proof).** A single-node GridSample ONNX (D-FINE's exact 4D / bilinear / zeros / align_corners=0), torch vs TRT on grids spanning borders + out-of-bounds:
`torch-vs-TRT max_abs = 2.4e-07` (in-bounds 2.4e-7, out-of-bounds/FILL 6e-8) — **bit-exact, even better than ORT (1.6e-5)**. So a GridSample/deformable plugin would fix nothing. The drift is downstream in the `bbox_head → FDR Integral → distance2bbox` path (the Integral amplifies tiny upstream FP differences into box shifts; scores via the linear head don't amplify).

**Every standard TRT control fails to close it** (COCO 100-subset, torch=0.647 / ORT=0.647):
| Build | AP |
|---|---|
| weakly-typed, TF32 on | 0.5475 (full-val 0.4455) |
| weakly-typed, **TF32 off** | 0.5475 |
| **+ PREFER_PRECISION_CONSTRAINTS + opt-level 5** | 0.5475 |
| **strongly-typed** (FP32 pinned) | 0.5559 |
| max-batch 1 (static-ish) | 0.4492 (worse) |
→ none recover the ~9–10 AP. The gap is a robust TRT decoder-execution divergence, not a precision-flag or tactic issue.

**Why the two D-FINE repos look "fine" on TRT but aren't validated:** `D-FINE-seg/src/dl/export.py:472 run_parity` checks **sorted top-K SCORES only** (gate cos≥0.99) and explicitly skips boxes as "noise" — a 10-AP box drop passes silently. Their `export_to_tensorrt` uses the same plain builder config. Only `make bench` (per-backend mAP) would surface it; evidence says **they have the same gap, unmeasured**.

### Localized to the op (forward-hook intermediates, TRT vs ORT on surviving queries, image 285)

Per **decoder layer** (hidden state):
| after layer | TRT-vs-ORT mean_abs | |
|---|---|---|
| L0 | 1.9e-06 | **exact** |
| L1 | 1.9e-01 | **jump** |
| L2 / L3 | ~0.19 | stays |

The FDR box path that feeds layer-1 reference points is **bit-exact**: `bbox_head[0]` out 9e-5, Integral out 1.3e-6. So FDR/Integral is **not** the culprit.

Per **layer-1 sub-op** (forward order):
| sub-op | TRT-vs-ORT mean_abs | |
|---|---|---|
| self-attn | 2.5e-05 | exact |
| norm1 | 8e-07 | exact |
| **deformable cross-attn** | **7.0e-02** | **divergence enters here** |
| Gate fusion | 2.5e-01 | amplifies |

**Final root cause:** the **multi-scale deformable cross-attention** diverges — but **not** its GridSample (bit-exact in isolation, and layer-0's identical cross-attn is exact). Same op + **exact inputs** (query, reference points, value all verified exact) computes exact in layer 0 but diverges in layers 1–3 → **TensorRT selects a different, less-accurate kernel for the deformable-attn matmuls / attention-weighted-sum at those layer positions.** D-FINE's FDR then bakes the layer-1 error into the accumulated box → ~10 AP.

**Every fix exhausted:** export mode (opset16 legacy / opset19 dynamo), TF32-off, PREFER_PRECISION_CONSTRAINTS, builder_optimization_level=5, strongly-typed, `set_tactic_sources` (cublas / cublas+edge+jit), min/opt/max batch — all give ~0.5475 (none recover).

### ✅ RESOLVED — explicit gather-bilinear deformable core (plugin-free)

Validation spike (`trt-files/scripts/spike_explicit_deform.py`): monkeypatch every `cross_attn.ms_deformable_attn_core` to an **explicit gather-bilinear** implementation of `grid_sample(bilinear, zeros, align_corners=False)` — same math, expressed as `Gather` + arithmetic instead of a `GridSample` node — then re-export (opset 16, GridSample=0, Gather=227). Torch parity of the rewrite: logits 2.3e-4, boxes 3.6e-6.

| Engine | COCO-100 AP | COCO **full-val** AP | latency N=1 |
|---|---|---|---|
| grid_sample (original) | 0.5475 | **0.4455** | 5.04 ms |
| **explicit gather-bilinear** | **0.6471** | **0.5507** | **4.92 ms** |
| torch / ORT reference | 0.6472 / 0.6471 | 0.5509 | — |

**The fix fully closes the gap** (full-val 0.4455 → **0.5507**, matching PyTorch 0.5509 to 0.0002) **at no latency cost** (4.92 vs 5.04 ms — TRT optimizes the Gather graph fine). Root cause confirmed: TRT compiles the **grid_sample-based deformable core divergently in-context** (GridSample is exact only in isolation); replacing it with `Gather` ops forces exact FP32.

**Validated across ALL sizes** (2000-img subset, ORT-GPU exact reference vs explicit TRT engine):

| size | layers/levels/hidden | ORT-GPU AP | TRT engine AP |
|---|---|---|---|
| n | 3 / 2 / 128 | 0.4440 | 0.4441 |
| s | 3 / 3 / 256 | 0.5235 | 0.5235 |
| l | 6 / 3 / 256 | 0.5918 | 0.5918 |
| x | 6 / 3 / 256 | 0.6067 | 0.6066 |

Engine == reference to ≤0.0001 AP for every variant, incl. nano's 2-level path → the fix is backbone/layer/reg_scale-agnostic. Engines built: `dfine_{n,s,l,x}.engine` + `dfine_m_fp32.engine`. (ORT-GPU enabled via `cuda_env.py` bootstrap + `onnxruntime-gpu==1.24.4` (CUDA-12 wheel) + `socksio`.)

**Plan impact (good):** the fix is **entirely in the Python export** — no TensorRT plugin, no C++ runtime change. The "zero-plugin C++ runtime" story (V02/V06) **holds**. `export_dfine_onnx.py` must apply the explicit-core patch (fold in from the spike) for every D-FINE engine; the C++ detector/preprocess/postprocess are unaffected. The deformable-attention-plugin direction is **withdrawn** — not needed.

## C++ build toolchain — VERIFIED (no sudo)
- cmake `/home/dxdxxd/miniconda3/envs/dfine/bin/cmake` (4.3.1).
- `TENSORRT_DIR=/home/dxdxxd/projects/custom-dfine/D-FINE-cpp/third_party/tensorrt` (10.13 public headers + symlinked libs); compile+link of `NvInfer.h`+`NvOnnxParser.h` proven (`getInferLibVersion()=101303`).
- Runtime `LD_LIBRARY_PATH=<seg .venv>/lib/python3.11/site-packages/tensorrt_libs`. M0 apps need no OpenCV.
- cmake: `-DTENSORRT_DIR=$TP -DCUDAToolkit_ROOT=/home/dxdxxd/miniconda3 -DCMAKE_CXX_FLAGS=-B/usr/bin`.

## Open for M1+

- COCO val mAP: PyTorch reference vs TRT FP32(no-TF32) — quantify the gap in (3)/(4).
- Then FP16 with the decoder kept FP32 (plan §10), re-measure mAP.
- Confirm the GridSample border-sampling delta is the dominant TRT-vs-torch source (isolate one sampler).
