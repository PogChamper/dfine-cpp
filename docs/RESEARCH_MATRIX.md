# Research matrix — v0.3.0 measurements

This document preserves the 2026-07-02/03 precision and throughput campaign that produced the
surgical-FP16 pipeline, export sliders, and v0.3.0 recipe. The current cross-domain preset study is
in the [v0.5 report](reports/v0.5.0-preset-evaluation.md).

**Common setup.** D-FINE-M `obj2coco` unless noted; RTX 4070 Ti SUPER (GeForce Ada, sm_89);
TensorRT 10.13.3; CUDA 12; COCO val2017. "Full-val" = mAP@[.50:.95] on all 5000 images
(`coco_eval.py --limit 0`); "subset" = the first 2000 (screening only, never a release gate).
All engines strongly-typed, `--no-tf32`. Benchmarks: `dfine_bench` (the C++ runtime), 500 iters,
medians of 3 interleaved rounds on an idle GPU. References (m): PyTorch subset base 0.5672,
full-val FP32 0.5506, production FP16-ST 0.5500. The PyTorch full-val reference is
session-dependent within 0.03 points: the bug-report era measured 0.5509, this campaign 0.5506,
and the v0.5 study 0.55067 with TF32 disabled — each document anchors its deltas to its own
measured base.

Reduced-query throughput rows report the measured `K=Q` decode configuration. Current published
results are in [Benchmarks](BENCHMARKS.md).

Engine configs referenced throughout:

| name | what it is | how to build it |
|---|---|---|
| **prod** | v0.2.0 production FP16-ST (backbone+encoder FP16, decoder FP32) | `convert_fp16.py` → `build_engine.py --strongly-typed --no-tf32` |
| **surgical** | whole net FP16 incl. decoder; FP32 = FDR scopes + glue + deform coord slice | opset-19 export → `convert_fp16_surgical.py` |
| **slim** | surgical with the glue tier FP16 too (FP32 = FDR scopes + coord slice) — **v0.3.0 default** | opset-19 export → `convert_fp16_surgical.py --slim` |
| **fast** | slim + `--num-queries 200 --cascade 1:100` | export with sliders → `--slim` convert |
| **max** | fast + `--eval-idx 2`, engine built `--opt-batch 8` | export with sliders → `--slim` convert → `--opt-batch 8` |

---

## 1. Where the time goes (per-layer profiling, fp16_st)

`ProfilingVerbosity.DETAILED` rebuild + `IProfiler`, fused-kernel time attributed fractionally to
ONNX origins via engine-inspector metadata. Shares are % of the profiled sum (b1 wall 3.46 ms,
profiled 5.16 ms; b8 wall 15.30 ms, profiled 16.41 ms).

| subsystem | b1 ms | b1 % | b8 ms | b8 % |
|---|---|---|---|---|
| decoder | 2.345 | 45.4% | 7.619 | 46.4% |
| backbone | 1.452 | 28.1% | 4.566 | 27.8% |
| encoder | 1.194 | 23.1% | 4.030 | 24.6% |
| reformats (all) | 0.052 | 1.0% | 0.127 | 0.8% |
| other | 0.118 | 2.3% | 0.065 | 0.4% |

Decoder breakdown (b8 ms): deform cross-attention 2.866 (17.5% — the single largest block),
self-attention 0.995, glue 0.881, FFN 0.475, bbox_head 0.421, enc_output 0.380, gateway 0.367,
norms 0.361, LQE 0.331, score_head 0.250, query_pos_head 0.236, FDR integral 0.100,
pre_bbox_head 0.069. The FDR tail (integral+LQE+bbox+pre_bbox) is only ≈5.6% of GPU time —
precision-critical, not time-critical.

**Deform kernel census.** The explicit gather-bilinear core is 4843 of 7631 ONNX nodes, but Myelin
fuses it to **16 runtime kernels** (4 large per-layer blobs carry ~88% of deform time). A 2×
fused-deform plugin would cap at ~3% (b1) / ~8% (b8) end to end. The measured ceiling did not
justify a plugin. Reformat/cast cleanup had a 0.7-1.0% ceiling.

## 2. Torch fake-quant ablation (what precision D-FINE actually needs)

Research harness (torch-side QDQ on weighted ops, per-subgraph dtype casts, 96-image calibration;
adversarially reviewed, AIFI in_proj escape fixed; not shipped — the shipped equivalents are
`convert_fp16.py` / `convert_fp16_surgical.py` / `convert_int8.py`). Subset-2000, base 0.5672:

| config | AP | Δ |
|---|---|---|
| decoder-FP32 wrap check | 0.5672 | ±0.0000 (methodology anchor) |
| backbone+encoder FP16 | 0.5671 | −0.0001 |
| backbone+encoder BF16 (strongly typed) | 0.5660 | −0.0012 |
| engineered FP16 decoder (FDR+coords FP32) | 0.5662 | −0.0010 |
| FP16 decoder, FDR-island v1 | 0.5665 | −0.0007 |
| FP16 decoder, minimal island {integral, LQE} | 0.5661 | −0.0011 |
| INT8 weights only (per-channel) | 0.5660 | −0.0012 |
| INT8 W+A, minmax activations | 0.2835 | −28 pts |
| INT8 W+A, robust-minmax activations | 0.5449 | −2.2 pts |
| FP8 W only (per-tensor) | 0.5514 | −1.6 pts |
| FP8 W only (per-channel) | 0.5537 | −1.35 pts |
| FP8 W+A, minmax | 0.3177 | −25 pts |
| FP8 W+A, percentile | 0.2053 | −36 pts (percentile makes FP8 *worse*) |

Results: an engineered FP16 decoder is ~lossless (→ §4); INT8 accuracy is an activation-
*calibration* problem, not a precision wall; FP8 loss is E4M3-mantissa-limited (scale-invariant
relative error — PTQ calibration does not address it); the earlier −27 AP BF16 result was
a weak-typing artifact. **F-1 hazard:** the explicit-deform gather *index* math (`y*w+x`, values
up to 6400) is FP16-unsafe (integers >2048 are inexact) — any reduced-precision deform must keep
index/coordinate math FP32. This shaped the converter's coordinate slice.

## 3. FP8 E4M3 (GeForce Ada)

Real TensorRT path: modelopt 0.44 QDQ (decoder excluded, 129 quantized nodes), opset-19,
strongly-typed build. Result: subset mAP **0.3909 (−17.6 pts)** and **7-9% slower** than fp16_st
(b1 3.99 vs 3.72 ms, b8 19.04 vs 17.48 ms). GeForce Ada runs FP8 tensor cores at FP16 rate (FP8
mandates FP32 accumulation, which GeForce halves), so you pay Q/DQ overhead for zero math gain.
Matches the torch ablation (§2): the accuracy loss is mantissa-limited. The path was rejected for
GeForce Ada.

## 4. Surgical FP16 — the bisect that found a TensorRT bug

Goal: FP16 for the ~46% of GPU time the decoder occupies, with an FP32 island for what §2 proved
precision-critical. `convert_fp16_surgical.py`: FP16 everywhere including the deform data path;
FP32 = FDR scopes (integral, LQE, bbox heads) + decoder glue leaves + the deform coordinate/index
slice (F-1). Implementation note: `onnxconverter_common` leaves *stale* value_info after
`node_block_list` conversion, so the converter re-derives true types topologically and fixes every
mixed-type input (the `harmonize` pass).

On **opset-16** exports the fine-grained config miscompiled — ONNXRuntime healthy, TensorRT broken
(the divergence is the bug signature):

| opset-16 variant | ORT (200-img) | TRT (subset-2000) |
|---|---|---|
| fine-grained island | 0.6148 healthy | 0.4354 broken |
| whole cross-attn FP32, rest FP16 | 0.6159 healthy | 0.0048 broken |
| coarse (only self_attn+FFN FP16) | 0.6097 healthy | 0.5662 **holds** (full-val 0.5498, −5.5% slower than fine) |
| bisect: coarse + gateway | — | 0.5662 passes |
| bisect: coarse + norms | 0.6152 healthy | 0.0050 **broken** ← root cause |
| same config, opset-19 (native LayerNormalization) | — | 0.5664 **passes** |

**Root cause: the opset-16 decomposed LayerNorm (ReduceMean/Sub/Pow/Sqrt/Div) in FP16 is
miscompiled by TRT 10.13's Myelin.** Opset ≥ 17 exports a native `LayerNormalization` node and the
same surgery compiles correctly (also shrinks the graph 5595 → 2226 nodes). This is why
`convert_fp16_surgical.py` **requires an opset-19 export** and hard-errors below it. A minimal
ORT-vs-TRT repro pair is archived; an NVIDIA bug report is in preparation. (Third member of the
family documented in [impl/M0_STATUS.md](impl/M0_STATUS.md): in-context GridSample and the kFP16
builder flag.)

### Surgical, gated on all five sizes

| size | surgical | fp16_st ref | verdict |
|---|---|---|---|
| n | 0.4276 | 0.4280 | lossless |
| s | 0.5065 | 0.5069 | lossless |
| m | 0.5502 | 0.5500 (fp32 0.5506) | lossless |
| l | 0.5724 | 0.5723 | lossless |
| x | 0.5929 | 0.5927 (fp32 0.5931) | lossless |

### m benchmark (medians of 3 rounds, 500 iters, idle GPU)

| engine | full-val | VRAM MiB | b1 p50/ips | b2 | b4 | b8 p50/ips |
|---|---|---|---|---|---|---|
| fp32 | 0.5506 | 674 | 5.61/178 | 9.38/213 | 17.73/226 | 34.85/230 |
| fp16_st (prod) | 0.5500 | 520 | 3.67/272 | 5.33/376 | 8.96/446 | 17.05/469 |
| fp16_st `--opt-batch 8` | = | 508 | 4.31/232 | 5.60/357 | 8.89/450 | 15.57/514 |
| surgical | 0.5502 | 456 | 3.60/278 | 4.92/406 | 8.05/497 | 15.16/528 |
| surgical `--opt-batch 8` | = | 426 | 3.90/256 | 5.13/390 | 7.75/516 | **14.26/561** |
| int8 (torch-calibrated, §6) | 0.5190 | 450 | 4.22/237 | 5.45/367 | 8.41/476 | 15.43/519 |

Opt-batch recipe: `--opt-batch 8` buys +6-10% b8 throughput and costs 8-15% of b1 throughput
(b1 p50 +8-19%) — build opt=8 for batch serving, opt=1 (default) for latency.

### Slim: drop the glue tier too

`--slim` leaves only the FDR scopes + coordinate slice FP32. Subset 0.5662 (lossless), **+2-3% b8
over full surgical**, and full-val **lossless on all five sizes**: n 0.4272, s 0.5060, m 0.5500
(exactly the prod reference), l 0.5723 (exact), x 0.5926. **This is the v0.3.0 production
default.**

### Full-pipeline CUDA graph on the surgical engine

The P1-P3 runtime machinery (GPU decode, `freeze()`, one-`cudaGraphLaunch`-per-frame) works
unchanged on a `--max-aux-streams 0` surgical build: byte-identical detections over 300 iters +
threshold probe, **b1 wall 2.474 ms (−27% vs split path; the prod-fp16_st graph was 2.55 ms)**,
CPU 0.164 ms/frame — this campaign's latency record. The current record is the v0.5 `max` graph
point, 1.99 ms ([Benchmarks](BENCHMARKS.md#runtime-modes-and-serving-profile)).

## 5. Export sliders (now `export_dfine_onnx.py` flags)

All three reshape the torch model before `deploy()`/tracing — the graph itself gets smaller.
Measured on the m surgical base (full-val where stated; b8 ips medians):

| slider | flag | mAP | b1 p50/ips | b8 ips¹ |
|---|---|---|---|---|
| (baseline: surgical) | — | 0.5502 full-val | 3.40/294 | 533 |
| queries 300→200 | `--num-queries 200` | **0.5487 full-val (−0.13)** | 3.44/291 | 563 |
| decoder 4→3 layers | `--eval-idx 2` | 0.5443 full-val (−0.57) | 3.29/304 | 546 |
| cascade prune | `--cascade 1:100` | 0.5456 full-val (−0.44) | — | ~530 |
| E2 + Q200 | `--eval-idx 2 --num-queries 200` | 0.5434 full-val (−0.66) | 3.04/330 | 608 (648 with opt8) |

¹ Same-run comparison block; run-to-run the surgical b8 median lands at 526-533 (the §7 full
ladder, a different session, has 526).

**Cascade** (`--cascade K:KEEP`) is the architecture-native slider: after decoder layer K, keep
only the top-KEEP queries ranked by layer K's *trained deep-supervision score head* (folded away
in a normal deploy), pruning every per-query tensor in-graph via TopK+Gather. Later layers
(self-attention is O(Q²)) and the decode run on KEEP queries. At the same speed it is *more
accurate* than dropping a layer (1:100 subset 0.5624 vs eval_idx=2's 0.5606) because the ranking
uses trained weights instead of removing computation.

Cascade curve (m, subset + b8 ips):

| cascade | subset AP | full-val | b8 ips |
|---|---|---|---|
| 1:75 | 0.5584 | — | 583 |
| 1:100 | 0.5624 | 0.5456 (−0.44) | ~530 |
| **1:150** | **0.5645** | **0.5482 (−0.18)** | 569 |
| 2:100 | 0.5644 | — | 563 |

This campaign selected 1:150 as its single-slider point: −0.18 AP full-val for +8% b8. The later
[v0.5 study](reports/v0.5.0-preset-evaluation.md) found a better measured balance for Q200 and is
the current selection reference.

## 6. INT8 — closed negative (standalone and combo)

The §2 sim said INT8 W8A8 could reach ≈−0.9 pt with SmoothQuant (α=0.7 → 0.5580) vs a pooled-
percentile plateau at ≈−1.9 pt; regional sensitivity (minmax worst-case): late backbone −12.4 pt,
FPN −9.0, early backbone −5.3, AIFI −0.4 — outliers are smeared across the CNN, so no region
exclusion saves you; scaling does. Real engines (torch-side calibration, scales injected into a
minmax QDQ graph — ORT histogram calibration OOMs a 16 GB host):

| engine | subset | full-val | b8 ips |
|---|---|---|---|
| int8 minmax | 0.1940 | — | — |
| int8 torch-percentile injection | 0.5361 | 0.5190 (−3.2) | 519 |
| int8 SmoothQuant α=0.7 | build failed¹ | — | — |
| int8 backbone/encoder + surgical FP16 decoder (combo, depthwise excluded²) | 0.5470 | — | 486 |

¹ TRT cannot fuse a per-input-channel `Mul(1/s)` into a per-output-channel weight-DQ scale
(`wtsOpUtils` broadcast assert). The fix design — folding 1/s into the *producer* conv through
ReLU commutation (`relu(x)·k == relu(x·k)`, k>0, valid for the whole HGNetv2 backbone) — is
documented but unimplemented; the measured speed ceiling did not justify the work.
² TRT 10.13 has no int8-depthwise kernels inside an FP16-typed context.

**Why closed:** the int8 conv gain is only ~1.23× real (Q/DQ overhead), so standalone INT8 (519
ips) loses to pure surgical FP16 (528/561), and the combo (486) loses harder while costing 2 pts
of mAP. On GeForce Ada, surgical FP16 strictly dominates every INT8 recipe we could build.
`convert_int8.py` stays in the tree as the reference implementation.

## 7. Size and configuration ladder

Medians of 3 interleaved rounds × 500 iters, idle GPU, `p50 ms / img/s`; VRAM = peak engine+buffers.

| size | config | b1 | b2 | b4 | b8 | VRAM MiB | full-val |
|---|---|---|---|---|---|---|---|
| n | prod | 2.21/452 | 2.76/726 | 3.74/1068 | 6.48/**1234** | 220 | 0.4280 |
| n | surgical | 2.15/465 | 2.70/740 | 3.70/1082 | 6.11/**1309** | 192 | 0.4276 |
| n | fast | 2.30/435 | 2.44/820 | 3.16/1266 | 5.14/**1556** | 192 | 0.4231 |
| s | prod | 3.00/333 | 4.11/486 | 6.64/603 | 12.56/**637** | 494 | 0.5069 |
| s | surgical | 2.67/375 | 3.67/545 | 5.76/694 | 10.55/**758** | 370 | 0.5065 |
| s | fast | 2.76/362 | 3.43/583 | 5.03/795 | 9.09/**880** | 364 | 0.5021 |
| m | prod | 3.56/281 | 5.36/373 | 9.00/444 | 17.04/**469** | 520 | 0.5500 |
| m | surgical | 3.47/288 | 4.92/406 | 8.11/493 | 15.20/**526** | 456 | 0.5502 |
| m | fast | 3.29/304 | 4.47/447 | 7.23/553 | 13.39/**598** | 416 | 0.5448 |
| m | max | 3.45/290 | 4.38/457 | 6.51/614 | 11.66/**686** | 416 | 0.5411 |
| l | prod | 4.32/232 | 6.67/300 | 11.93/335 | 22.41/**357** | 596 | 0.5723 |
| l | surgical | 4.38/228 | 6.42/311 | 10.98/364 | 20.50/**390** | 528 | 0.5724 |
| l | fast | 4.29/233 | 5.72/350 | 9.60/417 | 17.64/**453** | 520 | 0.5647 |
| x | prod | 5.71/175 | 9.45/212 | 17.18/233 | 32.76/**244** | 700 | 0.5927 |
| x | surgical | 5.62/178 | 9.07/220 | 16.09/248 | 30.28/**264** | 628 | 0.5929 |
| x | fast | 5.43/184 | 8.38/239 | 14.58/274 | 27.40/**292** | 622 | 0.5855 |

The m ladder end to end: PyTorch 66 → C++ fp32 230 → prod fp16_st 469 → surgical 526 (528 in the
§4 session; 561 with opt-batch 8) → fast 598 → **max 686 img/s (10.4× PyTorch, +46% over v0.2.0
prod)**. Subset deltas predicted the full-val deltas within ±0.05 pt on every gated config.

## 8. Fine-tuned production model

The whole stack, re-validated on a production fine-tuned **D-FINE-S (3 classes, food domain)**
against its own PyTorch checkpoint as pseudo-ground-truth (score ≥ 0.40), 2000 real 1810×1080
frames. Fidelity = AP of the engine's detections vs the .pt reference:

| stage | fidelity AP | b1 ms/FPS | b8 ips | VRAM MiB |
|---|---|---|---|---|
| fp32 | 0.9995 | 4.32/231 | 314 | 612 |
| fp16_st | 0.9989 | 3.20/312 | 500 | 476 |
| surgical | **0.9999** (AP50 = 1.0000 — the most faithful stage, incl. fp32) | 3.19/313 | 555 | 410 |
| Q100 (`--num-queries 100`) | 0.9964 | 3.03/330 | 619 | 404 |
| E2 + Q100 (`--eval-idx 2 --num-queries 100`) | 0.9966 | 2.99/334 | **638** | 370 |
| cascade 1:50 + slim | **0.9972** | — | 564 | 376 |

The measured point reached 2× FP32 throughput at 0.997+ fidelity and −40% VRAM. In this three-class
model, query reduction had a smaller fidelity cost than on COCO. CPU preprocessing of 1810×1080
frames then became the deployment bottleneck addressed by `full_pipeline_graph`.

## 9. Rejected or neutral probes

| probe | result | verdict |
|---|---|---|
| FP8 (real TRT, modelopt) | 0.3909 mAP, 7-9% slower | rejected on GeForce Ada (§3) |
| fused deform-attn plugin | Myelin already fuses to 16 kernels; ≤8% e2e ceiling | unjustified (§1) |
| INT8 standalone / combo | 519 / 486 ips < 528 surgical | closed (§6) |
| `builder_optimization_level=5` | 517 vs 562 ips — picked *worse* tactics | rejected |
| TF32 island in FP32 scopes | lossless, no speed | keep `--no-tf32` |
| L2 `persistent_cache_limit` | ≤ +1.2% (noise) | rejected |
| b16/b32 batches | +2-3% over b8 at 3.3× VRAM | plateau; not worth it |
| FP16 engine *input* (preprocess stores half) | lossless (fv 0.5503), perf-neutral, VRAM −18 MiB | memory-only; parked on a branch |
| 2-context ping-pong | +5.2% infer-ips | parked as a future runtime feature |
| BF16 | −0.0012 strongly-typed sim; the "−27 AP" was the weak-typing flag | superseded by FP16 paths |

## 10. Reproduce

```sh
: "${DFINE_CHECKPOINT:?set DFINE_CHECKPOINT to dfine_m_obj2coco.pt}"
: "${COCO_IMAGES:?set COCO_IMAGES to COCO val2017}"
: "${COCO_ANN:?set COCO_ANN to instances_val2017.json}"

uv sync --frozen --extra gpu --extra torch
./build.sh

# Opset-19 export required by the surgical converter.
uv run python trt-files/scripts/export_dfine_onnx.py \
    --model-name m --checkpoint "$DFINE_CHECKPOINT" \
    --opset 19 --output dfine_m_op19.onnx

# Fast adds --num-queries 200 --cascade 1:100.
# Max adds those flags plus --eval-idx 2, then builds with --opt-batch 8.

# Surgical FP16; --slim is the release recipe.
uv run python trt-files/scripts/convert_fp16_surgical.py --onnx dfine_m_op19.onnx \
    --output dfine_m_slim.onnx --slim

# Default latency-profile engine. Use --opt-batch 8 for the measured serving profile.
uv run python trt-files/scripts/build_engine.py --strongly-typed --no-tf32 --max-batch 8 \
    --onnx dfine_m_slim.onnx --output dfine_m_slim.engine

# Full-val accuracy and native runtime throughput.
uv run python trt-files/scripts/coco_eval.py --backends engine \
    --engine dfine_m_slim.engine --images "$COCO_IMAGES" --ann "$COCO_ANN" --limit 0
TRTLIB="$(uv run python -c 'import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))')"
LD_LIBRARY_PATH="$TRTLIB:${LD_LIBRARY_PATH:-}" \
    ./build/dfine_bench --engine dfine_m_slim.engine --batches 1,2,4,8 --iters 500
```

The flag-based export was verified to reproduce the gated research artifacts **bit-exactly**
(graph node/initializer hashes match), so these commands inherit the full-val gates above.
Profiling harnesses (§1) and the torch fake-quant rig (§2) are research scratch and not shipped;
`profile.py`, `coco_eval.py`, `dfine_bench` cover the reproducible surface.
