# Benchmarks

Results are grouped by question. Throughput, compiler, and accuracy tables use separate protocols
and are not combined into a single multiplier.

## Production accuracy

The released `slim` graphs retain the maintained full-COCO accuracy gate across all five sizes.

| Model | COCO AP |
|---|---:|
| D-FINE-N | 42.72 |
| D-FINE-S | 50.60 |
| D-FINE-M | 55.00 |
| D-FINE-L | 57.23 |
| D-FINE-X | 59.26 |

AP is reported in points. These values are the release gate, not the preset study below: the
study re-exports and rebuilds the same D-FINE-M checkpoint and measures 55.033, so gate and study
numbers are separate builds of one model.

## Preset surface

Throughput and COCO columns share one artifact chain: D-FINE-M, strongly typed `slim` (TensorRT
preserves the compute types encoded in the ONNX), RTX 4070 Ti SUPER, TensorRT 10.13, 640×640,
dynamic profile 1/1/8 (min/opt/max batch). Throughput is in-memory native C++
image-to-detections with CPU decode: the median of five independent 500-iteration process rounds,
interleaved across configurations, each preceded by an idle-GPU gate; the range is min–max across
rounds. Δ percentages are computed from the unrounded medians. Accuracy is full COCO `val2017` on the corresponding graph, evaluated through the TensorRT
Python session. RF4 is a separate cross-domain accuracy panel built from four independently
fine-tuned D-FINE-S checkpoints; its column contains only the minimum and maximum AP delta
against `base`.

Environment: WSL2 Linux 6.18, RTX 4070 Ti SUPER (SM 8.9), driver 581.15, CUDA runtime
12.8.90, and TensorRT 10.13.3.9.post1.

| Graph | Export flags | b8 img/s | b8 range | Δb8 | COCO AP | ΔAP | ΔAPs | ΔAR100 | ΔARs | RF4 ΔAP range |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | — | 533 | 492–537 | — | 55.033 | — | — | — | — | — |
| `Q200` | `--num-queries 200` | 560 | 558–568 | +5.1% | 54.849 | −0.184 | +0.286 | −0.562 | −1.015 | −0.152…+0.060 |
| `C300→150` | `--cascade 1:150` | 564 | 523–569 | +5.9% | 54.804 | −0.230 | −0.649 | −1.067 | −2.355 | −0.645…−0.017 |
| `C300→100` | `--cascade 1:100` | 576 | 555–579 | +8.1% | 54.536 | −0.497 | −1.157 | −2.379 | −3.841 | −0.733…−0.040 |
| `Q200→C100` | `--num-queries 200 --cascade 1:100` | 585 | 571–587 | +9.7% | 54.518 | −0.515 | −1.176 | −2.395 | −3.946 | −0.797…+0.053 |

`Q200` is the conservative measured point: its worst AP delta across COCO and RF4 is −0.184
points. `Q200→C100` is the fastest preset, and batch-8 medians rise monotonically with pruned
query count. Isolated rounds show host-side dips (the low range endpoints); medians across five
interleaved rounds absorb them. Batch-1 medians are 280–299 img/s for every preset with
overlapping round ranges: the presets are not separable at batch 1 in this protocol.

## Runtime modes and serving profile

The preset table uses the production 1/1/8 engine profile and the default enqueue path (one
`enqueueV3` call per batch). Two deployment axes stack on top of any preset: a batch-8-optimized engine profile (1/8/8, built with
`--opt-batch 8`) and full-pipeline CUDA Graph capture (FP32-output engines built with
`--cuda-graph`). `max` is `Q200→C100` plus `--eval-idx 2`. Same session and protocol as the
preset table.

| Config | Profile | Mode | b1 img/s | b1 p50 ms | b8 img/s | Δb8 vs `base` |
|---|---|---|---:|---:|---:|---:|
| `base` | 1/1/8 | CUDA graph | 432 | 2.31 | 558 | +4.7% |
| `base` | 1/8/8 | enqueue | 250 | 3.99 | 564 | +5.9% |
| `Q200→C100` | 1/1/8 | CUDA graph | 475 | 2.11 | 639 | +19.9% |
| `Q200→C100` | 1/8/8 | enqueue | 263 | 3.80 | 637 | +19.6% |
| `max` | 1/1/8 | CUDA graph | 501 | 1.99 | 654 | +22.7% |
| `max` | 1/8/8 | enqueue | 270 | 3.71 | 662 | +24.2% |

Against the same graph's 1/1/8 enqueue path in the preset table, CUDA Graph capture removes 34%
(`base`) to 38% (`Q200→C100`) of batch-1 latency; the 1/8/8 profile trades batch-1 latency for
batch-8 throughput. The fastest measured points are `max` under graph capture at batch 1
(1.99 ms) and `max` on 1/8/8 at batch 8 (662 img/s, +24.2% over `base`). `max` costs
−0.964 COCO AP against `base` (54.069 on TensorRT `slim`) — about twice the `Q200→C100`
movement — and its RF4 cross-domain deltas are unmeasured.

The [v0.5 preset report](reports/v0.5.0-preset-evaluation.md) contains AP50/AP75, all size and recall
slices, per-class sensitivity, backend deltas, dataset provenance, and the exact scope of each
claim.

## PyTorch baselines

D-FINE-M on full COCO `val2017`. Environment: WSL2 Linux 6.18, RTX 4070 Ti SUPER, driver
581.15, PyTorch 2.9.1+cu128, CUDA 12.8, and cuDNN 9.10.2; the GPU was otherwise idle. Each mode
and batch used three fresh 500-iteration processes. Compilation caches were new for every compiled
mode, batch, and round. First call includes lazy compilation and autotuning; it is excluded from
steady state.

| Mode | COCO AP | b1 forward | b1 E2E | b8 forward | b8 E2E | First call b1 / b8 |
|---|---:|---:|---:|---:|---:|---:|
| eager FP32 | 55.0680 | 50.89 | 39.99 | 153.09 | 88.23 | 4.23 s / 4.80 s |
| `torch.compile` FP32 | 55.0672 | 84.46 | 66.82 | 199.03 | 102.95 | 46.63 s / 51.10 s |
| eager FP16 | 55.0609 | 34.41 | 31.16 | 221.23 | 100.77 | 2.41 s / 2.36 s |
| `torch.compile` FP16 | 55.0755 | 74.08 | 58.88 | 420.91 | 146.92 | 57.42 s / 62.69 s |

Values are img/s. Compiled FP32 improved measured E2E throughput by 1.67× at batch 1 and 1.17× at
batch 8 while changing AP by −0.0008 points. This is a PyTorch baseline table, not a direct
TensorRT comparison: the native and PyTorch rows use different execution surfaces.

## Cross-GPU throughput

D-FINE-M, 640×640, steady-state native pipeline p50 including preprocess, TensorRT, transfer, and
CPU decode. Each row is the median of three interleaved 500-iteration rounds. The one-pass
compatibility report uses a separate protocol; [Validation](VALIDATION.md) shows how to generate it.
All rows used CUDA 12, TensorRT 10.13.3.9, and an otherwise-idle GPU.

| GPU | SM | Driver | Host | TRT | Recipe | b1 img/s | b8 img/s |
|---|---:|---|---|---|---|---:|---:|
| RTX 3090 | 8.6 | 550.107.02 | Ubuntu 22.04 | 10.13.3.9 | `slim` | 310 | 487 |
| RTX 4070 Ti SUPER | 8.9 | 581.15 | WSL2 6.18 | 10.13.3.9 | `slim` | 287 | 533 |
| RTX 5080 | 12.0 | 610.43.02 | Ubuntu 22.04 | 10.13.3.9 | `slim` | 456 | 676 |

The full Ada size/configuration ladder is preserved in the [research matrix](RESEARCH_MATRIX.md).
Engine-only, image-to-detections, and application E2E numbers must be reported as different scopes.
