# Conversion

The conversion toolchain produces a verified D-FINE TensorRT artifact without moving model-specific box math into the runtime.

```text
checkpoint
  → explicit-gather FP32 ONNX artifact
  → surgical strongly typed FP16 ONNX artifact
  → target-local TensorRT engine
```

An **ONNX artifact** is an `.onnx` graph and its required `.json` sidecar. An **engine** is a TensorRT plan compiled from that artifact for one target stack. A **model pack** is a published set of ONNX artifacts. A **preset** changes the exported model graph and may trade accuracy for speed.

## Production path

| Stage | Default | Output |
|---|---|---|
| Load | Strict checkpoint match | Loaded detection model |
| Export | Opset 19, explicit gather, trace batch 2 | FP32 ONNX artifact |
| Convert | Surgical FP16 with `--slim` | Typed FP16 ONNX artifact |
| Build | Strong typing, TF32 off, min/opt/max batch 1/1/8 | TensorRT engine |

Released model packs contain both FP32 (`dfine_<size>_op19`) and production FP16 (`dfine_<size>_slim`) artifacts for N, S, M, L, and X. Download a graph together with its same-stem JSON sidecar.

## Build an engine from a released artifact

The Python CLI is the shortest path:

```sh
dfine build --model m --onnx dfine_m_slim.onnx \
    --output dfine_m_slim.engine
```

The direct builder is equivalent:

```sh
python trt-files/scripts/build_engine.py \
    --onnx dfine_m_slim.onnx \
    --output dfine_m_slim.engine \
    --strongly-typed --no-tf32 \
    --min-batch 1 --opt-batch 1 --max-batch 8
```

Use `--opt-batch 8` for a batch-serving engine. It improves batch-8 throughput by approximately 6–10% in the measured D-FINE-M runs and increases batch-1 latency by 8–19%.
Keep `--min-batch 1` when targeting a larger batch. An equal profile such as `8/8/8` becomes a
static-batch engine and is rejected: the native runtime accepts static batch 1 or a dynamic range.

Add `--cuda-graph` to `dfine build` when either CUDA Graph mode will be used. It sets
`max_aux_streams=0`; the default cache output uses a separate `-g0` entry. With the direct builder,
use `--max-aux-streams 0` or its `--cuda-graph` alias. The flag does not change output types; active
capture also requires FP32 logits and boxes. The ordinary runtime and GPU decode do not require it.

Do not add `--fp16` to a typed FP16 artifact. That flag discards the encoded types and lets TensorRT choose precision freely — the weak-typing failure class this pipeline exists to prevent. The FP16 compute types are already encoded in `dfine_m_slim.onnx`; `--strongly-typed` tells TensorRT to preserve them.

## Export a checkpoint

Checkpoint export uses the detection-only D-FINE implementation bundled under
`trt-files/dfine_model`. Its upstream [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg)
revision is stored in [`trt-files/DFINE_SEG_REVISION`](../trt-files/DFINE_SEG_REVISION).
D-FINE-seg remains the training implementation and differential-test reference; users do not need
its source tree to export a checkpoint.

Prepare the locked tools environment:

```sh
uv sync --frozen --extra gpu --extra torch
```

Standard detection checkpoints are hosted in
[`ArgoSA/D-FINE-seg`](https://huggingface.co/ArgoSA/D-FINE-seg). Use
`dfine_{s,m,l,x}_obj2coco.pt`; nano has no `obj2coco` checkpoint, so use `dfine_n_coco.pt`. The
exporter does not download checkpoints; pass an existing standard or custom checkpoint explicitly.

Export the FP32 base:

```sh
uv run python trt-files/scripts/export_dfine_onnx.py \
    --model-name m \
    --checkpoint /path/to/dfine_m_obj2coco.pt \
    --opset 19 \
    --output dfine_m_op19.onnx
```

The exporter performs these postconditions before publishing the graph and sidecar:

1. Every checkpoint-owned detection tensor is loaded with a matching shape.
2. Graph inputs and outputs are `images`, `logits`, and `boxes`.
3. The batch dimension remains symbolic.
4. The graph contains only standard-domain ONNX operations.
5. The explicit deformable-attention core contains no `GridSample` node.
6. ONNX Runtime executes batch 1 and 2 with finite outputs whose shapes match the sidecar.

Failed postconditions do not publish or overwrite the output pair. `--allow-partial-checkpoint` exists for research and records a partial load in the sidecar; do not use it for a release or deployment artifact.
Checkpoint deserialization is weights-only by default. `--allow-unsafe-checkpoint` enables pickle
fallback and therefore permits arbitrary code execution; use it only for a trusted legacy file.
The sidecar records the selected checkpoint state, deserialization mode, and loaded and unused
tensor counts.

Convert the FP32 graph to the production recipe:

```sh
uv run python trt-files/scripts/convert_fp16_surgical.py \
    --onnx dfine_m_op19.onnx \
    --output dfine_m_slim.onnx \
    --slim
```

The converter carries the sidecar forward, updates its precision recipe, and records the source-graph
and converter SHA-256 plus the `onnxconverter-common` version. It keeps the FDR box-decoder scopes and
deform coordinate/index chain in FP32; backbone, encoder, decoder compute, and deform data flow use
FP16. Graph outputs remain FP32, preserving one output contract for CPU decode and the optional
GPU/graph paths.

The source-checkout CLI runs the same sequence in the locked tools environment:

```sh
PYTHONPATH=python uv run --frozen --extra gpu --extra torch \
  python -m dfine.cli export --model m --checkpoint /path/to/checkpoint.pt \
    --precision fp16 --output dfine_m_slim.onnx
```

An editable install exposes the shorter `dfine export` form. The runtime wheel does not contain the
checkpoint-export toolchain.

## Custom classes and checkpoints

Choose the model size and class count that match training. The strict loader reports classifier-shape and variant mismatches before export.

```sh
uv run python trt-files/scripts/export_dfine_onnx.py \
    --model-name s \
    --checkpoint food_s.pt \
    --num-classes 3 \
    --class-names burger,fries,drink \
    --output food_s_op19.onnx
```

`--class-names` accepts a comma-separated list or a file with one label per line. When it is omitted, COCO-80 names are recorded for an 80-class model; other class counts use `class_<id>` display names unless names are supplied at runtime.

The standard checkpoints are detection checkpoints. Extra tensors, such as a segmentation head, are reported and ignored when every detection tensor matches. Missing or shape-mismatched detection tensors remain fatal.

## Artifact contract

The raw ONNX interface is deliberately small:

| Tensor | Type and shape | Meaning |
|---|---|---|
| `images` | FP32 `[N,3,H,W]` | RGB, `/255`, NCHW input |
| `logits` | FP32 `[N,Q,C]` | Pre-sigmoid class logits |
| `boxes` | FP32 `[N,Q,4]` | Normalized `cxcywh` boxes |

Sigmoid, global top-k, thresholding, `cxcywh → xyxy`, and source-image scaling remain outside the engine. D-FINE's FDR/Integral/LQE box computation and deformable attention remain inside it. No NMS is applied.

The ONNX sidecar owns the model, preprocessing, export recipe, checkpoint hash, tool versions, and
source provenance. Engine builders carry that contract forward and add compiled IO/profile facts,
TensorRT settings, auxiliary streams, and source-ONNX identity. The runtime trusts the compiled
profile, validates compatible sidecar fields, and uses documented preprocessing defaults when no
sidecar is found. Explicit sidecar paths are strict.

Names remain discovery labels, not identity. [Artifact naming and identity](NAMING.md) defines field
ownership, discovery precedence, cache fingerprints, and schema evolution.

## Why the ordinary paths fail

Three failures define the maintained recipe:

| Path | Observed result | Maintained fix |
|---|---|---|
| Native `grid_sample` deform export | 0.4455 AP vs 0.5509 PyTorch | Explicit gather-bilinear ONNX |
| Weakly typed TensorRT FP16 flag | Large fixed AP loss in box decode | Precision encoded in typed ONNX |
| Fine-grained FP16 on opset-16 LayerNorm decomposition | TensorRT collapse while ONNX Runtime remains healthy | Opset 19 native `LayerNormalization` |

These are measured TensorRT behaviors for the recorded graph and stack. They are not general claims about every model using the same operators. Full isolation data is in [`impl/DFINE_SEG_TRT_BUG_REPORT.md`](impl/DFINE_SEG_TRT_BUG_REPORT.md) and the [research matrix](RESEARCH_MATRIX.md).

## Accuracy-traded presets

`slim` changes precision placement; the options below change the exported graph. They are explicit
operating points, not aliases:

| Graph | Export flags | Output queries | Measured D-FINE-M ΔAP | Measured b8 throughput |
|---|---|---:|---:|---:|
| `base` | — | 300 | — | 533 img/s |
| `Q200` | `--num-queries 200` | 200 | −0.184 | 560 img/s |
| `C300→150` | `--cascade 1:150` | 150 | −0.230 | 564 img/s |
| `C300→100` | `--cascade 1:100` | 100 | −0.497 | 576 img/s |
| `Q200→C100` | `--num-queries 200 --cascade 1:100` | 100 | −0.515 | 585 img/s |

Accuracy is full COCO `val2017` on TensorRT `slim`; throughput is the matching batch-8 native C++
comparison on RTX 4070 Ti SUPER (median of five idle-gated interleaved rounds). The
[benchmark table](BENCHMARKS.md#preset-surface) adds recall and object-size deltas plus the CUDA
Graph and batch-8-profile runtime modes; the
[sweep](VALIDATION.md#sweep-your-checkpoint) runs the same comparison on a custom checkpoint and
dataset in one command.

`--eval-idx` is an additional graph slider with a larger accuracy cost (the measured `max` preset
in [Benchmarks](BENCHMARKS.md#runtime-modes-and-serving-profile) uses `--eval-idx 2`). Apply every
graph change explicitly so the sidecar records the query and decoder contract.

## Reproducibility and release gates

`uv.lock` pins the Python toolchain. The sidecar records the upstream revision and a deterministic
SHA-256 over every bundled model source, in addition to the exporter and checkpoint hashes. Byte
comparison is meaningful only when those hashes, tool versions, export flags, and simplification path
match. Runtime correctness does not rely on byte identity alone: validate the produced engine against
the expected detections and dataset metric.

Release gates cover three boundaries:

1. ONNX behavior at batches 1 and 2.
2. Engine smoke at batches 1, 2, and 8 plus box/score parity.
3. Full COCO validation for graph or precision changes, followed by atomic graph/sidecar publication
   with SHA-256 coverage.

The executable release procedure is in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md); benchmark methodology is in [Validation](VALIDATION.md).
