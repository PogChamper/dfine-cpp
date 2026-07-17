# Validation

Validation has two outputs:

- an accuracy report for one checkpoint, graph, or engine on a COCO-format dataset;
- a compatibility report for one TensorRT stack and GPU.

Published measurements are in [Benchmarks](BENCHMARKS.md). Historical experiments are in the
[research matrix](RESEARCH_MATRIX.md).

## Environment

Dataset evaluation uses the locked maintainer environment:

```sh
uv sync --frozen --extra gpu --extra torch
```

Set the dataset paths once. `test.json` must use the COCO detection schema and reference files under
`test/images`.

```sh
export IMAGES=/path/to/test/images
export ANN=/path/to/test.json
export SAMPLE_IMAGE=/path/to/sample.jpg
export TRTLIB=/path/to/directory/containing/libnvinfer.so.10
mkdir -p validation
```

The scorer accepts any class count and any COCO-format detection dataset. Model input remains RGB,
640×640 stretch, and `/255`. Model class index `i` maps to the `i`th sorted COCO `category_id`;
that order must match the checkpoint's training label order.

## Sweep your checkpoint

One command runs the complete comparison below over a set of operating points and writes a
decision table (`sweep.md`, machine-readable `sweep.json`):

```sh
uv run --frozen python trt-files/scripts/preset_sweep.py \
    --model-name s --num-classes 3 --checkpoint model.pt \
    --images "$IMAGES" --ann "$ANN" \
    --ap-budget 0.3 \
    --out sweep/
```

`dfine sweep` forwards to the same tool from a source checkout. The default points are the
published presets (`base`, `q200`, `c150`, `c100`, `fast`). `--point NAME[:key=value,...]` adds
one operating point per flag; a preset name prefills the graph keys, a new name requires them
(or copies a preset via `graph=`):

| Key | Meaning | Default |
|---|---|---|
| `queries` | initial query count | 300 |
| `cascade` | `K:KEEP` — keep KEEP queries after decoder layer K | none |
| `eval-idx` | decoder exit layer | model default |
| `graph` | preset to copy the graph from, under a new point name | — |
| `precision` | `slim` or `fp32` | `slim` |
| `profile` | engine batch profile `MIN/OPT/MAX` | `1/1/8` |
| `mode` | `enqueue`, or `graph` — FP32-output build, CUDA Graph measurement | `enqueue` |

```sh
--point max                                # the measured record configuration
--point fast:profile=1/8/8                 # batch-8-optimized serving engine
--point max:mode=graph                     # full-pipeline CUDA Graph batch-1 latency
--point q150c75:queries=150,cascade=1:75   # a custom graph
--point base-fp32:graph=base,precision=fp32
```

Accuracy runs once per graph and precision; throughput once per point. Deltas and the
`--ap-budget` marking are anchored to the `base` point (`--baseline` overrides). Measurement
knobs default to the published protocol — batches 1 and 8, 50 warm-up, 3×500 iterations
(`--batches/--warmup/--iters/--rounds`) — and every fresh throughput step is preceded by a GPU
idle check: a warning by default, a refusal with `--strict-idle`.

A D-FINE-seg YOLO split converts on the fly: `--yolo-dataset data/dataset --split val
--class-names names.txt`, names in the training `label_to_name` order
(`trt-files/scripts/yolo_to_coco.py` is the standalone converter).

Every step is one of the tools documented below with its argv recorded in `sweep.json`;
artifacts and reports keep their canonical on-disk forms under `--out` (`graphs/`, `engines/`,
`reports/`, `logs/`). A step is reused on re-runs when its sidecar or report proves the expected
lineage; `--rebench` forces fresh latency. A failed step aborts the sweep with its log path;
`--keep-going` finishes the surviving points, marks failed rows under the table, and exits
nonzero.

The sections below are the workflow the sweep automates; use them to run or re-check any single
step.

## Evaluate one backend chain

Export and build the candidate as described in [Conversion](CONVERSION.md). The graph flags define
the operating point. ONNX and engine evaluation requires each artifact's adjacent JSON sidecar.

| Graph | Export flags |
|---|---|
| `base` | — |
| `Q200` | `--num-queries 200` |
| `C300→150` | `--cascade 1:150` |
| `C300→100` | `--cascade 1:100` |
| `Q200→C100` | `--num-queries 200 --cascade 1:100` |

The following example evaluates `Q200` for a three-class D-FINE-S checkpoint. The first command
compares PyTorch, explicit FP32 ONNX, and TensorRT FP32. The second evaluates the typed `slim`
artifacts under the same dataset contract.

```sh
uv run --frozen python trt-files/scripts/coco_eval.py \
    --backends torch onnx engine \
    --model-name s --num-classes 3 --num-queries 200 \
    --checkpoint model.pt \
    --onnx artifacts/q200-fp32.onnx \
    --engine artifacts/q200-fp32.engine \
    --images "$IMAGES" --ann "$ANN" \
    --report validation/q200-raw.json

uv run --frozen python trt-files/scripts/coco_eval.py \
    --backends onnx engine \
    --model-name s --num-classes 3 \
    --onnx artifacts/q200-slim.onnx \
    --engine artifacts/q200-slim.engine \
    --images "$IMAGES" --ann "$ANN" \
    --report validation/q200-slim.json
```

Assemble the adjacent deltas:

```sh
uv run --frozen python trt-files/scripts/accuracy_chain.py \
    --stage pytorch=validation/q200-raw.json::torch \
    --stage ort-fp32=validation/q200-raw.json::onnx \
    --stage trt-fp32=validation/q200-raw.json::engine \
    --stage trt-slim=validation/q200-slim.json::engine \
    --transition-kind export \
    --transition-kind runtime \
    --transition-kind precision \
    --output validation/q200-chain.json

uv run --frozen python trt-files/scripts/accuracy_chain.py \
    --stage ort-fp32=validation/q200-raw.json::onnx \
    --stage ort-slim=validation/q200-slim.json::onnx \
    --stage trt-slim=validation/q200-slim.json::engine \
    --transition-kind precision \
    --transition-kind runtime \
    --output validation/q200-slim-chain.json
```

The command rejects reports with different annotations, image bytes, preprocessing, thresholds,
Top-K, inference batch, evaluator source, GT population, model contract, or artifact lineage.
Reports that predate model-contract recording carry no lineage and are rejected outright;
`--allow-missing-lineage` compares them anyway and marks the affected transitions
`lineage_verified: false`.
Absolute AP describes the trained model. The first chain isolates export, FP32 TensorRT, and
TensorRT precision transitions; the second isolates ONNX precision and the final `slim` backend
transition.

To compare a candidate against `base`, assemble a two-stage chain from the final engine reports:

```sh
uv run --frozen python trt-files/scripts/accuracy_chain.py \
    --stage base=validation/base-slim.json::engine \
    --stage q200=validation/q200-slim.json::engine \
    --transition-kind preset \
    --transition-label 'Q200 vs base' \
    --output validation/base-to-q200.json
```

Repeat the two-stage comparison for each candidate. Select the fastest graph that satisfies the
application's AP, recall, object-size, and class budgets.

## Accuracy report

Reports use standard COCO bbox evaluation with `maxDets=[1,10,100]`.

| Field | Purpose |
|---|---|
| AP, AP50, AP75 | Overall localization quality |
| APs, APm, APl | Original-image object-size slices |
| AR1, AR10, AR100 | Recall under standard detection limits |
| ARs, ARm, ARl | Recall by object size |
| AP by IoU | Localization sensitivity from 0.50 through 0.95 |
| Per-class AP and GT count | Class-specific movement with sample size |
| GT density | Objects per image and density histogram |
| Model-space metrics | Secondary size view after resize/letterbox |

Every report records the annotation hash, selected image-content hash, evaluator hash, artifact and
sidecar hashes, preprocessing, threshold, Top-K, batch, and runtime versions. PyTorch and ONNX
Runtime entries record their numeric policy; TensorRT build policy remains in the hashed engine
sidecar and the profiler's embedded engine contract. Native reports also record requested CUDA
Graph, GPU-decode, frozen-memory, and preprocessing modes. Requested and verified runtime states
remain separate when a native path permits fallback.

## Native accuracy

Build `dfine_coco_eval`, then evaluate the C++ boundary:

```sh
uv run --frozen python trt-files/scripts/cpp_coco_eval.py \
    --binary build/dfine_coco_eval \
    --engine artifacts/q200-slim.engine \
    --images "$IMAGES" --ann "$ANN" \
    --ld-library-path "$TRTLIB" \
    --batch 1 --limit 0 \
    --out validation/q200-native-detections.json \
    --report validation/q200-native.json
```

Use native accuracy to verify the runtime boundary. Do not infer that GPU decode or CUDA Graph was
active unless the report contains the corresponding activation and replay evidence.

## Measure throughput

Use the same engine profile, sample image, warm-up, iteration count, round count, and measurement
scope for every candidate. Build `dfine_bench` with `./build.sh` first.

```sh
uv run --frozen python trt-files/scripts/profile.py \
    --backends cpp \
    --engine artifacts/q200-slim.engine \
    --model-name s --num-classes 3 \
    --sample-image "$SAMPLE_IMAGE" \
    --batches 1 8 \
    --warmup 50 --iters 500 --rounds 3 \
    --no-accuracy \
    --ld-library-path "$TRTLIB" \
    --out validation/q200-latency.json
```

`profile.py` keeps engine-device, transfer-inclusive, framework-call, and native
image-to-detections scopes distinct. Report batch latency and batch throughput separately. Build a
batch-targeted 1/B/B engine when comparing compilers at batch B; report the production 1/1/8
profile as a separate deployment result.

## Generate a compatibility report

The compatibility report builds a target-local engine and records the GPU, driver, CUDA, TensorRT,
artifact hashes, build recipe, and steady-state batch-1/batch-8 throughput.

```sh
git clone https://github.com/PogChamper/dfine-cpp && cd dfine-cpp
python -m pip install "tensorrt-cu12==10.13.*"
curl -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/dfine_m_slim.onnx \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/dfine_m_slim.json \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.5.0/SHA256SUMS
python trt-files/scripts/validation_report.py \
    --onnx dfine_m_slim.onnx \
    --check-sums SHA256SUMS \
    --out validation
```

Without TensorRT, the report records the environment and marks the build skipped. With TensorRT
but no usable GPU, the build is recorded as failed. Build `dfine_bench` first with `./build.sh` to
include throughput.

## Submit a compatibility report

Review `validation/report.md` and `validation/report.json`, then attach both to a GitHub issue titled
`validation: <GPU> / TRT <version>`. The report includes `nvidia-smi` and `platform.uname()` output;
inspect it before posting.
