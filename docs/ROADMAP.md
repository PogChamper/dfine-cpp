# Roadmap

D-FINE-cpp maintains one path: accurate checkpoint conversion, target-local TensorRT compilation,
and native in-process inference. [Runtime](RUNTIME.md) defines its current boundary;
[Unreleased](releases/UNRELEASED.md) tracks work awaiting the next release. This roadmap covers
adoption and scope beyond that boundary.

## Adoption work

Technical distribution now has higher value than adding model variants.

### Upstream the conversion finding

- Publish a minimal native-`GridSample` versus explicit-gather reproducer.
- Propose box-aware parity checks to the D-FINE export path; score-only comparison misses the observed failure.
- Coordinate the maintained recipe with D-FINE and D-FINE-seg rather than maintaining incompatible deployment guidance.
- Submit the isolated TensorRT divergence to NVIDIA with exact graph, versions, inputs, and output deltas.

The goal is a reviewable upstream fix backed by independently reproducible evidence.

### Reduce time to first verified engine

- Keep model graph and sidecar paired throughout download, build, and cache resolution.
- Separate stable model packs from runtime patch releases so unchanged model assets are not republished.
- Report artifact identity, engine profile, GPU, and TensorRT facts in one diagnostic output.
- Collect complete install reports from non-maintainer systems.

Any new download helper belongs here only after it exists, has checksum coverage, and replaces—not supplements—the manual path.

### Build one real video reference

Add an optional application-layer example that reads a file or camera, invokes the unchanged runtime, and reports measured end-to-end latency, throughput, and dropped frames. OpenCV or GStreamer may be an app dependency; `libdfine` remains independent of either.

The demo must process the displayed frames. Benchmark-derived counters are not a substitute for a running pipeline.

## Choose one integration lane from demand

The project should not build video and serving infrastructure in parallel. Revisit the choice after external issues, validation reports, and deployments identify a repeated constraint.

| Repeated demand | Candidate lane | First useful increment |
|---|---|---|
| Multi-stream video, camera, or Jetson | Video/edge | DeepStream parser/config or device-input adapter |
| Concurrent requests and latency SLOs | Serving | Shared engine state with separate execution contexts |
| Native mask consumption | Segmentation | Explicit mask-output and ABI design |
| Memory-bound edge deployment | Quantization | Hardware-specific INT8 evaluation |

### Video/edge lane

Start with an adapter around existing NVIDIA video infrastructure. DeepStream already owns decode, batching, surfaces, and scheduling; a configuration and output parser provide more evidence than duplicating that stack inside the core. Add device-pointer or NV12 input only after copy cost is measured in a real pipeline.

### Serving lane

The current detector owns one engine, execution context, and stream. Serving work begins by separating immutable engine/model state from per-execution context and buffer state. Integration with an existing serving system should precede a project-specific scheduler.

### Segmentation lane

Segmentation checkpoints now exist, so absence of weights is not a blocker. The feature remains deferred because it changes ONNX outputs, GPU postprocessing, result ownership, C ABI layout, and memory behavior without a demonstrated runtime user. Design it only with a concrete mask consumer.

## Research policy

Measured negative results remain part of the project record, not active product tiers.

| Topic | Current disposition |
|---|---|
| Weakly typed FP16 | Rejected for correctness |
| FP8 on GeForce Ada | Closed: lower accuracy and slower than surgical FP16 |
| Tested desktop INT8 PTQ recipes | Closed: slower and less accurate than surgical FP16 |
| Custom deform/FDR plugin | Closed: TensorRT fusion leaves a small end-to-end ceiling |
| Accuracy-traded export presets | Available for explicit evaluation; not the default |
| Browser inference | Feasibility work only after the native adoption path |

A closed result may be reopened for different hardware or a materially different method. It must return through the same artifact, accuracy, and performance gates as the production recipe.

## Evidence required to expand scope

Before adding a new maintained lane, require:

1. A concrete user and workload.
2. A measured bottleneck in the current path.
3. An API boundary that does not weaken the synchronous core.
4. Correctness parity against the existing result contract.
5. A reproducible benchmark on the target hardware.
6. An owner for tests, documentation, and compatibility after release.

## v1.0 criteria

Version 1.0 should describe stability, not feature count:

- the default checkpoint-to-engine path is reproducible and independently validated;
- supported platforms and TensorRT compatibility are explicit;
- C ABI evolution and artifact schemas have tested compatibility rules;
- clean-machine installation and out-of-tree CMake consumption are release gates;
- GPU correctness and recovery tests run routinely;
- at least one external deployment exercises the maintained runtime contract.

Released work is recorded in [release notes](releases/). Current measurements remain in
[Validation](VALIDATION.md) and the [research matrix](RESEARCH_MATRIX.md).
