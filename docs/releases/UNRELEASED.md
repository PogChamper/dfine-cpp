# v0.4.0 — Unreleased

These changes are present in the source tree but not in the published v0.3.3 wheel.

## Runtime contract

- Detector destruction drains pending CUDA work before releasing detector-owned buffers.
- Engine loading validates the D-FINE IO layout, tensor types, and single-profile batch contract.
- Batch limits come from the TensorRT optimization profile and are cross-checked against engine
  sidecars.
- Explicit metadata paths are strict; automatic discovery still supports appended and same-stem sidecars.
- ONNX and engine sidecars are distinguished so an ONNX build recommendation is not mistaken for
  the engine's actual optimization profile.
- Legacy untagged sidecars remain supported; only metadata with engine build facts asserts profile
  fields.
- Engines may expose additional outputs when logits and boxes are identified by canonical or
  sidecar names; shape-only discovery remains limited to exactly two outputs.
- Model input remains RGB. `ImageU8::is_bgr` describes source pixels; a sidecar that declares BGR
  model input is rejected rather than silently ignored.
- Decode uses a fixed `min(300, Q×C)` candidate limit; sidecars do not override it.
- Full-val mAP is retained across the five `slim` engines and two reduced-query checks. On the Ada
  reference system, the larger reduced-query decode set costs 2–5% end to end while TensorRT
  inference time remains unchanged.
- `FreezeSpec` rejects incomplete or negative source dimensions and applies explicit width and height
  bounds independently.
- A rejected enqueue or deferred CUDA execution failure makes the detector unusable; recreate it
  before retrying inference.

## Build and packaging

- Installed CMake consumers no longer require CUDA or TensorRT development packages in `dfineConfig.cmake`.
- The C++ FP32 builder disables TF32; `--cuda-graph` sets zero auxiliary streams, and both facts
  are recorded in its engine sidecar.
- The Python builder derives graph compatibility from `max_aux_streams == 0`; `--cuda-graph` is a
  compatibility alias for that setting rather than an advisory label.
- Wheel publication is atomic: a failed rebuild preserves the previous artifact. The wheel is a
  native `py3-none-linux_x86_64` distribution with LICENSE and NOTICE included.
- Graph converters and engine builders reject path collisions, fully stage both files, serialize
  publishers, and restore the previous pair after a reported publication failure.
- New checkpoint exports record the D-FINE source repository, revision, and dirty state, plus the
  exporter hash and ONNX simplification result.
- Surgical FP16 sidecars identify the source graph, converter, and tool version; research-only
  precision overrides are marked experimental rather than labeled as the release recipe.
- Export fails before model setup when the validated D-FINE source revision is missing or malformed.
- Release assembly validates the exact model contract, ONNX structure, FP32-to-slim lineage, wheel
  contents, official checkpoint/tool provenance, version, and complete asset set before staging
  upload bytes.

The Ada release candidate passes WERROR, GPU recovery, UBSan, compute-sanitizer, Python/C++ tests,
installed-consumer and wheel loading, interleaved throughput, and full COCO validation. The tag
remains gated on final review and release-asset verification of the committed bytes.
