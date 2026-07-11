# Unreleased

These changes are present in the source tree but not in the published v0.3.3 wheel.

## Runtime contract

- Detector destruction drains pending CUDA work before releasing detector-owned buffers.
- Engine loading validates the D-FINE IO layout, tensor types, and single-profile batch contract.
- Batch limits come from the TensorRT optimization profile and are cross-checked against the sidecar.
- Explicit metadata paths are strict; automatic discovery still supports appended and same-stem sidecars.
- Decode uses a fixed `min(300, Q×C)` candidate limit; sidecars do not override it.
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
- The wheel is emitted as a native `py3-none-linux_x86_64` distribution with LICENSE and NOTICE included.
- New checkpoint exports record the D-FINE source repository, revision, and dirty state, plus the
  exporter hash and ONNX simplification result.

The next release remains gated on GPU recovery tests, compute-sanitizer, batches 1/2/8, Python/C++ parity, installed-wheel loading, and release-asset verification.
