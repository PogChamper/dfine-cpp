# examples/

Documented, minimal usage snippets for the D-FINE C++ API — not built by the
project's CMake (see the header of each file for a manual compile/link line).

- `cpp_detector_example.cpp` — construct `dfine::DFineDetector` from an engine
  path, build an `ImageU8` from a raw HWC uint8 buffer, call `detect()`, print
  the resulting boxes/classes/scores.
- `python_quickstart.ipynb` — the zero-checkout Python path: install the release
  wheel, `dfine build` the release ONNX into an engine, detect, draw, and
  measure throughput on your GPU.

To build the real library (`libdfine.so`) these snippets link against, use
`./build.sh` (see the repo root `CMakeLists.txt`) — it configures cmake with
this project's toolchain gotchas already baked in.

Real apps (`apps/dfine_detect.cpp` etc.) decode images with the vendored
`third_party/stb` (`stb_image.h`) via `apps/image_io.cpp`; these examples skip
that and construct the pixel buffer directly to stay dependency-free.
