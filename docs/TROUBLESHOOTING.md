# Troubleshooting

Run this first:

```sh
dfine doctor
```

It reports the package version, Python, OS, GPU/driver, TensorRT import, candidate `libdfine`
paths, the selected library load status, TensorRT headers, and engine cache. Include the output in
bug reports.

## Installation

### `tensorrt-cu13-libs` or a CUDA 13 dependency conflict

The unqualified `tensorrt` metapackage may resolve to the CUDA-13 build. The published v0.4.0 wheel and source use CUDA 12:

```sh
python -m pip uninstall -y tensorrt tensorrt-cu13 tensorrt-cu13-libs tensorrt-cu13-bindings
python -m pip install "tensorrt-cu12==10.13.*"
```

Use the same Python environment for `dfine build` and `dfine doctor`.

### `libnvinfer.so.10: cannot open shared object file`

The TensorRT runtime directory is not visible to the dynamic loader. For a pip installation:

```sh
export LD_LIBRARY_PATH="$(python -c 'import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))'):${LD_LIBRARY_PATH}"
dfine doctor
```

For a system or tarball installation, add its `lib` directory instead.

### The wheel fails on Turing or Ampere

The release wheel contains an `sm_89` native build and forward PTX. It is validated on Ada and
Blackwell; PTX compatibility does not run backward on Turing or Ampere. Build the library for the
target architecture:

```sh
git clone https://github.com/PogChamper/dfine-cpp
cd dfine-cpp
CUDA_ARCH=86 ./build.sh   # Ampere; use 75 for Turing
python -m pip install -e "python[cli]"
export DFINE_LIBRARY="$PWD/build/libdfine.so"
```

### `DFINE_LIBRARY` points to a missing or unloadable file

An explicit `DFINE_LIBRARY` is strict and never falls through to another copy:

```sh
unset DFINE_LIBRARY
# or
export DFINE_LIBRARY=/absolute/path/to/libdfine.so
```

`dfine doctor` lists the candidate paths and reports the selected load failure, including a missing
transitive TensorRT or CUDA library.

### The driver or `nvidia-smi` fails

Fix the host driver before debugging D-FINE-cpp. RTX 50-series systems require an R570+ NVIDIA
driver; native `sm_120` builds require CUDA 12.8 or newer. Driver replacement is host-specific;
preserve package-manager and kernel logs instead of applying an unconditional purge recipe.

## Building from source

### `Could NOT find TensorRT (missing: TENSORRT_INCLUDE_DIR ...)`

The pip package supplies runtime libraries and Python bindings, not C++ headers. Choose one header source:

1. Install the TensorRT 10.13 development packages from NVIDIA's CUDA 12 apt repository.
2. Build inside an NVIDIA TensorRT development container.
3. Unpack a matching TensorRT GA tarball under `third_party/tensorrt/{include,lib}`.

For apt, pin the complete dependency chain. An unpinned install may select another TensorRT major:

```sh
V="$(apt-cache madison libnvinfer-dev | grep -oPm1 '10\.13\.[0-9.]+-1\+cuda12\.[0-9]+')"
sudo apt-get install -y \
  "libnvinfer10=$V" \
  "libnvinfer-headers-dev=$V" \
  "libnvinfer-dev=$V" \
  "libnvonnxparsers10=$V" \
  "libnvonnxparsers-dev=$V"
```

The parser package is plural: `libnvonnxparsers-dev`. It is required by the C++ engine builder;
`libdfine` itself links only the TensorRT runtime.

To build the library without the parser or command-line applications:

```sh
cmake -B build -S . -DDFINE_BUILD_APPS=OFF -DDFINE_BUILD_TESTS=OFF
cmake --build build -j
```

For a tarball:

```text
third_party/tensorrt/
├── include/NvInfer.h
└── lib/libnvinfer.so.10
```

Set `TENSORRT_DIR=/path/to/tensorrt` to use another layout.

### A wall of `math.h`, pthread, or glibc errors

A conda cross-compiler is likely shadowing the system toolchain while TensorRT uses system headers. `./build.sh` detects the common case. With plain CMake:

```sh
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=native
```

### `nvcc fatal: Don't know what to do with 'UNSET'`

A repeated conda activation may leak a sentinel into `NVCC_PREPEND_FLAGS`:

```sh
unset NVCC_PREPEND_FLAGS
./build.sh
```

### `CUDA_ARCHITECTURES=native` fails

`native` requires CMake ≥3.24 and a visible local GPU. Use an explicit architecture for older CMake, containers, or headless builders:

```sh
CUDA_ARCH=89 ./build.sh
```

## Engine build and load

### TensorRT cannot deserialize the engine

TensorRT plans are target-local build products. Rebuild from the ONNX artifact with the TensorRT stack used at runtime:

```sh
dfine build --model m --onnx dfine_m_slim.onnx \
    --output dfine_m_slim.engine
```

Do not copy an engine across incompatible TensorRT versions or GPU targets. Copy the ONNX artifact and sidecar, then rebuild.

### `meta sidecar contradicts the engine`

The JSON belongs to another graph or engine build. Keep these pairs together:

```text
dfine_m_slim.onnx + dfine_m_slim.json
dfine_m_slim.engine + dfine_m_slim.engine.json
```

Current sidecars identify themselves as `onnx` or `engine`. ONNX batch bounds are build recommendations;
engine sidecars are checked against the compiled profile. Rebuild the engine from the intended ONNX
artifact. Do not edit dimensions, class counts, or engine profile fields to suppress the error.

An explicit `meta_path` must exist and agree with the engine. Omit it to use normal sidecar discovery.

### `incompatible D-FINE engine`

The runtime requires a linear FP32 input `[B,3,H,W]` with static `B=1` or dynamic batch,
logits `[B,Q,C]`, and boxes `[B,Q,4]`. Additional outputs require canonical or explicit names for
the two detection tensors. Inspect the engine and rebuild it from a raw D-FINE ONNX artifact. A
fused-postprocessing or differently typed export is not interchangeable.

### A batch exceeds `max_batch`

The runtime reads min/opt/max batch from the engine profile. Rebuild with a larger profile:

```sh
dfine build --model m --onnx dfine_m_slim.onnx \
    --max-batch 8 --output dfine_m_slim.engine
```

After `freeze()`, the batch must also fit the frozen bound.

### TensorRT runtime/header version warning

The library was compiled against different TensorRT headers from the runtime library it loaded. Matching the major version is necessary but not a complete deployment guarantee. Rebuild `libdfine` and the engine against the installed TensorRT stack to remove ambiguity.

## Runtime behavior

### `full_pipeline_graph_active()` is false

Full-pipeline capture requires all of the following:

- `full_pipeline_graph=true`;
- FP32 engine outputs;
- an engine built with `--max-aux-streams 0`;
- `freeze()` with the intended batch and source bounds, either explicit or derived from the
  engine input dimensions;
- successful warmup and capture.

The detector remains usable: FP32 outputs use split GPU decode, while FP16 outputs use CPU decode. Check the warning log and rebuild the engine with `--max-aux-streams 0` when graph capture is required.

### A frame fails after `freeze()`

Frozen bounds are strict. A batch or source frame larger than the declared maximum is rejected instead of reallocating device memory. Create another detector and freeze it with the required maximum:

```cpp
detector.freeze(dfine::FreezeSpec{8, 3840, 2160, false});
```

Different dimensions within the bound use the split path when they do not match a full-graph capture exactly.

### mAP collapses after changing preprocessing

Published D-FINE weights use stretch resize and `/255` only. Do not add ImageNet mean/std. Letterbox is supported but costs approximately 1.7–2.0 AP on the published stretch-trained weights.

### FP16 detections differ across batch positions near `1e-3`

Small ULP-level score or box variation is expected from fully FP16 tactics and batch position. The `slim` recipe is gated by dataset mAP and box-aware tolerances, not universal bitwise identity. FP32 paths remain the strict parity reference.

### GPU decode scores differ by one ULP from CPU decode

GPU and host sigmoid implementations round differently. This is within the GPU-decode contract; compare decoded boxes, ordering, tolerances, and dataset metrics rather than raw bit patterns.

## Checkpoint export

### Exports differ byte-for-byte

Export serialization depends on the complete toolchain and source tree. Use the lockfile:

```sh
uv sync --frozen --extra gpu --extra torch
```

Compare the sidecars' `tool_versions`, `model_source`, checkpoint hash, opset, precision recipe, and export flags. A dirty D-FINE source checkout is recorded and should not produce a release artifact.

### `ModuleNotFoundError` during export

Install the checkpoint-export chain and point to a compatible D-FINE-seg checkout:

```sh
uv sync --frozen --extra gpu --extra torch
export DFINE_SEG_DIR=/path/to/D-FINE-seg
```

### Checkpoint tensors are missing or shape-mismatched

The exporter is strict. Select the correct `--model-name`, `--num-classes`, and class names. Do not use `--allow-partial-checkpoint` for deployment; it leaves unmatched tensors initialized outside the checkpoint and records the artifact as partial.

If the problem remains, open an issue with `dfine doctor`, the failing command, the complete error, and the ONNX/engine sidecar where relevant.
