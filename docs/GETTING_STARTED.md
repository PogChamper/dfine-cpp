# Getting started

D-FINE-cpp has two installation paths. Both consume the same ONNX artifacts and produce the same target-local TensorRT engine.

| Path | Use it when | Requirements |
|---|---|---|
| Release wheel | Python or CLI on Ada or Blackwell Linux x86_64 GPUs | CUDA 12, TensorRT 10.13 |
| Source build | C++ integration, Turing/Ampere, custom CUDA arch, or development | CUDA toolkit, TensorRT headers and libraries, CMake, C++17 compiler |

TensorRT engines are build outputs, not portable model files. Build each engine on the target TensorRT/GPU stack from a released or locally exported ONNX artifact.

## Release wheel

The latest published wheel is v0.3.3. It contains the Python package, CLI, `libdfine.so`, and the engine builder. The native library targets `sm_89` with forward PTX; it is validated on Ada and Blackwell. Turing and Ampere use the [source build](#source-build). Source-tree changes awaiting the next tag are listed under [Unreleased](releases/UNRELEASED.md).

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install "dfine[cli,tensorrt] @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine-0.3.3-py3-none-linux_x86_64.whl"
```

Download one ONNX artifact: the graph and its sidecar are a pair.

```sh
curl -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.onnx \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.json
dfine doctor
dfine build --model m --onnx dfine_m_slim.onnx --output dfine_m_slim.engine
dfine predict --engine dfine_m_slim.engine --image image.jpg --out result.jpg
```

`image.jpg` may be any JPEG or PNG. Engine compilation normally takes 1–3 minutes. Subsequent inference loads `dfine_m_slim.engine` directly.

Verify the downloaded pair against the release manifest when the artifacts enter a build or release pipeline:

```sh
curl -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/SHA256SUMS
grep -E '  dfine_m_slim\.(onnx|json)$' SHA256SUMS | sha256sum -c -
```

## Source build

### Prerequisites

| Dependency | Supported version | Used for |
|---|---|---|
| NVIDIA driver | CUDA-12-capable | Build and runtime |
| CUDA toolkit | 12.x | Native build |
| TensorRT | 10.x; validated on 10.13 | Native build, engine build, runtime |
| CMake | ≥3.24 for `native`; ≥3.20 with an explicit arch | Native build |
| C++ compiler | C++17; supported by the installed CUDA toolkit | Native build |
| Python | ≥3.9 for bindings; ≥3.11 for the locked conversion tools | Tooling and bindings |

The native build needs TensorRT headers as well as runtime libraries. `tensorrt-cu12` from pip supplies the Python builder and runtime libraries but not the C++ headers. `./build.sh` detects missing headers and prints the supported apt, tarball, and container routes. See [Troubleshooting](TROUBLESHOOTING.md#building-from-source) for exact packages.

### Build and test

```sh
git clone https://github.com/PogChamper/dfine-cpp
cd dfine-cpp
./build.sh                         # probes the local GPU
# CUDA_ARCH=86 ./build.sh          # explicit Ampere build
ctest --test-dir build --output-on-failure
```

`CUDA_ARCH` accepts one CMake architecture or a semicolon-separated list. Useful values are `75` (Turing), `86` (Ampere), `87` (Jetson Orin), `89` (Ada), and `120` (Blackwell; CUDA ≥12.8).

Plain CMake is equivalent:

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

Set `TENSORRT_DIR` when TensorRT is outside the system paths:

```sh
TENSORRT_DIR=/opt/tensorrt ./build.sh
```

The lookup order is explicit `TENSORRT_DIR`, `third_party/tensorrt`, then system installations.

### Build an engine

Install the TensorRT Python package into the environment used for the builder, then compile the released typed ONNX:

```sh
python -m pip install "tensorrt-cu12==10.13.*"
python trt-files/scripts/build_engine.py \
    --onnx dfine_m_slim.onnx \
    --output dfine_m_slim.engine \
    --strongly-typed --no-tf32 --max-batch 8
```

The default optimization batch is 1. Add `--opt-batch 8` for sustained batch-8 throughput; this trades some batch-1 latency for higher batch throughput.

Run the native app:

```sh
./build/dfine_detect --engine dfine_m_slim.engine \
    --image image.jpg --threshold 0.5
```

If TensorRT lives in a Python environment, its library directory may need to be visible at runtime:

```sh
export LD_LIBRARY_PATH="$(python -c 'import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))'):${LD_LIBRARY_PATH}"
```

## Use the C++ package

Install the build into a user-writable prefix:

```sh
cmake --install build --prefix "$HOME/.local"
```

Consume it from another CMake project:

```cmake
find_package(dfine CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE dfine::dfine)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH="$HOME/.local"` when the prefix is not searched by default. [`examples/consumer`](../examples/consumer/CMakeLists.txt) is the complete out-of-tree example used by CI. Runtime behavior and API contracts are in [Runtime](RUNTIME.md).

## Use the Python package from a source build

```sh
python -m pip install -e "python[cli]"
export DFINE_LIBRARY="$PWD/build/libdfine.so"
dfine doctor
```

The loader checks, in order, `DFINE_LIBRARY`, the library bundled with an installed wheel, and `<repo>/build/libdfine.so`. An explicit invalid `DFINE_LIBRARY` is an error; it never silently selects another library.

```python
from dfine import Detector

with Detector("dfine_m_slim.engine", threshold=0.5) as detector:
    detections = detector.detect(rgb_hwc_uint8)
```

`rgb_hwc_uint8` is a contiguous NumPy HWC `uint8` array. Pass `is_bgr=True` for BGR input. See the [Python reference](../python/README.md).

## CLI map

| Command | Input | Operation |
|---|---|---|
| `dfine doctor` | Environment | Diagnose library, GPU, TensorRT, and headers |
| `dfine build` | ONNX artifact | Build and cache or write an engine |
| `dfine predict` | Engine + image | Detect and optionally draw or emit JSON |
| `dfine info` | Engine | Print model and shape facts |
| `dfine bench` | Engine | Run the native benchmark binary |
| `dfine export` | Checkpoint + D-FINE source | Export an ONNX artifact from a source checkout |

`predict` may build from an explicit `--onnx` when no matching engine exists. `info` and `bench` never build. `predict --json` reserves stdout for JSON and sends diagnostics to stderr. Engines cached under `~/.cache/dfine` (or `DFINE_CACHE`) are keyed by artifact fingerprint, profile, GPU architecture, and TensorRT version. Exporting a checkpoint requires a repository checkout and a compatible D-FINE source tree; see [Conversion](CONVERSION.md).

Native source builds also provide:

| Binary | Purpose |
|---|---|
| `dfine_detect` | Single-image inference |
| `dfine_coco_eval` | COCO evaluation through the C++ runtime |
| `dfine_bench` | Batch, stage-timing, memory, and graph comparisons |
| `dfine_inspect` | TensorRT engine inspection |
| `dfine_smoke` | Engine load and inference smoke test |
| `dfine_build` | Pure-C++ FP32 engine build; typed FP16 uses the Python builder |

## Docker

The Dockerfile builds the native library and apps against pinned TensorRT 10.13 packages. It is a development image, not an ONNX export environment; mount an engine and image at runtime.

```sh
docker build --build-arg CUDA_ARCH=89 -t dfine-cpp .
docker run --rm --gpus all -v "$PWD:/data" dfine-cpp \
    dfine_detect --engine /data/dfine_m_slim.engine --image /data/image.jpg
```

## Platform status

| Platform or GPU | Status | Installation |
|---|---|---|
| Ubuntu 22.04 x86_64 | Validated | Wheel on Ada/Blackwell; source otherwise |
| WSL2 Ubuntu guest | Validated | Same as Linux |
| Ampere (`sm_86`) | Validated | Source build |
| Ada (`sm_89`) | Validated | Wheel or source |
| Blackwell (`sm_120`) | Validated | Wheel through PTX or source with CUDA ≥12.8 |
| Turing (`sm_75`) | Expected; not yet validated | Source build |
| Jetson Orin (`sm_87`) | Expected; not yet validated | JetPack TensorRT and source build |
| Windows native | Not supported | Use WSL2 |
| macOS | Not applicable | CUDA/TensorRT unavailable |

Measured compatibility reports are in [Validation](VALIDATION.md). Run `dfine doctor` first when an installation fails, then follow [Troubleshooting](TROUBLESHOOTING.md).
