# syntax=docker/dockerfile:1
#
# D-FINE-cpp — build + run the TensorRT C++ inference binaries.
#
# This is intentionally single-stage (build tools stay in the final image).
# A multi-stage split (builder -> slim runtime copying only build/ + the .so's)
# would shrink the final image, but is left as a follow-up since it's optional
# per the task and would need to duplicate the TensorRT runtime .so's into the
# runtime stage anyway (they aren't a separate apt package from the -devel bits
# in this base image).
#
# Base: CUDA 12.9 devel (nvcc + the NVIDIA apt repo); TensorRT 10.13 is
# installed below as a pinned apt chain — the same recipe CI and the docs use.
# The NGC tensorrt images that carry TRT 10.13 are all CUDA 13, and this stack
# stays on CUDA 12 (the wheel and every validated engine link libcudart.so.12).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# The whole chain carries one explicit version: apt does not down-resolve
# dependencies to a pin, and unpinned it installs TensorRT 11.
ARG TRT_DEB_VERSION=10.13.3.9-1+cuda12.9
RUN apt-get update && apt-get install -y --no-install-recommends \
      "libnvinfer10=${TRT_DEB_VERSION}" \
      "libnvinfer-headers-dev=${TRT_DEB_VERSION}" \
      "libnvinfer-dev=${TRT_DEB_VERSION}" \
      "libnvinfer-plugin10=${TRT_DEB_VERSION}" \
      "libnvinfer-headers-plugin-dev=${TRT_DEB_VERSION}" \
      "libnvinfer-plugin-dev=${TRT_DEB_VERSION}" \
      "libnvonnxparsers10=${TRT_DEB_VERSION}" \
      "libnvonnxparsers-dev=${TRT_DEB_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# GPU architecture for CMAKE_CUDA_ARCHITECTURES. 89 = Ada (RTX 40-series / L4),
# matching this repo's validated dev box (RTX 4070 Ti SUPER) and CMakeLists.txt's
# own default. Override at build time for other GPUs, e.g.:
#   docker build --build-arg CUDA_ARCH=86 .   # Ampere (RTX 30-series / A10)
#   docker build --build-arg CUDA_ARCH=75 .   # Turing (T4 / RTX 20-series)
ARG CUDA_ARCH=89

# ---------------------------------------------------------------------------
# Toolchain deps. The base image already provides nvcc + TensorRT dev headers
# and libraries, so only cmake (>=3.20, per this repo's cmake_minimum_required)
# and a host compiler are missing. Ubuntu 22.04 (jammy)'s apt cmake is 3.22,
# which satisfies >=3.20 — no Kitware apt repo needed.
#
# nlohmann-json3-dev is installed so CMakeLists.txt's `find_path(... nlohmann/json.hpp)`
# fallback finds a real system header (/usr/include/nlohmann/json.hpp) instead of
# falling through to its FetchContent path, which would clone from GitHub at
# `docker build` time and silently require network access mid-build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      build-essential \
      nlohmann-json3-dev \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/D-FINE-cpp

# Respect .dockerignore: build/, .git, trt-files/{onnx,engines},
# __pycache__ are excluded (the engines/onnx dirs alone are ~2 GB on the authoring
# host and are build outputs anyway — see limitation (c) below).
COPY . .

# ---------------------------------------------------------------------------
# Configure + build. Flags mirror build.sh's cmake invocation as closely as
# this environment allows:
#
#   -DTENSORRT_DIR=<repo>/third_party/tensorrt
#     Points at the repo's vendored TensorRT headers (real files: TensorRT 10.13
#     public headers) for consistency with the rest of the toolchain description
#     in docs/HANDOFF.md/build.sh, per the task's instruction to keep this flag
#     even though the base image ships its own TensorRT dev headers too.
#
#     HONEST LIMITATION: third_party/tensorrt/lib/*.so in this repo are symlinks
#     pointing OUTSIDE the repo, at the authoring host's
#     ../../../D-FINE-seg/.venv/lib/python3.11/site-packages/tensorrt_libs/*.so.10
#     — a path that does not exist in this image (and .dockerignore/.gitignore
#     wouldn't help even if it did, since it's a different repo entirely). Those
#     symlinks therefore arrive broken. CMake's find_library() (see
#     cmake/FindTensorRT.cmake) does not fail on that though — it silently keeps
#     searching its other hint dirs (/usr/local/TensorRT, /opt/tensorrt, /usr)
#     and resolves nvinfer/nvinfer_plugin/nvonnxparser to the libraries THIS BASE
#     IMAGE ships instead. Net effect: you get the repo's vendored 10.13 headers
#     compiled against whatever TensorRT minor version nvcr.io/nvidia/tensorrt:
#     24.10-py3 actually bundles (may or may not be 10.13 — not verified here).
#     TensorRT's ABI is normally stable across patch releases, but if you hit an
#     unresolved-symbol link error with a version suffix baked into the symbol
#     name (e.g. createInferBuilder_INTERNAL_10013), that's this headers/lib
#     skew. Fix by either (a) picking a base image tag known to ship 10.13, or
#     (b) passing --build-arg-equivalent / editing this file to use
#     -DTENSORRT_DIR=/usr so headers and libs both come from the base image.
#
#   -DCUDAToolkit_ROOT=/usr/local/cuda
#     Where NGC images install the CUDA toolkit (build.sh instead pointed this
#     at a miniconda env, since that host's CUDA came from conda, not apt/NGC).
#
#   -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
#     From the build ARG above (default 89).
#
#   -DCMAKE_CXX_FLAGS=-B/usr/bin
#     Carried over from build.sh for parity only. On the authoring host this
#     forced nvcc's host-link step to use system binutils instead of a conda
#     env's glibc-incompatible `ld` (see docs/HANDOFF.md's `__nptl_change_stack_
#     perm@GLIBC_PRIVATE` gotcha). There is no conda environment shadowing `ld`
#     inside this container — gcc/ld here are already the matched distro pair —
#     so this flag is expected to be a harmless no-op, not a required fix.
#
# Deliberately NOT passed: -DCMAKE_CUDA_HOST_COMPILER=cmake/cuda_host_ccbin.sh.
# That wrapper exists solely to route around the conda-vs-system-ld mismatch
# above and has nothing to fix in this image's toolchain.
RUN cmake -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DTENSORRT_DIR="$(pwd)/third_party/tensorrt" \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DCMAKE_CXX_FLAGS=-B/usr/bin \
    && cmake --build build -j"$(nproc)"

# Binaries land in ./build (dfine_inspect, dfine_detect, dfine_build, dfine_bench,
# dfine_coco_eval, dfine_smoke) alongside libdfine.so.
ENV PATH="/workspace/D-FINE-cpp/build:${PATH}"
# libdfine.so lives in build/ too; the apps link it via an RPATH set by CMake's
# default settings on most distros, but this is added defensively in case that
# isn't picked up in your derived image.
ENV LD_LIBRARY_PATH="/workspace/D-FINE-cpp/build:${LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# Runtime limitations — read before `docker run`:
#
#  (a) GPU + driver. `docker run --gpus all ...` (NVIDIA Container Toolkit) is
#      required, plus a host NVIDIA driver new enough for the CUDA/TensorRT
#      versions baked in above. Nothing here substitutes for that: there is no
#      GPU inside a container *build*, so only the CPU-side compile/link was
#      exercised while producing this image — none of the binaries have been
#      run against a real device as part of `docker build`.
#
#  (b) The WSL-specific libcuda path in docs/HANDOFF.md
#      (`/usr/lib/wsl/lib` for onnxruntime-gpu's `cuda_env.bootstrap()`) does
#      NOT apply inside a proper Linux container. That path is a WSL2
#      translation-layer artifact for the authoring host; here, `--gpus all`
#      has the NVIDIA Container Toolkit inject the host's real libcuda.so
#      directly, and this image's own base already provides the matching CUDA
#      runtime libs — no extra LD_LIBRARY_PATH wrangling needed for that part.
#
#  (c) ONNX export is NOT available in this image. trt-files/scripts/
#      export_dfine_onnx.py, build_engine.py, convert_fp16.py, etc. all depend
#      on the sibling D-FINE-seg python package/venv (PyTorch, the D-FINE model
#      code, onnxruntime-gpu) — a different repo, not copied into this build
#      context and not installed here. This image only builds+runs the C++
#      CONSUMER of an already-built .engine file. To export/build engines from
#      inside a container, bind-mount D-FINE-seg and install its deps
#      separately; trt-files/onnx and trt-files/engines (build outputs, ~2 GB
#      combined on the authoring host) are excluded from this build's context
#      via .dockerignore for exactly this reason.
#
# Supply a real engine at runtime via a bind mount, e.g.:
#   docker build --build-arg CUDA_ARCH=89 -t dfine-cpp .
#   docker run --rm --gpus all -v /host/path/to/engines:/engines dfine-cpp \
#     dfine_inspect /engines/dfine_m_fp32.engine
#
# The default CMD below is a smoke-test, not the intended real usage: dfine_inspect
# has no actual --help flag — apps/dfine_inspect.cpp treats argv[1] as an engine
# path unconditionally whenever argc==2 (see the source). Running it with
# "--help" therefore (1) prints the loaded TensorRT runtime + header version
# first — proving the binary links and the container's TensorRT/CUDA are wired
# up — and then (2) fails with "error: cannot open --help" / exit code 1,
# because it genuinely tries to open a file literally named "--help". That
# failure is expected and left as-is (not papered over) so the container's
# default behavior stays honest about what it is: a build/link smoke-test, not
# a working CLI invocation. Pass a real engine path (per the docker run example
# above) for actual inspection.
CMD ["dfine_inspect", "--help"]
