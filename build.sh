#!/usr/bin/env bash
# One-command C++ build for D-FINE-cpp. Bakes in the toolchain gotchas from
# docs/HANDOFF.md (conda cmake, vendored TRT headers, nvcc host-compiler wrapper that
# forces system binutils, Ada sm_89). Override the knobs below via env vars.
#
#   ./build.sh                      # configure + build (Release)
#   BUILD_TYPE=UBSAN ./build.sh     # sanitizer build
#   WERROR=ON ./build.sh            # warnings-as-errors
#   JOBS=8 ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

: "${CMAKE:=/home/dxdxxd/miniconda3/envs/dfine/bin/cmake}"
: "${NVCC:=/home/dxdxxd/miniconda3/bin/nvcc}"
: "${CUDA_ROOT:=/home/dxdxxd/miniconda3}"
: "${CUDA_ARCH:=89}"
: "${BUILD_TYPE:=Release}"
: "${WERROR:=OFF}"
: "${JOBS:=4}"
TP="$PWD/third_party/tensorrt"

if [[ ! -f "$CMAKE" ]]; then echo "cmake not found at $CMAKE (set \$CMAKE)"; exit 1; fi

echo "[build] configure (type=$BUILD_TYPE arch=$CUDA_ARCH werror=$WERROR)"
"$CMAKE" -B build -S . \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DTENSORRT_DIR="$TP" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_HOST_COMPILER="$PWD/cmake/cuda_host_ccbin.sh" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CXX_FLAGS=-B/usr/bin \
  -DDFINE_WARNINGS_AS_ERRORS="$WERROR"

echo "[build] compile (-j$JOBS)"
"$CMAKE" --build build -j"$JOBS"

echo
echo "[build] done. Binaries in ./build/ . Runtime needs on LD_LIBRARY_PATH:"
echo "  <D-FINE-seg>/.venv/lib/python3.11/site-packages/tensorrt_libs : $CUDA_ROOT/lib"
