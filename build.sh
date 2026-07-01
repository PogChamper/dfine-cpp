#!/usr/bin/env bash
# One-command C++ build for D-FINE-cpp.
#
#   ./build.sh                      # configure + build (Release)
#   BUILD_TYPE=UBSAN ./build.sh     # sanitizer build (UBSAN | ASAN)
#   WERROR=ON ./build.sh            # warnings-as-errors
#   CUDA_ARCH=86 ./build.sh         # target a specific SM (default: native GPU)
#   JOBS=8 ./build.sh
#
# Toolchain discovery order: $CMAKE/$NVCC env override -> PATH -> the author's
# conda locations. TensorRT: $TENSORRT_DIR -> third_party/tensorrt (if populated,
# see third_party/README.md) -> system paths via cmake/FindTensorRT.cmake.
# Conda-based nvcc needs system binutils for the host link (the conda ld lacks
# GLIBC_PRIVATE symbols) — the wrapper + -B/usr/bin below are applied only then.
set -euo pipefail
cd "$(dirname "$0")"

find_tool() { # name, fallback...
  local name=$1; shift
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
  local c
  for c in "$@"; do [[ -x "$c" ]] && { echo "$c"; return; }; done
  echo ""
}

: "${CMAKE:=$(find_tool cmake /home/dxdxxd/miniconda3/envs/dfine/bin/cmake)}"
: "${NVCC:=$(find_tool nvcc /usr/local/cuda/bin/nvcc /home/dxdxxd/miniconda3/bin/nvcc)}"
[[ -n "$CMAKE" ]] || { echo "error: cmake not found on PATH (set \$CMAKE)"; exit 1; }
[[ -n "$NVCC"  ]] || { echo "error: nvcc not found on PATH (set \$NVCC)"; exit 1; }
: "${CUDA_ROOT:=$(dirname "$(dirname "$NVCC")")}"
: "${CUDA_ARCH:=native}"       # 'native' probes the local GPU; set e.g. 89 (Ada) for CI/cross builds
: "${BUILD_TYPE:=Release}"
: "${WERROR:=OFF}"
: "${JOBS:=$(nproc 2>/dev/null || echo 4)}"

# TensorRT: explicit dir > populated vendored dir > system (find-module fallback).
if [[ -z "${TENSORRT_DIR:-}" && -f "$PWD/third_party/tensorrt/include/NvInfer.h" ]]; then
  TENSORRT_DIR="$PWD/third_party/tensorrt"
fi

EXTRA=()
[[ -n "${TENSORRT_DIR:-}" ]] && EXTRA+=(-DTENSORRT_DIR="$TENSORRT_DIR")
# Conda toolchains ship an ld that breaks nvcc's host link; force system binutils.
if [[ "$NVCC" == *conda* || "$NVCC" == *miniconda* ]]; then
  EXTRA+=(-DCMAKE_CUDA_HOST_COMPILER="$PWD/cmake/cuda_host_ccbin.sh"
          -DCMAKE_CXX_FLAGS=-B/usr/bin -DCMAKE_C_FLAGS=-B/usr/bin)
fi

echo "[build] cmake=$CMAKE nvcc=$NVCC arch=$CUDA_ARCH type=$BUILD_TYPE werror=$WERROR trt=${TENSORRT_DIR:-system}"
"$CMAKE" -B build -S . \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DDFINE_WARNINGS_AS_ERRORS="$WERROR" \
  "${EXTRA[@]}"

echo "[build] compile (-j$JOBS)"
"$CMAKE" --build build -j"$JOBS"

echo
echo "[build] done. Binaries in ./build/. At runtime libnvinfer/libcudart must be on"
echo "  LD_LIBRARY_PATH (e.g. a 'pip install tensorrt' venv's tensorrt_libs dir, or a"
echo "  system TensorRT install)."
