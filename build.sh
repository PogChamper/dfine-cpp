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

# --------------------------------------------------------------------------- #
# Preflight: the failures below otherwise surface as cmake walls of text.

# TensorRT HEADERS. The pip wheel (tensorrt-cu12) ships runtime .so's only;
# building needs NvInfer.h. Search the same places cmake/FindTensorRT.cmake
# will, and fail with the remedies instead of its "Could NOT find TensorRT".
trt_hdr=""
for d in "${TENSORRT_DIR:+$TENSORRT_DIR/include}" \
         /usr/local/TensorRT/include /opt/tensorrt/include \
         /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include; do
  [[ -n "$d" && -f "$d/NvInfer.h" ]] && { trt_hdr="$d"; break; }
done
if [[ -z "$trt_hdr" ]]; then
  cat >&2 <<'EOF'
error: TensorRT headers (NvInfer.h) not found. `pip install tensorrt-cu12`
provides the runtime libraries only — building needs the headers. Pick one:

  apt (NVIDIA CUDA repo; pin the WHOLE chain to your runtime, e.g. 10.13 —
  unpinned apt installs TensorRT 11):
    V="$(apt-cache madison libnvinfer-dev | grep -oPm1 '10\.13\.[0-9.]+-1\+cuda12\.[0-9]+')"
    sudo apt-get install -y "libnvinfer10=$V" "libnvinfer-headers-dev=$V" \
      "libnvinfer-dev=$V" "libnvinfer-plugin10=$V" "libnvinfer-headers-plugin-dev=$V" \
      "libnvinfer-plugin-dev=$V" "libnvonnxparsers10=$V" "libnvonnxparsers-dev=$V"

  container: nvcr.io/nvidia/tensorrt ships headers + libs preinstalled.

  no root: unpack a TensorRT GA tarball into third_party/tensorrt/{include,lib}
  (see third_party/README.md).
EOF
  exit 1
fi

# TensorRT 10 supports Turing (SM 7.5) and newer; a native probe on an older
# GPU wastes a full compile before failing at engine build.
if [[ "$CUDA_ARCH" == native ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "error: CUDA_ARCH=native probes the local GPU but nvidia-smi is missing;" >&2
    echo "  set CUDA_ARCH=<sm> for a cross build (86 Ampere, 89 Ada, 120 Blackwell)" >&2
    exit 1
  }
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
  if awk "BEGIN{exit !($cap < 7.5)}"; then
    echo "error: GPU compute capability $cap — TensorRT 10 needs Turing (7.5) or newer" >&2
    exit 1
  fi
fi

# A conda cross-g++ first on PATH compiles against conda's own sysroot; the
# system TensorRT/glibc include dirs then mix two libcs and die inside math.h.
# Prefer the system toolchain unless the caller pinned CC/CXX explicitly.
if [[ -z "${CC:-}${CXX:-}" && -x /usr/bin/g++ ]] \
   && command -v c++ 2>/dev/null | grep -q conda; then
  echo "[build] conda cross-g++ on PATH — using /usr/bin/g++ for host code (set CC/CXX to override)"
  export CC=/usr/bin/gcc CXX=/usr/bin/g++
fi

# Conda's nvcc activation injects -ccbin here (and a double activation leaks a
# literal 'UNSET' token nvcc dies on); the host-compiler choice belongs to the
# cmake flags below, so drop the injected value.
if [[ "${NVCC_PREPEND_FLAGS:-}" == *conda* || "${NVCC_PREPEND_FLAGS:-}" == *UNSET* ]]; then
  echo "[build] dropping conda-injected NVCC_PREPEND_FLAGS ('${NVCC_PREPEND_FLAGS}')"
  unset NVCC_PREPEND_FLAGS
fi
# --------------------------------------------------------------------------- #

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
