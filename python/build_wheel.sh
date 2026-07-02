#!/usr/bin/env bash
# Build the dfine Python wheel with libdfine.so bundled.
#
#   ./build_wheel.sh                # ../build.sh, then package the wheel
#   SKIP_BUILD=1 ./build_wheel.sh   # reuse an existing ../build/libdfine.so
#
# Env is passed through to ../build.sh (CUDA_ARCH, BUILD_TYPE, WERROR, JOBS,
# CMAKE, NVCC, TENSORRT_DIR). Note build.sh defaults CUDA_ARCH=native, which
# targets only the local GPU's SM — set CUDA_ARCH explicitly (e.g. 89) for a
# wheel meant to run elsewhere.
#
# The wheel bundles libdfine.so but NOT TensorRT/CUDA: the installing machine
# needs a TensorRT 10.x + CUDA 12 runtime on the loader path, e.g.
# `pip install tensorrt==10.13.*`.
#
# Requires the `build` module and `wheel` >= 0.40 (the `tags` subcommand):
# `pip install build 'wheel>=0.40'`.
# `python -m build` runs in an isolated env by default (downloads setuptools
# from PyPI), so network access is needed.
set -euo pipefail
cd "$(dirname "$0")"  # python/

REPO=$(cd .. && pwd)
PYTHON=${PYTHON:-python3}
BUNDLED=dfine/libdfine.so

# The .so lives in dfine/ only for the duration of the build; remove it on any
# exit so the dev tree stays clean and stale copies can't leak into later wheels.
# Also drop setuptools' staging dirs (python/build, dfine.egg-info) — only
# dist/ is the product. NB: "build" here is python/build, not the C++ ../build.
cleanup() { rm -rf "$BUNDLED" build dfine.egg-info; }
trap cleanup EXIT

"$PYTHON" -c "import build" 2>/dev/null \
    || { echo "error: python module 'build' missing — run: $PYTHON -m pip install build" >&2; exit 1; }
"$PYTHON" -c "import wheel" 2>/dev/null \
    || { echo "error: python module 'wheel' missing — run: $PYTHON -m pip install wheel" >&2; exit 1; }

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    "$REPO/build.sh"
fi
[[ -f "$REPO/build/libdfine.so" ]] \
    || { echo "error: $REPO/build/libdfine.so not found (run ../build.sh)" >&2; exit 1; }

# Dereference (cp -L): build/libdfine.so -> .so.<abi> -> .so.<version> is a symlink
# chain, wheels (zip) cannot hold symlinks, and _ffi.py loads the literal name
# libdfine.so. One real file avoids shipping three copies.
cp -L "$REPO/build/libdfine.so" "$BUNDLED"

"$PYTHON" -m build --wheel --outdir dist .

# setuptools tags the wheel py3-none-any because the .so is package-data, not
# an extension module. Retag to the platform the bundled ELF actually requires
# (`wheel tags` prints the new filename). manylinux is not possible: libdfine.so
# links CUDA/TensorRT libs that are non-redistributable and outside the
# manylinux policy.
WHEEL="dist/$("$PYTHON" -m wheel tags --platform-tag "linux_$(uname -m)" \
              --remove dist/dfine-*-py3-none-any.whl | tail -1)"

echo
echo "wheel: $PWD/$WHEEL"
echo "install: pip install $PWD/$WHEEL  # runtime needs local TensorRT 10.x + CUDA 12, e.g. pip install 'tensorrt==10.13.*'"
