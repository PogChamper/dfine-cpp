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
BUNDLED_SCRIPTS=dfine/_scripts

# The .so lives in dfine/ only for the duration of the build; remove it on any
# exit so the dev tree stays clean and stale copies can't leak into later wheels.
# Also drop setuptools' staging dirs (python/build, dfine.egg-info) — only
# dist/ is the product. NB: "build" here is python/build, not the C++ ../build.
cleanup() { rm -rf "$BUNDLED" "$BUNDLED_SCRIPTS" build dfine.egg-info "${RPATH_SCRIPT:-}"; }
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
chmod u+w "$BUNDLED"

# The dev build bakes an absolute RPATH (this machine's TensorRT/conda dirs).
# A wheel must not ship it: DT_RPATH is searched BEFORE LD_LIBRARY_PATH, so a
# path that happens to exist on the target machine would silently hijack
# TensorRT/CUDA resolution. Strip it from the bundled copy only — the dev-tree
# .so keeps its RPATH for local runs. (cmake for file(RPATH_REMOVE): PATH or
# $CMAKE, same override as ../build.sh.)
CMAKE=${CMAKE:-$(command -v cmake || true)}
[[ -n "$CMAKE" ]] || { echo "error: cmake not found (needed to strip the RPATH); set \$CMAKE" >&2; exit 1; }
RPATH_SCRIPT=$(mktemp)
printf '%s\n' "file(RPATH_REMOVE FILE \"\$ENV{DFINE_STRIP_LIB}\")" > "$RPATH_SCRIPT"
DFINE_STRIP_LIB="$PWD/$BUNDLED" "$CMAKE" -P "$RPATH_SCRIPT"
if readelf -d "$BUNDLED" | grep -qE 'RPATH|RUNPATH'; then
    echo "error: bundled libdfine.so still carries an RPATH/RUNPATH:" >&2
    readelf -d "$BUNDLED" | grep -E 'RPATH|RUNPATH' >&2
    exit 1
fi

# Snapshot the self-contained engine-build script so a wheel-only install can go
# release-ONNX -> .engine (`dfine build --onnx ...`) without a repo checkout.
# cli.py prefers the dev tree when present, so the snapshot never shadows it.
mkdir -p "$BUNDLED_SCRIPTS"
cp "$REPO/trt-files/scripts/build_engine.py" "$BUNDLED_SCRIPTS/"

"$PYTHON" -m build --wheel --outdir dist .

# setuptools tags the wheel py3-none-any because the .so is package-data, not
# an extension module. Retag to the platform the bundled ELF actually requires
# (`wheel tags` prints the new filename). manylinux is not possible: libdfine.so
# links CUDA/TensorRT libs that are non-redistributable and outside the
# manylinux policy.
# A same-named wheel already in dist/ (e.g. the tracked, gated release asset)
# is about to be replaced — make that loud, with both hashes, never silent.
PRE_TAG="dist/$(basename "$(ls dist/dfine-*-py3-none-any.whl)")"
TARGET="${PRE_TAG/py3-none-any/py3-none-linux_$(uname -m)}"
if [[ -f "$TARGET" ]]; then
    echo "WARNING: replacing existing $TARGET" >&2
    echo "  old sha256: $(sha256sum "$TARGET" | cut -d' ' -f1)" >&2
fi
WHEEL="dist/$("$PYTHON" -m wheel tags --platform-tag "linux_$(uname -m)" \
              --remove dist/dfine-*-py3-none-any.whl | tail -1)"
echo "  new sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)" >&2

echo
echo "wheel: $PWD/$WHEEL"
echo "install: pip install $PWD/$WHEEL  # runtime needs local TensorRT 10.x + CUDA 12, e.g. pip install 'tensorrt==10.13.*'"
