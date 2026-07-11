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
# `pip install tensorrt-cu12==10.13.*`.
#
# Requires the `build` module: `pip install build`.
# `python -m build` runs in an isolated env by default (downloads setuptools
# from PyPI), so network access is needed.
set -euo pipefail
cd "$(dirname "$0")"  # python/

REPO=$(cd .. && pwd)
PYTHON=${PYTHON:-python3}
BUNDLED=dfine/libdfine.so
BUNDLED_SCRIPTS=dfine/_scripts
STAGED_LICENSE=LICENSE
STAGED_NOTICE=NOTICE
LICENSES_STAGED=0

for staged in "$STAGED_LICENSE" "$STAGED_NOTICE"; do
    [[ ! -e "$staged" ]] \
        || { echo "error: refusing to overwrite $PWD/$staged" >&2; exit 1; }
done

# The .so lives in dfine/ only for the duration of the build; remove it on any
# exit so the dev tree stays clean and stale copies can't leak into later wheels.
# Also drop setuptools' staging dirs (python/build, dfine.egg-info) — only
# dist/ is the product. NB: "build" here is python/build, not the C++ ../build.
cleanup() {
    rm -rf "$BUNDLED" "$BUNDLED_SCRIPTS" build dfine.egg-info "${RPATH_SCRIPT:-}"
    if [[ "$LICENSES_STAGED" == 1 ]]; then
        rm -f "$STAGED_LICENSE" "$STAGED_NOTICE"
    fi
}
trap cleanup EXIT

"$PYTHON" -c "import build" 2>/dev/null \
    || { echo "error: python module 'build' missing — run: $PYTHON -m pip install build" >&2; exit 1; }

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
LICENSES_STAGED=1
cp "$REPO/LICENSE" "$STAGED_LICENSE"
cp "$REPO/NOTICE" "$STAGED_NOTICE"

# setup.py declares a platform wheel because the bundled ELF is package data,
# not a Python extension. The wheel remains Python-ABI independent. manylinux
# is not applicable: CUDA and TensorRT are external to the manylinux policy.
VERSION=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml)
[[ -n "$VERSION" ]] || { echo "error: package version not found in pyproject.toml" >&2; exit 1; }
TARGET="dist/dfine-${VERSION}-py3-none-linux_$(uname -m).whl"
if [[ -f "$TARGET" ]]; then
    echo "WARNING: replacing existing $TARGET" >&2
    echo "  old sha256: $(sha256sum "$TARGET" | cut -d' ' -f1)" >&2
    rm -f "$TARGET"
fi
"$PYTHON" -m build --wheel --outdir dist .
WHEEL="$TARGET"
[[ -f "$WHEEL" ]] || { echo "error: expected wheel not produced: $WHEEL" >&2; exit 1; }
echo "  new sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)" >&2

echo
echo "wheel: $PWD/$WHEEL"
echo "install: pip install $PWD/$WHEEL  # runtime needs local TensorRT 10.x + CUDA 12, e.g. pip install 'tensorrt-cu12==10.13.*'"
