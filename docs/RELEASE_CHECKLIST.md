# Release checklist

Run the gates in order on the release commit. The GPU gates remain release-machine checks until a
GPU runner is available. Publish the validated bytes; do not rebuild between validation and upload.

## Environment

```sh
export DFINE_SEG_DIR="${DFINE_SEG_DIR:?set DFINE_SEG_DIR to the tested D-FINE-seg checkout}"
export TRTLIB="${TRTLIB:?set TRTLIB to the directory containing libnvinfer.so.10}"
export ENGINE="${ENGINE:-trt-files/engines/dfine_m_slim.engine}"
export ENGINE_G0="${ENGINE_G0:-trt-files/engines/dfine_m_slim_g0.engine}"
export KNOWN_IMAGE="${KNOWN_IMAGE:?set KNOWN_IMAGE to a recorded COCO image}"
export FOOD_CHECKPOINT="${FOOD_CHECKPOINT:?set FOOD_CHECKPOINT to the 3-class checkpoint}"
export FOOD_CLASSES="${FOOD_CLASSES:-food,plate,tray}"
export FOOD_IMAGE="${FOOD_IMAGE:?set FOOD_IMAGE to the recorded food image}"
export RELEASE_DIR="${RELEASE_DIR:-/tmp/dfine-release}"
export MODEL_DIR="${MODEL_DIR:-$RELEASE_DIR/models}"
test ! -e "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR" "$MODEL_DIR"

export TRTLIB="$(realpath "$TRTLIB")"
export ENGINE="$(realpath "$ENGINE")"
export ENGINE_G0="$(realpath "$ENGINE_G0")"
export KNOWN_IMAGE="$(realpath "$KNOWN_IMAGE")"
export FOOD_CHECKPOINT="$(realpath "$FOOD_CHECKPOINT")"
export FOOD_IMAGE="$(realpath "$FOOD_IMAGE")"
export RELEASE_DIR="$(realpath "$RELEASE_DIR")"
export MODEL_DIR="$(realpath "$MODEL_DIR")"

test "$(git -C "$DFINE_SEG_DIR" rev-parse HEAD)" = "$(cat trt-files/DFINE_SEG_REVISION)"
test -z "$(git -C "$DFINE_SEG_DIR" status --porcelain)"
```

The revision file identifies the tested model source. A dirty or different checkout is not a
release input.

## 1. Finalize the release commit

Set the release version in all six sources, archive the current unreleased note as
`docs/releases/v$VERSION.md`, restore a fresh `docs/releases/UNRELEASED.md` for the next cycle, and
update active latest-release links before any build or gate below. Preserve explicitly historical
validation rows.

```sh
export VERSION="${VERSION:?set VERSION to the release version without the v prefix}"
test "$(sed -n 's/^project(dfine VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)" = "$VERSION"
test "$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)" = "$VERSION"
test "$(sed -n 's/^version = "\([^"]*\)"/\1/p' python/pyproject.toml)" = "$VERSION"
test "$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' python/dfine/__init__.py)" = "$VERSION"
test "$(sed -n 's/^[[:space:]]*return "\([^"]*\)";/\1/p' include/dfine/version.hpp)" = "$VERSION"
test "$(python -c 'import tomllib; d=tomllib.load(open("uv.lock", "rb")); print(next(p["version"] for p in d["package"] if p["name"] == "dfine-cpp-tools"))')" = "$VERSION"
test "$(PYTHONPATH="$PWD/python" python -m dfine.cli --version)" = "dfine $VERSION"
rg 'v0\.3\.3|0\.3\.3' README.md docs python/README.md examples
git diff --check
```

The release note must enumerate observable behavior and compatibility changes without relabeling an
unchanged model recipe. Review every old-version match and retain only explicitly historical
validation records. Commit these edits, then verify the release checkout is clean:

```sh
test -z "$(git status --porcelain)"
```

From this point through upload, do not modify source or rebuild validated artifacts.

## 2. Hosted gates

- [ ] `lint`, `compile-cuda`, `install-consumer`, `python-nogpu`, and `wheel` are green on the
      release commit.
- [ ] The wheel job verifies the bundled library and build script, absent RPATH/RUNPATH,
      `libdfine.so.1` SONAME, `Root-Is-Purelib: false`, LICENSE/NOTICE, and import outside the
      checkout.
- [ ] `CUDA_ARCH=89 WERROR=ON ./build.sh` is clean with the release toolchain and produces the
      native library that will be packaged.

## 3. CPU gates

```sh
ctest --test-dir build --output-on-failure
PYTHONPATH="$PWD/python" python -m pytest "$PWD/python/tests" -q
./build/tests/dfine_test_engine_meta trt-files/onnx/*.json trt-files/engines/*.json
```

- [ ] All checked sidecars parse and reject contradictory engine contracts.
- [ ] The out-of-tree CMake consumer links the installed package, not the source tree.

## 4. GPU runtime gates

```sh
LD_LIBRARY_PATH="$TRTLIB" DFINE_TEST_ENGINE="$ENGINE" \
    ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH="$TRTLIB" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_ENGINE_G0="$ENGINE_G0" \
    DFINE_TEST_REQUIRE_FULL_GRAPH=1 ./build/tests/dfine_test_detector_recovery
LD_LIBRARY_PATH="$TRTLIB" DFINE_TEST_ENGINE="$ENGINE" \
    compute-sanitizer --tool memcheck --error-exitcode 99 \
    ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH="$TRTLIB" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_ENGINE_G0="$ENGINE_G0" \
    DFINE_TEST_REQUIRE_FULL_GRAPH=1 compute-sanitizer --tool memcheck --error-exitcode 99 \
    ./build/tests/dfine_test_detector_recovery
python trt-files/scripts/verify_engine.py --engine "$ENGINE" --batches 1 2 8
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" \
    LD_LIBRARY_PATH="$TRTLIB" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_IMAGE="$KNOWN_IMAGE" \
    python -m pytest python/tests -q -ra
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" LD_LIBRARY_PATH="$TRTLIB" \
    python -m dfine.cli predict --engine "$ENGINE" --image "$KNOWN_IMAGE" --json \
    > "$RELEASE_DIR/native-detections.json"
```

- [ ] Recovery covers ordinary, GPU-decode, CUDA Graph, and required full-pipeline-graph paths.
- [ ] Repeated shape transitions and teardown report zero compute-sanitizer errors.
- [ ] Batches 1, 2, and 8 execute against the declared engine profile.
- [ ] The Python/C++ parity test runs against the recorded image; it is not skipped.

## 5. Official-model provenance and accuracy

```sh
PYTHONPATH="$PWD/python" python -m dfine.cli export \
    --model m --precision fp16 --output "$MODEL_DIR/dfine_m_slim.onnx"
PYTHONPATH="$PWD/python" python -m dfine.cli build \
    --model m --precision fp16 --onnx "$MODEL_DIR/dfine_m_slim.onnx" \
    --output "$RELEASE_DIR/dfine_m_slim.engine"
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" LD_LIBRARY_PATH="$TRTLIB" \
    python -m dfine.cli predict --engine "$RELEASE_DIR/dfine_m_slim.engine" \
    --image "$KNOWN_IMAGE" --json
python trt-files/scripts/verify_engine.py \
    --engine "$RELEASE_DIR/dfine_m_slim.engine" --batches 1 2 8
```

- [ ] The ONNX sidecar reports `checkpoint_load: strict`, the commit in
      `trt-files/DFINE_SEG_REVISION`, `model_source.dirty: false`, the checkpoint SHA-256, and
      complete `tool_versions`. `exporter_sha256` is a 64-character lowercase SHA-256 and
      `onnx_simplification` records `applied` for the locked release recipe. A slim sidecar also
      records `source_onnx_sha256`, `converter_sha256`, and the `onnxconverter-common` version.
      The exporter/converter hashes and official checkpoint hash must match the release sources.
- [ ] The engine sidecar reports `precision: fp16`,
      `precision_mode: strongly_typed_onnx_fp16_surgical_slim`, the actual TensorRT and SM target,
      the 1/2/8-compatible profile, and the ONNX SHA-256.
- [ ] Python and C++ return equivalent detections on the recorded image.

The validated D-FINE-M slim graph SHA-256 is
`0f0b8e9ecafa3112d3f7d983e52809c92514836ee1328b519fe81fe25abc7419`. If the ONNX bytes differ,
run the full COCO D-FINE-M gate before release:

```sh
python trt-files/scripts/profile.py --backends trt cpp \
    --engine "$RELEASE_DIR/dfine_m_slim.engine" --full
```

An unchanged graph may reuse the recorded accuracy result even when its sidecar gains new
provenance fields.

## 6. Custom-model gate

```sh
PYTHONPATH="$PWD/python" python -m dfine.cli export \
    --model s --checkpoint "$FOOD_CHECKPOINT" --num-classes 3 \
    --class-names "$FOOD_CLASSES" --precision fp16 --output "$RELEASE_DIR/food_s_slim.onnx"
PYTHONPATH="$PWD/python" python -m dfine.cli build \
    --model s --onnx "$RELEASE_DIR/food_s_slim.onnx" \
    --output "$RELEASE_DIR/food_s.engine"
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" LD_LIBRARY_PATH="$TRTLIB" \
    python -m dfine.cli predict --engine "$RELEASE_DIR/food_s.engine" --image "$FOOD_IMAGE"
```

- [ ] Strict checkpoint load covers every checkpoint-owned detection tensor; the sidecar carries
      the three class names and source provenance.
- [ ] Omitting `--num-classes` fails with the class-count diagnostic.

## 7. Five-size static gate

For n/s/m/l/x, require strict checkpoint load, zero `GridSample` nodes, symbolic batch, successful
ONNX Runtime execution at N=1 and N=2, and a clean `onnx.checker` result. A missing ONNX Runtime or
skipped behavioral postcondition fails the gate. Each published sidecar must pass the source,
exporter-hash, tool-version, and simplification checks from the D-FINE-M gate. Run the five-size full
COCO campaign only when the graph or precision recipe changes.

Generate the remaining pairs directly in the publication input directory; the M pair from the
previous gate stays untouched:

```sh
for size in n s l x; do
    PYTHONPATH="$PWD/python" python -m dfine.cli export \
        --model "$size" --precision fp16 \
        --output "$MODEL_DIR/dfine_${size}_slim.onnx"
done
```

## 8. Build and smoke the wheel

```sh
SKIP_BUILD=1 ./python/build_wheel.sh
export WHEEL="python/dist/dfine-$VERSION-py3-none-linux_x86_64.whl"

test ! -e "$RELEASE_DIR/wheel-smoke"
python -m venv "$RELEASE_DIR/wheel-smoke"
"$RELEASE_DIR/wheel-smoke/bin/python" -m pip install "$WHEEL" "tensorrt-cu12==10.13.*" "pillow>=9"
export WHEEL_TRTLIB="$("$RELEASE_DIR/wheel-smoke/bin/python" -c \
    'import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))')"
(cd /tmp && LD_LIBRARY_PATH="$WHEEL_TRTLIB" \
    "$RELEASE_DIR/wheel-smoke/bin/python" -c 'import dfine
from dfine.cli import _build_engine_script
assert dfine.library_version() == dfine.__version__
assert _build_engine_script().is_file()')
(cd /tmp && LD_LIBRARY_PATH="$WHEEL_TRTLIB" "$RELEASE_DIR/wheel-smoke/bin/dfine" doctor)
(cd /tmp && LD_LIBRARY_PATH="$WHEEL_TRTLIB" "$RELEASE_DIR/wheel-smoke/bin/dfine" predict \
    --engine "$ENGINE" --image "$KNOWN_IMAGE" --json \
    > "$RELEASE_DIR/wheel-detections.json")
cmp "$RELEASE_DIR/native-detections.json" "$RELEASE_DIR/wheel-detections.json"
```

- [ ] The wheel contains `dfine/libdfine.so`, `dfine/_scripts/build_engine.py`, LICENSE, and NOTICE.
- [ ] Its WHEEL metadata contains `Root-Is-Purelib: false`; the native library has no build-machine
      RPATH/RUNPATH and retains the `libdfine.so.1` SONAME.
- [ ] In a fresh environment outside the checkout, `dfine.library_version() ==
      dfine.__version__`, `dfine doctor` selects the bundled library, and `dfine build` resolves the
      bundled build script. The bundled library reproduces the native detection result.

## 9. Stage and publish

- [ ] `$MODEL_DIR` contains exactly the gated n/s/m/l/x FP32 and slim ONNX/JSON pairs.

```sh
python trt-files/scripts/release_assets.py assemble \
    --input "$MODEL_DIR" --wheel "$WHEEL" --version "$VERSION" --out "$RELEASE_DIR/upload"
```

- [ ] `assemble` requires a new or empty output directory, accepts exactly 20 model files, validates
      graph/sidecar pairing, canonical model facts, provenance, precision, opset 19, and each slim
      graph's FP32 source hash, then writes `SHA256SUMS` for the 20 model files and wheel.
- [ ] Tag the release commit and upload the contents of `$RELEASE_DIR/upload` without rebuilding.

```sh
python trt-files/scripts/release_assets.py verify --tag "v$VERSION"
```

- [ ] Verification downloads every published asset, checks every digest, and reports no uncovered
      or missing files.
