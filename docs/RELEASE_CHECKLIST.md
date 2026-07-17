# Release checklist

Run every shell block in order in one Bash session; the first enables fail-fast execution. The GPU
gates remain release-machine checks until a GPU runner is available. Publish the validated bytes;
do not rebuild between validation and upload.

## Environment

```sh
set -euo pipefail

uv sync --frozen --extra gpu --extra torch --group release
PY="$PWD/.venv/bin/python"
test -x "$PY"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the standard checkpoint directory}"
export DFINE_ORACLE_SRC="${DFINE_ORACLE_SRC:?set DFINE_ORACLE_SRC to the pinned D-FINE-seg checkout}"
export TRTLIB="${TRTLIB:?set TRTLIB to the directory containing libnvinfer.so.10}"
export ENGINE="${ENGINE:?set ENGINE to the validated D-FINE-M slim engine}"
export ENGINE_G0="${ENGINE_G0:?set ENGINE_G0 to the validated zero-aux-stream engine}"
export COCO_IMAGES="${COCO_IMAGES:?set COCO_IMAGES to COCO val2017}"
export COCO_ANN="${COCO_ANN:?set COCO_ANN to instances_val2017.json}"
export PREVIOUS_VERSION="${PREVIOUS_VERSION:?set PREVIOUS_VERSION without the v prefix}"
export KNOWN_IMAGE="${KNOWN_IMAGE:?set KNOWN_IMAGE to a recorded COCO image}"
export FOOD_CHECKPOINT="${FOOD_CHECKPOINT:?set FOOD_CHECKPOINT to the 3-class checkpoint}"
export FOOD_CLASSES="${FOOD_CLASSES:?set FOOD_CLASSES to comma-separated checkpoint labels}"
export FOOD_IMAGE="${FOOD_IMAGE:?set FOOD_IMAGE to the recorded food image}"
export RELEASE_DIR="${RELEASE_DIR:-/tmp/dfine-release}"
export MODEL_DIR="${MODEL_DIR:-$RELEASE_DIR/models}"
test ! -e "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR" "$MODEL_DIR"

test -d "$CHECKPOINT_DIR"
test -d "$DFINE_ORACLE_SRC"
test -e "$TRTLIB/libnvinfer.so.10"
test -f "$ENGINE"
test -f "$ENGINE_G0"
test -d "$COCO_IMAGES"
test -f "$COCO_ANN"
test -f "$KNOWN_IMAGE"
test -f "$FOOD_CHECKPOINT"
test -f "$FOOD_IMAGE"

CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR")"
DFINE_ORACLE_SRC="$(realpath "$DFINE_ORACLE_SRC")"
TRTLIB="$(realpath "$TRTLIB")"
ENGINE="$(realpath "$ENGINE")"
ENGINE_G0="$(realpath "$ENGINE_G0")"
COCO_IMAGES="$(realpath "$COCO_IMAGES")"
COCO_ANN="$(realpath "$COCO_ANN")"
KNOWN_IMAGE="$(realpath "$KNOWN_IMAGE")"
FOOD_CHECKPOINT="$(realpath "$FOOD_CHECKPOINT")"
FOOD_IMAGE="$(realpath "$FOOD_IMAGE")"
RELEASE_DIR="$(realpath "$RELEASE_DIR")"
MODEL_DIR="$(realpath "$MODEL_DIR")"
ENGINE_META="$ENGINE.json"
ENGINE_G0_META="$ENGINE_G0.json"
test -f "$ENGINE_META" || ENGINE_META="${ENGINE%.*}.json"
test -f "$ENGINE_G0_META" || ENGINE_G0_META="${ENGINE_G0%.*}.json"
test -f "$ENGINE_META"
test -f "$ENGINE_G0_META"
ENGINE_META="$(realpath "$ENGINE_META")"
ENGINE_G0_META="$(realpath "$ENGINE_G0_META")"
RUNTIME_LD_PATH="$TRTLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PY CHECKPOINT_DIR DFINE_ORACLE_SRC TRTLIB RUNTIME_LD_PATH
export ENGINE ENGINE_G0 ENGINE_META ENGINE_G0_META
export COCO_IMAGES COCO_ANN KNOWN_IMAGE FOOD_CHECKPOINT FOOD_IMAGE RELEASE_DIR MODEL_DIR

for checkpoint in dfine_n_coco.pt dfine_{s,m,l,x}_obj2coco.pt; do
    test -f "$CHECKPOINT_DIR/$checkpoint"
done
test "$(git -C "$DFINE_ORACLE_SRC" rev-parse HEAD)" = "$(cat trt-files/DFINE_SEG_REVISION)"
test -z "$(git -C "$DFINE_ORACLE_SRC" status --porcelain)"
```

The revision file identifies the upstream source of the bundled model implementation. The checkout
is used only for the differential release gate. Each export records the revision and exact bundled-source
hash; users do not need this checkout.

## 1. Finalize the release commit

Set the release version in all six sources, archive the current unreleased note as
`docs/releases/v$VERSION.md`, restore a fresh `docs/releases/UNRELEASED.md` for the next cycle, and
update active latest-release links before any build or gate below. Preserve version-scoped
validation rows.

```sh
export VERSION="${VERSION:?set VERSION to the release version without the v prefix}"
test "$(sed -n 's/^project(dfine VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)" = "$VERSION"
test "$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)" = "$VERSION"
test "$(sed -n 's/^version = "\([^"]*\)"/\1/p' python/pyproject.toml)" = "$VERSION"
test "$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' python/dfine/__init__.py)" = "$VERSION"
test "$(sed -n 's/^[[:space:]]*return "\([^"]*\)";/\1/p' include/dfine/version.hpp)" = "$VERSION"
test "$("$PY" -c 'import tomllib; d=tomllib.load(open("uv.lock", "rb")); print(next(p["version"] for p in d["package"] if p["name"] == "dfine-cpp-tools"))')" = "$VERSION"
test "$(PYTHONPATH="$PWD/python" "$PY" -m dfine.cli --version)" = "dfine $VERSION"
rg -n -F "$PREVIOUS_VERSION" README.md docs python/README.md examples
git diff --check
```

The release note must enumerate observable behavior and compatibility changes without relabeling an
unchanged model recipe. Review every old-version match and retain only version-scoped validation
records. Commit these edits, then verify the release checkout is clean:

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
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ctest --test-dir build --output-on-failure
PYTHONPATH="$PWD/python" LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    "$PY" -m pytest "$PWD/python/tests" -q
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    ./build/tests/dfine_test_engine_meta "$ENGINE_META" "$ENGINE_G0_META"
```

- [ ] All checked sidecars parse and reject contradictory engine contracts.
- [ ] The out-of-tree CMake consumer links the installed package, not the source tree.

## 4. GPU runtime gates

```sh
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" DFINE_TEST_ENGINE="$ENGINE" \
    ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_ENGINE_G0="$ENGINE_G0" \
    DFINE_TEST_REQUIRE_FULL_GRAPH=1 ./build/tests/dfine_test_detector_recovery
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" DFINE_TEST_ENGINE="$ENGINE" \
    compute-sanitizer --tool memcheck --error-exitcode 99 \
    ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_ENGINE_G0="$ENGINE_G0" \
    DFINE_TEST_REQUIRE_FULL_GRAPH=1 compute-sanitizer --tool memcheck --error-exitcode 99 \
    ./build/tests/dfine_test_detector_recovery
"$PY" trt-files/scripts/verify_engine.py --engine "$ENGINE" --batches 1 2 8
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" DFINE_TEST_ENGINE="$ENGINE" DFINE_TEST_IMAGE="$KNOWN_IMAGE" \
    "$PY" -m pytest python/tests -q -ra
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    "$PY" -m dfine.cli predict --engine "$ENGINE" --image "$KNOWN_IMAGE" --json \
    > "$RELEASE_DIR/native-detections.json"
"$PY" trt-files/scripts/cpp_coco_eval.py \
    --binary "$PWD/build/dfine_coco_eval" --engine "$ENGINE" \
    --images "$COCO_IMAGES" --ann "$COCO_ANN" --batch 8 --limit 0 \
    --ld-library-path "$TRTLIB" --out "$RELEASE_DIR/runtime-coco-detections.json"
```

- [ ] Recovery covers ordinary, GPU-decode, CUDA Graph, and required full-pipeline-graph paths.
- [ ] Repeated shape transitions and teardown report zero compute-sanitizer errors.
- [ ] Batches 1, 2, and 8 execute against the declared engine profile.
- [ ] The Python/C++ parity test runs against the recorded image; it is not skipped.
- [ ] Full COCO val2017 completes through the current C++ preprocessing and decode path; zero
      processed images or detections is a hard failure.

## 5. Official-model provenance and accuracy

```sh
PYTHONPATH="$PWD/python" "$PY" -m dfine.cli export \
    --model m --checkpoint "$CHECKPOINT_DIR/dfine_m_obj2coco.pt" \
    --precision fp16 --output "$MODEL_DIR/dfine_m_slim.onnx"
PYTHONPATH="$PWD/python" "$PY" -m dfine.cli build \
    --model m --precision fp16 --onnx "$MODEL_DIR/dfine_m_slim.onnx" \
    --output "$RELEASE_DIR/dfine_m_slim.engine"
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    "$PY" -m dfine.cli predict --engine "$RELEASE_DIR/dfine_m_slim.engine" \
    --image "$KNOWN_IMAGE" --json
"$PY" trt-files/scripts/verify_engine.py \
    --engine "$RELEASE_DIR/dfine_m_slim.engine" --batches 1 2 8
```

- [ ] The ONNX sidecar reports `checkpoint_load: strict`,
      `checkpoint_deserialization: weights_only`, `checkpoint_selected_state: checkpoint root`,
      the canonical loaded-tensor count, zero unused tensors, the commit in
      `trt-files/DFINE_SEG_REVISION`, the bundled model-source SHA-256, the checkpoint SHA-256, and
      complete `tool_versions`.
      `exporter_sha256` is a 64-character lowercase SHA-256 and
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
"$PY" trt-files/scripts/profile.py --backends trt cpp \
    --engine "$RELEASE_DIR/dfine_m_slim.engine" --full \
    --images "$COCO_IMAGES" --ann "$COCO_ANN" --ld-library-path "$TRTLIB" \
    --workdir "$RELEASE_DIR" --out "$RELEASE_DIR/dfine-m-full-coco.json"
"$PY" - "$RELEASE_DIR/dfine-m-full-coco.json" <<'PY'
import json
import math
import sys
from pathlib import Path

EXPECTED_AP = 0.550
TOLERANCE = 0.001

report = json.loads(Path(sys.argv[1]).read_text())
try:
    measured = {
        backend: float(report["backends"][backend]["map"]["AP"])
        for backend in ("trt", "cpp")
    }
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"invalid full-COCO report: {error}") from error

for backend, ap in measured.items():
    if not math.isfinite(ap) or abs(ap - EXPECTED_AP) > TOLERANCE:
        raise SystemExit(
            f"{backend} AP {ap:.6f} is outside {EXPECTED_AP:.3f} ± {TOLERANCE:.3f}"
        )
if abs(measured["trt"] - measured["cpp"]) > TOLERANCE:
    raise SystemExit(
        "TensorRT/C++ AP mismatch: "
        f"{measured['trt']:.6f} vs {measured['cpp']:.6f} "
        f"(tolerance {TOLERANCE:.3f})"
    )
print(
    f"full-COCO AP gate passed: TensorRT={measured['trt']:.6f}, "
    f"C++={measured['cpp']:.6f}"
)
PY
```

An unchanged graph may reuse the recorded accuracy result even when its sidecar gains new
provenance fields.

## 6. Custom-model gate

```sh
PYTHONPATH="$PWD/python" "$PY" -m dfine.cli export \
    --model s --checkpoint "$FOOD_CHECKPOINT" --num-classes 3 \
    --class-names "$FOOD_CLASSES" --precision fp16 --output "$RELEASE_DIR/food_s_slim.onnx"
PYTHONPATH="$PWD/python" "$PY" -m dfine.cli build \
    --model s --onnx "$RELEASE_DIR/food_s_slim.onnx" \
    --output "$RELEASE_DIR/food_s.engine"
PYTHONPATH="$PWD/python" DFINE_LIBRARY="$PWD/build/libdfine.so" \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    "$PY" -m dfine.cli predict --engine "$RELEASE_DIR/food_s.engine" --image "$FOOD_IMAGE"
```

- [ ] Strict checkpoint load covers every checkpoint-owned detection tensor; the sidecar carries
      the three class names and source provenance.
- [ ] Omitting `--num-classes` fails with the class-count diagnostic.

## 7. Five-size static gate

For n/s/m/l/x, require strict checkpoint load, zero `GridSample` nodes, symbolic batch, finite ONNX
Runtime outputs at N=1 and N=2, and a clean `onnx.checker` result. A missing ONNX Runtime or skipped
behavioral postcondition fails the gate. Each published sidecar must pass the source,
exporter-hash, tool-version, and simplification checks from the D-FINE-M gate. Run the five-size full
COCO campaign only when the graph or precision recipe changes.

First prove the bundled factory against the pinned upstream implementation and all five standard
checkpoints:

```sh
DFINE_SEG_SRC="$DFINE_ORACLE_SRC" DFINE_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    PYTHONPATH="$PWD/python" "$PY" -m pytest -q \
    python/tests/test_bundled_model.py::test_pinned_dfine_seg_oracle
```

Generate the remaining pairs directly in the publication input directory; the M pair from the
previous gate stays untouched:

```sh
declare -A CHECKPOINT=(
    [n]=dfine_n_coco.pt
    [s]=dfine_s_obj2coco.pt
    [l]=dfine_l_obj2coco.pt
    [x]=dfine_x_obj2coco.pt
)
for size in n s l x; do
    PYTHONPATH="$PWD/python" "$PY" -m dfine.cli export \
        --model "$size" --checkpoint "$CHECKPOINT_DIR/${CHECKPOINT[$size]}" --precision fp16 \
        --output "$MODEL_DIR/dfine_${size}_slim.onnx"
done
```

## 8. Build and smoke the wheel

```sh
PYTHON="$PY" SKIP_BUILD=1 ./python/build_wheel.sh
export WHEEL="python/dist/dfine-$VERSION-py3-none-linux_x86_64.whl"

test ! -e "$RELEASE_DIR/wheel-smoke"
"$PY" -m venv "$RELEASE_DIR/wheel-smoke"
"$RELEASE_DIR/wheel-smoke/bin/python" -m pip install "$WHEEL" "tensorrt-cu12==10.13.*" "pillow>=9"
WHEEL_TRTLIB="$("$RELEASE_DIR/wheel-smoke/bin/python" -c \
    'import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))')"
export WHEEL_TRTLIB
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
"$PY" trt-files/scripts/release_assets.py assemble \
    --input "$MODEL_DIR" --wheel "$WHEEL" --version "$VERSION" --out "$RELEASE_DIR/upload"
```

- [ ] `assemble` requires a new or empty output directory, accepts exactly 20 model files, validates
      graph/sidecar pairing, canonical model facts, provenance, precision, opset 19, and each slim
      graph's FP32 source hash, then writes `SHA256SUMS` for the 20 model files and wheel.
- [ ] **Republishing the frozen v0.3.1 model pack** (byte-identical models, only the wheel is new):
      add `--frozen-model-pack`. Those artifacts predate the provenance schema and are admitted by
      their pinned published SHA-256 instead; the wheel is still fully validated, and any byte drift
      fails. Use the canonical form above only for a freshly exported pack.
- [ ] Tag the release commit and upload the contents of `$RELEASE_DIR/upload` without rebuilding.

```sh
"$PY" trt-files/scripts/release_assets.py verify --tag "v$VERSION"
```

- [ ] Verification downloads every published asset, checks every digest, and reports no uncovered
      or missing files.
