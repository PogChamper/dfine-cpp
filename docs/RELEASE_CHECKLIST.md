# Release checklist

Executable, in order. The GPU gates run on the release machine until a GPU runner exists;
everything else is covered by CI. Rule zero: **the bytes that were validated are the bytes
that get published — no rebuild between the gate and the upload.**

Environment used below:

```sh
export TRTLIB="${TRTLIB:?set TRTLIB to the directory containing libnvinfer.so.10}"
export ENGINE="${ENGINE:-trt-files/engines/dfine_m_fp16_st.engine}"
export ENGINE_G0="${ENGINE_G0:-trt-files/engines/dfine_m_fp16_g0.engine}"
export KNOWN_IMAGE="${KNOWN_IMAGE:?set KNOWN_IMAGE to a recorded COCO image}"
export FOOD_CHECKPOINT="${FOOD_CHECKPOINT:?set FOOD_CHECKPOINT to the 3-class checkpoint}"
export FOOD_CLASSES="${FOOD_CLASSES:-food,plate,tray}"
export FOOD_IMAGE="${FOOD_IMAGE:?set FOOD_IMAGE to the recorded food image}"
export RELEASE_DIR="${RELEASE_DIR:-/tmp/dfine-v0.3.1-release}"
mkdir -p "$RELEASE_DIR"
```

## 1. Hosted gates (must be green in CI on the release commit)

- [ ] `lint`, `compile-cuda`, `python-nogpu`, `wheel` jobs green — the wheel job includes the
      hygiene gate (bundled `.so` + `_scripts/build_engine.py`, no RPATH, SONAME intact)
      and the outside-checkout smoke.
- [ ] `WERROR=ON ./build.sh` clean locally.

## 2. CPU test gates

```sh
ctest --test-dir build --output-on-failure          # image_layout, engine_meta run anywhere
PYTHONPATH="$PWD/python" python -m pytest "$PWD/python/tests" -q
./build/tests/dfine_test_engine_meta trt-files/onnx/*.json trt-files/engines/*.json  # all sidecars parse
```

## 3. GPU runtime-safety gates

```sh
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE DFINE_TEST_ENGINE_G0=$ENGINE_G0 \
    DFINE_TEST_REQUIRE_FULL_GRAPH=1 ./build/tests/dfine_test_detector_recovery
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE \
    compute-sanitizer --tool memcheck --error-exitcode 99 ./build/tests/dfine_test_shape_transitions
```

- [ ] All pass; memcheck reports 0 errors. `dfine_test_detector_recovery` must exercise
      the full-graph section: an unset `DFINE_TEST_ENGINE_G0` or an inactive full graph is
      a failed release gate, not a passing partial run.

## 4. Official-model gate (D-FINE-M)

```sh
dfine export --model m --precision fp16 --output "$RELEASE_DIR/dfine_m_slim.onnx"
dfine build --model m --precision fp16 --onnx "$RELEASE_DIR/dfine_m_slim.onnx" \
    --output "$RELEASE_DIR/dfine_m_slim.engine"
dfine predict --engine "$RELEASE_DIR/dfine_m_slim.engine" --image "$KNOWN_IMAGE" --json
python trt-files/scripts/verify_engine.py \
    --engine "$RELEASE_DIR/dfine_m_slim.engine" --batches 1 2 8
```

- [ ] Engine sidecar says `precision: fp16`, `precision_mode: strongly_typed_onnx_fp16_surgical_slim`,
      carries `onnx_sha256`.
- [ ] `sha256sum "$RELEASE_DIR/dfine_m_slim.onnx"` is
      `0f0b8e9ecafa3112d3f7d983e52809c92514836ee1328b519fe81fe25abc7419`, the gated
      v0.3.0 m-slim asset. If it differs, run the full-COCO m gate against the newly built
      engine (`profile.py --backends trt --engine "$RELEASE_DIR/dfine_m_slim.engine" --full`)
      before shipping; byte-identical graph = skip.

## 5. Custom-model gate (3-class food checkpoint)

```sh
dfine export --model s --checkpoint "$FOOD_CHECKPOINT" --num-classes 3 \
    --class-names "$FOOD_CLASSES" --precision fp16 --output "$RELEASE_DIR/food_s_slim.onnx"
dfine build --model s --onnx "$RELEASE_DIR/food_s_slim.onnx" \
    --output "$RELEASE_DIR/food_s.engine"
dfine predict --engine "$RELEASE_DIR/food_s.engine" --image "$FOOD_IMAGE"
```

- [ ] Strict load reports full tensor coverage; sidecar carries the 3 class names.
- [ ] A deliberately wrong invocation (no `--num-classes`) aborts with the class-count hint.

## 6. All-size static gate

For each of n/s/m/l/x: strict-load passes (`export` runs through the checkpoint check), the
export postconditions pass (GridSample=0, symbolic batch, ORT N=1/2 run), `onnx.checker` clean.
The exporter log must contain `dynamic batch OK: N=1 and N=2 run`; a missing onnxruntime or
any skipped behavioral check fails the gate.
Full five-size COCO campaign only when the precision recipe itself changed.

## 7. Package and publish

```sh
CUDA_ARCH=89 ./python/build_wheel.sh    # or (preferred) take the CI artifact: same script,
                                        # but a different toolchain — the artifacts are NOT
                                        # byte-identical; gate and publish ONE of them
```

- [ ] Wheel SHA256 recorded in SHA256SUMS alongside the ONNX assets.
- [ ] Core upload is exactly 20 model files (`dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json}`),
      the one gated `dfine-0.3.1-py3-none-linux_x86_64.whl`, and `SHA256SUMS`; the manifest
      contains all 21 payload files. Engines and custom-model artifacts are not uploaded.
- [ ] Release notes name every behavior change (this release: `dfine export --precision fp16`
      now = surgical/slim; re-export custom models to pick up the production tier).
- [ ] Tag; upload the *gated* artifacts; download every asset into a fresh directory and run
      `sha256sum -c SHA256SUMS` against the pre-upload manifest.
