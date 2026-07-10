# Release checklist

Executable, in order. The GPU gates run on the release machine until a GPU runner exists;
everything else is covered by CI. Rule zero: **the bytes that were validated are the bytes
that get published — no rebuild between the gate and the upload.**

Environment used below:

```sh
export TRTLIB=<dir with libnvinfer.so.10>          # e.g. a pip tensorrt venv's tensorrt_libs
export ENGINE=trt-files/engines/dfine_m_fp16_st.engine     # any dynamic-batch engine
export ENGINE_G0=trt-files/engines/dfine_m_fp16_g0.engine  # --max-aux-streams 0 build
```

## 1. Hosted gates (must be green in CI on the release commit)

- [ ] `compile-cuda`, `python-nogpu`, `wheel` jobs green — the wheel job includes the
      hygiene gate (bundled `.so` + `_scripts/build_engine.py`, no RPATH, SONAME intact)
      and the outside-checkout smoke.
- [ ] `WERROR=ON ./build.sh` clean locally.

## 2. CPU test gates

```sh
cd build && ctest --output-on-failure          # image_layout, engine_meta run anywhere
PYTHONPATH=python python -m pytest python/tests -q   # CPU subset passes without GPU
./build/tests/dfine_test_engine_meta trt-files/onnx/*.json trt-files/engines/*.json  # all sidecars parse
```

## 3. GPU runtime-safety gates

```sh
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE ./build/tests/dfine_test_shape_transitions
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE DFINE_TEST_ENGINE_G0=$ENGINE_G0 \
    ./build/tests/dfine_test_detector_recovery
LD_LIBRARY_PATH=$TRTLIB DFINE_TEST_ENGINE=$ENGINE \
    compute-sanitizer --tool memcheck --error-exitcode 99 ./build/tests/dfine_test_shape_transitions
```

- [ ] All pass; memcheck reports 0 errors.

## 4. Official-model gate (D-FINE-M)

```sh
dfine export --model m --precision fp16            # strict load, opset 19, surgical --slim
dfine build  --model m --precision fp16            # engine into the hash-keyed cache
dfine predict --model m --image <known.jpg> --json # detections match the recorded reference
python trt-files/scripts/verify_engine.py --engine <built.engine> --batches 1 2 8
```

- [ ] Engine sidecar says `precision: fp16`, `precision_mode: strongly_typed_onnx_fp16_surgical_slim`,
      carries `onnx_sha256`.
- [ ] If the ONNX graph hash differs from the previously gated slim asset, run the full-COCO
      m gate (`profile.py --backends trt --full`) before shipping; byte-identical graph = skip.

## 5. Custom-model gate (3-class food checkpoint)

```sh
dfine export --model s --checkpoint <food.pt> --num-classes 3 --class-names <a,b,c> --precision fp16
dfine build  --model s --onnx <exported.onnx> --output <engine>
dfine predict --engine <engine> --image <food.jpg>   # classes/labels sane vs the .pt reference
```

- [ ] Strict load reports full tensor coverage; sidecar carries the 3 class names.
- [ ] A deliberately wrong invocation (no `--num-classes`) aborts with the class-count hint.

## 6. All-size static gate

For each of n/s/m/l/x: strict-load passes (`export` runs through the checkpoint check), the
export postconditions pass (GridSample=0, symbolic batch, ORT N=1/2 run), `onnx.checker` clean.
Full five-size COCO campaign only when the precision recipe itself changed.

## 7. Package and publish

```sh
CUDA_ARCH=89 ./python/build_wheel.sh    # or take the CI artifact — same script, same bytes
```

- [ ] Wheel SHA256 recorded in SHA256SUMS alongside the ONNX assets.
- [ ] Release notes name every behavior change (this release: `dfine export --precision fp16`
      now = surgical/slim; re-export custom models to pick up the production tier).
- [ ] Tag; upload the *gated* artifacts; download one asset back and compare hashes.
