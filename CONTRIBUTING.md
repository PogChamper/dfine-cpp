# Contributing to D-FINE-cpp

This is a performance- and correctness-critical inference library. Maintained conversion and
runtime changes must remain warning-clean and sanitizer-clean. Changes to the default inference
path must remain mAP-neutral; an explicit accuracy-traded preset requires its own measured
contract.

## Correctness invariants

- **Preprocessing is `/255` only** — no ImageNet mean/std. Do not "fix" this.
- **FP16 is achieved via strong typing, never the weakly-typed `kFP16` builder flag** (that flag
  measured approximately 6.8 AP below FP32; native `GridSample` caused a separate approximately
  10.5 AP regression). The production recipe is the surgical converter (`convert_fp16_surgical.py --slim`,
  opset ≥ 19): the decoder runs FP16 *except* the FDR/deform-coordinate FP32 island the converter pins.
  Do not widen or shrink that island without a full-COCO gate.
- **The FDR/Integral/LQE box math stays inside the engine** — never reimplement it in C++.
- Current artifact and runtime contracts live in [docs/CONVERSION.md](docs/CONVERSION.md),
  [docs/RUNTIME.md](docs/RUNTIME.md), and [docs/NAMING.md](docs/NAMING.md). The research matrix records
  measured alternatives. `docs/HANDOFF.md` and `docs/impl/` are historical engineering records.

## Build

```sh
./build.sh                  # Release; CUDA arch defaults to 'native' (the local GPU)
WERROR=ON ./build.sh        # warnings-as-errors (CI/PR bar; must pass clean)
BUILD_TYPE=UBSAN ./build.sh # UndefinedBehaviorSanitizer (safe on the full pipeline)
BUILD_TYPE=ASAN  ./build.sh # AddressSanitizer (host-isolated: decode/postprocess)
CUDA_ARCH=89 ./build.sh     # explicit SM (CI/cross builds; 'native' needs CMake >= 3.24)
```

`build.sh` discovers `cmake`/`nvcc` from `PATH` (`$CMAKE`/`$NVCC` override) and resolves TensorRT as
`$TENSORRT_DIR` → a populated `third_party/tensorrt` → system paths via `cmake/FindTensorRT.cmake`; the
conda-`ld` workaround is applied only for conda toolchains. Plain CMake works too:
`cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build -j`.

## Style

- **C++17.** The whole tree is formatted with the repo `.clang-format` and CI gates on it
  (`clang-format --dry-run --Werror`, clang-format 18) — run `clang-format -i` on changed files before
  committing.
- RAII for all CUDA/TensorRT resources (`cuda_raii.hpp`); no raw `cudaFree`/`cudaStreamDestroy`. Comments say
  *why*, not *what*. Match the surrounding idiom.
- Python (scripts): 4-space, ≤100 col; `ruff` clean.

## Testing / validation

Fast tests run everywhere; accuracy validation is empirical against the model:

- **`ctest` green.** `tests/` holds CPU-only tests (image-layout validation, sidecar parsing) that run
  anywhere, and GPU tests (shape-transition recovery, detector error recovery) that skip without a GPU —
  set `DFINE_TEST_ENGINE` (and optionally `DFINE_TEST_ENGINE_G0`) to run them. Python-side:
  `pytest python/tests` (CPU subset needs no GPU).
- **Build clean** with `WERROR=ON`.
- **Default-path mAP unchanged** for changes to export, build, or decode:
  `trt-files/scripts/profile.py --backends trt cpp --subset 2000` (or `--full`) must hold the
  reference AP for the affected size. Accuracy-traded presets require measured before/after results.
- **Sanitizers clean** for C++ changes on the exercised paths (UBSAN build; `compute-sanitizer` for new kernels).
- Engine-graph changes: `dfine_bench --graph-compare` must keep the raw-output path byte-identical
  to ordinary enqueue; it requires a `--max-aux-streams 0` engine.
- GPU-decode or full-pipeline-graph changes: use box-aware tolerances and dataset mAP against CPU
  decode. A one-ULP score difference is allowed; bitwise identity is not the contract. Require an
  active full graph when validating `dfine_coco_eval --full-graph`.

## PRs

Keep commits focused and messages terse (why, not what). Reference the affected contract or validation gate. If
a change alters validated numbers, include the before/after `profile.py` table in the PR description.
