# Contributing to D-FINE-cpp

This is a performance- and correctness-critical inference library, so the bar is high:
**every change must keep the C++ warning-clean, sanitizer-clean, and mAP-neutral.**

## Ground rules (the hard-won ones)

- **Preprocessing is `/255` only** — no ImageNet mean/std. Do not "fix" this.
- **The decoder must stay FP32.** FP16 is achieved via strong typing (`convert_fp16.py`), never the weakly-typed
  `kFP16` builder flag. Never ship a non-FP32-validated engine from `dfine_build`.
- **The FDR/Integral/LQE box math stays inside the engine** — never reimplement it in C++.
- Read [docs/HANDOFF.md](docs/HANDOFF.md) before touching anything — it is the single source of truth, and the
  "Decisions & gotchas" section will save you a day.

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

There are no unit tests (the model *is* the test); validation is empirical:

- **Build clean** with `WERROR=ON`.
- **mAP unchanged** for anything touching export/build/decode: `profile.py --backends trt cpp --subset 2000`
  (or `--full`) must hold the reference AP for the size you touched.
- **Sanitizers clean** for C++ changes on the exercised paths (UBSAN build; `compute-sanitizer` for new kernels).
- CUDA-graph changes: `dfine_coco_eval` graph-vs-no-graph must stay **byte-identical**, and
  `dfine_bench --graph-compare` (needs a `--max-aux-streams 0` engine) for latency.

## PRs

Keep commits focused and messages terse (why, not what). Reference the relevant HANDOFF/M-section. If a change
alters the validated numbers, include the before/after `profile.py` table in the PR description.
