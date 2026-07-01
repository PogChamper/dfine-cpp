# third_party/

Vendored / externally-provided dependencies.

## `stb/` — tracked ✅

`stb_image.h` (v2.30, public domain) — single-header JPEG/PNG decoder used by the image-consuming apps
(`dfine_detect`, `dfine_bench`, `dfine_coco_eval`) via `apps/image_io.cpp`. The core `libdfine` is
OpenCV-free and does **not** need this — it takes a raw `ImageU8`. Committed so a fresh clone builds.

## `tensorrt/` — NOT tracked (populate locally) ⚠️

Ignored by `.gitignore` because it mixes machine-specific symlinks with re-fetchable headers. To build,
recreate it as:

```
third_party/tensorrt/
├── include/    # TensorRT 10.13 public headers (NvInfer.h, NvOnnxParser.h, NvInferRuntime.h, ...)
└── lib/        # symlinks to the runtime .so's
```

- **Headers:** copy from your TensorRT install, or from the TensorRT OSS repo at the matching tag
  (`github.com/NVIDIA/TensorRT`, `include/`), for the version you link against (here 10.13). These are the
  public API headers; the compile only needs `NvInfer*.h` + `NvOnnxParser.h`.
- **Libraries:** symlink the `.so`s from wherever your TensorRT runtime lives. In this project they point at
  the sibling D-FINE-seg venv's wheel libs, e.g.:
  ```sh
  mkdir -p third_party/tensorrt/lib && cd third_party/tensorrt/lib
  ln -sf <D-FINE-seg>/.venv/lib/python3.11/site-packages/tensorrt_libs/*.so* .
  ```
- At **runtime**, the same `tensorrt_libs` dir must be on `LD_LIBRARY_PATH` (see `build.sh` / HANDOFF).

`CMakeLists.txt` finds this tree via `-DTENSORRT_DIR=$PWD/third_party/tensorrt` (baked into `build.sh`).
Inside a container the base image (`nvcr.io/nvidia/tensorrt:*`) already provides the headers/libs — see the
`Dockerfile`.
