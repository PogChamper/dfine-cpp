# third_party/

Vendored / externally-provided dependencies.

## `stb/` — tracked ✅

`stb_image.h` (v2.30, public domain) — single-header JPEG/PNG decoder used by the image-consuming apps
(`dfine_detect`, `dfine_bench`, `dfine_coco_eval`) via `apps/image_io.cpp`. The core `libdfine` is
OpenCV-free and does **not** need this — it takes a raw `ImageU8`. Committed so a fresh clone builds.

## `tensorrt/` — NOT tracked, and NOT required

Ignored by `.gitignore` (machine-specific symlinks + re-fetchable headers). **You do not need it**: a system
TensorRT install is found automatically by `cmake/FindTensorRT.cmake` (searches `$TENSORRT_DIR`,
`/usr/local/TensorRT`, `/opt/tensorrt`, `/usr`). Populate this directory only as an alternative — e.g. no
root access, or you get TensorRT from the pip wheel (which ships the `.so`s but no headers). Layout:

```
third_party/tensorrt/
├── include/    # TensorRT 10.x public headers (NvInfer.h, NvOnnxParser.h, NvInferRuntime.h, ...)
└── lib/        # symlinks to the runtime .so's
```

- **Headers:** copy from a TensorRT install, or from the TensorRT OSS repo at the matching tag
  (`github.com/NVIDIA/TensorRT`, `include/`), for the version you link against (validated here: 10.13). The
  compile only needs `NvInfer*.h` + `NvOnnxParser.h`.
- **Libraries:** symlink the `.so`s from wherever your TensorRT runtime lives — e.g. a venv after
  `python -m pip install "tensorrt==10.13.*"`:
  ```sh
  mkdir -p third_party/tensorrt/lib && cd third_party/tensorrt/lib
  ln -sf <venv>/lib/python3.11/site-packages/tensorrt_libs/*.so* .
  ```
- At **runtime**, the same lib dir must be on `LD_LIBRARY_PATH`.

`build.sh` auto-detects a populated tree (checks `third_party/tensorrt/include/NvInfer.h`) and passes
`-DTENSORRT_DIR`; an explicit `$TENSORRT_DIR` wins. Inside a container the base image
(`nvcr.io/nvidia/tensorrt:*`) already provides the headers/libs — see the `Dockerfile`.
