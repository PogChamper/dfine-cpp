# Troubleshooting

Field notes from real clean-machine installs (two rented GPU boxes, native
Ubuntu 22.04, drivers from scratch). Start with `dfine doctor` — it prints the
environment facts every entry below keys on, and its output belongs in any bug
report.

## Python / pip

**`tensorrt-cu13-libs ... requires cuda-toolkit>=13` dependency conflict, or
TensorRT needing a ≥580 driver you don't have.**
The bare `tensorrt` metapackage resolves to the CUDA-13 flavor since
10.13.3.9.post1. This stack is CUDA-12; always install the flavored package:

```sh
pip uninstall -y tensorrt tensorrt-cu13 tensorrt-cu13-libs tensorrt-cu13-bindings
pip install "tensorrt-cu12==10.13.*"
```

**Exports differ byte-for-byte from the released ONNX.**
Byte-reproducibility holds only under the locked toolchain — a different torch
serializes a different graph. Use `uv sync --frozen --extra gpu --extra torch`
(the committed `uv.lock` pins torch 2.9.1+cu128) and compare the
`tool_versions` field the sidecar records.

**`ModuleNotFoundError` (torchvision / loguru / scipy) during export.**
The D-FINE-seg model sources import more than torch. The `torch` extra carries
the whole chain: `uv sync --frozen --extra gpu --extra torch`.

## Building from source

**cmake: `Could NOT find TensorRT (missing: TENSORRT_INCLUDE_DIR ...)`.**
`pip install tensorrt-cu12` ships runtime `.so`'s only — building needs the
headers. `./build.sh` now fails before cmake with the full recipe; the short
version, in preference order:

1. NVIDIA apt repo, chain pinned to your runtime (apt does **not** down-resolve
   dependencies to a pin, and unpinned it installs TensorRT 11):
   ```sh
   V="$(apt-cache madison libnvinfer-dev | grep -oPm1 '10\.13\.[0-9.]+-1\+cuda12\.[0-9]+')"
   sudo apt-get install -y "libnvinfer10=$V" "libnvinfer-headers-dev=$V" "libnvinfer-dev=$V" \
     "libnvinfer-plugin10=$V" "libnvinfer-headers-plugin-dev=$V" "libnvinfer-plugin-dev=$V" \
     "libnvonnxparsers10=$V" "libnvonnxparsers-dev=$V"
   ```
2. `nvcr.io/nvidia/tensorrt` container (headers preinstalled).
3. No root: a TensorRT GA tarball unpacked into `third_party/tensorrt/{include,lib}`
   (`third_party/README.md`).

**`E: Unable to locate package libnvonnxparser-dev`.**
The parser package is plural: `libnvonnxparsers-dev`.

**A wall of `math.h` / `__PTHREAD_SPINS` errors mid-compile.**
A conda cross-g++ leads your PATH (conda's `cuda-toolkit` drags one in) and its
sysroot clashes with the system TensorRT/glibc headers. `./build.sh` detects
this and switches to `/usr/bin/g++`; if you invoke cmake directly, pass
`CC=/usr/bin/gcc CXX=/usr/bin/g++`.

**`nvcc fatal : Don't know what to do with 'UNSET'`.**
A double conda activation (shell + tmux is the classic) leaks the activation
script's sentinel into `NVCC_PREPEND_FLAGS`. `./build.sh` drops the injected
value; elsewhere, `unset NVCC_PREPEND_FLAGS`.

## Driver

**`nvidia-smi` fails, or apt driver installs die in dependency conflicts.**
Hosting images often carry remnants of several driver stacks (an old
proprietary set, a vendor repo pinning different versions). Clean slate:

```sh
sudo apt-get purge -y '*nvidia*' && sudo apt-get autoremove -y
# disable third-party repos pinning their own driver versions, then:
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) nvidia-open
sudo reboot
```

RTX 50xx (Blackwell) needs driver ≥ 570; `nvidia-open` from the NVIDIA CUDA
apt repo tracks the current branch and supports it.

## Runtime

**mAP collapses after "fixing" preprocessing.**
D-FINE is `/255` only — no ImageNet mean/std. The sidecar records the contract;
the runtime validates it.

**CMake error on `CUDA_ARCHITECTURES=native`.**
Needs CMake ≥ 3.24; on older CMake pass an explicit arch (`CUDA_ARCH=89 ./build.sh`).

**`TRT runtime X != compile-time headers Y` warning at load.**
Advisory: the binary was compiled against different (older) TensorRT headers
than the libraries it found at runtime. Same major version works; rebuild
against matching headers to silence it.

**Detections on an fp16 engine differ across batch positions at ~1e-3.**
Expected ULP-level jitter of fully-FP16 engines (batch-position tactic
variance). Correctness for the slim tier is gated by mAP parity, not bitwise
equality; the fp32 and fp16_st flavors are bitwise-deterministic.

## Networks and rented boxes

- `curl: (35) ... unexpected eof` from GitHub releases: the CDN drops TLS on
  some hosting networks. Everything in `remote_matrix.sh` retries and every
  download is skipped when the file is already in place — `scp` assets in from
  a machine with working access and only the hash check runs.
- Flapping DNS: `echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf`.
- Web-terminal paste mangles multi-line commands (backslash continuations
  merge): paste one line at a time.
- Always run long jobs inside `tmux` — an SSH drop kills the pipeline.

## conda fallback (no root only)

Prefer uv + the apt toolchain above. If you cannot install system packages,
conda can still provide nvcc:

```sh
conda create -y -n dfine python=3.11 'cuda-toolkit>=12.8,<13' -c nvidia
```

Pin `<13` (unpinned resolves to CUDA 13), accept the ToS prompts conda demands
on first use, and expect the toolchain quirks documented above — `build.sh`
works around the known ones.

Something not covered here: open an issue with the `dfine doctor` output and,
for build failures, the cmake error text.
