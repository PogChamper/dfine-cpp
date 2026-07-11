# syntax=docker/dockerfile:1

# Native build environment and command-line tools. ONNX export is intentionally
# outside this image because it also requires a D-FINE source checkout and the
# locked PyTorch tools environment.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ARG TRT_DEB_VERSION=10.13.3.9-1+cuda12.9
ARG CUDA_ARCH=89

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      nlohmann-json3-dev \
      "libnvinfer10=${TRT_DEB_VERSION}" \
      "libnvinfer-headers-dev=${TRT_DEB_VERSION}" \
      "libnvinfer-dev=${TRT_DEB_VERSION}" \
      "libnvonnxparsers10=${TRT_DEB_VERSION}" \
      "libnvonnxparsers-dev=${TRT_DEB_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/dfine-cpp
COPY . .

RUN cmake -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DDFINE_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

ENV PATH="/workspace/dfine-cpp/build:${PATH}"
ENV LD_LIBRARY_PATH="/workspace/dfine-cpp/build:${LD_LIBRARY_PATH}"

# A no-argument container verifies that the native binaries and their runtime
# dependencies load without requiring an engine or a GPU.
CMD ["dfine_detect", "--help"]
