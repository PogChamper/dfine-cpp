#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#define DFINE_CUDA_CHECK(call)                                                             \
    do {                                                                                   \
        const cudaError_t _dfine_err = (call);                                             \
        if (_dfine_err != cudaSuccess) {                                                   \
            throw std::runtime_error(std::string("CUDA error: ") +                         \
                                     cudaGetErrorString(_dfine_err) + " (" #call ") at " + \
                                     __FILE__ ":" + std::to_string(__LINE__));             \
        }                                                                                  \
    } while (0)

#define DFINE_TRT_CHECK(expr)                                                                 \
    do {                                                                                      \
        if (!(expr)) {                                                                        \
            throw std::runtime_error(std::string("TensorRT call failed: ") + #expr + " at " + \
                                     __FILE__ ":" + std::to_string(__LINE__));                \
        }                                                                                     \
    } while (0)
