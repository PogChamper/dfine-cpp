// Derived from rf-detr-cpp (github.com/infracv/rf-detr-cpp, Apache-2.0); modified for D-FINE-cpp.
#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <utility>

namespace dfine {
namespace detail {

struct DevFree {
    void operator()(void* p) const noexcept {
        if (p) cudaFree(p);
    }
};

struct HostFree {
    void operator()(void* p) const noexcept {
        if (p) cudaFreeHost(p);
    }
};

}  // namespace detail

// Owning device-memory pointer. Drop-in for void* managed with cudaMalloc/cudaFree.
using DevPtr = std::unique_ptr<void, detail::DevFree>;

// Owning pinned-host-memory pointer. Managed with cudaMallocHost/cudaFreeHost.
using HostPtr = std::unique_ptr<void, detail::HostFree>;

// Owning CUDA stream. Non-copyable, movable — adopts a stream created by the
// caller so it is destroyed even if construction later throws. Move nulls the
// source so the handle is never double-destroyed.
class CudaStream {
 public:
    CudaStream() = default;
    explicit CudaStream(cudaStream_t s) noexcept : s_(s) {}
    ~CudaStream() {
        if (s_) cudaStreamDestroy(s_);
    }

    CudaStream(CudaStream&& o) noexcept : s_(std::exchange(o.s_, nullptr)) {}
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) {
            if (s_) cudaStreamDestroy(s_);
            s_ = std::exchange(o.s_, nullptr);
        }
        return *this;
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return s_; }
    explicit operator bool() const noexcept { return s_ != nullptr; }

 private:
    cudaStream_t s_{nullptr};
};

// Owning CUDA event, same ownership contract as CudaStream.
class CudaEvent {
 public:
    CudaEvent() = default;
    explicit CudaEvent(cudaEvent_t e) noexcept : e_(e) {}
    ~CudaEvent() {
        if (e_) cudaEventDestroy(e_);
    }

    CudaEvent(CudaEvent&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
    CudaEvent& operator=(CudaEvent&& o) noexcept {
        if (this != &o) {
            if (e_) cudaEventDestroy(e_);
            e_ = std::exchange(o.e_, nullptr);
        }
        return *this;
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return e_; }
    explicit operator bool() const noexcept { return e_ != nullptr; }

 private:
    cudaEvent_t e_{nullptr};
};

// Owning captured CUDA graph (the template produced by stream capture), same
// ownership contract as CudaStream. Destroyed with cudaGraphDestroy.
class CudaGraph {
 public:
    CudaGraph() = default;
    explicit CudaGraph(cudaGraph_t g) noexcept : g_(g) {}
    ~CudaGraph() {
        if (g_) cudaGraphDestroy(g_);
    }

    CudaGraph(CudaGraph&& o) noexcept : g_(std::exchange(o.g_, nullptr)) {}
    CudaGraph& operator=(CudaGraph&& o) noexcept {
        if (this != &o) {
            if (g_) cudaGraphDestroy(g_);
            g_ = std::exchange(o.g_, nullptr);
        }
        return *this;
    }
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    cudaGraph_t get() const noexcept { return g_; }
    cudaGraph_t* addr() noexcept { return &g_; }  // for cudaStreamEndCapture(&g_)
    explicit operator bool() const noexcept { return g_ != nullptr; }

 private:
    cudaGraph_t g_{nullptr};
};

// Owning instantiated (executable) CUDA graph. Destroyed with cudaGraphExecDestroy.
class CudaGraphExec {
 public:
    CudaGraphExec() = default;
    explicit CudaGraphExec(cudaGraphExec_t e) noexcept : e_(e) {}
    ~CudaGraphExec() {
        if (e_) cudaGraphExecDestroy(e_);
    }

    CudaGraphExec(CudaGraphExec&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
    CudaGraphExec& operator=(CudaGraphExec&& o) noexcept {
        if (this != &o) {
            if (e_) cudaGraphExecDestroy(e_);
            e_ = std::exchange(o.e_, nullptr);
        }
        return *this;
    }
    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    cudaGraphExec_t get() const noexcept { return e_; }
    explicit operator bool() const noexcept { return e_ != nullptr; }

 private:
    cudaGraphExec_t e_{nullptr};
};

}  // namespace dfine
