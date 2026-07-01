#pragma once

// DeviceArena — a one-shot device/host memory arena for the frozen-pipeline
// contract (intensive-core P2). Every buffer the detector needs at steady state
// is sized up front (all maxima are computable without running: TRT's activation
// size comes from getDeviceMemorySize, binding/scratch sizes from max_batch), so
// the arena makes exactly ONE cudaMalloc and hands out slices of it. After lock()
// any further reservation is a contract breach (policy: warn/throw) — this is what
// guarantees no allocation on the hot path, no fragmentation, and stable addresses
// for CUDA-graph capture (P3).
//
// Usage:
//   DeviceArena a;                          // Kind::kDevice by default
//   auto off = a.sub(bytes);                // reserve; returns a byte offset
//   ...reserve every slab...
//   a.commit();                             // one cudaMalloc(high_water)
//   T* p = a.at<T>(off);                    // resolve (valid after commit)
//   a.lock(DeviceArena::Policy::kThrow);    // freeze; sub() now throws

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "internal/cuda_check.hpp"

namespace dfine {

class DeviceArena {
   public:
    enum class Policy { kWarn, kThrow };  // never abort() in a library
    enum class Kind { kDevice, kHost };   // cudaMalloc vs cudaMallocHost (pinned)

    explicit DeviceArena(Kind kind = Kind::kDevice) noexcept : kind_(kind) {}
    ~DeviceArena() { free_(); }
    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // Reserve a `bytes` slab (aligned) and return its byte offset. Valid to
    // dereference (via at()) only after commit(). Must be called before lock():
    // a post-lock reservation means a new allocation appeared at steady state,
    // which is the contract breach this arena exists to catch.
    std::size_t sub(std::size_t bytes, std::size_t align = 256) {
        if (bytes == 0) bytes = 1;
        if (locked_) {
            const std::string msg =
                "dfine: DeviceArena is locked but a " + std::to_string(bytes) +
                "-byte allocation was requested (steady-state allocation / warmup did "
                "not cover the max shape). Increase the reserved size or unlock.";
            if (policy_ == Policy::kThrow) throw std::runtime_error(msg);
            return align_up(high_water_, align);  // kWarn: hand out an UNBACKED offset
        }
        if (committed_) {
            throw std::runtime_error("dfine: DeviceArena::sub after commit (reserve before commit)");
        }
        const std::size_t off = align_up(high_water_, align);
        high_water_           = off + bytes;
        return off;
    }

    // Allocate the single backing block sized to the current high-water mark.
    void commit() {
        if (committed_) return;
        committed_ = true;
        capacity_  = high_water_;
        if (capacity_ == 0) return;
        void* p = nullptr;
        if (kind_ == Kind::kDevice) {
            DFINE_CUDA_CHECK(cudaMalloc(&p, capacity_));
        } else {
            DFINE_CUDA_CHECK(cudaMallocHost(&p, capacity_));
        }
        base_ = static_cast<std::uint8_t*>(p);
    }

    // Freeze: any subsequent sub() fires the policy.
    void lock(Policy p = Policy::kThrow) noexcept {
        policy_ = p;
        locked_ = true;
    }

    template <class T = void>
    [[nodiscard]] T* at(std::size_t offset) const noexcept {
        return reinterpret_cast<T*>(base_ + offset);
    }

    [[nodiscard]] void*       base()       const noexcept { return base_; }
    [[nodiscard]] std::size_t high_water() const noexcept { return high_water_; }
    [[nodiscard]] std::size_t capacity()   const noexcept { return capacity_; }
    [[nodiscard]] bool        committed()  const noexcept { return committed_; }
    [[nodiscard]] bool        locked()     const noexcept { return locked_; }

   private:
    static std::size_t align_up(std::size_t x, std::size_t a) noexcept {
        return a == 0 ? x : (x + a - 1) / a * a;
    }
    void free_() noexcept {
        if (!base_) return;
        if (kind_ == Kind::kDevice) {
            cudaFree(base_);
        } else {
            cudaFreeHost(base_);
        }
        base_ = nullptr;
    }

    Kind          kind_;
    std::uint8_t* base_{nullptr};
    std::size_t   high_water_{0};
    std::size_t   capacity_{0};
    bool          committed_{false};
    bool          locked_{false};
    Policy        policy_{Policy::kThrow};
};

}  // namespace dfine
