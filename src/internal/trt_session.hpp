// Derived from rf-detr-cpp (github.com/infracv/rf-detr-cpp, Apache-2.0); substantially
// extended for D-FINE-cpp (frozen-memory contract, transactional shape transitions,
// user-managed activation memory).
#pragma once

#include "cuda_raii.hpp"
#include "trt_logger.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

// This library uses the TensorRT 10+ tensor-based API (enqueueV3, setTensorAddress,
// getNbIOTensors). TRT 9 and below are not supported.
#if NV_TENSORRT_MAJOR < 10
#error "D-FINE-cpp requires TensorRT >= 10.0. Please upgrade your TensorRT installation."
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dfine {

// Resolved description of a single TRT IO tensor.
struct BindingInfo {
    std::string name;
    nvinfer1::DataType dtype;
    nvinfer1::Dims shape;  // resolved shape; dims may be -1 if dynamic and not yet set
    nvinfer1::TensorLocation location{nvinfer1::TensorLocation::kDEVICE};
    nvinfer1::TensorFormat format{nvinfer1::TensorFormat::kLINEAR};
    std::string format_desc;
    int vectorized_dim{-1};
    bool is_input{false};
    int64_t element_count{0};  // 0 = shape has unresolved dim(s)
    std::size_t bytes{0};      // element_count * sizeof(dtype)
};

struct InputProfileInfo {
    nvinfer1::Dims min{};
    nvinfer1::Dims opt{};
    nvinfer1::Dims max{};
};

// Owns a TRT runtime, engine, execution context, per-binding device + pinned-host
// buffers, and a CUDA stream. Supports device-resident, linear IO tensors; the
// task layer validates the narrower D-FINE tensor contract.
class TrtSession {
 public:
    // Shape-transition state. set_input_shape is transactional: it either
    // commits (context, buffers, and the bindings_ cache all agree) or leaves
    // the previous buffers live and the cache untouched.
    //   kClean:    bindings_ mirrors the execution context; the de-dup fast
    //              path may trust the cache and infer may run.
    //   kDirty:    a transition failed after the context was touched; cache and
    //              context may disagree. Recoverable — the next set_input_shape
    //              re-issues the shape unconditionally; until it commits, every
    //              infer-path entry point throws instead of running.
    //   kPoisoned: an address rollback failed; the context may reference freed
    //              memory. Unrecoverable — every entry point throws; destroy
    //              and recreate the session.
    enum class ShapeState : std::uint8_t { kClean, kDirty, kPoisoned };

    // user_managed_memory: create the execution context with kUSER_MANAGED so TRT
    // does not allocate activation memory; the caller must supply it once via
    // set_device_memory() before the first infer (see device_memory_size()).
    explicit TrtSession(
        const std::filesystem::path& engine_path,
        nvinfer1::ILogger::Severity log_severity = nvinfer1::ILogger::Severity::kWARNING,
        bool user_managed_memory = false);
    ~TrtSession();

    // Non-movable by choice: the type is always owned in place (behind a PIMPL),
    // and the destructor drains/frees in a specific order. All members (incl. the
    // CudaStream) are RAII, so this is a policy choice, not a correctness crutch.
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;
    TrtSession(TrtSession&&) = delete;
    TrtSession& operator=(TrtSession&&) = delete;

    const std::vector<BindingInfo>& bindings() const noexcept { return bindings_; }
    const std::vector<int>& input_indices() const noexcept { return input_indices_; }
    const std::vector<int>& output_indices() const noexcept { return output_indices_; }

    int find_index(std::string_view name) const noexcept;
    const BindingInfo* find(std::string_view name) const noexcept;

    // Dynamic-shape input: set the actual shape before infer. Re-resolves all bindings;
    // grows device + pinned-host buffers if needed and rebinds context tensor addresses.
    // Transactional (see ShapeState): on failure the previous buffers/addresses stay
    // live, the cache stays at the last committed shape, and the session is kDirty
    // until a subsequent call commits. Frozen violations on the input binding are
    // rejected before the context is touched (kClean is preserved).
    void set_input_shape(std::string_view name, const nvinfer1::Dims& dims);

    [[nodiscard]] ShapeState shape_state() const noexcept { return shape_state_; }
    void require_ready(const char* operation) const { require_clean_(operation); }

    [[nodiscard]] int num_optimization_profiles() const noexcept;
    [[nodiscard]] InputProfileInfo input_profile(std::string_view name,
                                                 int profile_index = 0) const;

    // Copy host bytes -> pinned staging -> device. host_bytes must equal binding.bytes.
    void set_input(std::string_view name, const void* host_data, std::size_t bytes);

    // Enqueue inference without waiting. A rejected enqueue makes the execution
    // context unusable; subsequent operations fail until the session is recreated.
    void enqueue(const char* operation);

    // Sync infer: enqueueV3 on the stream and wait. H2D copies are already on the
    // stream from set_input.
    void infer();

    // Device -> pinned staging -> host. host_bytes must equal binding.bytes.
    void get_output(std::string_view name, void* host_data, std::size_t bytes);

    // Like get_output but always delivers float32 regardless of engine output dtype.
    // `host_float32` must hold `element_count` floats. Handles FP16->FP32 conversion.
    void get_output_f32(std::string_view name, float* host_float32, std::size_t element_count);

    // Direct access to a binding's device buffer. Use this when an upstream CUDA
    // kernel (e.g. preprocessing) wants to write into the TRT input without the
    // host-staging round-trip. Caller must enqueue work on the session's stream
    // (`stream()`) so the writes are visible to `infer()`.
    void* device_buffer(std::string_view name);
    const void* device_buffer(std::string_view name) const;

    cudaStream_t stream() const noexcept { return stream_.get(); }
    void synchronize(const char* operation);
    [[nodiscard]] bool drain_noexcept() noexcept;

    // --- Frozen-memory contract -----------------------------------------------------
    // Device-memory size TRT needs for activation across all profiles (upper bound).
    [[nodiscard]] int64_t device_memory_size() const noexcept;
    // Supply user-managed activation memory (kUSER_MANAGED contexts only). `ptr` must
    // stay alive for the context's lifetime; `size` >= device_memory_size().
    void set_device_memory(void* ptr, int64_t size);
    // Freeze: after this, a shape change that would grow a binding buffer throws
    // instead of reallocating — so device addresses never move (needed for CUDA-graph
    // capture) and no allocation happens on the steady-state path. Warm up at the max
    // shape first so every buffer has already reached peak capacity.
    void freeze() noexcept { frozen_ = true; }
    [[nodiscard]] bool frozen() const noexcept { return frozen_; }

    // Internal use — CUDA Graph capture hook.
    nvinfer1::IExecutionContext* context() const {
        require_clean_("context");
        return context_.get();
    }

    // Number of auxiliary streams the engine may launch kernels on. CUDA-graph
    // capture is only safe when this is 0 (or when those streams are wired as
    // non-blocking via setAuxStreams); the task layer gates capture on it.
    int num_aux_streams() const noexcept { return engine_ ? engine_->getNbAuxStreams() : 0; }

    // Helpers callers may want.
    static std::size_t dtype_bytes(nvinfer1::DataType d) noexcept;
    static int64_t volume(const nvinfer1::Dims& dims) noexcept;
    static const char* dtype_name(nvinfer1::DataType d) noexcept;

 private:
    void load_engine_(const std::filesystem::path& path);
    bool user_managed_memory_{false};
    bool frozen_{false};
    ShapeState shape_state_{ShapeState::kClean};
    std::string poison_reason_;
    void parse_bindings_();
    void allocate_buffers_();
    void free_buffers_() noexcept;
    void bind_address_(int idx);
    // Throws unless kClean: infer-path entry points must not run against a
    // context whose shape/addresses may not match the cached bindings.
    void require_clean_(const char* what) const;
    void poison_(std::string reason);
    [[noreturn]] void throw_frozen_grow_(const std::string& binding, std::size_t bytes) const;

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::vector<BindingInfo> bindings_;
    std::vector<DevPtr> device_buffers_;        // RAII cudaMalloc, parallel to bindings_
    std::vector<HostPtr> host_buffers_;         // RAII cudaMallocHost (pinned)
    std::vector<std::size_t> buffer_capacity_;  // current allocation in bytes
    std::vector<int> input_indices_;
    std::vector<int> output_indices_;

    CudaStream stream_;
};

}  // namespace dfine
