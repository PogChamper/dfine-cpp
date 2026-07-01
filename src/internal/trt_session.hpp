#pragma once

#include "cuda_raii.hpp"
#include "trt_logger.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

// This library uses the TensorRT 10+ tensor-based API (enqueueV3, setTensorAddress,
// getNbIOTensors). TRT 9 and below are not supported.
#if NV_TENSORRT_MAJOR < 10
#  error "D-FINE-cpp requires TensorRT >= 10.0. Please upgrade your TensorRT installation."
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
    nvinfer1::Dims shape;     // resolved shape; dims may be -1 if dynamic and not yet set
    bool is_input{false};
    int64_t element_count{0}; // 0 = shape has unresolved dim(s)
    std::size_t bytes{0};     // element_count * sizeof(dtype)
};

// Owns a TRT runtime, engine, execution context, per-binding device + pinned-host
// buffers, and a CUDA stream. Does sync inference (H2D -> enqueueV3 -> D2H + sync).
// Name-driven and model-agnostic: callers pass the D-FINE tensor names
// ("images", "logits", "boxes"); nothing here is hardcoded.
class TrtSession {
   public:
    explicit TrtSession(const std::filesystem::path& engine_path,
                        nvinfer1::ILogger::Severity log_severity =
                            nvinfer1::ILogger::Severity::kWARNING);
    ~TrtSession();

    // Non-movable by choice: the type is always owned in place (behind a PIMPL),
    // and the destructor drains/frees in a specific order. All members (incl. the
    // CudaStream) are RAII, so this is a policy choice, not a correctness crutch.
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;
    TrtSession(TrtSession&&) = delete;
    TrtSession& operator=(TrtSession&&) = delete;

    const std::vector<BindingInfo>& bindings()        const noexcept { return bindings_; }
    const std::vector<int>&         input_indices()   const noexcept { return input_indices_; }
    const std::vector<int>&         output_indices()  const noexcept { return output_indices_; }

    int find_index(std::string_view name) const noexcept;
    const BindingInfo* find(std::string_view name) const noexcept;

    // Dynamic-shape input: set the actual shape before infer. Re-resolves all bindings;
    // grows device + pinned-host buffers if needed and rebinds context tensor addresses.
    void set_input_shape(std::string_view name, const nvinfer1::Dims& dims);

    // Copy host bytes -> pinned staging -> device. host_bytes must equal binding.bytes.
    void set_input(std::string_view name, const void* host_data, std::size_t bytes);

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
    void*       device_buffer(std::string_view name);
    const void* device_buffer(std::string_view name) const;

    cudaStream_t stream() const noexcept { return stream_.get(); }

    // Internal use — CUDA Graph capture hook.
    nvinfer1::IExecutionContext* context() const noexcept { return context_.get(); }

    // Number of auxiliary streams the engine may launch kernels on. CUDA-graph
    // capture is only safe when this is 0 (or when those streams are wired as
    // non-blocking via setAuxStreams); the task layer gates capture on it.
    int num_aux_streams() const noexcept { return engine_ ? engine_->getNbAuxStreams() : 0; }

    // Helpers callers may want.
    static std::size_t dtype_bytes(nvinfer1::DataType d) noexcept;
    static int64_t     volume(const nvinfer1::Dims& dims) noexcept;
    static const char* dtype_name(nvinfer1::DataType d) noexcept;

   private:
    void load_engine_(const std::filesystem::path& path);
    void parse_bindings_();
    void allocate_buffers_();
    void free_buffers_() noexcept;
    void update_binding_shape_(int idx, const nvinfer1::Dims& dims);
    void bind_address_(int idx);

    TrtLogger                                       logger_;
    std::unique_ptr<nvinfer1::IRuntime>             runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>          engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>    context_;

    std::vector<BindingInfo> bindings_;
    std::vector<DevPtr>      device_buffers_;    // RAII cudaMalloc, parallel to bindings_
    std::vector<HostPtr>     host_buffers_;      // RAII cudaMallocHost (pinned)
    std::vector<std::size_t> buffer_capacity_;   // current allocation in bytes
    std::vector<int>         input_indices_;
    std::vector<int>         output_indices_;

    CudaStream stream_;
};

}  // namespace dfine
