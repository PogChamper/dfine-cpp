#pragma once

#include "dfine/core/types.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dfine {

struct DetectorOptions {
    float threshold{0.5f};  // default score threshold; override per detect() call

    // Opt-in CUDA-graph replay of the engine (enqueueV3 + output D2H). Captures one
    // graph per batch size after warm-up and replays it, cutting per-frame launch
    // overhead; preprocess/H2D stay outside the graph. Falls back to plain enqueueV3
    // if capture fails (e.g. data-dependent internal shapes) or outputs aren't FP32.
    // Best for fixed-resolution, steady-batch streaming.
    bool use_cuda_graph{false};
    int  graph_warmup_iters{3};  // full enqueue cycles before capture (TRT needs >=2)

    // Opt-in GPU-side decode (Zero-D2H): run sigmoid/top-k/threshold/box-decode as
    // CUDA kernels reading the engine outputs on-device, so only the survivors cross
    // PCIe instead of the full logits, and the CPU does no per-frame decode work.
    // Requires FP32 engine outputs (falls back to the CPU decode otherwise). Results
    // are mAP-equivalent to the CPU decode (the score differs by <=1 ULP: GPU expf
    // vs libm), not byte-identical. Takes precedence over use_cuda_graph for now.
    bool gpu_decode{false};

    // Own TRT's activation memory in a single detector-allocated block (kUSER_MANAGED
    // + setDeviceMemoryV2) instead of letting TRT self-manage it — part of the
    // frozen-memory contract (see freeze()). No accuracy or mean-latency effect.
    bool own_device_memory{false};
};

// Public D-FINE detector. Hides all TensorRT/CUDA (and OpenCV) headers behind a
// PIMPL — consumers work with plain `ImageU8` views and `Detection` structs.
//
// Pipeline: CUDA stretch-resize + /255 preprocess -> TRT engine -> C++ decode
// (sigmoid, top-k, cxcywh->xyxy). The engine owns the deformable-attention core;
// this class only orchestrates preprocess, inference, and decode.
//
// Thread safety: not thread-safe. Use one instance per thread.
class DFineDetector {
   public:
    // Load engine + sidecar `<engine_path>.json`.
    explicit DFineDetector(const std::filesystem::path& engine_path,
                           const DetectorOptions& opts = {});

    // Load engine with an explicit sidecar path.
    DFineDetector(const std::filesystem::path& engine_path,
                  const std::filesystem::path& meta_path, const DetectorOptions& opts = {});

    ~DFineDetector();

    DFineDetector(const DFineDetector&)            = delete;
    DFineDetector& operator=(const DFineDetector&) = delete;
    DFineDetector(DFineDetector&&) noexcept;
    DFineDetector& operator=(DFineDetector&&) noexcept;

    // Detect on one HWC uint8 image. `threshold` overrides the constructor option
    // (pass < 0 to use the default).
    [[nodiscard]] Detections detect(const ImageU8& image, float threshold = -1.0f);

    // Batch detect. Requires an engine built with max_batch >= images.size().
    // results[i] holds the detections for images[i].
    [[nodiscard]] std::vector<Detections> detect_batch(const std::vector<ImageU8>& images,
                                                        float threshold = -1.0f);

    // Freeze the memory footprint: warm every grow-only buffer to peak (at `batch`,
    // default the engine max), then lock so the steady-state path performs no device
    // allocation and device addresses never move. A subsequent detect with a larger
    // batch than the frozen peak throws. Idempotent. (Intensive-core P2.)
    void freeze(int batch = 0);

    [[nodiscard]] const std::string& variant()     const noexcept;
    [[nodiscard]] int                input_h()     const noexcept;
    [[nodiscard]] int                input_w()     const noexcept;
    [[nodiscard]] int                num_queries() const noexcept;
    [[nodiscard]] int                num_classes() const noexcept;
    // 1 for a static engine; the profile max for a dynamic engine; 0 if dynamic
    // but the max is unknown (no/partial sidecar).
    [[nodiscard]] int                max_batch()   const noexcept;

    struct Timings {
        double preprocess_ms{0};   // merged into infer_ms (async on the stream)
        double infer_ms{0};
        double postprocess_ms{0};
        double total_ms{0};
    };
    [[nodiscard]] const Timings& last_timings() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void init_(const std::filesystem::path& engine_path,
               const std::filesystem::path& meta_path, const DetectorOptions& opts);
};

}  // namespace dfine
