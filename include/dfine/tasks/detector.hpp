#pragma once

#include "dfine/core/types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dfine {

// Preprocessing geometry. D-FINE is trained with stretch-resize, which is the
// default and the accuracy-optimal choice for the published weights (measured:
// letterbox costs ~1.7-2.0 AP on COCO val). Letterbox exists for pipelines that
// standardize on aspect-preserving inputs; box coordinates are un-mapped (and
// clipped to the frame) automatically.
struct PreprocessSpec {
    enum class Resize {
        kAuto,      // follow the engine sidecar's "resize" field (default: stretch)
        kStretch,   // force stretch (the training convention)
        kLetterbox  // force letterbox with the fields below
    };
    Resize resize{Resize::kAuto};
    bool anchor_topleft{false};  // letterbox: false = centered content
    int pad_value{114};          // letterbox padding, 0..255
    bool allow_upscale{true};    // letterbox: false = paste 1:1 when the frame fits
};

struct DetectorOptions {
    float threshold{0.5f};  // finite default score threshold in [0,1]

    // Opt-in CUDA-graph replay of the engine (enqueueV3 + output D2H). Captures one
    // graph per batch size after warm-up and replays it, cutting per-frame launch
    // overhead; preprocess/H2D stay outside the graph. Falls back to plain enqueueV3
    // if capture fails (e.g. data-dependent internal shapes) or outputs aren't FP32.
    // Best for fixed-resolution, steady-batch streaming.
    bool use_cuda_graph{false};
    int graph_warmup_iters{3};  // full enqueue cycles before capture (TRT needs >=2)

    // Decode on the GPU and copy fixed top-K records instead of full logits/boxes.
    // Requires FP32 outputs; otherwise the detector uses CPU decode. Scores can
    // differ from CPU decode by one ULP. Takes precedence over use_cuda_graph.
    bool gpu_decode{false};

    // Own TRT's activation memory in one detector-allocated block (kUSER_MANAGED
    // + setDeviceMemoryV2). See freeze() for the fixed-memory contract.
    bool own_device_memory{false};

    // Capture input transfer, preprocessing, inference, GPU decode, and result
    // transfer in one CUDA graph, replayed with one cudaGraphLaunch per call.
    // Implies gpu_decode. The capture happens inside freeze(FreezeSpec) (before the
    // allocation lock) and requires a 0-aux-stream engine with FP32 outputs. After
    // freeze, calls whose batch/source-resolution/channel-order match the FreezeSpec
    // replay the graph. Other valid calls and recoverable capture failures use split
    // GPU decode with FP32 outputs, or CPU decode otherwise. Execution failures throw.
    // The score threshold remains configurable per call.
    bool full_pipeline_graph{false};

    // Preprocessing geometry (stretch by default; see PreprocessSpec).
    PreprocessSpec preprocess;
};

// Steady-state configuration freeze() warms, captures, and locks for. Zeros mean
// "engine defaults": batch = the engine's max batch, src_w/src_h = the engine input
// size. src_w/src_h must be both zero or both positive. Positive dimensions bound
// the largest source frame the frozen detector will accept — a larger frame after
// freeze() throws instead of allocating on the hot path. Negative values are invalid.
struct FreezeSpec {
    int batch{0};
    int src_w{0};
    int src_h{0};
    bool src_is_bgr{false};  // channel order the full-pipeline graph is captured for
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
    // Load an engine and auto-discover an optional sidecar: `<engine>.json`, then
    // the same-stem JSON.
    explicit DFineDetector(const std::filesystem::path& engine_path,
                           const DetectorOptions& opts = {});

    // Load engine with an explicit sidecar path.
    DFineDetector(const std::filesystem::path& engine_path, const std::filesystem::path& meta_path,
                  const DetectorOptions& opts = {});

    ~DFineDetector();

    DFineDetector(const DFineDetector&) = delete;
    DFineDetector& operator=(const DFineDetector&) = delete;
    DFineDetector(DFineDetector&&) noexcept;
    DFineDetector& operator=(DFineDetector&&) noexcept;

    // Detect on one HWC uint8 image. A finite threshold in [0,1] overrides the
    // constructor option; pass a finite negative value to use the default.
    [[nodiscard]] Detections detect(const ImageU8& image, float threshold = -1.0f);

    // Batch detect with the same threshold contract as detect(). An empty input returns
    // an empty result. Otherwise the count must be within the engine profile and frozen
    // bound. results[i] corresponds to images[i].
    [[nodiscard]] std::vector<Detections> detect_batch(const std::vector<ImageU8>& images,
                                                       float threshold = -1.0f);

    // Freeze the memory footprint: warm every grow-only buffer to peak (at `batch`,
    // default the engine max), then lock so the steady-state path performs no device
    // allocation and device addresses never move. A subsequent detect with a larger
    // batch than the frozen peak throws.
    //
    // Explicit src_w/src_h also bound and lock the preprocessing staging buffer and
    // define the configuration captured by full_pipeline_graph. freeze(batch) leaves
    // source dimensions unbounded, so a larger source may grow preprocessing staging.
    //
    // Re-freezing with the same resolved configuration is a no-op; a different
    // configuration throws (locked buffers cannot be re-sized — create a new
    // detector to reconfigure).
    void freeze(int batch = 0);
    void freeze(const FreezeSpec& spec);

    // True once freeze(FreezeSpec) captured the full-pipeline graph. Matching calls
    // then run as one cudaGraphLaunch; other calls use the configured fallback path.
    [[nodiscard]] bool full_pipeline_graph_active() const noexcept;
    // Number of calls served by full-graph replay so far (observability: a frozen
    // fixed-resolution pipeline should show this equal to its frame count).
    [[nodiscard]] std::uint64_t full_graph_replays() const noexcept;
    // Number of calls served by enqueue-plus-output-copy CUDA Graph replay.
    [[nodiscard]] std::uint64_t cuda_graph_replays() const noexcept;

    [[nodiscard]] const std::string& variant() const noexcept;
    [[nodiscard]] int input_h() const noexcept;
    [[nodiscard]] int input_w() const noexcept;
    [[nodiscard]] int num_queries() const noexcept;
    [[nodiscard]] int num_classes() const noexcept;
    // Display name for a class id: the sidecar's class_names entry when present,
    // else the COCO-80 table when the engine has 80 classes, else "class_<id>".
    [[nodiscard]] std::string class_name(int class_id) const;
    // 1 for a static engine; profile 0 max for a dynamic engine.
    [[nodiscard]] int max_batch() const noexcept;

    struct Timings {
        double preprocess_ms{0};  // merged into infer_ms (async on the stream)
        double infer_ms{0};
        double postprocess_ms{0};
        double total_ms{0};

        // Host-side (CPU) wall time per stage of the last call. Separates what the
        // CPU spends ISSUING work from what it spends WAITING on the GPU — the
        // dispatch cost is what the full-pipeline graph collapses (hundreds of
        // kernel launches -> one cudaGraphLaunch). preprocess_cpu_ms covers the
        // frame pack + H2D/kernel issue loop (or the pinned-slab pack on the
        // full-graph path); dispatch_ms covers enqueueV3 + decode-kernel issue (or
        // the single cudaGraphLaunch); wait_ms is the final stream sync;
        // decode_host_ms is the CPU decode or survivor->Detections conversion.
        double preprocess_cpu_ms{0};
        double dispatch_ms{0};
        double wait_ms{0};
        double decode_host_ms{0};
    };
    [[nodiscard]] const Timings& last_timings() const noexcept;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void init_(const std::filesystem::path& engine_path, const std::filesystem::path& meta_path,
               bool explicit_meta, const DetectorOptions& opts);
};

}  // namespace dfine
