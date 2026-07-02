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
    float threshold{0.5f};  // default score threshold; override per detect() call

    // Opt-in CUDA-graph replay of the engine (enqueueV3 + output D2H). Captures one
    // graph per batch size after warm-up and replays it, cutting per-frame launch
    // overhead; preprocess/H2D stay outside the graph. Falls back to plain enqueueV3
    // if capture fails (e.g. data-dependent internal shapes) or outputs aren't FP32.
    // Best for fixed-resolution, steady-batch streaming.
    bool use_cuda_graph{false};
    int graph_warmup_iters{3};  // full enqueue cycles before capture (TRT needs >=2)

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

    // Opt-in single-launch steady state (intensive-core P3): capture the ENTIRE
    // per-frame pipeline — input H2D, preprocess, enqueueV3, GPU decode, survivor
    // D2H — into one CUDA graph, replayed with a single cudaGraphLaunch per call.
    // Implies gpu_decode. The capture happens inside freeze(FreezeSpec) (before the
    // allocation lock) and requires a 0-aux-stream engine with FP32 outputs. After
    // freeze, calls whose batch/source-resolution/channel-order match the FreezeSpec
    // replay the graph; anything else falls back (no-throw) to the split gpu_decode
    // path. The score threshold stays a live per-call knob (read by the captured
    // decode kernel through mapped pinned memory, not baked into the graph).
    bool full_pipeline_graph{false};

    // Preprocessing geometry (stretch by default; see PreprocessSpec).
    PreprocessSpec preprocess;
};

// Steady-state configuration freeze() warms, captures, and locks for. Zeros mean
// "engine defaults": batch = the engine's max batch, src_w/src_h = the engine input
// size. src_w/src_h bound the LARGEST source frame the frozen detector will accept —
// a larger frame after freeze() throws instead of allocating on the hot path.
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
    // Load engine + sidecar `<engine_path>.json`.
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
    // batch than the frozen peak throws. (Intensive-core P2.)
    //
    // A spec with explicit src_w/src_h additionally bounds the steady-state SOURCE
    // frame: the preprocessor staging is warmed to that peak and locked (a larger
    // frame afterwards throws instead of allocating on the hot path), and it is the
    // configuration DetectorOptions.full_pipeline_graph captures its graph for
    // (intensive-core P3). freeze(batch) leaves the source size unbounded — the
    // legacy P2 behavior, where an oversized frame still grows the preprocessor
    // staging on the hot path (the one documented exception to the zero-allocation
    // contract).
    //
    // Re-freezing with the same resolved configuration is a no-op; a DIFFERENT
    // configuration throws (locked buffers cannot be re-sized — create a new
    // detector to reconfigure).
    void freeze(int batch = 0);
    void freeze(const FreezeSpec& spec);

    // True once freeze(FreezeSpec) captured the full-pipeline graph (P3): matching
    // calls now run as one cudaGraphLaunch. False = not requested / not capturable
    // (aux streams, non-FP32 outputs) / capture failed — split path in effect.
    [[nodiscard]] bool full_pipeline_graph_active() const noexcept;
    // Number of calls served by full-graph replay so far (observability: a frozen
    // fixed-resolution pipeline should show this equal to its frame count).
    [[nodiscard]] std::uint64_t full_graph_replays() const noexcept;

    [[nodiscard]] const std::string& variant() const noexcept;
    [[nodiscard]] int input_h() const noexcept;
    [[nodiscard]] int input_w() const noexcept;
    [[nodiscard]] int num_queries() const noexcept;
    [[nodiscard]] int num_classes() const noexcept;
    // Display name for a class id: the sidecar's class_names entry when present,
    // else the COCO-80 table when the engine has 80 classes, else "class_<id>".
    [[nodiscard]] std::string class_name(int class_id) const;
    // 1 for a static engine; the profile max for a dynamic engine; 0 if dynamic
    // but the max is unknown (no/partial sidecar).
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
               const DetectorOptions& opts);
};

}  // namespace dfine
