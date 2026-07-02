#include "dfine/tasks/detector.hpp"

#include "dfine/core/coco_classes.hpp"
#include "dfine/core/engine_meta.hpp"
#include "dfine/core/log.hpp"
#include "dfine/core/postprocess.hpp"
#include "internal/cuda_check.hpp"
#include "internal/cuda_preprocess.cuh"
#include "internal/decode_gpu.cuh"
#include "internal/device_arena.hpp"
#include "internal/trt_session.hpp"

#include "internal/cuda_raii.hpp"

#include <memory>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dfine {

struct DFineDetector::Impl {
    static double ms_(std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    // Declared first so it is destroyed LAST (after `session` and its context): TRT
    // requires user-managed device memory to outlive the execution context.
    DevPtr act_mem_;  // TRT activation block (own_device_memory)

    TrtSession session;
    EngineMeta meta;
    std::unique_ptr<ImagePreprocessor> preprocessor;
    DetectorOptions opts;

    std::string input_name;
    const BindingInfo* b_logits{nullptr};
    const BindingInfo* b_boxes{nullptr};
    int N{0};  // num_queries
    int C{0};  // num_classes
    int in_h_{640};
    int in_w_{640};
    bool input_dynamic_{false};  // batch axis is dynamic (-1)
    int max_batch_{1};           // enforced upper bound for detect_batch

    std::vector<float> h_logits;
    std::vector<float> h_boxes;

    PostprocessParams pp;
    Timings timings;

    // --- CUDA-graph replay state (opt-in via DetectorOptions.use_cuda_graph) ------
    // One instantiated graph per batch size, each capturing enqueueV3 + the output
    // D2H copies into detector-owned pinned buffers. The graph bakes device/host
    // addresses, so we record them and re-capture if a buffer realloc moves them.
    struct GraphEntry {
        CudaGraphExec exec;
        void* d_input{nullptr};
        void* d_logits{nullptr};
        void* d_boxes{nullptr};
        void* p_logits{nullptr};
        void* p_boxes{nullptr};
    };
    std::unordered_map<int, GraphEntry> graphs_;
    HostPtr pinned_logits_;
    HostPtr pinned_boxes_;
    std::size_t pinned_logits_cap_{0};
    std::size_t pinned_boxes_cap_{0};
    int graph_ctx_batch_{-1};      // batch the context is configured + flushed for
    bool graph_supported_{false};  // FP32 outputs and no aux streams
    bool graph_disabled_{false};   // a capture attempt failed unrecoverably; stop trying
    bool graph_warned_{false};

    // --- GPU decode (Zero-D2H) state (opt-in via DetectorOptions.gpu_decode) -------
    // Kernels read the engine's FP32 logits/boxes on-device and emit only the compact
    // survivors ([B*N] DetectionGPU + counts) — no full-logits D2H, no CPU decode.
    bool gpu_decode_supported_{false};  // FP32 outputs
    bool gpu_decode_warned_{false};
    GpuDecodeScratch gdec_;                     // raw device pointers into gdec_arena_
    std::unique_ptr<DeviceArena> gdec_arena_;   // one block backing all 9 scratch slabs
    int gdec_cap_batch_{0};                     // buffers sized for this many images
    std::vector<DetectionGPU> gdec_host_out_;   // D2H compact results [cap*N]
    std::vector<uint32_t> gdec_host_counts_;    // survivors per image [cap]
    std::vector<DecodeMapGPU> gdec_maps_host_;  // per-image coordinate maps, H2D staging

    // --- Preprocessing geometry (resolved once: options override > sidecar) --------
    bool letterbox_{false};
    bool lb_topleft_{false};
    int lb_pad_{114};
    bool lb_upscale_{true};

    // --- Frozen-memory contract (P2) ----------------------------------------------
    // (act_mem_ is declared at the top of Impl for destruction ordering.)
    bool frozen_{false};   // freeze() called: no further device allocation allowed
    int frozen_batch_{0};  // resolved config freeze() locked (re-freeze must match)
    int frozen_src_w_{0};
    int frozen_src_h_{0};
    bool frozen_bgr_{false};

    // --- Full-pipeline graph (P3) state (opt-in via full_pipeline_graph) -----------
    // One graph spanning H2D(frames) -> preprocess -> enqueueV3 -> GPU decode ->
    // survivor D2H, captured inside freeze_() BEFORE the allocation lock. Every
    // address it bakes is frozen by construction, so there is no staleness tracking
    // (contrast graphs_/graph_stale_): the graph stays valid for the process
    // lifetime. Replays are gated on the exact frozen configuration (batch, source
    // size, channel order); anything else falls back to the split gpu_decode path.
    CudaGraphExec full_exec_;
    int full_batch_{0};
    int full_src_w_{0};
    int full_src_h_{0};
    bool full_is_bgr_{false};
    bool full_ready_{false};
    std::uint64_t full_replays_{0};
    std::size_t frame_slot_bytes_{0};  // packed src_h * src_w * 3; one slot per batch image
    HostPtr h_frames_;                 // pinned input slab [full_batch_ slots]; host packs here
    DevPtr d_frames_;                  // device twin; the graph H2Ds slot-by-slot from h_frames_
    HostPtr h_survivors_;              // pinned [full_batch_ * N] DetectionGPU (graph D2H target)
    HostPtr h_counts_;                 // pinned [full_batch_] uint32_t (graph D2H target)
    HostPtr h_scale_;                  // pinned [full_batch_] DecodeMapGPU, constant per config
    HostPtr h_thr_;                    // mapped pinned float: live threshold (see decode_gpu.cuh)
    float* d_thr_{nullptr};            // device alias of h_thr_

    Impl(const std::filesystem::path& engine_path, const std::filesystem::path& meta_path,
         const DetectorOptions& options)
        : session(engine_path, nvinfer1::ILogger::Severity::kWARNING, options.own_device_memory),
          opts(options) {
        // Resolve the sidecar: the given path, else `<engine-stem>.json` (the
        // convention the Python build_engine.py writes, e.g. dfine_m_fp32.json).
        std::filesystem::path mp = meta_path;
        if (!std::filesystem::is_regular_file(mp)) {
            std::filesystem::path alt = engine_path;
            alt.replace_extension(".json");
            if (std::filesystem::is_regular_file(alt)) mp = alt;
        }
        bool have_meta = false;
        if (std::filesystem::is_regular_file(mp)) {
            meta = EngineMeta::from_json_file(mp);
            have_meta = true;
        } else {
            log_message(LogSeverity::kWarning,
                        "no meta sidecar found; assuming D-FINE defaults "
                        "(/255, RGB, dims/classes read from engine bindings)");
        }

        // Resolve the input tensor name (sidecar, else first input binding).
        input_name = meta.input_names.empty() ? std::string{"images"} : meta.input_names.front();
        const BindingInfo* in_b = session.find(input_name);
        if (!in_b) {
            const auto& ins = session.input_indices();
            if (ins.empty()) throw std::runtime_error("dfine: engine has no input tensor");
            in_b = &session.bindings()[ins.front()];
            input_name = in_b->name;
        }
        // The CUDA preprocess writes float32 straight into the input device buffer,
        // so a non-FP32 input binding would be a size mismatch / out-of-bounds write.
        if (in_b->dtype != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "dfine: input tensor '" + input_name +
                "' is not FP32; rebuild the engine with an FP32 (fp32:chw) input.");
        }

        // Trust the engine bindings for shape facts — a stale/absent sidecar must
        // not silently misconfigure preprocessing or leave a dynamic input unset.
        input_dynamic_ = in_b->shape.nbDims >= 1 && in_b->shape.d[0] < 0;
        if (in_b->shape.nbDims == 4) {
            if (in_b->shape.d[2] > 0) in_h_ = static_cast<int>(in_b->shape.d[2]);
            if (in_b->shape.d[3] > 0) in_w_ = static_cast<int>(in_b->shape.d[3]);
        } else {
            in_h_ = meta.input_h;
            in_w_ = meta.input_w;
        }
        max_batch_ = (have_meta && meta.dynamic_batch) ? meta.max_batch : (input_dynamic_ ? 0 : 1);

        resolve_outputs_();

        // num_queries / num_classes are batch-independent — read from the (possibly
        // still dynamic-batch) logits binding: shape [*, N, C].
        const int nb = b_logits->shape.nbDims;
        N = (nb >= 2) ? static_cast<int>(b_logits->shape.d[nb - 2]) : meta.num_queries;
        C = (nb >= 1) ? static_cast<int>(b_logits->shape.d[nb - 1]) : meta.num_classes;
        if (N <= 0) N = meta.num_queries;
        if (C <= 0) C = meta.num_classes;

        preprocessor = std::make_unique<ImagePreprocessor>(in_h_, in_w_);
        preprocessor->set_mean(meta.mean[0], meta.mean[1], meta.mean[2]);
        preprocessor->set_std(meta.std[0], meta.std[1], meta.std[2]);

        // Preprocessing geometry: an explicit option wins, else the sidecar's
        // "resize" field, else stretch (the training convention). The stretch
        // path is untouched — its detections stay byte-identical.
        using Resize = PreprocessSpec::Resize;
        if (opts.preprocess.resize == Resize::kLetterbox) {
            letterbox_ = true;
            lb_topleft_ = opts.preprocess.anchor_topleft;
            lb_pad_ = opts.preprocess.pad_value;
            lb_upscale_ = opts.preprocess.allow_upscale;
        } else if (opts.preprocess.resize == Resize::kAuto && meta.resize == "letterbox") {
            letterbox_ = true;
            lb_topleft_ = meta.letterbox_anchor == "topleft";
            lb_pad_ = meta.letterbox_pad;
            lb_upscale_ = meta.letterbox_upscale;
        }
        if (letterbox_) preprocessor->set_letterbox(lb_topleft_, lb_pad_, lb_upscale_);

        pp.num_queries = N;
        pp.num_classes = C;
        pp.topk = N;
        pp.threshold = opts.threshold;

        h_logits.resize(static_cast<std::size_t>(N) * C);
        h_boxes.resize(static_cast<std::size_t>(N) * 4);

        // CUDA graph is only viable when both outputs are FP32 (the graph does a raw
        // D2H, no dtype conversion) and the engine spawns no auxiliary streams that a
        // ThreadLocal capture would miss (P12 §5c). Otherwise we silently use the
        // plain enqueueV3 path, which handles FP16 outputs via get_output_f32.
        graph_supported_ = b_logits->dtype == nvinfer1::DataType::kFLOAT &&
                           b_boxes->dtype == nvinfer1::DataType::kFLOAT &&
                           session.num_aux_streams() == 0;

        // GPU decode only needs FP32 outputs — it runs after enqueueV3 on the main
        // stream, so (unlike graph capture) it is fine with aux streams.
        gpu_decode_supported_ = b_logits->dtype == nvinfer1::DataType::kFLOAT &&
                                b_boxes->dtype == nvinfer1::DataType::kFLOAT;

        // full_pipeline_graph is a superset of gpu_decode (the captured tail IS the
        // GPU decode), so requesting it implies gpu_decode as the warmup/fallback
        // path. Capturability (0 aux streams + FP32 outputs) is checked at freeze.
        if (opts.full_pipeline_graph) opts.gpu_decode = true;

        // Own TRT's activation memory in a single block (context was created
        // kUSER_MANAGED). The size is static (all profiles), so allocate once now and
        // hand it to the context before any inference.
        if (opts.own_device_memory) {
            const int64_t sz = session.device_memory_size();
            if (sz > 0) {
                void* p = nullptr;
                DFINE_CUDA_CHECK(cudaMalloc(&p, static_cast<std::size_t>(sz)));
                act_mem_.reset(p);
                session.set_device_memory(p, sz);
            }
        }
    }

    // Locate the logits/boxes output bindings by name, falling back to shape
    // (the box tensor is the output whose last dim == 4).
    void resolve_outputs_() {
        b_logits = session.find("logits");
        b_boxes = session.find("boxes");
        if (meta.output_names.size() >= 2) {
            if (!b_logits) b_logits = session.find(meta.output_names[0]);
            if (!b_boxes) b_boxes = session.find(meta.output_names[1]);
        }
        if (!b_logits || !b_boxes) {
            const auto& outs = session.output_indices();
            if (outs.size() < 2) throw std::runtime_error("dfine: engine has fewer than 2 outputs");
            auto last_dim = [](const BindingInfo* x) {
                return x->shape.nbDims > 0 ? x->shape.d[x->shape.nbDims - 1] : -1;
            };
            // Shape heuristic, restricted to the documented raw D-FINE contract of
            // EXACTLY two outputs where boxes is the unique [..., 4] tensor. More
            // outputs, or a 4-class model (both outputs [*, N, 4]), cannot be
            // resolved by shape — fail loudly rather than guess either tensor.
            if (outs.size() != 2) {
                throw std::runtime_error(
                    "dfine: engine has " + std::to_string(outs.size()) +
                    " outputs and no "
                    "'logits'/'boxes' tensor names; name the tensors at export or provide "
                    "the .json sidecar with output_names");
            }
            const BindingInfo* box_cand = nullptr;
            const BindingInfo* other = nullptr;
            int n_box = 0;
            for (int oi : outs) {
                const BindingInfo* x = &session.bindings()[oi];
                if (last_dim(x) == 4) {
                    box_cand = x;
                    ++n_box;
                } else {
                    other = x;
                }
            }
            if (n_box != 1 || !other) {
                throw std::runtime_error(
                    "dfine: cannot disambiguate logits/boxes outputs by shape (e.g. a 4-class "
                    "model has two [*, N, 4] outputs); name the tensors 'logits'/'boxes' at "
                    "export or provide the .json sidecar with output_names");
            }
            b_boxes = box_cand;
            b_logits = other;
        }
    }

    // Per-image normalized-canvas -> original-pixels map for the decode. Stretch
    // yields the historical {W, 0, H, 0, no-clip} identity; letterbox un-maps
    // through the same LetterboxMap the preprocessor used and clips to the frame.
    DecodeMap make_map_(int src_w, int src_h) const noexcept {
        DecodeMap m;
        if (letterbox_) {
            const LetterboxMap lb =
                compute_letterbox_map(src_w, src_h, in_w_, in_h_, lb_topleft_, lb_upscale_);
            m.sx = static_cast<float>(in_w_) / lb.s;
            m.ox = static_cast<float>(lb.dx) / lb.s;
            m.sy = static_cast<float>(in_h_) / lb.s;
            m.oy = static_cast<float>(lb.dy) / lb.s;
            m.clip_w = static_cast<float>(src_w);
            m.clip_h = static_cast<float>(src_h);
        } else {
            m.sx = static_cast<float>(src_w);
            m.sy = static_cast<float>(src_h);
        }
        return m;
    }

    static DecodeMapGPU to_gpu_(const DecodeMap& m) noexcept {
        return DecodeMapGPU{m.sx, m.ox, m.sy, m.oy, m.clip_w, m.clip_h};
    }

    // Set the dynamic input shape for batch B (or validate a static engine).
    void set_batch_(int B) {
        if (input_dynamic_) {
            if (max_batch_ > 0 && B > max_batch_) {
                throw std::runtime_error("dfine: batch size " + std::to_string(B) +
                                         " exceeds engine max_batch " + std::to_string(max_batch_));
            }
            session.set_input_shape(input_name, nvinfer1::Dims4{B, 3, in_h_, in_w_});
        } else if (B != 1) {
            throw std::runtime_error(
                "dfine: engine is static-batch; rebuild with a dynamic batch profile for B>1");
        }
    }

    // Grow the detector-owned pinned output buffers to hold at least the given sizes.
    // Returns false (no throw) on allocation failure so capture_graph_ can fall back.
    bool ensure_pinned_(std::size_t logits_bytes, std::size_t boxes_bytes) {
        if (pinned_logits_cap_ < logits_bytes) {
            void* p = nullptr;
            if (cudaMallocHost(&p, logits_bytes) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            pinned_logits_.reset(p);
            pinned_logits_cap_ = logits_bytes;
        }
        if (pinned_boxes_cap_ < boxes_bytes) {
            void* p = nullptr;
            if (cudaMallocHost(&p, boxes_bytes) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            pinned_boxes_.reset(p);
            pinned_boxes_cap_ = boxes_bytes;
        }
        return true;
    }

    // A cached graph is stale once any baked device/host address moves (a grow-only
    // buffer realloc after a larger batch was seen). Cheap 5-pointer check per replay.
    bool graph_stale_(const GraphEntry& g) const {
        return g.d_input != session.device_buffer(input_name) ||
               g.d_logits != session.device_buffer(b_logits->name) ||
               g.d_boxes != session.device_buffer(b_boxes->name) ||
               g.p_logits != pinned_logits_.get() || g.p_boxes != pinned_boxes_.get();
    }

    // Capture enqueueV3 + output D2H for the current (already-set, already-flushed)
    // batch B into a replayable graph. Returns false on any capture failure, leaving
    // the context usable for the plain enqueueV3 path. H2D/preprocess stay outside.
    bool capture_graph_(int B) {
        const std::size_t logits_bytes = static_cast<std::size_t>(B) * N * C * sizeof(float);
        const std::size_t boxes_bytes = static_cast<std::size_t>(B) * N * 4 * sizeof(float);
        // No-throw on any CUDA failure below: a false return lets try_graph_infer_ set
        // graph_disabled_ and run_batch degrade to the plain enqueueV3 path, which
        // reuses the session's own pinned buffers and allocates nothing new.
        if (!ensure_pinned_(logits_bytes, boxes_bytes)) return false;

        void* d_input = session.device_buffer(input_name);
        void* d_logits = session.device_buffer(b_logits->name);
        void* d_boxes = session.device_buffer(b_boxes->name);
        cudaStream_t stream = session.stream();
        auto* ctx = session.context();

        ctx->setEnqueueEmitsProfile(false);  // profiling is not capturable (P12 §5b)

        // Warm-up: >=2 full enqueue cycles at this exact shape flush TRT's deferred
        // shape setup, which would otherwise sync-abort the capture (P12 §5a).
        const int warm = opts.graph_warmup_iters < 2 ? 2 : opts.graph_warmup_iters;
        auto d2h = [&](void* dst, void* src, std::size_t n) {
            return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        };
        for (int w = 0; w < warm; ++w) {
            if (!ctx->enqueueV3(stream) || !d2h(pinned_logits_.get(), d_logits, logits_bytes) ||
                !d2h(pinned_boxes_.get(), d_boxes, boxes_bytes)) {
                cudaGetLastError();
                return false;
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError();  // clear sticky error
            return false;
        }
        const bool enq_ok = ctx->enqueueV3(stream);
        cudaMemcpyAsync(pinned_logits_.get(), d_logits, logits_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(pinned_boxes_.get(), d_boxes, boxes_bytes, cudaMemcpyDeviceToHost, stream);
        cudaGraph_t graph_raw = nullptr;
        const cudaError_t end_err = cudaStreamEndCapture(stream, &graph_raw);
        if (!enq_ok || end_err != cudaSuccess || graph_raw == nullptr) {
            cudaGetLastError();
            if (graph_raw) cudaGraphDestroy(graph_raw);
            return false;
        }
        CudaGraph graph(graph_raw);

        cudaGraphExec_t exec_raw = nullptr;
        if (cudaGraphInstantiate(&exec_raw, graph.get(), 0) != cudaSuccess || exec_raw == nullptr) {
            cudaGetLastError();
            return false;
        }
        GraphEntry e;
        e.exec = CudaGraphExec(exec_raw);
        e.d_input = d_input;
        e.d_logits = d_logits;
        e.d_boxes = d_boxes;
        e.p_logits = pinned_logits_.get();
        e.p_boxes = pinned_boxes_.get();
        graphs_[B] = std::move(e);
        return true;
    }

    // Try to run inference for batch B via graph replay. Returns true if the graph
    // path produced the outputs (into h_logits/h_boxes); false means the caller must
    // use the plain enqueueV3 path. Only replays when the context shape is stable
    // (== last flushed batch), so a graph never runs against an unflushed shape.
    // dispatch_ms/wait_ms report the replay's CPU issue/wait cost (Timings contract).
    bool try_graph_infer_(int B, double& dispatch_ms, double& wait_ms) {
        if (B != graph_ctx_batch_) return false;  // shape just (re)set; let enqueueV3 flush it

        auto it = graphs_.find(B);
        if (it != graphs_.end() && graph_stale_(it->second)) {
            graphs_.erase(it);
            it = graphs_.end();
        }
        if (it == graphs_.end()) {
            if (!capture_graph_(B)) {
                graph_disabled_ = true;
                return false;
            }
            it = graphs_.find(B);
        }

        const auto td0 = std::chrono::steady_clock::now();
        DFINE_CUDA_CHECK(cudaGraphLaunch(it->second.exec.get(), session.stream()));
        const auto td1 = std::chrono::steady_clock::now();
        DFINE_CUDA_CHECK(cudaStreamSynchronize(session.stream()));

        const std::size_t logits_n = static_cast<std::size_t>(B) * N * C;
        const std::size_t boxes_n = static_cast<std::size_t>(B) * N * 4;
        if (h_logits.size() < logits_n) h_logits.resize(logits_n);
        if (h_boxes.size() < boxes_n) h_boxes.resize(boxes_n);
        std::memcpy(h_logits.data(), it->second.p_logits, logits_n * sizeof(float));
        std::memcpy(h_boxes.data(), it->second.p_boxes, boxes_n * sizeof(float));
        const auto td2 = std::chrono::steady_clock::now();
        dispatch_ms = ms_(td0, td1);
        wait_ms = ms_(td1, td2);  // sync + pinned->host copies (matches the plain path)
        return true;
    }

    // (Re)allocate the GPU-decode scratch to hold at least `B` images (grow-only).
    // seg_off is batch-invariant, so a scratch sized for `cap` serves any B <= cap.
    void ensure_gpu_decode_(int B) {
        if (B <= gdec_cap_batch_) return;
        if (frozen_) {
            throw std::runtime_error("dfine: detector is frozen but gpu_decode needs batch " +
                                     std::to_string(B) + " > frozen max " +
                                     std::to_string(gdec_cap_batch_));
        }
        const int cap = B > max_batch_ ? B : (max_batch_ > 0 ? max_batch_ : B);
        const int n_cand = N * C;
        // The GPU decode carries the candidate count (cap * n_cand) in 32-bit int
        // (kernel indices, grid dims, CUB num_items). Unreachable for D-FINE
        // (N*C=24000), but fail loudly rather than silently overflow on a huge engine.
        if (static_cast<long long>(cap) * n_cand > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "dfine: gpu_decode candidate count (batch*queries*classes) "
                "exceeds INT_MAX; reduce batch or disable gpu_decode");
        }
        const auto total = static_cast<std::size_t>(cap) * n_cand;
        const std::size_t cub_bytes = gpu_decode_temp_bytes(cap, n_cand);

        // Pack all nine scratch buffers into ONE device allocation. sub() offsets are
        // 256-byte aligned (safe for CUB temp + coalescing); commit() does one cudaMalloc.
        auto arena = std::make_unique<DeviceArena>();
        const auto o_keys = arena->sub(total * sizeof(float));
        const auto o_vals = arena->sub(total * sizeof(uint32_t));
        const auto o_ko = arena->sub(total * sizeof(float));
        const auto o_vo = arena->sub(total * sizeof(uint32_t));
        const auto o_seg = arena->sub(static_cast<std::size_t>(cap + 1) * sizeof(int));
        const auto o_out = arena->sub(static_cast<std::size_t>(cap) * N * sizeof(DetectionGPU));
        const auto o_cnt = arena->sub(static_cast<std::size_t>(cap) * sizeof(uint32_t));
        const auto o_scale = arena->sub(static_cast<std::size_t>(cap) * sizeof(DecodeMapGPU));
        const auto o_temp = arena->sub(cub_bytes);
        arena->commit();

        // Commit-last: run the only throwing setup (fill seg_off + sync) on LOCAL
        // pointers first, so a throw here unwinds the local `arena` while gdec_,
        // gdec_arena_, and gdec_cap_batch_ all still reference the prior valid arena.
        int* seg = arena->at<int>(o_seg);
        gpu_decode_fill_segoff(seg, cap, n_cand, session.stream());
        DFINE_CUDA_CHECK(
            cudaStreamSynchronize(session.stream()));  // seg_off ready before first use

        gdec_.keys = arena->at<float>(o_keys);
        gdec_.vals = arena->at<uint32_t>(o_vals);
        gdec_.keys_out = arena->at<float>(o_ko);
        gdec_.vals_out = arena->at<uint32_t>(o_vo);
        gdec_.seg_off = seg;
        gdec_.out = arena->at<DetectionGPU>(o_out);
        gdec_.counts = arena->at<uint32_t>(o_cnt);
        gdec_.maps = arena->at<DecodeMapGPU>(o_scale);
        gdec_.cub_temp = arena->at(o_temp);
        gdec_.cub_temp_bytes = cub_bytes;

        gdec_arena_ = std::move(arena);  // frees the previous block (grow-only replacement)
        gdec_host_out_.resize(static_cast<std::size_t>(cap) * N);
        gdec_host_counts_.resize(static_cast<std::size_t>(cap));
        gdec_maps_host_.resize(static_cast<std::size_t>(cap));
        gdec_cap_batch_ = cap;
    }

    // All frames exactly match the frozen capture configuration (size, channel
    // order, 3-channel). Data pointers are already validated by the public API.
    bool full_frames_match_(const std::vector<ImageU8>& images) const noexcept {
        for (const auto& im : images) {
            if (im.width != full_src_w_ || im.height != full_src_h_ || im.channels != 3 ||
                im.is_bgr != full_is_bgr_) {
                return false;
            }
        }
        return true;
    }

    // Enqueue the ENTIRE per-frame pipeline on the session stream (capture-safe:
    // enqueues only — no host sync, no allocation): per-image H2D from the pinned
    // frame slab + fused preprocess into the engine input, the constant scale H2D
    // (re-uploaded inside the graph so a fallback call's scale overwrite cannot
    // leak into a later replay), enqueueV3, the GPU decode reading the live
    // threshold, and the survivor/count D2H into pinned buffers.
    bool enqueue_full_sequence_(int B) {
        cudaStream_t stream = session.stream();
        auto* d_input = static_cast<float*>(session.device_buffer(input_name));
        auto* h_slab = static_cast<std::uint8_t*>(h_frames_.get());
        auto* d_slab = static_cast<std::uint8_t*>(d_frames_.get());
        const std::size_t single = static_cast<std::size_t>(3) * in_h_ * in_w_;
        for (int i = 0; i < B; ++i) {
            if (cudaMemcpyAsync(d_slab + i * frame_slot_bytes_, h_slab + i * frame_slot_bytes_,
                                frame_slot_bytes_, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                return false;
            }
            if (letterbox_) {
                const LetterboxMap lb = compute_letterbox_map(full_src_w_, full_src_h_, in_w_,
                                                              in_h_, lb_topleft_, lb_upscale_);
                launch_letterbox_resize_normalize(stream, d_slab + i * frame_slot_bytes_,
                                                  full_src_h_, full_src_w_, full_src_w_ * 3,
                                                  d_input + i * single, in_h_, in_w_, lb, lb_pad_,
                                                  full_is_bgr_, meta.mean.data(), meta.std.data());
            } else {
                launch_stretch_resize_normalize(stream, d_slab + i * frame_slot_bytes_, full_src_h_,
                                                full_src_w_, full_src_w_ * 3, d_input + i * single,
                                                in_h_, in_w_, full_is_bgr_, meta.mean.data(),
                                                meta.std.data());
            }
        }
        if (cudaMemcpyAsync(gdec_.maps, h_scale_.get(),
                            static_cast<std::size_t>(B) * sizeof(DecodeMapGPU),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
        if (!session.context()->enqueueV3(stream)) return false;
        const auto* d_logits = static_cast<const float*>(session.device_buffer(b_logits->name));
        const auto* d_boxes = static_cast<const float*>(session.device_buffer(b_boxes->name));
        gpu_decode_enqueue(d_logits, d_boxes, B, N, C, /*topk=*/N, /*threshold=*/0.0f, d_thr_,
                           gdec_, stream);
        if (cudaMemcpyAsync(h_survivors_.get(), gdec_.out,
                            static_cast<std::size_t>(B) * N * sizeof(DetectionGPU),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            return false;
        }
        if (cudaMemcpyAsync(h_counts_.get(), gdec_.counts,
                            static_cast<std::size_t>(B) * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess) {
            return false;
        }
        return true;
    }

    // Drop everything capture_full_graph_ allocated. Used when the capture fails:
    // the split path must not carry dead pinned/device slabs for the process
    // lifetime (a failed batch-8 1080p capture would otherwise strand ~95 MiB).
    void release_full_graph_state_() noexcept {
        full_exec_ = CudaGraphExec();
        h_frames_.reset();
        d_frames_.reset();
        h_survivors_.reset();
        h_counts_.reset();
        h_scale_.reset();
        h_thr_.reset();
        d_thr_ = nullptr;
        frame_slot_bytes_ = 0;
    }

    // Capture the full pipeline for the frozen configuration. Called ONLY from
    // freeze_ BEFORE the allocation lock: everything it allocates (staging slabs,
    // pinned outputs, the instantiated graph) exists before steady state begins,
    // so the post-lock path stays allocation-free. No-throw contract: false leaves
    // the split gpu_decode path fully functional — and holding nothing.
    bool capture_full_graph_(int B) {
        // Any failure below (early return or throw) releases the partial state.
        struct ReleaseGuard {
            Impl* self;
            bool dismiss{false};
            ~ReleaseGuard() {
                if (!dismiss) self->release_full_graph_state_();
            }
        } guard{this};
        try {
            cudaStream_t stream = session.stream();
            frame_slot_bytes_ = static_cast<std::size_t>(full_src_h_) * full_src_w_ * 3;
            const std::size_t slab_bytes = static_cast<std::size_t>(B) * frame_slot_bytes_;
            const std::size_t out_bytes = static_cast<std::size_t>(B) * N * sizeof(DetectionGPU);
            const std::size_t cnt_bytes = static_cast<std::size_t>(B) * sizeof(uint32_t);
            const std::size_t scale_bytes = static_cast<std::size_t>(B) * sizeof(DecodeMapGPU);

            auto pin = [](HostPtr& h, std::size_t bytes) {
                void* p = nullptr;
                if (cudaMallocHost(&p, bytes) != cudaSuccess) {
                    cudaGetLastError();
                    return false;
                }
                h.reset(p);
                return true;
            };
            if (!pin(h_frames_, slab_bytes) || !pin(h_survivors_, out_bytes) ||
                !pin(h_counts_, cnt_bytes) || !pin(h_scale_, scale_bytes)) {
                return false;
            }
            void* p = nullptr;
            if (cudaMalloc(&p, slab_bytes) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            d_frames_.reset(p);
            // The live-threshold knob must be device-readable: mapped pinned memory.
            if (cudaHostAlloc(&p, sizeof(float), cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            h_thr_.reset(p);
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            d_thr_ = static_cast<float*>(dp);

            *static_cast<float*>(h_thr_.get()) = opts.threshold;
            auto* sc = static_cast<DecodeMapGPU*>(h_scale_.get());
            for (int i = 0; i < B; ++i) {
                sc[i] = to_gpu_(make_map_(full_src_w_, full_src_h_));
            }
            // Warmup/capture slab content: neutral gray. The graph records
            // operations, not data — any valid frame content works.
            std::memset(h_frames_.get(), 114, slab_bytes);

            // Warm the exact captured sequence (the context shape is already set and
            // flushed at B by freeze_'s warm runs). >=2 iterations flush TRT's
            // deferred shape setup, which would otherwise sync-abort the capture.
            const int warm = opts.graph_warmup_iters < 2 ? 2 : opts.graph_warmup_iters;
            session.context()->setEnqueueEmitsProfile(false);
            for (int w = 0; w < warm; ++w) {
                if (!enqueue_full_sequence_(B)) {
                    cudaGetLastError();
                    return false;
                }
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    cudaGetLastError();
                    return false;
                }
            }

            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // Between Begin and End the stream is in capture mode; if the sequence
            // throws (DFINE_CUDA_CHECK inside a launch helper), EndCapture must
            // STILL run or every later op on the session stream — including the
            // split-path fallback — would fail with a capture-in-progress error.
            bool seq_ok = false;
            try {
                seq_ok = enqueue_full_sequence_(B);
            } catch (...) {
                seq_ok = false;
            }
            cudaGraph_t graph_raw = nullptr;
            const cudaError_t end_err = cudaStreamEndCapture(stream, &graph_raw);
            if (!seq_ok || end_err != cudaSuccess || graph_raw == nullptr) {
                cudaGetLastError();
                if (graph_raw) cudaGraphDestroy(graph_raw);
                return false;
            }
            CudaGraph graph(graph_raw);

            cudaGraphExec_t exec_raw = nullptr;
            if (cudaGraphInstantiate(&exec_raw, graph.get(), 0) != cudaSuccess ||
                exec_raw == nullptr) {
                cudaGetLastError();
                return false;
            }
            full_exec_ = CudaGraphExec(exec_raw);

            // Upload + one replay BEFORE the lock: any lazy driver-side allocation
            // tied to the first launch happens now, keeping steady state clean.
            if (cudaGraphUpload(full_exec_.get(), stream) != cudaSuccess ||
                cudaGraphLaunch(full_exec_.get(), stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
                cudaGetLastError();
                full_exec_ = CudaGraphExec();
                return false;
            }

            full_batch_ = B;
            full_ready_ = true;
            guard.dismiss = true;
            return true;
        } catch (...) {
            cudaGetLastError();
            return false;
        }
    }

    // Steady-state single-launch path (P3): no TensorRT API calls, no allocation —
    // pack the frames into the pinned slab, refresh the live threshold, one
    // cudaGraphLaunch, sync, convert the pinned survivors.
    std::vector<Detections> run_full_graph_(const std::vector<ImageU8>& images, int B, float thr,
                                            std::chrono::steady_clock::time_point t0) {
        using Clock = std::chrono::steady_clock;
        auto* slab = static_cast<std::uint8_t*>(h_frames_.get());
        for (int i = 0; i < B; ++i) {
            const ImageU8& im = images[static_cast<std::size_t>(i)];
            const std::size_t packed_row = static_cast<std::size_t>(im.width) * 3;
            std::uint8_t* dst = slab + static_cast<std::size_t>(i) * frame_slot_bytes_;
            if (static_cast<std::size_t>(im.row_bytes()) == packed_row) {
                std::memcpy(dst, im.data, frame_slot_bytes_);
            } else {
                for (int r = 0; r < im.height; ++r) {
                    std::memcpy(dst + static_cast<std::size_t>(r) * packed_row,
                                im.data + static_cast<std::size_t>(r) * im.row_bytes(), packed_row);
                }
            }
        }
        *static_cast<float*>(h_thr_.get()) = thr;  // read by k_decode_topk at execution
        const auto tp = Clock::now();

        DFINE_CUDA_CHECK(cudaGraphLaunch(full_exec_.get(), session.stream()));
        const auto td = Clock::now();
        DFINE_CUDA_CHECK(cudaStreamSynchronize(session.stream()));
        const auto t1 = Clock::now();
        ++full_replays_;

        const auto* survivors = static_cast<const DetectionGPU*>(h_survivors_.get());
        const auto* counts = static_cast<const uint32_t*>(h_counts_.get());
        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        for (int i = 0; i < B; ++i) {
            const uint32_t m = counts[static_cast<std::size_t>(i)];
            const DetectionGPU* base = survivors + static_cast<std::size_t>(i) * N;
            Detections dets;
            dets.reserve(m);
            for (uint32_t k = 0; k < m; ++k) {
                const DetectionGPU& g = base[k];
                Detection d;
                d.box.x1 = g.x1;
                d.box.y1 = g.y1;
                d.box.x2 = g.x2;
                d.box.y2 = g.y2;
                d.class_id = g.class_id;
                d.score = g.score;
                dets.push_back(d);
            }
            results.push_back(std::move(dets));
        }
        const auto t2 = Clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timings.preprocess_ms = 0.0;
        timings.infer_ms = ms(t0, t1);
        timings.postprocess_ms = ms(t1, t2);
        timings.total_ms = ms(t0, t2);
        timings.preprocess_cpu_ms = ms(t0, tp);
        timings.dispatch_ms = ms(tp, td);
        timings.wait_ms = ms(td, t1);
        timings.decode_host_ms = ms(t1, t2);
        return results;
    }

    // Warm every grow-only buffer to peak (bindings, decode scratch, and the
    // preprocessor staging — at the steady-state SOURCE size, not just the engine
    // input size), optionally capture the full-pipeline graph (P3), then lock:
    // no further device allocation.
    void freeze_(const FreezeSpec& spec) {
        const int b = spec.batch > 0 ? spec.batch : (max_batch_ > 0 ? max_batch_ : 1);
        const int sw = spec.src_w > 0 ? spec.src_w : in_w_;
        const int sh = spec.src_h > 0 ? spec.src_h : in_h_;
        if (frozen_) {
            // Idempotent only for the same resolved configuration. A different
            // spec cannot be honored (buffers and the captured graph are locked) —
            // fail fast here instead of surfacing later as a grow-guard throw in
            // the middle of the steady-state loop.
            if (b != frozen_batch_ || sw != frozen_src_w_ || sh != frozen_src_h_ ||
                spec.src_is_bgr != frozen_bgr_) {
                throw std::runtime_error(
                    "dfine: freeze() called again with a different configuration; the detector "
                    "is already frozen (create a new detector to reconfigure)");
            }
            return;
        }
        const std::size_t px = static_cast<std::size_t>(sh) * sw * 3;
        std::vector<std::uint8_t> gray(px, 114);
        std::vector<ImageU8> imgs(static_cast<std::size_t>(b));
        for (auto& im : imgs) {
            im.data = gray.data();
            im.height = sh;
            im.width = sw;
            im.channels = 3;
            im.is_bgr = spec.src_is_bgr;
        }
        // Warm enough to settle every grow-only allocation. M2.2 CUDA-graph capture
        // is deferred to the 2nd enqueue at a batch (the 1st flushes the shape), and
        // it allocates pinned buffers + a graph exec, so warm 3x when it's enabled
        // to complete capture before locking — otherwise the first real frame would
        // allocate. gpu_decode settles in a single pass.
        const int warm = opts.use_cuda_graph ? 3 : 1;
        for (int w = 0; w < warm; ++w) (void)run_batch(imgs, opts.threshold);

        if (opts.full_pipeline_graph) {
            full_src_w_ = sw;
            full_src_h_ = sh;
            full_is_bgr_ = spec.src_is_bgr;
            if (!gpu_decode_supported_ || session.num_aux_streams() != 0) {
                log_message(LogSeverity::kWarning,
                            "dfine: full_pipeline_graph set but the engine is not capturable "
                            "(needs FP32 outputs and a --max-aux-streams 0 build); "
                            "split gpu_decode path in effect");
            } else if (!capture_full_graph_(b)) {
                log_message(LogSeverity::kWarning,
                            "dfine: full-pipeline graph capture failed; "
                            "split gpu_decode path in effect");
            }
        }

        session.freeze();  // binding grow -> throw hereafter
        frozen_ = true;    // gpu-decode grow -> throw hereafter
        if (gdec_arena_) gdec_arena_->lock();
        // Only a spec that explicitly bounds the source size locks the preprocessor
        // staging. freeze(int)/freeze({batch}) keeps the legacy P2 behavior — an
        // oversized source frame still grows staging on the hot path (a documented
        // exception to the zero-allocation contract) — because locking at the
        // engine-input default would turn every pre-existing freeze(batch) caller
        // with larger-than-network frames into a hard runtime error.
        if (spec.src_w > 0 || spec.src_h > 0) preprocessor->freeze();
        frozen_batch_ = b;
        frozen_src_w_ = sw;
        frozen_src_h_ = sh;
        frozen_bgr_ = spec.src_is_bgr;
    }

    // Zero-D2H path: enqueueV3 -> on-device decode -> D2H only the survivors.
    // pre_cpu_ms: host cost of the preprocess/H2D issue loop, measured by run_batch.
    std::vector<Detections> run_gpu_decode_(const std::vector<ImageU8>& images, int B, float thr,
                                            std::chrono::steady_clock::time_point t0,
                                            double pre_cpu_ms) {
        ensure_gpu_decode_(B);
        cudaStream_t stream = session.stream();

        const auto td0 = std::chrono::steady_clock::now();
        DFINE_TRT_CHECK(session.context()->enqueueV3(stream));  // outputs -> device_buffer(name)
        graph_ctx_batch_ = B;  // context is now enqueued/flushed for this shape

        for (int i = 0; i < B; ++i) {
            gdec_maps_host_[static_cast<std::size_t>(i)] =
                to_gpu_(make_map_(images[i].width, images[i].height));
        }
        DFINE_CUDA_CHECK(cudaMemcpyAsync(gdec_.maps, gdec_maps_host_.data(),
                                         static_cast<std::size_t>(B) * sizeof(DecodeMapGPU),
                                         cudaMemcpyHostToDevice, stream));

        const auto* d_logits = static_cast<const float*>(session.device_buffer(b_logits->name));
        const auto* d_boxes = static_cast<const float*>(session.device_buffer(b_boxes->name));
        gpu_decode_enqueue(d_logits, d_boxes, B, N, C, /*topk=*/N, thr, /*threshold_dev=*/nullptr,
                           gdec_, stream);

        const auto out_n = static_cast<std::size_t>(B) * N;
        DFINE_CUDA_CHECK(cudaMemcpyAsync(gdec_host_out_.data(), gdec_.out,
                                         out_n * sizeof(DetectionGPU), cudaMemcpyDeviceToHost,
                                         stream));
        DFINE_CUDA_CHECK(cudaMemcpyAsync(gdec_host_counts_.data(), gdec_.counts,
                                         static_cast<std::size_t>(B) * sizeof(uint32_t),
                                         cudaMemcpyDeviceToHost, stream));
        const auto td1 = std::chrono::steady_clock::now();  // all GPU work issued
        DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t1 = std::chrono::steady_clock::now();

        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        for (int i = 0; i < B; ++i) {
            const uint32_t m = gdec_host_counts_[static_cast<std::size_t>(i)];
            const DetectionGPU* base = gdec_host_out_.data() + static_cast<std::size_t>(i) * N;
            Detections dets;
            dets.reserve(m);
            for (uint32_t k = 0; k < m; ++k) {
                const DetectionGPU& g = base[k];
                Detection d;
                d.box.x1 = g.x1;
                d.box.y1 = g.y1;
                d.box.x2 = g.x2;
                d.box.y2 = g.y2;
                d.class_id = g.class_id;
                d.score = g.score;
                dets.push_back(d);
            }
            results.push_back(std::move(dets));
        }
        const auto t2 = std::chrono::steady_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timings.preprocess_ms = 0.0;
        timings.infer_ms = ms(t0, t1);
        timings.postprocess_ms = ms(t1, t2);
        timings.total_ms = ms(t0, t2);
        timings.preprocess_cpu_ms = pre_cpu_ms;
        timings.dispatch_ms = ms(td0, td1);
        timings.wait_ms = ms(td1, t1);
        timings.decode_host_ms = ms(t1, t2);
        return results;
    }

    std::vector<Detections> run_batch(const std::vector<ImageU8>& images, float threshold) {
        const int B = static_cast<int>(images.size());
        if (B == 0) return {};
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        const float thr = (threshold >= 0.0f) ? threshold : opts.threshold;

        // Single-launch steady state (P3): when the frozen full-pipeline graph
        // matches this call exactly, skip ALL TensorRT/preprocess host work — pack
        // the frames into the pinned slab and replay. graph_ctx_batch_ must equal
        // the frozen batch: after a non-matching call re-flushed the context for a
        // different shape, one split-path call at the frozen batch restores it
        // (run_gpu_decode_ sets graph_ctx_batch_), so the gate self-heals.
        if (full_ready_ && B == full_batch_ && graph_ctx_batch_ == full_batch_ &&
            full_frames_match_(images)) {
            return run_full_graph_(images, B, thr, t0);
        }

        set_batch_(B);

        const std::size_t single = static_cast<std::size_t>(3) * in_h_ * in_w_;
        float* d_input = static_cast<float*>(session.device_buffer(input_name));
        for (int i = 0; i < B; ++i) {
            preprocessor->process(images[i], d_input + i * single, session.stream());
        }
        const auto tp = Clock::now();  // preprocess/H2D issue loop done (host cost)

        // Zero-D2H GPU decode takes precedence when requested + supported (FP32 outputs).
        if (opts.gpu_decode) {
            if (gpu_decode_supported_) return run_gpu_decode_(images, B, thr, t0, ms_(t0, tp));
            if (!gpu_decode_warned_) {
                gpu_decode_warned_ = true;
                log_message(
                    LogSeverity::kWarning,
                    "dfine: gpu_decode set but engine outputs aren't FP32; using CPU decode");
            }
        }

        // Preprocess/H2D stayed outside; the graph (if any) covers enqueueV3 + D2H.
        bool used_graph = false;
        double dispatch_ms = 0.0;
        double wait_ms = 0.0;
        if (opts.use_cuda_graph) {
            if (graph_supported_ && !graph_disabled_) {
                used_graph = try_graph_infer_(B, dispatch_ms, wait_ms);
                if (!used_graph && graph_disabled_ && !graph_warned_) {
                    graph_warned_ = true;
                    log_message(LogSeverity::kWarning,
                                "dfine: CUDA-graph capture failed; using enqueueV3 "
                                "(correct, no graph speedup)");
                }
            } else if (!graph_supported_ && !graph_warned_) {
                graph_warned_ = true;
                log_message(LogSeverity::kWarning,
                            "dfine: use_cuda_graph set but engine isn't graph-capturable "
                            "(needs FP32 outputs and 0 aux streams); using enqueueV3");
            }
        }
        if (!used_graph) {
            // enqueueV3 + D2H(+sync) split so Timings can attribute the CPU issue
            // cost vs the GPU wait separately. Identical semantics to the previous
            // session.infer() + get_output_f32 sequence: enqueueV3 flushes any
            // deferred shape setup, and get_output_f32 syncs the stream before
            // reading its pinned staging.
            const auto td0 = Clock::now();
            DFINE_TRT_CHECK(session.context()->enqueueV3(session.stream()));
            const auto td1 = Clock::now();
            const std::size_t logits_n = static_cast<std::size_t>(B) * N * C;
            const std::size_t boxes_n = static_cast<std::size_t>(B) * N * 4;
            if (h_logits.size() < logits_n) h_logits.resize(logits_n);
            if (h_boxes.size() < boxes_n) h_boxes.resize(boxes_n);
            session.get_output_f32(b_logits->name, h_logits.data(), logits_n);
            session.get_output_f32(b_boxes->name, h_boxes.data(), boxes_n);
            const auto td2 = Clock::now();
            dispatch_ms = ms_(td0, td1);
            wait_ms = ms_(td1, td2);
            graph_ctx_batch_ = B;  // context now enqueued/flushed for this shape
        }
        const auto t1 = Clock::now();

        pp.threshold = thr;
        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        for (int i = 0; i < B; ++i) {
            const float* l = h_logits.data() + static_cast<std::size_t>(i) * N * C;
            const float* b = h_boxes.data() + static_cast<std::size_t>(i) * N * 4;
            results.push_back(
                decode_detections(l, b, make_map_(images[i].width, images[i].height), pp));
        }
        const auto t2 = Clock::now();

        timings.preprocess_ms = 0.0;  // merged into infer_ms (async on the stream)
        timings.infer_ms = ms_(t0, t1);
        timings.postprocess_ms = ms_(t1, t2);
        timings.total_ms = ms_(t0, t2);
        timings.preprocess_cpu_ms = ms_(t0, tp);
        timings.dispatch_ms = dispatch_ms;
        timings.wait_ms = wait_ms;
        timings.decode_host_ms = ms_(t1, t2);
        return results;
    }
};

DFineDetector::DFineDetector(const std::filesystem::path& engine_path,
                             const DetectorOptions& opts) {
    init_(engine_path, engine_path.string() + ".json", opts);
}

DFineDetector::DFineDetector(const std::filesystem::path& engine_path,
                             const std::filesystem::path& meta_path, const DetectorOptions& opts) {
    init_(engine_path, meta_path, opts);
}

void DFineDetector::init_(const std::filesystem::path& engine_path,
                          const std::filesystem::path& meta_path, const DetectorOptions& opts) {
    impl_ = std::make_unique<Impl>(engine_path, meta_path, opts);
}

DFineDetector::~DFineDetector() = default;
DFineDetector::DFineDetector(DFineDetector&&) noexcept = default;
DFineDetector& DFineDetector::operator=(DFineDetector&&) noexcept = default;

Detections DFineDetector::detect(const ImageU8& image, float threshold) {
    if (!impl_) throw std::runtime_error("dfine: DFineDetector: moved-from object");
    if (!image.data) throw std::runtime_error("dfine: DFineDetector::detect: empty image");
    std::vector<ImageU8> one{image};
    return std::move(impl_->run_batch(one, threshold).front());
}

std::vector<Detections> DFineDetector::detect_batch(const std::vector<ImageU8>& images,
                                                    float threshold) {
    if (!impl_) throw std::runtime_error("dfine: DFineDetector: moved-from object");
    if (images.empty()) return {};
    for (const auto& img : images) {
        if (!img.data) throw std::runtime_error("dfine: detect_batch: image in batch is empty");
    }
    return impl_->run_batch(images, threshold);
}

void DFineDetector::freeze(int batch) {
    if (!impl_) throw std::runtime_error("dfine: DFineDetector: moved-from object");
    impl_->freeze_(FreezeSpec{batch, 0, 0, false});
}

void DFineDetector::freeze(const FreezeSpec& spec) {
    if (!impl_) throw std::runtime_error("dfine: DFineDetector: moved-from object");
    impl_->freeze_(spec);
}

bool DFineDetector::full_pipeline_graph_active() const noexcept {
    return impl_ && impl_->full_ready_;
}

std::uint64_t DFineDetector::full_graph_replays() const noexcept {
    return impl_ ? impl_->full_replays_ : 0;
}

// The noexcept accessors return inert defaults on a moved-from object instead of
// dereferencing a null impl_ (they cannot throw, and terminating on an accessor
// of a moved-from detector would be disproportionate).
const std::string& DFineDetector::variant() const noexcept {
    static const std::string kEmpty;
    return impl_ ? impl_->meta.variant : kEmpty;
}
int DFineDetector::input_h() const noexcept {
    return impl_ ? impl_->in_h_ : 0;
}
int DFineDetector::input_w() const noexcept {
    return impl_ ? impl_->in_w_ : 0;
}
int DFineDetector::num_queries() const noexcept {
    return impl_ ? impl_->N : 0;
}
int DFineDetector::num_classes() const noexcept {
    return impl_ ? impl_->C : 0;
}

std::string DFineDetector::class_name(int class_id) const {
    if (impl_ && class_id >= 0) {
        const auto& names = impl_->meta.class_names;
        if (class_id < static_cast<int>(names.size()))
            return names[static_cast<std::size_t>(class_id)];
        if (names.empty() && impl_->C == 80 && class_id < 80) return coco_class_name(class_id);
    }
    return "class_" + std::to_string(class_id);
}

int DFineDetector::max_batch() const noexcept {
    // 0 = dynamic engine whose profile max is unknown (no/partial sidecar);
    // detect_batch then defers the bound to TensorRT's setInputShape.
    if (!impl_) return 0;
    return impl_->input_dynamic_ ? impl_->max_batch_ : 1;
}

const DFineDetector::Timings& DFineDetector::last_timings() const noexcept {
    static const Timings kNone{};
    return impl_ ? impl_->timings : kNone;
}

}  // namespace dfine
