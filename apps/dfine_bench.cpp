// dfine_bench — precise per-stage latency + GPU-memory benchmark of the D-FINE
// detector's C++ path. Times each stage on the CUDA stream with events
// (preprocess+H2D / infer / D2H) plus host-side decode, with warm-up and
// percentiles, across a set of batch sizes.
//
// usage:
//   dfine_bench --engine E.engine [--meta E.json] [--image img.jpg]
//               [--src-size WxH] [--batches 1,2,4,8] [--warmup 20] [--iters 200]
//               [--json out.json] [--threshold 0.001] [execution mode]

#include "cli_helpers.hpp"
#include "image_io.hpp"

#include "dfine/core/engine_meta.hpp"
#include "dfine/core/postprocess.hpp"
#include "dfine/tasks/detector.hpp"
#include "dfine/version.hpp"
#include "internal/cuda_check.hpp"
#include "internal/cuda_preprocess.cuh"
#include "internal/cuda_raii.hpp"
#include "internal/decode_gpu.cuh"
#include "internal/engine_meta_detail.hpp"
#include "internal/trt_session.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Stats {
    double mean{0}, p50{0}, p90{0}, p99{0}, min{0}, max{0};
};

Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto at = [&](double q) {
        const std::size_t i = static_cast<std::size_t>(q * (v.size() - 1) + 0.5);
        return v[std::min(i, v.size() - 1)];
    };
    s.mean = sum / v.size();
    s.p50 = at(0.50);
    s.p90 = at(0.90);
    s.p99 = at(0.99);
    s.min = v.front();
    s.max = v.back();
    return s;
}

double ev_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    DFINE_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms);
}

dfine::CudaEvent make_cuda_event() {
    cudaEvent_t event = nullptr;
    DFINE_CUDA_CHECK(cudaEventCreate(&event));
    return dfine::CudaEvent(event);
}

std::vector<int> parse_batches(std::string_view s) {
    std::vector<int> out;
    std::string cur;
    const auto append = [&] {
        if (cur.empty()) {
            throw std::runtime_error("--batches expects comma-separated positive integers");
        }
        std::size_t consumed = 0;
        int batch = 0;
        try {
            batch = std::stoi(cur, &consumed);
        } catch (...) {
            throw std::runtime_error("--batches expects comma-separated positive integers");
        }
        if (consumed != cur.size() || batch <= 0) {
            throw std::runtime_error("--batches expects comma-separated positive integers");
        }
        out.push_back(batch);
        cur.clear();
    };
    for (char c : s) {
        if (c == ',') {
            append();
        } else
            cur += c;
    }
    append();
    return out;
}

struct OutputBindings {
    const dfine::BindingInfo* logits{nullptr};
    const dfine::BindingInfo* boxes{nullptr};
};

OutputBindings resolve_outputs(const dfine::TrtSession& session, const dfine::EngineMeta& meta,
                               bool names_asserted) {
    const auto named = [&](std::string_view name) {
        const auto* binding = session.find(name);
        return binding && !binding->is_input ? binding : nullptr;
    };
    if (names_asserted) {
        if (meta.output_names.size() != 2 || meta.output_names[0] == meta.output_names[1])
            return {};
        return {named(meta.output_names[0]), named(meta.output_names[1])};
    }
    OutputBindings resolved{named("logits"), named("boxes")};
    if (resolved.logits && resolved.boxes) return resolved;

    if (session.output_indices().size() != 2) return {};
    int boxes_candidates = 0;
    for (int index : session.output_indices()) {
        const auto* binding = &session.bindings()[static_cast<std::size_t>(index)];
        const int last = binding->shape.nbDims > 0
                             ? static_cast<int>(binding->shape.d[binding->shape.nbDims - 1])
                             : -1;
        if (last == 4) {
            resolved.boxes = binding;
            ++boxes_candidates;
        } else {
            resolved.logits = binding;
        }
    }
    return boxes_candidates == 1 && resolved.logits ? resolved : OutputBindings{};
}

// Bit-pattern float compare: the parity contract is byte-exactness, and operator==
// would report bit-identical NaNs (degenerate engine outputs) as a mismatch.
bool same_bits(float a, float b) {
    std::uint32_t x, y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

bool detections_equal(const std::vector<dfine::Detections>& a,
                      const std::vector<dfine::Detections>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (std::size_t k = 0; k < a[i].size(); ++k) {
            const auto& x = a[i][k];
            const auto& y = b[i][k];
            if (x.class_id != y.class_id || !same_bits(x.score, y.score) ||
                !same_bits(x.box.x1, y.box.x1) || !same_bits(x.box.y1, y.box.y1) ||
                !same_bits(x.box.x2, y.box.x2) || !same_bits(x.box.y2, y.box.y2)) {
                return false;
            }
        }
    }
    return true;
}

// --pipeline-compare: the split gpu_decode path vs the full-pipeline graph,
// measured through the public DFineDetector API. Two detector instances on the
// same engine (one frozen with full_pipeline_graph), interleaved per iteration so
// GPU clock drift cancels. Per-stage CPU columns come from Timings; every
// iteration's detections are compared byte-exact (same kernels either way — any
// mismatch is a capture bug).
int run_pipeline_compare(const std::filesystem::path& engine, const std::filesystem::path& meta,
                         const std::vector<int>& batches, const std::filesystem::path& image,
                         int src_w, int src_h, int warmup, int iters, float threshold) {
    dfine_app::LoadedImage loaded;
    std::vector<std::uint8_t> synth;
    dfine::ImageU8 base;
    if (!image.empty()) {
        loaded = dfine_app::load_image_rgb(image.string());
        if (!loaded) throw std::runtime_error("cannot decode image: " + image.string());
        base = loaded.view();
        src_w = base.width;
        src_h = base.height;
    } else {
        synth.resize(static_cast<std::size_t>(src_w) * src_h * 3);
        for (std::size_t i = 0; i < synth.size(); ++i)
            synth[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
        base = dfine::ImageU8{synth.data(), src_h, src_w, 3, src_w * 3, false};
    }

    for (int B : batches) {
        dfine::DetectorOptions oa;
        oa.threshold = threshold;
        oa.gpu_decode = true;
        oa.own_device_memory = true;
        dfine::DetectorOptions ob = oa;
        ob.full_pipeline_graph = true;

        // freeze() is one-way and per-configuration, so each batch size gets fresh
        // detector instances (one engine deserialize each — seconds, not per-iter).
        dfine::DFineDetector split = meta.empty() ? dfine::DFineDetector(engine, oa)
                                                  : dfine::DFineDetector(engine, meta, oa);
        dfine::DFineDetector graph = meta.empty() ? dfine::DFineDetector(engine, ob)
                                                  : dfine::DFineDetector(engine, meta, ob);
        dfine::FreezeSpec fs;
        fs.batch = B;
        fs.src_w = src_w;
        fs.src_h = src_h;
        split.freeze(fs);
        graph.freeze(fs);
        if (!graph.full_pipeline_graph_active()) {
            std::printf(
                "[pipeline-compare] batch %d: full-pipeline graph NOT active — the "
                "engine must have FP32 outputs and be built with --max-aux-streams 0\n",
                B);
            return 1;
        }

        const std::vector<dfine::ImageU8> frames(static_cast<std::size_t>(B), base);
        for (int w = 0; w < warmup; ++w) {
            (void)split.detect_batch(frames, threshold);
            (void)graph.detect_batch(frames, threshold);
        }
        // Snapshot after warmup: full_graph_replays() is a lifetime counter, and
        // counting the warmup replays would let up to `warmup` fallback iterations
        // slip past the contamination check below.
        const std::uint64_t replays_before = graph.full_graph_replays();

        std::vector<double> s_pre, s_disp, s_wait, s_dec, s_tot;
        std::vector<double> g_pre, g_disp, g_wait, g_dec, g_tot;
        long long mismatches = 0;
        std::size_t n_dets = 0;
        for (int it = 0; it < iters; ++it) {
            const auto ra = split.detect_batch(frames, threshold);
            const auto ts = split.last_timings();
            const auto rb = graph.detect_batch(frames, threshold);
            const auto tg = graph.last_timings();
            s_pre.push_back(ts.preprocess_cpu_ms);
            s_disp.push_back(ts.dispatch_ms);
            s_wait.push_back(ts.wait_ms);
            s_dec.push_back(ts.decode_host_ms);
            s_tot.push_back(ts.total_ms);
            g_pre.push_back(tg.preprocess_cpu_ms);
            g_disp.push_back(tg.dispatch_ms);
            g_wait.push_back(tg.wait_ms);
            g_dec.push_back(tg.decode_host_ms);
            g_tot.push_back(tg.total_ms);
            if (!detections_equal(ra, rb)) ++mismatches;
            n_dets = ra.empty() ? 0 : ra.front().size();
        }
        const std::uint64_t measured_replays = graph.full_graph_replays() - replays_before;
        if (measured_replays != static_cast<std::uint64_t>(iters)) {
            std::printf(
                "[pipeline-compare] batch %d: only %llu/%d measured iterations replayed "
                "the graph\n",
                B, static_cast<unsigned long long>(measured_replays), iters);
            return 1;
        }

        const Stats sp = summarize(s_pre), sd = summarize(s_disp), sw = summarize(s_wait),
                    sc = summarize(s_dec), st = summarize(s_tot);
        const Stats gp = summarize(g_pre), gd = summarize(g_disp), gw = summarize(g_wait),
                    gc = summarize(g_dec), gt = summarize(g_tot);
        std::printf("[pipeline-compare] batch %d (%d iters, p50 ms, src %dx%d):\n", B, iters, src_w,
                    src_h);
        std::printf("  %-18s %-12s %-12s %s\n", "stage (CPU)", "split", "full-graph", "delta");
        auto row = [](const char* name, double a, double b) {
            std::printf("  %-18s %-12.3f %-12.3f %+.3f\n", name, a, b, b - a);
        };
        row("pre (pack+issue)", sp.p50, gp.p50);
        row("dispatch", sd.p50, gd.p50);
        row("wait (GPU-bound)", sw.p50, gw.p50);
        row("decode host", sc.p50, gc.p50);
        const double cpu_s = sp.p50 + sd.p50 + sc.p50;
        const double cpu_g = gp.p50 + gd.p50 + gc.p50;
        std::printf("  %-18s %-12.3f %-12.3f %+.3f  <- CPU freed per call\n", "CPU total", cpu_s,
                    cpu_g, cpu_g - cpu_s);
        std::printf("  %-18s %-12.3f %-12.3f %+.3f (%+.1f%%)\n", "total wall", st.p50, gt.p50,
                    gt.p50 - st.p50, st.p50 > 0 ? (gt.p50 / st.p50 - 1) * 100 : 0);
        std::printf("  parity: %lld/%d iterations mismatched (%zu detections/frame)%s\n",
                    mismatches, iters, n_dets,
                    mismatches == 0 ? " — byte-identical" : " — capture mismatch");
        if (mismatches != 0) return 1;

        // Live-threshold probe: a per-call override differing from the frozen
        // opts.threshold must match on both paths — the captured decode kernel
        // reads the threshold through mapped pinned memory at execution time, so
        // a baked (capture-time) value would show up here as a count mismatch.
        const float probe = threshold == 0.25f ? 0.75f : 0.25f;
        const std::uint64_t replays_before_probe = graph.full_graph_replays();
        const auto pa = split.detect_batch(frames, probe);
        const auto pb = graph.detect_batch(frames, probe);
        const bool ok = detections_equal(pa, pb);
        std::printf("  threshold probe (%.4f vs frozen %.4f): %s\n", probe, threshold,
                    ok ? "byte-identical" : "mismatch (threshold was captured)");
        if (!ok) return 1;
        if (graph.full_graph_replays() != replays_before_probe + 1) {
            std::printf("  threshold probe did not replay the full graph\n");
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path engine, meta, image;
    std::string batches_arg = "1,2,4,8";
    int warmup = 20, iters = 200, src_w = 640, src_h = 480;
    float threshold = 0.001f;
    bool cuda_graph = false;
    bool require_cuda_graph = false;
    bool graph_compare = false;
    bool gpu_decode = false;
    bool pipeline_compare = false;
    std::filesystem::path json_out;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf(
                    "usage: %s --engine E [--meta M] [--image img] [--src-size WxH] "
                    "[--batches 1,2,4,8] [--warmup 20] [--iters 200] [--threshold 0.001] "
                    "[--json out]\n"
                    "  --cuda-graph  replay enqueueV3+D2H from a captured CUDA graph "
                    "(infer col then includes D2H)\n"
                    "  --require-cuda-graph  fail unless every batch uses graph replay\n"
                    "  --graph-compare  compare enqueue and graph replay in one run\n"
                    "  --gpu-decode  benchmark device-side decode and compact result transfer\n"
                    "  --pipeline-compare  compare split decode with the full-pipeline graph\n"
                    "  raw benchmark modes require FP32 logits and boxes outputs\n"
                    "  dfine v%s\n",
                    argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine"))
                engine = next_value(argc, argv, i, "--engine");
            else if (starts_with(a, "--meta"))
                meta = next_value(argc, argv, i, "--meta");
            else if (starts_with(a, "--image"))
                image = next_value(argc, argv, i, "--image");
            else if (starts_with(a, "--batches"))
                batches_arg = next_value(argc, argv, i, "--batches");
            else if (starts_with(a, "--warmup"))
                warmup = parse_int(next_value(argc, argv, i, "--warmup"), "--warmup");
            else if (starts_with(a, "--iters"))
                iters = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
            else if (starts_with(a, "--threshold"))
                threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            else if (starts_with(a, "--json"))
                json_out = next_value(argc, argv, i, "--json");
            else if (a == "--cuda-graph")
                cuda_graph = true;
            else if (a == "--require-cuda-graph") {
                cuda_graph = true;
                require_cuda_graph = true;
            } else if (a == "--graph-compare") {
                cuda_graph = true;
                graph_compare = true;
            } else if (a == "--gpu-decode")
                gpu_decode = true;
            else if (a == "--pipeline-compare")
                pipeline_compare = true;
            else if (starts_with(a, "--src-size")) {
                std::string v = next_value(argc, argv, i, "--src-size");
                const auto x = v.find('x');
                if (x == std::string::npos || v.find('x', x + 1) != std::string::npos) {
                    throw std::runtime_error("--src-size expects WxH");
                }
                src_w = parse_int(v.substr(0, x).c_str(), "--src-size");
                src_h = parse_int(v.substr(x + 1).c_str(), "--src-size");
            } else
                throw std::runtime_error("unknown arg: " + std::string(a));
        }
        if (engine.empty()) {
            std::fprintf(stderr, "error: --engine required\n");
            return 2;
        }
        if (warmup < 0) throw std::runtime_error("--warmup must be non-negative");
        if (iters <= 0) throw std::runtime_error("--iters must be positive");
        if (src_w <= 0 || src_h <= 0) {
            throw std::runtime_error("--src-size dimensions must be positive");
        }
        if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
            throw std::runtime_error("--threshold must be finite and within [0,1]");
        }
        if (pipeline_compare && (cuda_graph || gpu_decode || !json_out.empty())) {
            throw std::runtime_error(
                "--pipeline-compare cannot be combined with raw execution modes or --json");
        }
        if (graph_compare && (require_cuda_graph || !json_out.empty())) {
            throw std::runtime_error(
                "--graph-compare cannot be combined with --require-cuda-graph or --json");
        }
        if (cuda_graph && gpu_decode) {
            throw std::runtime_error("CUDA Graph replay and --gpu-decode are separate modes");
        }
        const auto batches = parse_batches(batches_arg);

        // Detector-level split-vs-full-graph comparison; self-contained mode.
        if (pipeline_compare) {
            return run_pipeline_compare(engine, meta, batches, image, src_w, src_h, warmup, iters,
                                        threshold);
        }

        // Baseline free memory before we build anything (force context init first).
        DFINE_CUDA_CHECK(cudaFree(nullptr));
        std::size_t free_before = 0, total_mem = 0;
        DFINE_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

        // The engine owns tensor shapes and names; the sidecar supplies preprocessing metadata.
        dfine::detail::EngineMetaDocument meta_doc;
        bool have_meta = false;
        if (meta.empty()) {
            std::filesystem::path alt = engine;
            alt.replace_extension(".json");
            std::filesystem::path discovered = engine.string() + ".json";
            if (!std::filesystem::is_regular_file(discovered)) discovered = alt;
            if (std::filesystem::is_regular_file(discovered)) {
                meta_doc = dfine::detail::load_engine_meta(discovered);
                have_meta = true;
            }
        } else {
            if (!std::filesystem::is_regular_file(meta)) {
                throw std::runtime_error("cannot open explicit sidecar: " + meta.string());
            }
            meta_doc = dfine::detail::load_engine_meta(meta);
            have_meta = true;
        }
        const dfine::EngineMeta& m = meta_doc.meta;
        dfine::TrtSession session(engine);
        if (session.input_indices().size() != 1) {
            throw std::runtime_error("dfine_bench requires exactly one input tensor");
        }
        if (session.num_optimization_profiles() != 1) {
            throw std::runtime_error("dfine_bench requires exactly one optimization profile");
        }
        const dfine::BindingInfo& input =
            session.bindings()[static_cast<std::size_t>(session.input_indices().front())];
        if (input.dtype != nvinfer1::DataType::kFLOAT || input.shape.nbDims != 4 ||
            input.shape.d[1] != 3 || input.shape.d[2] <= 0 || input.shape.d[3] <= 0 ||
            (input.shape.d[0] != -1 && input.shape.d[0] != 1)) {
            throw std::runtime_error("dfine_bench requires an FP32 [B,3,H,W] input with fixed H/W");
        }
        const std::string& in_name = input.name;
        const bool dynamic = input.shape.d[0] == -1;
        const int H = static_cast<int>(input.shape.d[2]);
        const int W = static_cast<int>(input.shape.d[3]);
        int min_batch = 1;
        int opt_batch = 1;
        int max_batch = 1;
        if (dynamic) {
            const dfine::InputProfileInfo profile = session.input_profile(in_name);
            const auto valid_profile_shape = [&](const nvinfer1::Dims& shape) {
                return shape.nbDims == 4 && shape.d[0] > 0 && shape.d[1] == 3 && shape.d[2] == H &&
                       shape.d[3] == W;
            };
            if (!valid_profile_shape(profile.min) || !valid_profile_shape(profile.opt) ||
                !valid_profile_shape(profile.max)) {
                throw std::runtime_error(
                    "dfine_bench requires a profile that varies only the batch dimension");
            }
            min_batch = static_cast<int>(profile.min.d[0]);
            opt_batch = static_cast<int>(profile.opt.d[0]);
            max_batch = static_cast<int>(profile.max.d[0]);
            if (min_batch > opt_batch || opt_batch > max_batch) {
                throw std::runtime_error("dfine_bench: engine batch profile is not ordered");
            }
        }
        if (have_meta && meta_doc.has_input_names &&
            (m.input_names.size() != 1 || m.input_names.front() != in_name)) {
            throw std::runtime_error("dfine_bench: sidecar input_names contradict the engine");
        }
        if (have_meta && meta_doc.has_input_hw && (m.input_h != H || m.input_w != W)) {
            throw std::runtime_error("dfine_bench: sidecar input dimensions contradict the engine");
        }
        if (have_meta && meta_doc.batch_facts_describe_engine()) {
            const auto conflict = [](const char* field, bool asserted, int sidecar, int actual) {
                if (asserted && sidecar != actual) {
                    throw std::runtime_error(std::string("dfine_bench: sidecar ") + field + " " +
                                             std::to_string(sidecar) + " contradicts engine " +
                                             std::to_string(actual));
                }
            };
            if (meta_doc.has_dynamic_batch && m.dynamic_batch != dynamic) {
                throw std::runtime_error("dfine_bench: sidecar dynamic_batch " +
                                         std::string(m.dynamic_batch ? "true" : "false") +
                                         " contradicts engine " + (dynamic ? "true" : "false"));
            }
            conflict("min_batch", meta_doc.has_min_batch, m.min_batch, min_batch);
            conflict("opt_batch", meta_doc.has_opt_batch, m.opt_batch, opt_batch);
            conflict("max_batch", meta_doc.has_max_batch, m.max_batch, max_batch);
        }

        // Source image (real, repeated) or synthetic gradient.
        dfine_app::LoadedImage loaded;
        std::vector<std::uint8_t> synth;
        dfine::ImageU8 base;
        if (!image.empty()) {
            loaded = dfine_app::load_image_rgb(image.string());
            if (!loaded) throw std::runtime_error("cannot decode image: " + image.string());
            base = loaded.view();
            src_w = base.width;
            src_h = base.height;
        } else {
            synth.resize(static_cast<std::size_t>(src_w) * src_h * 3);
            for (std::size_t i = 0; i < synth.size(); ++i)
                synth[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
            base = dfine::ImageU8{synth.data(), src_h, src_w, 3, src_w * 3, false};
        }

        dfine::ImagePreprocessor pre(H, W);
        pre.set_mean(m.mean[0], m.mean[1], m.mean[2]);
        pre.set_std(m.std[0], m.std[1], m.std[2]);

        const OutputBindings outputs =
            resolve_outputs(session, m, have_meta && meta_doc.has_output_names);
        const dfine::BindingInfo* b_logits = outputs.logits;
        const dfine::BindingInfo* b_boxes = outputs.boxes;
        if (!b_logits || !b_boxes)
            throw std::runtime_error("dfine_bench: cannot resolve logits/boxes outputs");
        if (b_logits->dtype != nvinfer1::DataType::kFLOAT ||
            b_boxes->dtype != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("dfine_bench requires FP32 logits and boxes outputs");
        }
        const int batch_dim = dynamic ? -1 : 1;
        if (b_logits->shape.nbDims != 3 || b_logits->shape.d[0] != batch_dim ||
            b_logits->shape.d[1] <= 0 || b_logits->shape.d[2] <= 0) {
            throw std::runtime_error("dfine_bench requires logits shaped [B,Q,C]");
        }
        if (b_boxes->shape.nbDims != 3 || b_boxes->shape.d[0] != batch_dim ||
            b_boxes->shape.d[1] != b_logits->shape.d[1] || b_boxes->shape.d[2] != 4) {
            throw std::runtime_error("dfine_bench requires boxes shaped [B,Q,4]");
        }
        const int N = static_cast<int>(b_logits->shape.d[1]);
        const int C = static_cast<int>(b_logits->shape.d[2]);
        if (have_meta && ((meta_doc.has_num_queries && m.num_queries != N) ||
                          (meta_doc.has_num_classes && m.num_classes != C))) {
            throw std::runtime_error(
                "dfine_bench: sidecar output dimensions contradict the engine");
        }
        const int K = dfine::detection_limit(N, C);

        cudaStream_t stream = session.stream();
        const dfine::CudaEvent e0 = make_cuda_event();
        const dfine::CudaEvent e1 = make_cuda_event();
        const dfine::CudaEvent e2 = make_cuda_event();
        const dfine::CudaEvent e3 = make_cuda_event();

        std::printf(
            "dfine_bench: engine=%s  variant=%s  input=%dx%d  src=%dx%d  warmup=%d iters=%d%s\n",
            engine.filename().c_str(), m.variant.empty() ? "?" : m.variant.c_str(), W, H, src_w,
            src_h, warmup, iters, cuda_graph ? "  [cuda-graph: infer_ms includes D2H]" : "");
        std::printf("%-6s %-11s %-11s %-11s %-11s %-11s %-11s %-8s\n", "batch", "total_p50",
                    "total_p90", "total_p99", "pre_ms", "infer_ms", "decode_ms", "img/s");

        std::string json = "{\"engine\":\"" + engine.string() + "\",\"input\":[" +
                           std::to_string(W) + "," + std::to_string(H) + "]," +
                           "\"cuda_graph\":" + (cuda_graph ? "true" : "false") +
                           ",\"cuda_graph_required\":" + (require_cuda_graph ? "true" : "false") +
                           ",\"results\":[";
        bool first_json = true;
        std::size_t peak_used_mib = 0;

        for (int B : batches) {
            if (dynamic)
                session.set_input_shape(in_name, nvinfer1::Dims4{B, 3, H, W});
            else if (B != 1) {
                if (graph_compare || require_cuda_graph) {
                    throw std::runtime_error(
                        "the requested graph mode requires every batch; "
                        "the static engine only supports batch 1");
                }
                std::printf("(static engine — skipping batch %d)\n", B);
                continue;
            }

            const std::size_t single = static_cast<std::size_t>(3) * H * W;
            float* d_input = static_cast<float*>(session.device_buffer(in_name));
            std::vector<float> h_logits(static_cast<std::size_t>(B) * N * C);
            std::vector<float> h_boxes(static_cast<std::size_t>(B) * N * 4);
            void* d_logits = session.device_buffer(b_logits->name);
            void* d_boxes = session.device_buffer(b_boxes->name);
            const std::size_t logits_bytes = h_logits.size() * sizeof(float);
            const std::size_t boxes_bytes = h_boxes.size() * sizeof(float);

            // CUDA-graph replay copies into pinned buffers (a captured graph cannot D2H
            // into pageable memory — it would force a sync and abort capture).
            dfine::HostPtr p_logits, p_boxes;
            if (cuda_graph) {
                void* p = nullptr;
                DFINE_CUDA_CHECK(cudaMallocHost(&p, logits_bytes));
                p_logits.reset(p);
                DFINE_CUDA_CHECK(cudaMallocHost(&p, boxes_bytes));
                p_boxes.reset(p);
            }
            float* pl = static_cast<float*>(p_logits.get());
            float* pb = static_cast<float*>(p_boxes.get());
            dfine::CudaGraphExec graph_exec;  // empty => plain enqueueV3 path

            dfine::PostprocessParams pp;
            pp.num_queries = N;
            pp.num_classes = C;
            pp.topk = K;
            pp.threshold = threshold;

            // GPU-decode scratch replaces the full-logits D2H and CPU decode
            // with on-device kernels + a compact survivor D2H, folded into the d2h stage.
            dfine::DevPtr g_keys, g_vals, g_ko, g_vo, g_seg, g_temp, g_out, g_counts, g_scale;
            dfine::GpuDecodeScratch gdec;
            std::vector<dfine::DetectionGPU> gh_out;
            std::vector<uint32_t> gh_counts;
            if (gpu_decode) {
                const int n_cand = N * C;
                const std::size_t tot = static_cast<std::size_t>(B) * n_cand;
                auto da = [](dfine::DevPtr& p, std::size_t bytes) -> void* {
                    void* q = nullptr;
                    DFINE_CUDA_CHECK(cudaMalloc(&q, bytes));
                    p.reset(q);
                    return q;
                };
                gdec.keys = static_cast<float*>(da(g_keys, tot * sizeof(float)));
                gdec.vals = static_cast<uint32_t*>(da(g_vals, tot * sizeof(uint32_t)));
                gdec.keys_out = static_cast<float*>(da(g_ko, tot * sizeof(float)));
                gdec.vals_out = static_cast<uint32_t*>(da(g_vo, tot * sizeof(uint32_t)));
                gdec.seg_off =
                    static_cast<int*>(da(g_seg, static_cast<std::size_t>(B + 1) * sizeof(int)));
                gdec.out = static_cast<dfine::DetectionGPU*>(
                    da(g_out, static_cast<std::size_t>(B) * K * sizeof(dfine::DetectionGPU)));
                gdec.counts = static_cast<uint32_t*>(
                    da(g_counts, static_cast<std::size_t>(B) * sizeof(uint32_t)));
                gdec.maps = static_cast<dfine::DecodeMapGPU*>(
                    da(g_scale, static_cast<std::size_t>(B) * sizeof(dfine::DecodeMapGPU)));
                gdec.cub_temp_bytes = dfine::gpu_decode_temp_bytes(B, n_cand);
                gdec.cub_temp = da(g_temp, gdec.cub_temp_bytes);
                dfine::gpu_decode_fill_segoff(gdec.seg_off, B, n_cand, stream);
                std::vector<dfine::DecodeMapGPU> hs(
                    static_cast<std::size_t>(B),
                    dfine::DecodeMapGPU{static_cast<float>(src_w), 0.0f, static_cast<float>(src_h),
                                        0.0f, -1.0f, -1.0f});
                DFINE_CUDA_CHECK(cudaMemcpyAsync(
                    gdec.maps, hs.data(), static_cast<std::size_t>(B) * sizeof(dfine::DecodeMapGPU),
                    cudaMemcpyHostToDevice, stream));
                DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                gh_out.resize(static_cast<std::size_t>(B) * K);
                gh_counts.resize(static_cast<std::size_t>(B));
            }

            auto one_iter = [&](double& pre_ms, double& inf_ms, double& d2h_ms, double& dec_ms) {
                DFINE_CUDA_CHECK(cudaEventRecord(e0.get(), stream));
                for (int b = 0; b < B; ++b) pre.process(base, d_input + b * single, stream);
                DFINE_CUDA_CHECK(cudaEventRecord(e1.get(), stream));
                if (gpu_decode) {
                    // infer -> on-device decode -> compact survivor D2H (no CPU decode).
                    if (!session.context()->enqueueV3(stream))
                        throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaEventRecord(e2.get(), stream));
                    dfine::gpu_decode_enqueue(static_cast<const float*>(d_logits),
                                              static_cast<const float*>(d_boxes), B, N, C, K,
                                              threshold,
                                              /*threshold_dev=*/nullptr, gdec, stream);
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(gh_out.data(), gdec.out,
                                                     gh_out.size() * sizeof(dfine::DetectionGPU),
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(gh_counts.data(), gdec.counts,
                                                     gh_counts.size() * sizeof(uint32_t),
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3.get(), stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    pre_ms = ev_ms(e0.get(), e1.get());
                    inf_ms = ev_ms(e1.get(), e2.get());
                    d2h_ms = ev_ms(e2.get(), e3.get());
                    dec_ms = 0.0;
                    return;
                }
                if (graph_exec) {
                    // replay = enqueueV3 + both D2H copies fused; d2h folds into infer.
                    DFINE_CUDA_CHECK(cudaGraphLaunch(graph_exec.get(), stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e2.get(), stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3.get(), stream));
                } else {
                    if (!session.context()->enqueueV3(stream))
                        throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaEventRecord(e2.get(), stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), d_logits, logits_bytes,
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(), d_boxes, boxes_bytes,
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3.get(), stream));
                }
                DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                pre_ms = ev_ms(e0.get(), e1.get());
                inf_ms = ev_ms(e1.get(), e2.get());
                d2h_ms = ev_ms(e2.get(), e3.get());
                const float* Lsrc = graph_exec ? pl : h_logits.data();
                const float* Bsrc = graph_exec ? pb : h_boxes.data();
                const auto t0 = std::chrono::steady_clock::now();
                for (int b = 0; b < B; ++b)
                    (void)dfine::decode_detections(Lsrc + static_cast<std::size_t>(b) * N * C,
                                                   Bsrc + static_cast<std::size_t>(b) * N * 4,
                                                   src_w, src_h, pp);
                dec_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
            };

            double a, bb, c, d;
            for (int w = 0; w < warmup; ++w)
                one_iter(a, bb, c, d);  // enqueueV3 path: flush tactics + shape

            if (cuda_graph && session.num_aux_streams() != 0) {
                std::printf(
                    "(engine uses %d aux streams — ThreadLocal capture unsafe, using enqueueV3)\n",
                    session.num_aux_streams());
            } else if (cuda_graph) {
                // Capture enqueueV3 + D2H after the warm-up has flushed deferred setup.
                session.context()->setEnqueueEmitsProfile(false);
                cudaGraph_t g = nullptr;
                bool ok =
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
                if (ok) {
                    ok = session.context()->enqueueV3(stream);
                    cudaMemcpyAsync(pl, d_logits, logits_bytes, cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(pb, d_boxes, boxes_bytes, cudaMemcpyDeviceToHost, stream);
                    if (cudaStreamEndCapture(stream, &g) != cudaSuccess) ok = false;
                }
                cudaGraphExec_t exec = nullptr;
                if (ok && g && cudaGraphInstantiate(&exec, g, 0) == cudaSuccess && exec)
                    graph_exec = dfine::CudaGraphExec(exec);
                if (g) cudaGraphDestroy(g);
                cudaGetLastError();  // clear any sticky capture error
                if (!graph_exec)
                    std::printf("(cuda-graph capture failed for batch %d — using enqueueV3)\n", B);
                else
                    for (int w = 0; w < 3; ++w) one_iter(a, bb, c, d);  // prime the replay
            }

            if (require_cuda_graph && !graph_exec) {
                throw std::runtime_error("--require-cuda-graph: capture failed for batch " +
                                         std::to_string(B));
            }

            // Peak memory after warm-up and optional graph capture.
            std::size_t free_now = 0;
            DFINE_CUDA_CHECK(cudaMemGetInfo(&free_now, &total_mem));
            const std::size_t used_mib =
                free_before > free_now ? (free_before - free_now) / (1024 * 1024) : 0;
            peak_used_mib = std::max(peak_used_mib, used_mib);

            if (graph_compare) {
                if (!graph_exec) {
                    throw std::runtime_error(
                        "--graph-compare: CUDA Graph capture failed for batch " +
                        std::to_string(B));
                }
                // Rigorous same-run comparison. Per iteration, run BOTH the enqueueV3+D2H path
                // and the graph-replay path back-to-back (microseconds apart → identical GPU-clock
                // state; drift cancels). Time each on the CPU (dispatch/launch cost — exactly what
                // the graph removes) and on the stream (GPU wall). The graph only wins end-to-end
                // when CPU dispatch is on the critical path (GPU starved); when the GPU is busy the
                // dispatch overlaps and is hidden.
                using Clock = std::chrono::steady_clock;
                auto cpu_ms = [](Clock::time_point x, Clock::time_point y) {
                    return std::chrono::duration<double, std::milli>(y - x).count();
                };
                std::vector<double> ng_cpu, g_cpu, ng_wall, g_wall, ng_gpu, g_gpu;
                for (int it = 0; it < iters; ++it) {
                    for (int b = 0; b < B; ++b) pre.process(base, d_input + b * single, stream);
                    DFINE_CUDA_CHECK(
                        cudaStreamSynchronize(stream));  // exclude preprocess from both
                    // no-graph: enqueueV3 + D2H
                    DFINE_CUDA_CHECK(cudaEventRecord(e0.get(), stream));
                    const auto c0 = Clock::now();
                    if (!session.context()->enqueueV3(stream))
                        throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), d_logits, logits_bytes,
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(), d_boxes, boxes_bytes,
                                                     cudaMemcpyDeviceToHost, stream));
                    const auto c1 = Clock::now();  // CPU dispatch done (work is async)
                    DFINE_CUDA_CHECK(cudaEventRecord(e1.get(), stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto c1s = Clock::now();  // full wall done
                    // graph: single replay (enqueueV3+D2H fused)
                    DFINE_CUDA_CHECK(cudaEventRecord(e2.get(), stream));
                    const auto c2 = Clock::now();
                    DFINE_CUDA_CHECK(cudaGraphLaunch(graph_exec.get(), stream));
                    const auto c3 = Clock::now();  // CPU dispatch done
                    DFINE_CUDA_CHECK(cudaEventRecord(e3.get(), stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto c3s = Clock::now();
                    if (std::memcmp(h_logits.data(), pl, logits_bytes) != 0 ||
                        std::memcmp(h_boxes.data(), pb, boxes_bytes) != 0) {
                        throw std::runtime_error("--graph-compare: output mismatch at batch " +
                                                 std::to_string(B) + ", iteration " +
                                                 std::to_string(it));
                    }
                    ng_cpu.push_back(cpu_ms(c0, c1));
                    g_cpu.push_back(cpu_ms(c2, c3));
                    ng_wall.push_back(cpu_ms(c0, c1s));
                    g_wall.push_back(cpu_ms(c2, c3s));
                    ng_gpu.push_back(ev_ms(e0.get(), e1.get()));
                    g_gpu.push_back(ev_ms(e2.get(), e3.get()));
                }
                const Stats nc = summarize(ng_cpu), gc = summarize(g_cpu);
                const Stats nw = summarize(ng_wall), gw = summarize(g_wall);
                const Stats ng = summarize(ng_gpu), gg = summarize(g_gpu);
                std::printf("[graph-compare] batch %d (%d iters, p50 ms):\n", B, iters);
                std::printf(
                    "  CPU dispatch : enqueueV3 %.3f  vs graphLaunch %.3f   -> graph removes %.3f "
                    "ms of CPU launch\n",
                    nc.p50, gc.p50, nc.p50 - gc.p50);
                std::printf("  GPU wall     : no-graph  %.3f  vs graph       %.3f   (Δ %.3f)\n",
                            ng.p50, gg.p50, ng.p50 - gg.p50);
                std::printf(
                    "  full wall    : no-graph  %.3f  vs graph       %.3f   (Δ %.3f ms, %+.1f%%)\n",
                    nw.p50, gw.p50, nw.p50 - gw.p50, nw.p50 > 0 ? (gw.p50 / nw.p50 - 1) * 100 : 0);
                continue;
            }

            std::vector<double> totals, pres, infs, d2hs, decs;
            totals.reserve(iters);
            for (int it = 0; it < iters; ++it) {
                one_iter(a, bb, c, d);
                pres.push_back(a);
                infs.push_back(bb);
                d2hs.push_back(c);
                decs.push_back(d);
                totals.push_back(a + bb + c + d);
            }
            const Stats st = summarize(totals);
            const Stats sp = summarize(pres), si = summarize(infs), s2 = summarize(d2hs),
                        sd = summarize(decs);
            const double imgs_per_s = 1000.0 * B / st.p50;
            std::printf("%-6d %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-8.1f\n", B, st.p50,
                        st.p90, st.p99, sp.p50, si.p50, sd.p50, imgs_per_s);

            if (!first_json) json += ",";
            first_json = false;
            char buf[512];
            std::snprintf(
                buf, sizeof buf,
                "{\"batch\":%d,\"cuda_graph_replay\":%s,\"total_p50\":%.4f,"
                "\"total_mean\":%.4f,\"total_p90\":%.4f,\"total_p99\":%.4f,"
                "\"preprocess_p50\":%.4f,\"infer_p50\":%.4f,\"d2h_p50\":%.4f,\"decode_p50\":%.4f,"
                "\"img_per_s\":%.2f,\"gpu_mem_mib\":%zu}",
                B, graph_exec ? "true" : "false", st.p50, st.mean, st.p90, st.p99, sp.p50, si.p50,
                s2.p50, sd.p50, imgs_per_s, used_mib);
            json += buf;
        }
        json += "],\"peak_gpu_mem_mib\":" + std::to_string(peak_used_mib) + "}\n";
        std::printf("peak GPU mem (engine+buffers): %zu MiB / %zu total\n", peak_used_mib,
                    total_mem / (1024 * 1024));

        if (!json_out.empty()) {
            std::ofstream jf(json_out);
            if (!jf) throw std::runtime_error("cannot open JSON output: " + json_out.string());
            jf << json;
            jf.close();
            if (!jf) {
                throw std::runtime_error("failed to write JSON output: " + json_out.string());
            }
            std::printf("wrote %s\n", json_out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
