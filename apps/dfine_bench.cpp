// dfine_bench — precise per-stage latency + GPU-memory benchmark of the D-FINE
// detector's C++ path. Times each stage on the CUDA stream with events
// (preprocess+H2D / infer / D2H) plus host-side decode, with warm-up and
// percentiles, across a set of batch sizes.
//
// usage:
//   dfine_bench --engine E.engine [--meta E.json] [--image img.jpg]
//               [--src-size WxH] [--batches 1,2,4,8] [--warmup 20] [--iters 200]
//               [--json out.json] [--threshold 0.001]

#include "cli_helpers.hpp"
#include "image_io.hpp"

#include "dfine/core/engine_meta.hpp"
#include "dfine/core/postprocess.hpp"
#include "dfine/version.hpp"
#include "internal/cuda_check.hpp"
#include "internal/cuda_preprocess.cuh"
#include "internal/cuda_raii.hpp"
#include "internal/decode_gpu.cuh"
#include "internal/trt_session.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

std::vector<int> parse_batches(std::string_view s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stoi(cur)); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    if (out.empty()) out = {1};
    return out;
}

const dfine::BindingInfo* find_output(const dfine::TrtSession& s, const char* name, int want_last) {
    if (const auto* b = s.find(name)) return b;
    for (int i : s.output_indices()) {
        const auto& b = s.bindings()[i];
        if (b.shape.nbDims > 0 && b.shape.d[b.shape.nbDims - 1] == want_last) return &b;
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path engine, meta, image;
    std::string batches_arg = "1,2,4,8";
    int warmup = 20, iters = 200, src_w = 640, src_h = 480;
    float threshold = 0.001f;
    bool cuda_graph = false;
    bool graph_compare = false;
    bool gpu_decode = false;
    std::filesystem::path json_out;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf("usage: %s --engine E [--meta M] [--image img] [--src-size WxH] "
                            "[--batches 1,2,4,8] [--warmup 20] [--iters 200] [--json out] [--cuda-graph]\n"
                            "  --cuda-graph  replay enqueueV3+D2H from a captured CUDA graph "
                            "(infer col then includes D2H)\n  dfine v%s\n",
                            argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine"))    engine = next_value(argc, argv, i, "--engine");
            else if (starts_with(a, "--meta"))         meta = next_value(argc, argv, i, "--meta");
            else if (starts_with(a, "--image"))        image = next_value(argc, argv, i, "--image");
            else if (starts_with(a, "--batches"))      batches_arg = next_value(argc, argv, i, "--batches");
            else if (starts_with(a, "--warmup"))       warmup = parse_int(next_value(argc, argv, i, "--warmup"), "--warmup");
            else if (starts_with(a, "--iters"))        iters = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
            else if (starts_with(a, "--threshold"))    threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            else if (starts_with(a, "--json"))         json_out = next_value(argc, argv, i, "--json");
            else if (a == "--cuda-graph")              cuda_graph = true;
            else if (a == "--graph-compare")           { cuda_graph = true; graph_compare = true; }
            else if (a == "--gpu-decode")              gpu_decode = true;
            else if (starts_with(a, "--src-size")) {
                std::string v = next_value(argc, argv, i, "--src-size");
                const auto x = v.find('x');
                if (x == std::string::npos) throw std::runtime_error("--src-size expects WxH");
                src_w = std::stoi(v.substr(0, x));
                src_h = std::stoi(v.substr(x + 1));
            } else throw std::runtime_error("unknown arg: " + std::string(a));
        }
        if (engine.empty()) { std::fprintf(stderr, "error: --engine required\n"); return 2; }

        // Baseline free memory before we build anything (force context init first).
        DFINE_CUDA_CHECK(cudaFree(nullptr));
        std::size_t free_before = 0, total_mem = 0;
        DFINE_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

        // Meta (sidecar or default) for input dims + tensor names.
        dfine::EngineMeta m;
        if (meta.empty()) {
            std::filesystem::path alt = engine; alt.replace_extension(".json");
            if (std::filesystem::is_regular_file(engine.string() + ".json"))
                m = dfine::EngineMeta::from_json_file(engine.string() + ".json");
            else if (std::filesystem::is_regular_file(alt))
                m = dfine::EngineMeta::from_json_file(alt);
        } else if (std::filesystem::is_regular_file(meta)) {
            m = dfine::EngineMeta::from_json_file(meta);
        }
        const int H = m.input_h, W = m.input_w;

        dfine::TrtSession session(engine);
        const std::string in_name = m.input_names.empty() ? "images" : m.input_names.front();
        const bool dynamic = [&] {
            const auto* b = session.find(in_name);
            return b && b->shape.nbDims >= 1 && b->shape.d[0] < 0;
        }();

        // Source image (real, repeated) or synthetic gradient.
        dfine_app::LoadedImage loaded;
        std::vector<std::uint8_t> synth;
        dfine::ImageU8 base;
        if (!image.empty()) {
            loaded = dfine_app::load_image_rgb(image.string());
            if (!loaded) throw std::runtime_error("cannot decode image: " + image.string());
            base = loaded.view();
            src_w = base.width; src_h = base.height;
        } else {
            synth.resize(static_cast<std::size_t>(src_w) * src_h * 3);
            for (std::size_t i = 0; i < synth.size(); ++i) synth[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
            base = dfine::ImageU8{synth.data(), src_h, src_w, 3, src_w * 3, false};
        }

        dfine::ImagePreprocessor pre(H, W);
        pre.set_mean(m.mean[0], m.mean[1], m.mean[2]);
        pre.set_std(m.std[0], m.std[1], m.std[2]);

        const dfine::BindingInfo* b_logits = find_output(session, "logits", 80);
        const dfine::BindingInfo* b_boxes  = find_output(session, "boxes", 4);
        if (!b_logits || !b_boxes) throw std::runtime_error("dfine_bench: cannot resolve logits/boxes outputs");
        const int N = static_cast<int>(b_logits->shape.d[b_logits->shape.nbDims - 2]);
        const int C = static_cast<int>(b_logits->shape.d[b_logits->shape.nbDims - 1]);

        cudaStream_t stream = session.stream();
        cudaEvent_t e0, e1, e2, e3;
        DFINE_CUDA_CHECK(cudaEventCreate(&e0)); DFINE_CUDA_CHECK(cudaEventCreate(&e1));
        DFINE_CUDA_CHECK(cudaEventCreate(&e2)); DFINE_CUDA_CHECK(cudaEventCreate(&e3));

        const auto batches = parse_batches(batches_arg);
        std::printf("dfine_bench: engine=%s  variant=%s  input=%dx%d  src=%dx%d  warmup=%d iters=%d%s\n",
                    engine.filename().c_str(), m.variant.empty() ? "?" : m.variant.c_str(),
                    W, H, src_w, src_h, warmup, iters,
                    cuda_graph ? "  [cuda-graph: infer_ms includes D2H]" : "");
        std::printf("%-6s %-11s %-11s %-11s %-11s %-11s %-11s %-8s\n",
                    "batch", "total_p50", "total_p90", "total_p99", "pre_ms", "infer_ms", "decode_ms", "img/s");

        std::string json = "{\"engine\":\"" + engine.string() + "\",\"input\":[" +
                           std::to_string(W) + "," + std::to_string(H) + "]," +
                           "\"cuda_graph\":" + (cuda_graph ? "true" : "false") + ",\"results\":[";
        bool first_json = true;
        std::size_t peak_used_mib = 0;

        for (int B : batches) {
            if (dynamic) session.set_input_shape(in_name, nvinfer1::Dims4{B, 3, H, W});
            else if (B != 1) { std::printf("(static engine — skipping batch %d)\n", B); continue; }

            const std::size_t single = static_cast<std::size_t>(3) * H * W;
            float* d_input = static_cast<float*>(session.device_buffer(in_name));
            std::vector<float> h_logits(static_cast<std::size_t>(B) * N * C);
            std::vector<float> h_boxes(static_cast<std::size_t>(B) * N * 4);
            void* d_logits = session.device_buffer(b_logits->name);
            void* d_boxes  = session.device_buffer(b_boxes->name);
            const std::size_t logits_bytes = h_logits.size() * sizeof(float);
            const std::size_t boxes_bytes  = h_boxes.size() * sizeof(float);

            // CUDA-graph replay copies into pinned buffers (a captured graph cannot D2H
            // into pageable memory — it would force a sync and abort capture).
            dfine::HostPtr p_logits, p_boxes;
            if (cuda_graph) {
                void* p = nullptr;
                DFINE_CUDA_CHECK(cudaMallocHost(&p, logits_bytes)); p_logits.reset(p);
                DFINE_CUDA_CHECK(cudaMallocHost(&p, boxes_bytes));  p_boxes.reset(p);
            }
            float* pl = static_cast<float*>(p_logits.get());
            float* pb = static_cast<float*>(p_boxes.get());
            dfine::CudaGraphExec graph_exec;  // empty => plain enqueueV3 path

            dfine::PostprocessParams pp; pp.num_queries = N; pp.num_classes = C; pp.topk = N; pp.threshold = threshold;

            // GPU-decode scratch (Zero-D2H): replaces the full-logits D2H + CPU decode
            // with on-device kernels + a compact survivor D2H, folded into the d2h stage.
            dfine::DevPtr g_keys, g_vals, g_ko, g_vo, g_seg, g_temp, g_out, g_counts, g_scale;
            dfine::GpuDecodeScratch gdec;
            std::vector<dfine::DetectionGPU> gh_out;
            std::vector<uint32_t>            gh_counts;
            if (gpu_decode) {
                const int    n_cand = N * C;
                const std::size_t tot = static_cast<std::size_t>(B) * n_cand;
                auto da = [](dfine::DevPtr& p, std::size_t bytes) -> void* {
                    void* q = nullptr; DFINE_CUDA_CHECK(cudaMalloc(&q, bytes)); p.reset(q); return q; };
                gdec.keys     = static_cast<float*>(da(g_keys, tot * sizeof(float)));
                gdec.vals     = static_cast<uint32_t*>(da(g_vals, tot * sizeof(uint32_t)));
                gdec.keys_out = static_cast<float*>(da(g_ko, tot * sizeof(float)));
                gdec.vals_out = static_cast<uint32_t*>(da(g_vo, tot * sizeof(uint32_t)));
                gdec.seg_off  = static_cast<int*>(da(g_seg, static_cast<std::size_t>(B + 1) * sizeof(int)));
                gdec.out      = static_cast<dfine::DetectionGPU*>(
                    da(g_out, static_cast<std::size_t>(B) * N * sizeof(dfine::DetectionGPU)));
                gdec.counts   = static_cast<uint32_t*>(da(g_counts, static_cast<std::size_t>(B) * sizeof(uint32_t)));
                gdec.scale_wh = static_cast<float2*>(da(g_scale, static_cast<std::size_t>(B) * sizeof(float2)));
                gdec.cub_temp_bytes = dfine::gpu_decode_temp_bytes(B, n_cand);
                gdec.cub_temp = da(g_temp, gdec.cub_temp_bytes);
                dfine::gpu_decode_fill_segoff(gdec.seg_off, B, n_cand, stream);
                std::vector<float2> hs(static_cast<std::size_t>(B),
                                       float2{static_cast<float>(src_w), static_cast<float>(src_h)});
                DFINE_CUDA_CHECK(cudaMemcpyAsync(gdec.scale_wh, hs.data(),
                                                 static_cast<std::size_t>(B) * sizeof(float2),
                                                 cudaMemcpyHostToDevice, stream));
                DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                gh_out.resize(static_cast<std::size_t>(B) * N);
                gh_counts.resize(static_cast<std::size_t>(B));
            }

            auto one_iter = [&](double& pre_ms, double& inf_ms, double& d2h_ms, double& dec_ms) {
                DFINE_CUDA_CHECK(cudaEventRecord(e0, stream));
                for (int b = 0; b < B; ++b) pre.process(base, d_input + b * single, stream);
                DFINE_CUDA_CHECK(cudaEventRecord(e1, stream));
                if (gpu_decode) {
                    // infer -> on-device decode -> compact survivor D2H (no CPU decode).
                    if (!session.context()->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaEventRecord(e2, stream));
                    dfine::gpu_decode_enqueue(static_cast<const float*>(d_logits),
                                              static_cast<const float*>(d_boxes), B, N, C, N, threshold,
                                              gdec, stream);
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(gh_out.data(), gdec.out,
                                                     gh_out.size() * sizeof(dfine::DetectionGPU),
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(gh_counts.data(), gdec.counts,
                                                     gh_counts.size() * sizeof(uint32_t),
                                                     cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3, stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    pre_ms = ev_ms(e0, e1); inf_ms = ev_ms(e1, e2); d2h_ms = ev_ms(e2, e3); dec_ms = 0.0;
                    return;
                }
                if (graph_exec) {
                    // replay = enqueueV3 + both D2H copies fused; d2h folds into infer.
                    DFINE_CUDA_CHECK(cudaGraphLaunch(graph_exec.get(), stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e2, stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3, stream));
                } else {
                    if (!session.context()->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaEventRecord(e2, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), d_logits, logits_bytes, cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(), d_boxes, boxes_bytes, cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaEventRecord(e3, stream));
                }
                DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                pre_ms = ev_ms(e0, e1); inf_ms = ev_ms(e1, e2); d2h_ms = ev_ms(e2, e3);
                const float* Lsrc = graph_exec ? pl : h_logits.data();
                const float* Bsrc = graph_exec ? pb : h_boxes.data();
                const auto t0 = std::chrono::steady_clock::now();
                for (int b = 0; b < B; ++b)
                    (void)dfine::decode_detections(Lsrc + static_cast<std::size_t>(b) * N * C,
                                                   Bsrc + static_cast<std::size_t>(b) * N * 4, src_w, src_h, pp);
                dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            };

            double a, bb, c, d;
            for (int w = 0; w < warmup; ++w) one_iter(a, bb, c, d);  // enqueueV3 path: flush tactics + shape

            if (cuda_graph && session.num_aux_streams() != 0) {
                std::printf("(engine uses %d aux streams — ThreadLocal capture unsafe, using enqueueV3)\n",
                            session.num_aux_streams());
            } else if (cuda_graph) {
                // Capture enqueueV3 + D2H after the warm-up has flushed deferred setup.
                session.context()->setEnqueueEmitsProfile(false);
                cudaGraph_t g = nullptr;
                bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
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

            if (graph_compare) {
                if (!graph_exec) { std::printf("[graph-compare] batch %d: capture failed, skipping\n", B); continue; }
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
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));  // exclude preprocess from both
                    // no-graph: enqueueV3 + D2H
                    DFINE_CUDA_CHECK(cudaEventRecord(e0, stream));
                    const auto c0 = Clock::now();
                    if (!session.context()->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), d_logits, logits_bytes, cudaMemcpyDeviceToHost, stream));
                    DFINE_CUDA_CHECK(cudaMemcpyAsync(h_boxes.data(), d_boxes, boxes_bytes, cudaMemcpyDeviceToHost, stream));
                    const auto c1 = Clock::now();               // CPU dispatch done (work is async)
                    DFINE_CUDA_CHECK(cudaEventRecord(e1, stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto c1s = Clock::now();              // full wall done
                    // graph: single replay (enqueueV3+D2H fused)
                    DFINE_CUDA_CHECK(cudaEventRecord(e2, stream));
                    const auto c2 = Clock::now();
                    DFINE_CUDA_CHECK(cudaGraphLaunch(graph_exec.get(), stream));
                    const auto c3 = Clock::now();               // CPU dispatch done
                    DFINE_CUDA_CHECK(cudaEventRecord(e3, stream));
                    DFINE_CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto c3s = Clock::now();
                    ng_cpu.push_back(cpu_ms(c0, c1));   g_cpu.push_back(cpu_ms(c2, c3));
                    ng_wall.push_back(cpu_ms(c0, c1s)); g_wall.push_back(cpu_ms(c2, c3s));
                    ng_gpu.push_back(ev_ms(e0, e1));    g_gpu.push_back(ev_ms(e2, e3));
                }
                const Stats nc = summarize(ng_cpu), gc = summarize(g_cpu);
                const Stats nw = summarize(ng_wall), gw = summarize(g_wall);
                const Stats ng = summarize(ng_gpu),  gg = summarize(g_gpu);
                std::printf("[graph-compare] batch %d (%d iters, p50 ms):\n", B, iters);
                std::printf("  CPU dispatch : enqueueV3 %.3f  vs graphLaunch %.3f   -> graph removes %.3f ms of CPU launch\n",
                            nc.p50, gc.p50, nc.p50 - gc.p50);
                std::printf("  GPU wall     : no-graph  %.3f  vs graph       %.3f   (Δ %.3f)\n",
                            ng.p50, gg.p50, ng.p50 - gg.p50);
                std::printf("  full wall    : no-graph  %.3f  vs graph       %.3f   (Δ %.3f ms, %+.1f%%)\n",
                            nw.p50, gw.p50, nw.p50 - gw.p50, nw.p50 > 0 ? (gw.p50 / nw.p50 - 1) * 100 : 0);
                continue;
            }

            // Peak memory after warm-up (engine + context + all buffers resident).
            std::size_t free_now = 0;
            DFINE_CUDA_CHECK(cudaMemGetInfo(&free_now, &total_mem));
            const std::size_t used_mib = (free_before - free_now) / (1024 * 1024);
            peak_used_mib = std::max(peak_used_mib, used_mib);

            std::vector<double> totals, pres, infs, d2hs, decs;
            totals.reserve(iters);
            for (int it = 0; it < iters; ++it) {
                one_iter(a, bb, c, d);
                pres.push_back(a); infs.push_back(bb); d2hs.push_back(c); decs.push_back(d);
                totals.push_back(a + bb + c + d);
            }
            const Stats st = summarize(totals);
            const Stats sp = summarize(pres), si = summarize(infs), s2 = summarize(d2hs), sd = summarize(decs);
            const double imgs_per_s = 1000.0 * B / st.p50;
            std::printf("%-6d %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-8.1f\n",
                        B, st.p50, st.p90, st.p99, sp.p50, si.p50, sd.p50, imgs_per_s);

            if (!first_json) json += ",";
            first_json = false;
            char buf[512];
            std::snprintf(buf, sizeof buf,
                "{\"batch\":%d,\"total_p50\":%.4f,\"total_p90\":%.4f,\"total_p99\":%.4f,"
                "\"preprocess_p50\":%.4f,\"infer_p50\":%.4f,\"d2h_p50\":%.4f,\"decode_p50\":%.4f,"
                "\"img_per_s\":%.2f,\"gpu_mem_mib\":%zu}",
                B, st.p50, st.p90, st.p99, sp.p50, si.p50, s2.p50, sd.p50, imgs_per_s, used_mib);
            json += buf;
        }
        json += "],\"peak_gpu_mem_mib\":" + std::to_string(peak_used_mib) + "}\n";
        std::printf("peak GPU mem (engine+buffers): %zu MiB / %zu total\n", peak_used_mib, total_mem / (1024 * 1024));

        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
        if (!json_out.empty()) {
            std::ofstream jf(json_out);
            jf << json;
            std::printf("wrote %s\n", json_out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
