// dfine_detect — run the D-FINE detector on a single image and print detections.
//
// usage: dfine_detect --engine E.engine --image img.jpg [--meta E.json]
//                     [--threshold 0.5] [--cuda-graph] [--gpu-decode]

#include "cli_helpers.hpp"
#include "image_io.hpp"

#include "dfine/core/coco_classes.hpp"
#include "dfine/tasks/detector.hpp"
#include "dfine/version.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    std::filesystem::path engine, meta, image;
    float threshold = 0.5f;
    bool cuda_graph = false;
    bool gpu_decode = false;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf(
                    "usage: %s --engine E.engine --image img.jpg [--meta E.json] "
                    "[--threshold 0.5] [--cuda-graph] [--gpu-decode]\n  dfine v%s\n",
                    argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine"))
                engine = next_value(argc, argv, i, "--engine");
            else if (starts_with(a, "--meta"))
                meta = next_value(argc, argv, i, "--meta");
            else if (starts_with(a, "--image"))
                image = next_value(argc, argv, i, "--image");
            else if (starts_with(a, "--threshold"))
                threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            else if (a == "--cuda-graph")
                cuda_graph = true;
            else if (a == "--gpu-decode")
                gpu_decode = true;
            else
                throw std::runtime_error("unknown arg: " + std::string(a));
        }
        if (engine.empty() || image.empty()) {
            std::fprintf(stderr, "error: --engine and --image are required\n");
            return 2;
        }

        dfine_app::LoadedImage img = dfine_app::load_image_rgb(image.string());
        if (!img) throw std::runtime_error("cannot decode image: " + image.string());

        dfine::DetectorOptions opts;
        opts.threshold = threshold;
        opts.use_cuda_graph = cuda_graph;
        opts.gpu_decode = gpu_decode;
        dfine::DFineDetector det = meta.empty() ? dfine::DFineDetector(engine, opts)
                                                : dfine::DFineDetector(engine, meta, opts);

        std::printf("engine: %s  variant=%s  input=%dx%d  queries=%d  classes=%d\n", engine.c_str(),
                    det.variant().empty() ? "?" : det.variant().c_str(), det.input_w(),
                    det.input_h(), det.num_queries(), det.num_classes());
        std::printf("image : %s  %dx%d\n", image.c_str(), img.width(), img.height());

        dfine::Detections dets = det.detect(img.view(), threshold);
        std::sort(
            dets.begin(), dets.end(),
            [](const dfine::Detection& a, const dfine::Detection& b) { return a.score > b.score; });

        const auto& t = det.last_timings();
        std::printf("%zu detections (thr=%.2f) | infer=%.2fms decode=%.2fms total=%.2fms\n",
                    dets.size(), threshold, t.infer_ms, t.postprocess_ms, t.total_ms);
        for (const auto& d : dets) {
            std::printf("  %-16s %.3f  [%.1f, %.1f, %.1f, %.1f]\n",
                        dfine::coco_class_name(d.class_id), d.score, d.box.x1, d.box.y1, d.box.x2,
                        d.box.y2);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
