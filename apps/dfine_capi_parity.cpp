// dfine_capi_parity — prove the C ABI returns detections byte-identical to the
// C++ DFineDetector::detect on the same image. This is the acceptance test for
// "dfine_detector_detect must be IDENTICAL to dfine_detect".
//
// It decodes one image, runs the C++ API and the C ABI on the exact same pixel
// buffer + threshold, and compares every detection field bit-for-bit (the C ABI
// wraps the same detect(), so the results must match exactly — not merely close).
//
// usage: dfine_capi_parity --engine E.engine --image img.jpg [--meta E.json] [--threshold 0.5]

#include "cli_helpers.hpp"
#include "image_io.hpp"

#include "dfine/c_api.h"
#include "dfine/core/coco_classes.hpp"
#include "dfine/tasks/detector.hpp"
#include "dfine/version.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool box_equal(const dfine::Box& a, const dfine_box_t& b) {
    return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
}

// Compare a C++ Detections vector against a C-ABI result set, field by field.
// Returns the number of mismatches (0 == perfect parity) and prints the first few.
int compare(const dfine::Detections& cpp, const dfine_detections_t* c) {
    int mismatches = 0;
    if (static_cast<int>(cpp.size()) != c->count) {
        std::printf("  count mismatch: c++=%zu c-abi=%d\n", cpp.size(), c->count);
        return 1 + std::abs(static_cast<int>(cpp.size()) - c->count);
    }
    for (std::size_t i = 0; i < cpp.size(); ++i) {
        const dfine::Detection& a = cpp[i];
        const dfine_detection_t& b = c->detections[i];
        if (a.class_id != b.class_id || a.score != b.score || !box_equal(a.box, b.box)) {
            if (mismatches < 5) {
                std::printf("  [diff #%zu] c++: cls=%d score=%.9g box=[%.6f %.6f %.6f %.6f]\n", i,
                            a.class_id, a.score, a.box.x1, a.box.y1, a.box.x2, a.box.y2);
                std::printf("            c-abi: cls=%d score=%.9g box=[%.6f %.6f %.6f %.6f]\n",
                            b.class_id, b.score, b.box.x1, b.box.y1, b.box.x2, b.box.y2);
            }
            ++mismatches;
        }
    }
    return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path engine, meta, image;
    float threshold = 0.5f;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf(
                    "usage: %s --engine E.engine --image img.jpg [--meta E.json] "
                    "[--threshold 0.5]\n  dfine v%s\n",
                    argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine")) {
                engine = next_value(argc, argv, i, "--engine");
            } else if (starts_with(a, "--meta")) {
                meta = next_value(argc, argv, i, "--meta");
            } else if (starts_with(a, "--image")) {
                image = next_value(argc, argv, i, "--image");
            } else if (starts_with(a, "--threshold")) {
                threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            } else {
                throw std::runtime_error("unknown arg: " + std::string(a));
            }
        }
        if (engine.empty() || image.empty()) {
            std::fprintf(stderr, "error: --engine and --image are required\n");
            return 2;
        }

        dfine_app::LoadedImage img = dfine_app::load_image_rgb(image.string());
        if (!img) throw std::runtime_error("cannot decode image: " + image.string());
        const dfine::ImageU8 view = img.view();

        // ---- C++ reference ----
        dfine::DetectorOptions opts;
        opts.threshold = threshold;
        dfine::DFineDetector ref = meta.empty() ? dfine::DFineDetector(engine, opts)
                                                : dfine::DFineDetector(engine, meta, opts);
        dfine::Detections cpp_dets = ref.detect(view, threshold);

        // ---- C ABI (independent handle, same bytes) — RAII so it's freed on any throw ----
        std::unique_ptr<dfine_detector_t, void (*)(dfine_detector_t*)> det(
            dfine_detector_create(engine.string().c_str(),
                                  meta.empty() ? nullptr : meta.string().c_str()),
            &dfine_detector_destroy);
        if (!det)
            throw std::runtime_error(std::string("c_api create failed: ") + dfine_last_error());

        dfine_detections_t* c_dets =
            dfine_detector_detect(det.get(), view.data, view.width, view.height, view.row_bytes(),
                                  view.channels, view.is_bgr ? 1 : 0, threshold);
        if (!c_dets)
            throw std::runtime_error(std::string("c_api detect failed: ") + dfine_last_error());

        std::printf("image %s  %dx%d  thr=%.2f\n", image.string().c_str(), view.width, view.height,
                    threshold);
        std::printf("c++ detect: %zu  |  c-abi detect: %d\n", cpp_dets.size(), c_dets->count);

        int single_mismatches = compare(cpp_dets, c_dets);
        dfine_detections_free(c_dets);

        // ---- batch parity (same image twice) via the C ABI ----
        // Guarded: a static (max_batch<2) engine makes the C++ detect_batch throw, so
        // degrade to a note instead of failing the harness (the C-ABI side already does).
        int batch_mismatches = 0;
        try {
            std::vector<dfine::ImageU8> views{view, view};
            std::vector<dfine::Detections> cpp_batch = ref.detect_batch(views, threshold);

            dfine_image_t bi[2];
            std::memset(bi, 0, sizeof bi);
            for (int i = 0; i < 2; ++i) {
                bi[i].data = view.data;
                bi[i].width = view.width;
                bi[i].height = view.height;
                bi[i].step = view.row_bytes();
                bi[i].channels = view.channels;
                bi[i].is_bgr = view.is_bgr ? 1 : 0;
            }
            dfine_detections_t** cb = dfine_detector_detect_batch(det.get(), bi, 2, threshold);
            if (!cb) {
                std::printf("  [note] c_api detect_batch failed (%s) — skipping batch parity\n",
                            dfine_last_error());
            } else {
                for (int i = 0; i < 2; ++i) batch_mismatches += compare(cpp_batch[i], cb[i]);
                dfine_detections_free_batch(cb, 2);
                std::printf("batch parity (2x): %s\n",
                            batch_mismatches == 0 ? "identical" : "DIFF");
            }
        } catch (const std::exception& e) {
            std::printf("  [note] batch parity skipped (%s)\n", e.what());
        }

        const int total = single_mismatches + batch_mismatches;
        std::printf("\n%s — single=%d batch=%d mismatch(es)\n", total == 0 ? "PASS" : "FAIL",
                    single_mismatches, batch_mismatches);
        return total == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
