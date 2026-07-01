// c_api.cpp — implementation of the stable C ABI declared in dfine/c_api.h.
//
// A thin, exception-safe shim over dfine::DFineDetector. No exception ever
// crosses the extern "C" boundary: every entry point that can fail catches
// std::exception, records it in thread-local storage (dfine_last_error), and
// returns NULL / a safe default. The library stays OpenCV-free — images arrive
// as raw HWC uint8 bytes and are wrapped in a non-owning dfine::ImageU8 view.

#include "dfine/c_api.h"

#include "dfine/core/coco_classes.hpp"
#include "dfine/core/log.hpp"
#include "dfine/tasks/detector.hpp"
#include "dfine/version.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Thread-local error storage
// ---------------------------------------------------------------------------

namespace {

thread_local std::string t_last_error;

void set_error(const std::string& msg) noexcept {
    try {
        t_last_error = msg;
    } catch (...) {
        // A failed assignment (bad_alloc) must not throw across the boundary.
        // Leave whatever was there; the caller still gets *some* diagnostic.
    }
}

// ---------------------------------------------------------------------------
// Log callback bridge — the C ABI callback carries an int severity; forward the
// typed C++ severity through unchanged (dfine::LogSeverity is int-backed).
// ---------------------------------------------------------------------------

std::atomic<dfine_log_fn_t> g_c_log_callback{nullptr};

void c_log_adapter(dfine::LogSeverity sev, const char* msg) noexcept {
    if (auto cb = g_c_log_callback.load(std::memory_order_acquire)) {
        cb(static_cast<int>(sev), msg ? msg : "");
    }
}

// ---------------------------------------------------------------------------
// Detection packing: dfine::Detections -> heap dfine_detections_t
// ---------------------------------------------------------------------------

dfine_detections_t* pack_detections(const dfine::Detections& dets) {
    auto* out = new dfine_detections_t;
    out->count = static_cast<int>(dets.size());
    try {
        out->detections = (out->count > 0)
                              ? new dfine_detection_t[static_cast<std::size_t>(out->count)]
                              : nullptr;
    } catch (...) {
        delete out;
        throw;
    }
    for (int i = 0; i < out->count; ++i) {
        const dfine::Detection& d = dets[static_cast<std::size_t>(i)];
        dfine_detection_t& o = out->detections[i];
        o.box.x1 = d.box.x1;
        o.box.y1 = d.box.y1;
        o.box.x2 = d.box.x2;
        o.box.y2 = d.box.y2;
        o.class_id = d.class_id;
        o.score = d.score;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Build a validated, non-owning ImageU8 view over caller-owned bytes.
// ---------------------------------------------------------------------------

dfine::ImageU8 make_view(const uint8_t* data, int width, int height, int step, int channels,
                         int is_bgr) {
    if (!data) throw std::invalid_argument("dfine: image data is NULL");
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("dfine: invalid image size (width=" + std::to_string(width) +
                                    " height=" + std::to_string(height) + ")");
    }
    if (channels != 3) {
        throw std::invalid_argument("dfine: channels must be 3 (got " + std::to_string(channels) +
                                    ")");
    }
    // 64-bit to avoid signed-int overflow (UB) for pathological widths.
    const long long min_step = static_cast<long long>(width) * channels;
    if (step > 0 && step < min_step) {
        throw std::invalid_argument("dfine: step (" + std::to_string(step) +
                                    ") is smaller than width*channels (" +
                                    std::to_string(min_step) + ")");
    }
    dfine::ImageU8 view;
    view.data = data;
    view.height = height;
    view.width = width;
    view.channels = channels;
    view.stride = step > 0 ? step : 0;  // 0 => tightly packed (width*channels)
    view.is_bgr = (is_bgr != 0);
    return view;
}

// Opaque handle: owns the C++ detector.
struct DetectorHandle {
    dfine::DFineDetector obj;
    explicit DetectorHandle(dfine::DFineDetector&& d) : obj(std::move(d)) {}
};

}  // namespace

// The opaque struct the header forward-declares.
struct dfine_detector_s {
    DetectorHandle h;
    // Class-name strings cached per handle so dfine_detector_class_name can hand
    // out pointers that stay valid for the detector's lifetime.
    std::vector<std::string> class_names;
    explicit dfine_detector_s(dfine::DFineDetector&& d) : h(std::move(d)) {
        const int c = h.obj.num_classes();
        class_names.reserve(c > 0 ? static_cast<std::size_t>(c) : 0);
        for (int i = 0; i < c; ++i) class_names.push_back(h.obj.class_name(i));
    }
};

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

const char* dfine_last_error(void) {
    return t_last_error.c_str();
}

void dfine_set_log_callback(dfine_log_fn_t callback) {
    g_c_log_callback.store(callback, std::memory_order_release);
    dfine::set_log_callback(callback ? &c_log_adapter : nullptr);
}

const char* dfine_version(void) {
    return dfine::version();
}

const char* dfine_class_name(int class_id) {
    return dfine::coco_class_name(class_id);
}

const char* dfine_detector_class_name(const dfine_detector_t* det, int class_id) {
    if (!det || class_id < 0 || static_cast<std::size_t>(class_id) >= det->class_names.size()) {
        return "";
    }
    return det->class_names[static_cast<std::size_t>(class_id)].c_str();
}

// ----- lifecycle -----

dfine_detector_t* dfine_detector_create_ex(const char* engine_path, const char* meta_path,
                                           const dfine_options_t* opts) {
    t_last_error.clear();
    if (!engine_path) {
        set_error("dfine_detector_create: engine_path is NULL");
        return nullptr;
    }
    try {
        dfine::DetectorOptions dopts;  // threshold 0.5, graph off, warmup 3
        if (opts) {
            if (opts->threshold > 0.0f) dopts.threshold = opts->threshold;
            dopts.use_cuda_graph = (opts->use_cuda_graph != 0);
            if (opts->graph_warmup_iters > 0) dopts.graph_warmup_iters = opts->graph_warmup_iters;
        }
        dfine::DFineDetector det = (meta_path && meta_path[0] != '\0')
                                       ? dfine::DFineDetector(engine_path, meta_path, dopts)
                                       : dfine::DFineDetector(engine_path, dopts);
        return new dfine_detector_s(std::move(det));
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("dfine_detector_create: unknown error");
        return nullptr;
    }
}

dfine_detector_t* dfine_detector_create(const char* engine_path, const char* meta_path) {
    return dfine_detector_create_ex(engine_path, meta_path, nullptr);
}

void dfine_detector_destroy(dfine_detector_t* det) {
    delete det;  // ~DFineDetector releases the engine/context/CUDA resources
}

// ----- introspection -----

const char* dfine_detector_variant(const dfine_detector_t* det) {
    return det ? det->h.obj.variant().c_str() : "";
}
int dfine_detector_input_width(const dfine_detector_t* det) {
    return det ? det->h.obj.input_w() : 0;
}
int dfine_detector_input_height(const dfine_detector_t* det) {
    return det ? det->h.obj.input_h() : 0;
}
int dfine_detector_num_queries(const dfine_detector_t* det) {
    return det ? det->h.obj.num_queries() : 0;
}
int dfine_detector_num_classes(const dfine_detector_t* det) {
    return det ? det->h.obj.num_classes() : 0;
}
int dfine_detector_max_batch(const dfine_detector_t* det) {
    return det ? det->h.obj.max_batch() : 0;
}

// ----- inference -----

dfine_detections_t* dfine_detector_detect(dfine_detector_t* det, const uint8_t* data, int width,
                                          int height, int step, int channels, int is_bgr,
                                          float threshold) {
    t_last_error.clear();
    if (!det) {
        set_error("dfine_detector_detect: detector is NULL");
        return nullptr;
    }
    try {
        dfine::ImageU8 view = make_view(data, width, height, step, channels, is_bgr);
        dfine::Detections dets = det->h.obj.detect(view, threshold);
        return pack_detections(dets);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("dfine_detector_detect: unknown error");
        return nullptr;
    }
}

dfine_detections_t** dfine_detector_detect_batch(dfine_detector_t* det, const dfine_image_t* images,
                                                 int count, float threshold) {
    t_last_error.clear();
    if (!det) {
        set_error("dfine_detector_detect_batch: detector is NULL");
        return nullptr;
    }
    if (!images || count <= 0) {
        set_error("dfine_detector_detect_batch: images is NULL or count <= 0");
        return nullptr;
    }
    // Packed C result-sets accumulate here; freed on any failure before release.
    std::vector<dfine_detections_t*> packed;
    try {
        std::vector<dfine::ImageU8> views;
        views.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            const dfine_image_t& im = images[i];
            views.push_back(
                make_view(im.data, im.width, im.height, im.step, im.channels, im.is_bgr));
        }
        std::vector<dfine::Detections> results = det->h.obj.detect_batch(views, threshold);
        packed.reserve(results.size());
        for (const dfine::Detections& r : results) packed.push_back(pack_detections(r));

        auto** out = new dfine_detections_t*[static_cast<std::size_t>(packed.size())];
        for (std::size_t i = 0; i < packed.size(); ++i) out[i] = packed[i];
        return out;  // ownership transferred; skip the cleanup below
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("dfine_detector_detect_batch: unknown error");
    }
    for (dfine_detections_t* p : packed) dfine_detections_free(p);
    return nullptr;
}

// ----- result memory management -----

void dfine_detections_free(dfine_detections_t* dets) {
    if (!dets) return;
    delete[] dets->detections;
    delete dets;
}

void dfine_detections_free_batch(dfine_detections_t** results, int count) {
    if (!results) return;
    for (int i = 0; i < count; ++i) dfine_detections_free(results[i]);
    delete[] results;
}

}  // extern "C"
