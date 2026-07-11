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

#include "internal/image_check.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
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
    dfine::ImageU8 view;
    view.data = data;
    view.height = height;
    view.width = width;
    view.channels = channels;
    view.stride = step > 0 ? step : 0;  // documented C contract: <=0 => tightly packed
    view.is_bgr = (is_bgr != 0);
    // Same validator as the C++ entry points, so C and C++ cannot drift.
    dfine::validate_image_layout(view);
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
        dfine::DetectorOptions dopts;  // threshold 0.5, everything else off
        if (opts) {
            if (opts->struct_size == 0) {
                set_error(
                    "dfine_detector_create: options.struct_size is 0 — zero-initialize the "
                    "struct AND set struct_size = sizeof(dfine_options_t); binaries built "
                    "against the pre-0.2.0 header must be rebuilt");
                return nullptr;
            }
            // Plausibility bound. A pre-0.2.0 binary (v1 layout: float threshold
            // first) that dodged the SONAME bump presents its threshold bits here
            // — e.g. 0.5f reads as 1,056,964,608 — and would otherwise cause an
            // out-of-bounds read of the caller's 12-byte struct. Any real
            // struct_size stays tiny; reject the absurd loudly.
            if (opts->struct_size > 4096) {
                set_error(
                    "dfine_detector_create: options.struct_size is implausibly large — the "
                    "calling binary was likely built against the pre-0.2.0 header; rebuild "
                    "against the current dfine/c_api.h");
                return nullptr;
            }
            // Versioned copy: read exactly the bytes the caller's binary knows
            // about; fields appended after its build stay zero (= defaults).
            dfine_options_t o{};
            std::memcpy(&o, opts, std::min(opts->struct_size, sizeof(dfine_options_t)));
            if (!std::isfinite(o.threshold)) {
                set_error("dfine_detector_create: options.threshold must be finite");
                return nullptr;
            }
            if (o.threshold > 0.0f) dopts.threshold = o.threshold;
            dopts.use_cuda_graph = (o.use_cuda_graph != 0);
            if (o.graph_warmup_iters > 0) dopts.graph_warmup_iters = o.graph_warmup_iters;
            dopts.gpu_decode = (o.gpu_decode != 0);
            dopts.own_device_memory = (o.own_device_memory != 0);
            dopts.full_pipeline_graph = (o.full_pipeline_graph != 0);
            if (o.resize == 1) dopts.preprocess.resize = dfine::PreprocessSpec::Resize::kStretch;
            if (o.resize == 2) {
                dopts.preprocess.resize = dfine::PreprocessSpec::Resize::kLetterbox;
                dopts.preprocess.anchor_topleft = (o.letterbox_topleft != 0);
                if (o.letterbox_pad > 255) {
                    set_error(
                        "dfine_detector_create: options.letterbox_pad must be in 0..255 or "
                        "negative for the default");
                    return nullptr;
                }
                dopts.preprocess.pad_value = o.letterbox_pad < 0 ? 114 : o.letterbox_pad;
                dopts.preprocess.allow_upscale = (o.letterbox_no_upscale == 0);
            }
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

// ----- frozen pipeline -----

int dfine_detector_freeze(dfine_detector_t* det, const dfine_freeze_spec_t* spec) {
    t_last_error.clear();
    if (!det) {
        set_error("dfine_detector_freeze: det is NULL");
        return -1;
    }
    try {
        dfine::FreezeSpec fs;
        if (spec) {
            if (spec->struct_size == 0) {
                set_error(
                    "dfine_detector_freeze: spec.struct_size is 0 — set it to "
                    "sizeof(dfine_freeze_spec_t)");
                return -1;
            }
            dfine_freeze_spec_t s{};
            std::memcpy(&s, spec, std::min(spec->struct_size, sizeof(dfine_freeze_spec_t)));
            fs.batch = s.batch;
            fs.src_w = s.src_w;
            fs.src_h = s.src_h;
            fs.src_is_bgr = (s.src_is_bgr != 0);
        }
        det->h.obj.freeze(fs);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("dfine_detector_freeze: unknown error");
        return -1;
    }
}

int dfine_detector_full_graph_active(const dfine_detector_t* det) {
    return det && det->h.obj.full_pipeline_graph_active() ? 1 : 0;
}

int dfine_detector_last_timings(const dfine_detector_t* det, dfine_timings_t* out) {
    t_last_error.clear();
    if (!det || !out || out->struct_size < sizeof(size_t)) {
        set_error("dfine_detector_last_timings: NULL argument or struct_size not set");
        return -1;
    }
    const auto& t = det->h.obj.last_timings();
    dfine_timings_t full{};
    const std::size_t n = std::min(out->struct_size, sizeof(dfine_timings_t));
    full.struct_size = n;  // reports how many bytes were actually filled
    full.preprocess_ms = t.preprocess_ms;
    full.infer_ms = t.infer_ms;
    full.postprocess_ms = t.postprocess_ms;
    full.total_ms = t.total_ms;
    full.preprocess_cpu_ms = t.preprocess_cpu_ms;
    full.dispatch_ms = t.dispatch_ms;
    full.wait_ms = t.wait_ms;
    full.decode_host_ms = t.decode_host_ms;
    std::memcpy(out, &full, n);
    return 0;
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
