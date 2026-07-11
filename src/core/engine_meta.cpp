#include "dfine/core/engine_meta.hpp"

#include "internal/engine_meta_detail.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace dfine {

using nlohmann::json;

namespace {

std::vector<std::string> read_string_array(const json& j, const char* key,
                                           std::vector<std::string> fallback) {
    if (j.contains(key) && j[key].is_array() && !j[key].empty()) {
        std::vector<std::string> out;
        for (const auto& e : j[key]) {
            if (e.is_string()) out.push_back(e.get<std::string>());
        }
        if (!out.empty()) return out;
    }
    return fallback;
}

[[noreturn]] void bad_meta(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("dfine: invalid meta sidecar " + path.string() + ": " + what);
}

// Value validation. Absent fields stay allowed (old sidecars default), but a
// PRESENT value that would silently corrupt inference — a zero std dividing to
// inf, a typo'd resize mode falling back to stretch, an inverted batch range —
// must be a load-time error, not garbage detections an hour later.
void validate_meta(const EngineMeta& m, bool has_num_classes, const std::filesystem::path& path) {
    if (m.schema_version > kEngineMetaSchemaVersion) {
        bad_meta(path, "schema_version " + std::to_string(m.schema_version) +
                           " is newer than this runtime supports (" +
                           std::to_string(kEngineMetaSchemaVersion) + ") — upgrade dfine");
    }
    if (m.task != "detect") bad_meta(path, "task '" + m.task + "' (this runtime does detect)");
    if (m.input_h <= 0 || m.input_w <= 0) {
        bad_meta(path, "input_h/input_w must be positive (got " + std::to_string(m.input_h) + "x" +
                           std::to_string(m.input_w) + ")");
    }
    if (m.num_classes <= 0) bad_meta(path, "num_classes must be positive");
    if (m.num_queries <= 0) bad_meta(path, "num_queries must be positive");
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(m.mean[i])) bad_meta(path, "mean has a non-finite component");
        if (!std::isfinite(m.std[i]) || m.std[i] <= 0.0f) {
            bad_meta(path,
                     "std must be positive and finite (a zero collapses the "
                     "normalization to inf)");
        }
    }
    if (m.color_order != "RGB" && m.color_order != "BGR") {
        bad_meta(path, "color_order '" + m.color_order + "' (RGB or BGR)");
    }
    if (m.resize != "stretch" && m.resize != "letterbox") {
        bad_meta(path, "resize '" + m.resize + "' (stretch or letterbox)");
    }
    if (m.letterbox_anchor != "center" && m.letterbox_anchor != "topleft") {
        bad_meta(path, "letterbox_anchor '" + m.letterbox_anchor + "' (center or topleft)");
    }
    if (m.letterbox_pad < 0 || m.letterbox_pad > 255) {
        bad_meta(path, "letterbox_pad " + std::to_string(m.letterbox_pad) + " (0..255)");
    }
    if (m.min_batch < 1 || m.opt_batch < m.min_batch || m.max_batch < m.opt_batch) {
        bad_meta(path, "batch profile must satisfy 1 <= min <= opt <= max (got " +
                           std::to_string(m.min_batch) + "/" + std::to_string(m.opt_batch) + "/" +
                           std::to_string(m.max_batch) + ")");
    }
    if (has_num_classes && !m.class_names.empty() &&
        m.class_names.size() != static_cast<std::size_t>(m.num_classes)) {
        bad_meta(path, std::to_string(m.class_names.size()) + " class_names for " +
                           std::to_string(m.num_classes) + " classes");
    }
}

}  // namespace

detail::EngineMetaDocument detail::load_engine_meta(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("dfine: cannot open meta sidecar: " + path.string());
    }
    json j;
    in >> j;

    EngineMetaDocument doc;
    EngineMeta& m = doc.meta;
    m.schema_version = j.value("schema_version", kEngineMetaSchemaVersion);
    m.variant = j.value("variant", std::string{});
    m.task = j.value("task", std::string{"detect"});
    m.input_h = j.value("input_h", 640);
    m.input_w = j.value("input_w", 640);
    m.num_classes = j.value("num_classes", 80);
    m.num_queries = j.value("num_queries", 300);
    if (j.contains("input_h") != j.contains("input_w")) {
        bad_meta(path, "input_h and input_w must be specified together");
    }
    doc.has_input_hw = j.contains("input_h");
    doc.has_num_classes = j.contains("num_classes");
    doc.has_num_queries = j.contains("num_queries");
    m.has_input_hw = doc.has_input_hw;
    m.has_num_classes = doc.has_num_classes;
    m.has_num_queries = doc.has_num_queries;

    // A PRESENT mean/std that is not a 3-element numeric array must be an error,
    // not a silent fall-through to the defaults — the value the author wrote
    // would otherwise be ignored and every frame mis-normalized quietly.
    for (const char* key : {"mean", "std"}) {
        if (j.contains(key) && !(j[key].is_array() && j[key].size() == 3)) {
            bad_meta(path,
                     std::string(key) + " must be a 3-element array (got " + j[key].dump() + ")");
        }
    }
    if (j.contains("mean")) m.mean = j["mean"].get<std::array<float, 3>>();
    if (j.contains("std")) m.std = j["std"].get<std::array<float, 3>>();

    m.color_order = j.value("color_order", std::string{"RGB"});
    m.resize = j.value("resize", std::string{"stretch"});
    m.letterbox_anchor = j.value("letterbox_anchor", std::string{"center"});
    m.letterbox_pad = j.value("letterbox_pad", 114);
    m.letterbox_upscale = j.value("letterbox_upscale", true);
    m.precision = j.value("precision", std::string{"fp32"});
    m.dynamic_batch = j.value("dynamic_batch", false);
    m.min_batch = j.value("min_batch", 1);
    m.opt_batch = j.value("opt_batch", m.min_batch);
    m.max_batch = j.value("max_batch", m.opt_batch);
    doc.has_dynamic_batch = j.contains("dynamic_batch");
    doc.has_min_batch = j.contains("min_batch");
    doc.has_opt_batch = j.contains("opt_batch");
    doc.has_max_batch = j.contains("max_batch");
    m.cuda_graph_compat = j.value("cuda_graph_compat", false);

    for (const char* key : {"input_names", "output_names"}) {
        if (!j.contains(key)) continue;
        if (!j[key].is_array() || j[key].empty()) {
            bad_meta(path, std::string(key) + " must be a non-empty string array");
        }
        for (const auto& value : j[key]) {
            if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
                bad_meta(path, std::string(key) + " must contain non-empty strings");
            }
        }
    }
    m.input_names = read_string_array(j, "input_names", {"images"});
    m.output_names = read_string_array(j, "output_names", {"logits", "boxes"});
    doc.has_input_names = j.contains("input_names");
    doc.has_output_names = j.contains("output_names");
    if (j.contains("class_names")) {
        if (!j["class_names"].is_array()) {
            bad_meta(path, "class_names must be a string array");
        }
        for (const auto& value : j["class_names"]) {
            if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
                bad_meta(path, "class_names must contain non-empty strings");
            }
        }
    }
    m.class_names = read_string_array(j, "class_names", {});
    if (!doc.has_num_classes && !m.class_names.empty()) {
        m.num_classes = static_cast<int>(m.class_names.size());
    }
    validate_meta(m, doc.has_num_classes, path);
    return doc;
}

EngineMeta EngineMeta::from_json_file(const std::filesystem::path& path) {
    return detail::load_engine_meta(path).meta;
}

void EngineMeta::to_json_file(const std::filesystem::path& path) const {
    json j = {
        {"schema_version", schema_version},
        {"variant", variant},
        {"task", task},
        {"input_h", input_h},
        {"input_w", input_w},
        {"num_classes", num_classes},
        {"num_queries", num_queries},
        {"mean", mean},
        {"std", std},
        {"color_order", color_order},
        {"resize", resize},
        {"precision", precision},
        {"dynamic_batch", dynamic_batch},
        {"min_batch", min_batch},
        {"opt_batch", opt_batch},
        {"max_batch", max_batch},
        {"cuda_graph_compat", cuda_graph_compat},
        {"input_names", input_names},
        {"output_names", output_names},
    };
    if (!class_names.empty()) j["class_names"] = class_names;
    if (resize == "letterbox") {
        j["letterbox_anchor"] = letterbox_anchor;
        j["letterbox_pad"] = letterbox_pad;
        j["letterbox_upscale"] = letterbox_upscale;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("dfine: cannot write meta sidecar: " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace dfine
