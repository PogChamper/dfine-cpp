#include "dfine/core/engine_meta.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

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

}  // namespace

EngineMeta EngineMeta::from_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("dfine: cannot open meta sidecar: " + path.string());
    }
    json j;
    in >> j;

    EngineMeta m;
    m.schema_version = j.value("schema_version", kEngineMetaSchemaVersion);
    m.variant     = j.value("variant", std::string{});
    m.task        = j.value("task", std::string{"detect"});
    m.input_h     = j.value("input_h", 640);
    m.input_w     = j.value("input_w", 640);
    m.num_classes = j.value("num_classes", 80);
    m.num_queries = j.value("num_queries", 300);

    if (j.contains("mean") && j["mean"].is_array() && j["mean"].size() == 3) {
        m.mean = j["mean"].get<std::array<float, 3>>();
    }
    if (j.contains("std") && j["std"].is_array() && j["std"].size() == 3) {
        m.std = j["std"].get<std::array<float, 3>>();
    }

    m.color_order = j.value("color_order", std::string{"RGB"});
    m.resize      = j.value("resize", std::string{"stretch"});
    m.precision   = j.value("precision", std::string{"fp32"});
    m.dynamic_batch = j.value("dynamic_batch", false);
    m.min_batch     = j.value("min_batch", 1);
    m.opt_batch     = j.value("opt_batch", 1);
    m.max_batch     = j.value("max_batch", 1);
    m.cuda_graph_compat = j.value("cuda_graph_compat", false);

    m.input_names  = read_string_array(j, "input_names", {"images"});
    m.output_names = read_string_array(j, "output_names", {"logits", "boxes"});
    return m;
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
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("dfine: cannot write meta sidecar: " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace dfine
