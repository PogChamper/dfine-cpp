#pragma once

#include "dfine/core/engine_meta.hpp"

#include <filesystem>

namespace dfine::detail {

struct EngineMetaDocument {
    EngineMeta meta;
    bool has_input_hw{false};
    bool has_num_classes{false};
    bool has_num_queries{false};
    bool has_dynamic_batch{false};
    bool has_min_batch{false};
    bool has_opt_batch{false};
    bool has_max_batch{false};
    bool has_input_names{false};
    bool has_output_names{false};
};

[[nodiscard]] EngineMetaDocument load_engine_meta(const std::filesystem::path& path);

}  // namespace dfine::detail
