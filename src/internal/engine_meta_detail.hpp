#pragma once

#include "dfine/core/engine_meta.hpp"

#include <filesystem>

namespace dfine::detail {

enum class MetaArtifactKind { kUnknown, kOnnx, kEngine };

struct EngineMetaDocument {
    EngineMeta meta;
    MetaArtifactKind artifact_kind{MetaArtifactKind::kUnknown};
    bool has_trt_version{false};
    bool has_input_hw{false};
    bool has_num_classes{false};
    bool has_num_queries{false};
    bool has_dynamic_batch{false};
    bool has_min_batch{false};
    bool has_opt_batch{false};
    bool has_max_batch{false};
    bool has_input_names{false};
    bool has_output_names{false};

    [[nodiscard]] bool batch_facts_describe_engine() const noexcept {
        return artifact_kind == MetaArtifactKind::kEngine ||
               (artifact_kind == MetaArtifactKind::kUnknown && has_trt_version &&
                (has_min_batch || has_opt_batch));
    }
};

[[nodiscard]] EngineMetaDocument load_engine_meta(const std::filesystem::path& path);

}  // namespace dfine::detail
