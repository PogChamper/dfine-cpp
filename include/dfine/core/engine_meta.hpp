#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace dfine {

// Bump when the JSON schema gains fields that would break older readers.
inline constexpr int kEngineMetaSchemaVersion = 1;

// Runtime view of the `.json` sidecar written next to a D-FINE engine/ONNX by
// the Python export/build scripts (trt-files/scripts/). Only the fields the C++
// runtime needs are modelled; the reader is tolerant of missing keys and of the
// extra descriptive fields (reg_max, feat_strides, …) the scripts also emit.
//
// D-FINE defaults differ from ImageNet-style detectors on purpose:
//   mean = {0,0,0}, std = {1,1,1}  → preprocessing is `/255` only. ImageNet
//   normalization is incompatible with the published weights (docs/RUNTIME.md).
struct EngineMeta {
    int schema_version{kEngineMetaSchemaVersion};
    std::string variant;  // "n"/"s"/"m"/"l"/"x" (informational)
    std::string task{"detect"};
    int input_h{640};
    int input_w{640};
    int num_classes{80};
    int num_queries{300};

    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> std{1.0f, 1.0f, 1.0f};
    std::string color_order{"RGB"};
    std::string resize{"stretch"};  // "stretch" (training convention) or "letterbox"
    // Letterbox geometry, consulted only when resize == "letterbox":
    std::string letterbox_anchor{"center"};  // "center" | "topleft"
    int letterbox_pad{114};                  // padding value 0..255
    bool letterbox_upscale{true};            // false = paste 1:1 when the frame fits

    std::string precision{"fp32"};  // "fp32" / "fp16" / "int8"
    bool dynamic_batch{false};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph_compat{false};  // advisory: FP32 outputs and zero auxiliary streams

    // Engine IO tensor names (raw D-FINE contract).
    std::vector<std::string> input_names{"images"};
    std::vector<std::string> output_names{"logits", "boxes"};

    // Optional display names for class ids 0..num_classes-1 (custom label sets).
    // Empty = unknown; consumers fall back to COCO-80 when num_classes == 80.
    std::vector<std::string> class_names;

    // Distinguish fields declared by the sidecar from reader defaults. These
    // members are part of the public v0.3.3 layout and must remain in place.
    bool has_input_hw{false};
    bool has_num_classes{false};
    bool has_num_queries{false};

    [[nodiscard]] static EngineMeta from_json_file(const std::filesystem::path& path);
    // Writes an engine sidecar. ONNX metadata is produced by the export pipeline.
    void to_json_file(const std::filesystem::path& path) const;
};

}  // namespace dfine
