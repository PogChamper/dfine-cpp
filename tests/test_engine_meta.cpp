// EngineMeta sidecar parsing: absent fields default (old sidecars keep
// loading), but a present value that would silently corrupt inference — zero
// std, a typo'd resize mode, an inverted batch range — is a load-time error.
// CPU-only. With file arguments, parses each and reports (release-gate helper:
//   dfine_test_engine_meta trt-files/onnx/*.json trt-files/engines/*.json).

#include "testing.hpp"

#include "dfine/core/engine_meta.hpp"
#include "internal/engine_meta_detail.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

using dfine::EngineMeta;

namespace {

namespace fs = std::filesystem;

fs::path write_json(const fs::path& dir, const std::string& body) {
    static int n = 0;
    const fs::path p = dir / ("meta_" + std::to_string(n++) + ".json");
    std::ofstream(p) << body;
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        int rc = 0;
        for (int i = 1; i < argc; ++i) {
            try {
                (void)EngineMeta::from_json_file(argv[i]);
                std::printf("OK   %s\n", argv[i]);
            } catch (const std::exception& e) {
                std::printf("FAIL %s: %s\n", argv[i], e.what());
                rc = 1;
            }
        }
        return rc;
    }

    // Unique per-process dir: parallel ctest runs must not share/collide.
    const fs::path dir =
        fs::temp_directory_path() / ("dfine_meta_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);

    // Defaults: an empty object is a valid (fully defaulted) sidecar.
    {
        const EngineMeta m = EngineMeta::from_json_file(write_json(dir, "{}"));
        DFINE_CHECK(m.input_h == 640 && m.num_classes == 80 && m.resize == "stretch");
    }
    // A realistic exporter sidecar parses and keeps its values.
    {
        const auto doc = dfine::detail::load_engine_meta(write_json(dir, R"({
            "variant": "s", "num_classes": 3, "num_queries": 300,
            "class_names": ["a", "b", "c"], "resize": "letterbox",
            "letterbox_anchor": "topleft", "letterbox_pad": 114,
            "dynamic_batch": true, "min_batch": 1, "opt_batch": 1,
            "max_batch": 8})"));
        const EngineMeta& m = doc.meta;
        DFINE_CHECK(m.num_classes == 3 && m.class_names.size() == 3);
        DFINE_CHECK(m.resize == "letterbox" && m.letterbox_anchor == "topleft");
        DFINE_CHECK(doc.has_dynamic_batch && doc.has_min_batch && doc.has_opt_batch &&
                    doc.has_max_batch);
    }

    // Malformed JSON and unopenable files are errors (not silent defaults).
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, "{ nope")), "");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(dir / "missing.json"), "cannot open");

    // Value validation: each of these silently corrupted inference before.
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"std": [0.0, 1.0, 1.0]})")), "std");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"std": [-1.0, 1.0, 1.0]})")), "std");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"resize": "letterbux"})")), "resize");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"schema_version": 99})")), "newer");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"min_batch": 4, "opt_batch": 1})")),
        "batch");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"letterbox_pad": 300})")), "pad");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(
                           write_json(dir, R"({"num_classes": 3, "class_names": ["a", "b"]})")),
                       "class_names");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"task": "segment"})")),
                       "task");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"num_queries": 0})")),
                       "num_queries");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"input_h": 640})")),
                       "together");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"color_order": "BRG"})")),
        "color_order");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"color_order": "BGR"})")),
        "model input must be RGB");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"artifact_kind": "plan"})")),
        "artifact_kind");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"trt_version": ""})")),
                       "trt_version");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"trt_version": null})")),
        "trt_version");
    // A PRESENT mean/std of the wrong shape is an error, not a silent default.
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"mean": [1.0, 2.0]})")),
                       "3-element");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"std": 255})")),
                       "3-element");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"input_names": []})")),
                       "input_names");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"output_names": ["logits", 3]})")),
        "output_names");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"class_names": ["person", 3]})")),
        "class_names");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"class_names": [""]})")),
        "class_names");

    // Labels may be supplied without duplicating num_classes. Their count is
    // reconciled with the engine after its output shape has been resolved.
    {
        const auto labels_only =
            dfine::detail::load_engine_meta(write_json(dir, R"({"class_names": ["a", "b", "c"]})"));
        DFINE_CHECK(labels_only.meta.num_classes == 3 && labels_only.meta.class_names.size() == 3);
        DFINE_CHECK(!labels_only.has_num_classes);
        const fs::path roundtrip = write_json(dir, "{}");
        labels_only.meta.to_json_file(roundtrip);
        DFINE_CHECK(EngineMeta::from_json_file(roundtrip).class_names.size() == 3);
        DFINE_CHECK(dfine::detail::load_engine_meta(roundtrip).batch_facts_describe_engine());
    }

    // Partial batch declarations stay coherent without asserting absent fields.
    {
        const auto min_only =
            dfine::detail::load_engine_meta(write_json(dir, R"({"min_batch": 2})"));
        DFINE_CHECK(min_only.meta.min_batch == 2 && min_only.meta.opt_batch == 2 &&
                    min_only.meta.max_batch == 2);
        DFINE_CHECK(min_only.has_min_batch && !min_only.has_opt_batch && !min_only.has_max_batch);

        const auto opt_only =
            dfine::detail::load_engine_meta(write_json(dir, R"({"opt_batch": 4})"));
        DFINE_CHECK(opt_only.meta.min_batch == 1 && opt_only.meta.opt_batch == 4 &&
                    opt_only.meta.max_batch == 4);
        DFINE_CHECK(!opt_only.has_min_batch && opt_only.has_opt_batch && !opt_only.has_max_batch);

        const auto max_only =
            dfine::detail::load_engine_meta(write_json(dir, R"({"max_batch": 8})"));
        DFINE_CHECK(max_only.meta.min_batch == 1 && max_only.meta.opt_batch == 1 &&
                    max_only.meta.max_batch == 8);
        DFINE_CHECK(!max_only.has_min_batch && !max_only.has_opt_batch && max_only.has_max_batch);
    }

    // Presence tracking: a facts-only sidecar (no contract fields) loads and
    // asserts nothing, so the detector's cross-check must not invent defaults.
    {
        const auto partial = dfine::detail::load_engine_meta(
            write_json(dir, R"({"trt_version": "10.13", "max_batch": 8})"));
        DFINE_CHECK(!partial.has_input_hw && !partial.has_num_classes && !partial.has_num_queries);
        DFINE_CHECK(!partial.meta.has_input_hw && !partial.meta.has_num_classes &&
                    !partial.meta.has_num_queries);
        const auto full = dfine::detail::load_engine_meta(
            write_json(dir, R"({"num_classes": 3, "class_names": ["a","b","c"]})"));
        DFINE_CHECK(full.has_num_classes && !full.has_num_queries);
        DFINE_CHECK(full.meta.has_num_classes && !full.meta.has_num_queries);
        const auto doc = dfine::detail::load_engine_meta(write_json(dir, "{}"));
        DFINE_CHECK(!doc.has_dynamic_batch && !doc.has_max_batch);
    }

    // Batch fields on an ONNX sidecar describe export bounds, not the profile
    // selected later by an engine builder. Explicit engine metadata and legacy
    // sidecars with TensorRT plus min/opt profile facts describe the engine.
    {
        using dfine::detail::MetaArtifactKind;
        const auto onnx = dfine::detail::load_engine_meta(
            write_json(dir, R"({"artifact_kind": "onnx", "trt_version": "10.13",
                               "min_batch": 1, "max_batch": 8})"));
        DFINE_CHECK(onnx.artifact_kind == MetaArtifactKind::kOnnx);
        DFINE_CHECK(!onnx.batch_facts_describe_engine());

        const auto engine = dfine::detail::load_engine_meta(
            write_json(dir, R"({"artifact_kind": "engine", "min_batch": 1, "max_batch": 16})"));
        DFINE_CHECK(engine.artifact_kind == MetaArtifactKind::kEngine);
        DFINE_CHECK(engine.batch_facts_describe_engine());

        const auto legacy = dfine::detail::load_engine_meta(
            write_json(dir, R"({"trt_version": "10.13", "min_batch": 1, "max_batch": 4})"));
        DFINE_CHECK(legacy.artifact_kind == MetaArtifactKind::kUnknown);
        DFINE_CHECK(legacy.batch_facts_describe_engine());

        const auto trt_ambiguous = dfine::detail::load_engine_meta(
            write_json(dir, R"({"trt_version": "10.13", "max_batch": 8})"));
        DFINE_CHECK(!trt_ambiguous.batch_facts_describe_engine());

        const auto ambiguous =
            dfine::detail::load_engine_meta(write_json(dir, R"({"max_batch": 8})"));
        DFINE_CHECK(ambiguous.artifact_kind == MetaArtifactKind::kUnknown);
        DFINE_CHECK(!ambiguous.batch_facts_describe_engine());
    }

    fs::remove_all(dir);
    return dfine::testing::finish("test_engine_meta");
}
