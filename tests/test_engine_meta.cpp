// EngineMeta sidecar parsing: absent fields default (old sidecars keep
// loading), but a present value that would silently corrupt inference — zero
// std, a typo'd resize mode, an inverted batch range — is a load-time error.
// CPU-only. With file arguments, parses each and reports (release-gate helper:
//   dfine_test_engine_meta trt-files/onnx/*.json trt-files/engines/*.json).

#include "testing.hpp"

#include "dfine/core/engine_meta.hpp"

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
        const EngineMeta m = EngineMeta::from_json_file(write_json(dir, R"({
            "variant": "s", "num_classes": 3, "num_queries": 300,
            "class_names": ["a", "b", "c"], "resize": "letterbox",
            "letterbox_anchor": "topleft", "letterbox_pad": 114,
            "dynamic_batch": true, "min_batch": 1, "opt_batch": 1, "max_batch": 8})"));
        DFINE_CHECK(m.num_classes == 3 && m.class_names.size() == 3);
        DFINE_CHECK(m.resize == "letterbox" && m.letterbox_anchor == "topleft");
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
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(
            write_json(dir, R"({"num_classes": 3, "class_names": ["a", "b"]})")),
        "class_names");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"task": "segment"})")), "task");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"num_queries": 0})")), "num_queries");
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"color_order": "BRG"})")),
        "color_order");
    // A PRESENT mean/std of the wrong shape is an error, not a silent default.
    DFINE_EXPECT_THROW(
        (void)EngineMeta::from_json_file(write_json(dir, R"({"mean": [1.0, 2.0]})")),
        "3-element");
    DFINE_EXPECT_THROW((void)EngineMeta::from_json_file(write_json(dir, R"({"std": 255})")),
                       "3-element");

    // Presence tracking: a facts-only sidecar (no contract fields) loads and
    // asserts nothing, so the detector's cross-check must not invent defaults.
    {
        const EngineMeta m = EngineMeta::from_json_file(
            write_json(dir, R"({"trt_version": "10.13", "max_batch": 8})"));
        DFINE_CHECK(!m.has_input_hw && !m.has_num_classes && !m.has_num_queries);
        const EngineMeta full = EngineMeta::from_json_file(
            write_json(dir, R"({"num_classes": 3, "class_names": ["a","b","c"]})"));
        DFINE_CHECK(full.has_num_classes && !full.has_num_queries);
    }

    fs::remove_all(dir);
    return dfine::testing::finish("test_engine_meta");
}
