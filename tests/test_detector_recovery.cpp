// DFineDetector error-recovery contract through the public API: a rejected or
// failed call leaves the detector fully serviceable at its previous
// configuration, with results bit-identical to before the error. Covers the
// plain, gpu_decode, and full-pipeline-graph paths.
//
// Needs a GPU and DFINE_TEST_ENGINE pointing at a dynamic-batch D-FINE engine.
// The full-graph section additionally uses DFINE_TEST_ENGINE_G0 (an engine
// built with --max-aux-streams 0) and is skipped when unset.
// Exits 77 (skip) when prerequisites are missing.

#include "testing.hpp"

#include "dfine/tasks/detector.hpp"
#include "internal/failpoint.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

using dfine::Detections;
using dfine::DetectorOptions;
using dfine::DFineDetector;
using dfine::FreezeSpec;
using dfine::ImageU8;
using dfine::testing::arm_failpoint;

// Low threshold: a synthetic gradient frame has no confident objects, but the
// full top-K candidate set gives the comparisons a rich surface.
constexpr float kThr = 0.001f;

namespace {

// Deterministic RGB gradient frame; optional row padding exercises stride
// handling. The backing buffer outlives every view (function-local static maps
// keyed by geometry would be overkill for a test — callers keep it alive).
struct Frame {
    std::vector<std::uint8_t> bytes;
    ImageU8 view;
};

Frame make_frame(int w, int h, int row_pad = 0) {
    Frame f;
    const int stride = w * 3 + row_pad;
    f.bytes.resize(static_cast<std::size_t>(h) * stride);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::uint8_t* p = f.bytes.data() + static_cast<std::size_t>(y) * stride + x * 3;
            p[0] = static_cast<std::uint8_t>((x * 7 + y * 3) % 256);
            p[1] = static_cast<std::uint8_t>((x * 5 + y * 11) % 256);
            p[2] = static_cast<std::uint8_t>((x * 13 + y * 17) % 256);
        }
    }
    f.view.data = f.bytes.data();
    f.view.width = w;
    f.view.height = h;
    f.view.channels = 3;
    f.view.stride = row_pad > 0 ? stride : 0;
    return f;
}

// Byte-exact on purpose — and therefore calibrated for bitwise-deterministic
// engine flavors (fp32-faithful, fp16_st with its FP32 decoder), which is what
// DFINE_TEST_ENGINE should point at. A fully-FP16 slim engine shows ULP-level
// batch-position variance (sub-pixel box jitter, ~1e-3 score jitter) that
// reorders the low-score tail; slim correctness is gated by mAP parity, not here.
bool equal(const Detections& a, const Detections& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].class_id != b[i].class_id ||
            std::memcmp(&a[i].score, &b[i].score, sizeof(float)) != 0 ||
            std::memcmp(&a[i].box, &b[i].box, sizeof(dfine::Box)) != 0) {
            return false;
        }
    }
    return true;
}

// Frozen batch-1 detector: an over-batch call is rejected, and the next
// batch-1 call reproduces the pre-error result exactly. This is the public
// face of the shape-transition regression.
void frozen_recovery(const char* engine, const DetectorOptions& opts, const char* label) {
    DFineDetector det(engine, opts);
    Frame f = make_frame(512, 384);
    const Detections base = det.detect(f.view, kThr);
    DFINE_CHECK(!base.empty());  // a gradient frame yields at least low-score boxes

    det.freeze(1);
    std::vector<ImageU8> two{f.view, f.view};
    DFINE_EXPECT_THROW((void)det.detect_batch(two, kThr), "frozen");
    const Detections again = det.detect(f.view, kThr);
    if (!equal(base, again)) {
        std::fprintf(stderr, "  [%s] post-error result diverged from baseline\n", label);
        DFINE_CHECK(false);
    }
}

}  // namespace

int main() {
    const char* engine = std::getenv("DFINE_TEST_ENGINE");
    if (!engine || !*engine) {
        std::fprintf(stderr, "skip: set DFINE_TEST_ENGINE to a dynamic-batch engine\n");
        return dfine::testing::kSkipExitCode;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "skip: no CUDA device\n");
        return dfine::testing::kSkipExitCode;
    }

    // --- frozen recovery on the plain and gpu_decode paths ---------------------
    frozen_recovery(engine, DetectorOptions{}, "plain");
    {
        DetectorOptions o;
        o.gpu_decode = true;
        frozen_recovery(engine, o, "gpu_decode");
    }

    // --- padded rows decode identically to packed rows -------------------------
    {
        DFineDetector det(engine);
        Frame packed = make_frame(512, 384);
        Frame padded = make_frame(512, 384, /*row_pad=*/13);
        DFINE_CHECK(equal(det.detect(packed.view, kThr), det.detect(padded.view, kThr)));
    }

    // --- per-call threshold validation is side-effect free ---------------------
    {
        DFineDetector det(engine);
        Frame f = make_frame(512, 384);
        const Detections base = det.detect(f.view);
        std::vector<ImageU8> one{f.view};

        DFINE_EXPECT_THROW((void)det.detect(f.view, 1.1F), "0..1");
        DFINE_EXPECT_THROW((void)det.detect_batch(one, std::numeric_limits<float>::quiet_NaN()),
                           "finite");
        DFINE_CHECK(equal(det.detect(f.view, -1.0F), base));
        DFINE_CHECK(equal(det.detect_batch(one, -1.0F).front(), base));
    }

    // --- FreezeSpec validation is side-effect free ------------------------------
    {
        DFineDetector det(engine);
        Frame f = make_frame(512, 384);
        const Detections base = det.detect(f.view, kThr);

        DFINE_EXPECT_THROW(det.freeze(FreezeSpec{-1, 0, 0, false}), "non-negative");
        DFINE_EXPECT_THROW(det.freeze(FreezeSpec{1, -1, 384, false}), "non-negative");
        DFINE_EXPECT_THROW(det.freeze(FreezeSpec{1, 512, 0, false}), "both zero");
        DFINE_EXPECT_THROW(det.freeze(FreezeSpec{1, 0, 384, false}), "both zero");
        DFINE_EXPECT_THROW(det.freeze(FreezeSpec{det.max_batch() + 1, 0, 0, false}),
                           "outside engine profile");
        DFINE_CHECK(equal(det.detect(f.view, kThr), base));
    }

    // --- explicit source bounds are enforced per dimension ----------------------
    {
        DFineDetector det(engine);
        Frame exact = make_frame(512, 384);
        const Detections base = det.detect(exact.view, kThr);
        det.freeze(FreezeSpec{1, exact.view.width, exact.view.height, false});

        // Each rejected frame has fewer pixels than the frozen frame. A byte-capacity
        // check alone would accept it; the width/height contract must reject it.
        Frame too_wide = make_frame(513, 383);
        Frame too_tall = make_frame(510, 385);
        DFINE_EXPECT_THROW((void)det.detect(too_wide.view, kThr), "exceeds frozen bound");
        DFINE_EXPECT_THROW((void)det.detect(too_tall.view, kThr), "exceeds frozen bound");
        DFINE_CHECK(equal(det.detect(exact.view, kThr), base));
    }

    // An unbounded legacy freeze may be tightened at the already warmed source
    // size; the stricter per-dimension contract then takes effect.
    {
        DFineDetector det(engine);
        det.freeze(1);
        det.freeze(FreezeSpec{1, det.input_w(), det.input_h(), false});
        Frame too_wide = make_frame(det.input_w() + 1, det.input_h() - 1);
        DFINE_EXPECT_THROW((void)det.detect(too_wide.view, kThr), "exceeds frozen bound");
    }

    // --- non-frozen: B1 -> B8 -> B1 round trip is loss-free --------------------
    {
        DFineDetector det(engine);
        Frame f = make_frame(512, 384);
        const Detections base = det.detect(f.view, kThr);
        std::vector<ImageU8> eight(8, f.view);
        const auto batch = det.detect_batch(eight, kThr);
        DFINE_CHECK(batch.size() == 8);
        DFINE_CHECK(equal(batch.front(), batch.back()));  // same frame, same result
        DFINE_CHECK(equal(det.detect(f.view, kThr), base));
    }

    // --- allocation failure mid-grow is recoverable through the public API -----
    {
        DFineDetector det(engine);
        Frame f = make_frame(512, 384);
        const Detections base = det.detect(f.view, kThr);

        arm_failpoint("trt_session.dev_alloc", 1);
        std::vector<ImageU8> four(4, f.view);
        DFINE_EXPECT_THROW((void)det.detect_batch(four, kThr), "failpoint");
        DFINE_CHECK(equal(det.detect(f.view, kThr), base));     // batch-1 service restored
        DFINE_CHECK(det.detect_batch(four, kThr).size() == 4);  // and the grow now works
        DFINE_CHECK(equal(det.detect(f.view, kThr), base));
    }

    // --- a stale sidecar contradicting the engine must refuse to load ----------
    {
        namespace fs = std::filesystem;
        const fs::path dir =
            fs::temp_directory_path() / ("dfine_rec_test_" + std::to_string(::getpid()));
        fs::create_directories(dir);
        const fs::path eng = dir / "stale.engine";
        fs::copy_file(engine, eng, fs::copy_options::overwrite_existing);
        const int engine_max = DFineDetector(engine).max_batch();
        DFINE_CHECK(engine_max > 1);

        // Profile facts come from the engine even without a sidecar. An explicit
        // sidecar path is never replaced by an auto-discovered neighbor.
        DFINE_CHECK(DFineDetector(eng).max_batch() == engine_max);
        DFINE_EXPECT_THROW((void)DFineDetector(eng, dir / "missing.json"), "explicit");

        std::ofstream(dir / "stale.json")
            << R"({"num_classes": 81, "class_names": []})";  // engine has 80
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "contradicts");

        // A labels-only sidecar is valid, but its count must agree with the
        // resolved engine output rather than the parser's default class count.
        std::ofstream(dir / "stale.json") << R"({"class_names": ["a", "b", "c"]})";
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "class_names");

        std::ofstream(dir / "stale.json")
            << "{\"artifact_kind\": \"engine\", \"max_batch\": " << engine_max + 1 << "}";
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "max_batch");

        // A same-stem ONNX sidecar carries export bounds. A different profile
        // may be selected by trtexec or another builder, so those bounds are not
        // engine assertions.
        std::ofstream(dir / "stale.json")
            << "{\"artifact_kind\": \"onnx\", \"max_batch\": " << engine_max + 1 << "}";
        DFINE_CHECK(DFineDetector(eng).max_batch() == engine_max);

        // Legacy ONNX sidecars have no artifact_kind or TensorRT build facts.
        std::ofstream(dir / "stale.json") << "{\"max_batch\": " << engine_max + 1 << "}";
        DFINE_CHECK(DFineDetector(eng).max_batch() == engine_max);

        // Engine sidecars emitted before artifact_kind remain strict when their
        // TensorRT build facts identify them unambiguously.
        std::ofstream(dir / "stale.json")
            << "{\"trt_version\": \"10.13\", \"min_batch\": " << engine_max + 1
            << ", \"max_batch\": " << engine_max + 1 << "}";
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "min_batch");

        std::ofstream(dir / "stale.json")
            << R"({"input_names": ["missing"], "max_batch": )" << engine_max << "}";
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "missing input");

        std::ofstream(dir / "stale.json")
            << R"({"output_names": ["missing", "boxes"], "max_batch": )" << engine_max << "}";
        DFINE_EXPECT_THROW((void)DFineDetector(eng), "missing logits");

        // A partial sidecar asserts only the fields it contains.
        std::ofstream(dir / "stale.json")
            << "{\"artifact_kind\": \"engine\", \"max_batch\": " << engine_max << "}";
        {
            DFineDetector ok(eng);
            DFINE_CHECK(ok.max_batch() == engine_max);
        }
        fs::remove_all(dir);
    }

    // --- full-pipeline graph: rejected call neither replays nor corrupts -------
    if (const char* g0 = std::getenv("DFINE_TEST_ENGINE_G0"); g0 && *g0) {
        Frame f = make_frame(512, 384);
        {
            DetectorOptions o;
            o.use_cuda_graph = true;
            DFineDetector det(g0, o);
            const Detections first = det.detect(f.view, kThr);
            DFINE_CHECK(det.cuda_graph_replays() == 0);
            DFINE_CHECK(equal(first, det.detect(f.view, kThr)));
            DFINE_CHECK(det.cuda_graph_replays() == 1);
        }
        {
            DetectorOptions o;
            o.full_pipeline_graph = true;
            o.own_device_memory = true;
            DFineDetector det(g0, o);
            arm_failpoint("detector.full_graph_sequence", 1);
            det.freeze(FreezeSpec{1, f.view.width, f.view.height, false});
            arm_failpoint("detector.full_graph_sequence", 0);
            DFINE_CHECK(!det.full_pipeline_graph_active());
            const Detections first = det.detect(f.view, kThr);
            DFINE_CHECK(equal(first, det.detect(f.view, kThr)));
        }

        // A fatal replay synchronization error poisons the detector. No later
        // call may enqueue the captured graph again.
        {
            DetectorOptions o;
            o.full_pipeline_graph = true;
            o.own_device_memory = true;
            DFineDetector det(g0, o);
            det.freeze(FreezeSpec{1, f.view.width, f.view.height, false});
            DFINE_CHECK(det.full_pipeline_graph_active());
            if (det.full_pipeline_graph_active()) {
                const std::uint64_t replays = det.full_graph_replays();
                arm_failpoint("trt_session.sync_poison", 1);
                DFINE_EXPECT_THROW((void)det.detect(f.view, kThr), "unusable");
                DFINE_CHECK(det.full_graph_replays() == replays);
                DFINE_EXPECT_THROW((void)det.detect(f.view, kThr), "recreate");
                DFINE_CHECK(det.full_graph_replays() == replays);
            }
        }

        DetectorOptions o;
        o.full_pipeline_graph = true;
        o.own_device_memory = true;
        DFineDetector det(g0, o);
        det.freeze(FreezeSpec{1, f.view.width, f.view.height, false});
        if (det.full_pipeline_graph_active()) {
            const Detections base = det.detect(f.view, kThr);
            const std::uint64_t replays = det.full_graph_replays();
            DFINE_CHECK(replays >= 1);

            std::vector<ImageU8> two{f.view, f.view};
            DFINE_EXPECT_THROW((void)det.detect_batch(two, kThr), "frozen");
            DFINE_CHECK(det.full_graph_replays() == replays);  // no replay ran

            // Self-heal: one split-path call at the frozen batch (identical
            // result — both paths use the GPU decode), then replay resumes.
            const Detections heal = det.detect(f.view, kThr);
            DFINE_CHECK(det.full_graph_replays() == replays);
            DFINE_CHECK(equal(heal, base));
            const Detections replayed = det.detect(f.view, kThr);
            DFINE_CHECK(det.full_graph_replays() == replays + 1);
            DFINE_CHECK(equal(replayed, base));
        } else {
            std::fprintf(stderr,
                         "full graph inactive on DFINE_TEST_ENGINE_G0 — "
                         "configured release-gate coverage was not exercised\n");
            DFINE_CHECK(det.full_pipeline_graph_active());
        }
    } else if (std::getenv("DFINE_TEST_REQUIRE_FULL_GRAPH")) {
        // Release-gate mode: the full-graph section is mandatory, a silently
        // skipped section must fail the run, not pass it.
        std::fprintf(stderr,
                     "FAIL: DFINE_TEST_REQUIRE_FULL_GRAPH set but "
                     "DFINE_TEST_ENGINE_G0 is unset\n");
        DFINE_CHECK(false);
    } else {
        std::fprintf(stderr, "note: DFINE_TEST_ENGINE_G0 unset — full-graph section skipped\n");
    }

    return dfine::testing::finish("test_detector_recovery");
}
