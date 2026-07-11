// TrtSession shape-transition contract: a transition either commits or leaves
// the session at the last committed state (old buffers live, cache untouched);
// a failed transition can never lead to enqueueV3 running against mismatched
// shapes/addresses. Regression for the frozen-grow poisoning and the
// exception-unsafe grow path (v0.3.1).
//
// Needs a GPU and DFINE_TEST_ENGINE pointing at a dynamic-batch D-FINE engine
// (e.g. trt-files/engines/dfine_m_fp16_st.engine); exits 77 (skip) otherwise.

#include "testing.hpp"

#include "internal/failpoint.hpp"
#include "internal/trt_session.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using dfine::TrtSession;
using dfine::testing::arm_failpoint;
using ShapeState = dfine::TrtSession::ShapeState;

namespace {

struct Io {
    std::string input;
    std::string logits;
    std::string boxes;
    int h{640};
    int w{640};
};

Io resolve_io(const TrtSession& s) {
    Io io;
    io.input = s.bindings()[s.input_indices().front()].name;
    for (int oi : s.output_indices()) {
        const auto& b = s.bindings()[oi];
        (b.shape.nbDims > 0 && b.shape.d[b.shape.nbDims - 1] == 4 ? io.boxes : io.logits) = b.name;
    }
    const auto& in = s.bindings()[s.input_indices().front()];
    if (in.shape.nbDims == 4) {
        if (in.shape.d[2] > 0) io.h = static_cast<int>(in.shape.d[2]);
        if (in.shape.d[3] > 0) io.w = static_cast<int>(in.shape.d[3]);
    }
    return io;
}

struct Outputs {
    std::vector<float> logits;
    std::vector<float> boxes;
    bool operator==(const Outputs& o) const {
        return logits.size() == o.logits.size() && boxes.size() == o.boxes.size() &&
               std::memcmp(logits.data(), o.logits.data(), logits.size() * sizeof(float)) == 0 &&
               std::memcmp(boxes.data(), o.boxes.data(), boxes.size() * sizeof(float)) == 0;
    }
};

// Deterministic input, full sync infer, both outputs as f32.
Outputs run(TrtSession& s, const Io& io, int batch) {
    s.set_input_shape(io.input, nvinfer1::Dims4{batch, 3, io.h, io.w});
    const auto* in = s.find(io.input);
    std::vector<float> pixels(static_cast<std::size_t>(in->element_count));
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<float>(i % 251) / 255.0f;
    }
    s.set_input(io.input, pixels.data(), in->bytes);
    s.infer();
    Outputs out;
    out.logits.resize(static_cast<std::size_t>(s.find(io.logits)->element_count));
    out.boxes.resize(static_cast<std::size_t>(s.find(io.boxes)->element_count));
    s.get_output_f32(io.logits, out.logits.data(), out.logits.size());
    s.get_output_f32(io.boxes, out.boxes.data(), out.boxes.size());
    return out;
}

std::unique_ptr<TrtSession> make_session(const char* engine) {
    return std::make_unique<TrtSession>(engine);
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

    // Every committed device-buffer replacement advances one session-wide
    // generation. CUDA Graph caches use it to cover all bindings, including
    // additional outputs whose addresses are not otherwise inspected.
    {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        (void)run(*s, io, 1);
        const std::uint64_t b1_generation = s->buffer_generation();
        (void)run(*s, io, 1);
        DFINE_CHECK(s->buffer_generation() == b1_generation);
        (void)run(*s, io, 4);
        DFINE_CHECK(s->buffer_generation() > b1_generation);
        const std::uint64_t b4_generation = s->buffer_generation();
        (void)run(*s, io, 1);
        DFINE_CHECK(s->buffer_generation() == b4_generation);
    }

    // --- frozen violation is rejected with zero side effects -------------------
    {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        const Outputs base = run(*s, io, 1);
        s->freeze();
        DFINE_EXPECT_THROW(s->set_input_shape(io.input, nvinfer1::Dims4{2, 3, io.h, io.w}),
                           "frozen");
        DFINE_CHECK(s->shape_state() == ShapeState::kClean);
        // The regression this file exists for: the next batch-1 call must run
        // against the batch-1 context, not a context left at batch 2.
        DFINE_CHECK(run(*s, io, 1) == base);
    }

    // --- recoverable failures: dev alloc, host alloc, rebind -------------------
    for (const char* point :
         {"trt_session.dev_alloc", "trt_session.host_alloc", "trt_session.rebind"}) {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        const Outputs base = run(*s, io, 1);

        arm_failpoint(point, 1);
        DFINE_EXPECT_THROW(s->set_input_shape(io.input, nvinfer1::Dims4{4, 3, io.h, io.w}), "");
        arm_failpoint(point, 0);  // defensive: a regressed fault path must not leak the arm
        DFINE_CHECK(s->shape_state() == ShapeState::kDirty);
        // Until a transition commits, infer paths refuse to run...
        DFINE_EXPECT_THROW(s->infer(), "stale");
        DFINE_EXPECT_THROW(s->device_buffer(io.input), "stale");
        // ...and a successful transition (here: back to the committed batch-1
        // shape, which needs no growth) restores service bit-exactly.
        DFINE_CHECK(run(*s, io, 1) == base);
        DFINE_CHECK(s->shape_state() == ShapeState::kClean);
        // The originally requested growth also works once the fault is gone.
        const Outputs b4 = run(*s, io, 4);
        DFINE_CHECK(b4.logits.size() == 4 * base.logits.size());
        DFINE_CHECK(run(*s, io, 1) == base);
    }

    // --- unrecoverable failure: restore fails after a failed rebind ------------
    {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        (void)run(*s, io, 1);

        // Batch 1 -> 8 grows input+logits+boxes: the 1st rebind succeeds, the
        // 2nd fails, and restoring the 1st fails too -> the context may point
        // at memory the unwind frees -> the session must refuse to ever run.
        arm_failpoint("trt_session.rebind", 2);
        arm_failpoint("trt_session.rebind_restore", 1);
        DFINE_EXPECT_THROW(s->set_input_shape(io.input, nvinfer1::Dims4{8, 3, io.h, io.w}),
                           "unusable");
        DFINE_CHECK(s->shape_state() == ShapeState::kPoisoned);
        DFINE_EXPECT_THROW(s->infer(), "recreate");
        DFINE_EXPECT_THROW(s->set_input_shape(io.input, nvinfer1::Dims4{1, 3, io.h, io.w}),
                           "recreate");
        arm_failpoint("trt_session.rebind", 0);  // disarm leftovers
        arm_failpoint("trt_session.rebind_restore", 0);
    }

    // --- asynchronous execution failure poisons the session -------------------
    {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        (void)run(*s, io, 1);

        const auto* in = s->find(io.input);
        std::vector<float> pixels(static_cast<std::size_t>(in->element_count), 0.5f);
        s->set_input(io.input, pixels.data(), in->bytes);
        arm_failpoint("trt_session.sync_poison", 1);
        DFINE_EXPECT_THROW(s->infer(), "unusable");
        DFINE_CHECK(s->shape_state() == ShapeState::kPoisoned);
        DFINE_EXPECT_THROW(s->infer(), "recreate");
        DFINE_EXPECT_THROW((void)s->context(), "recreate");
    }

    // --- a rejected enqueue poisons the execution context ---------------------
    {
        auto s = make_session(engine);
        const Io io = resolve_io(*s);
        s->set_input_shape(io.input, nvinfer1::Dims4{1, 3, io.h, io.w});

        const auto* in = s->find(io.input);
        std::vector<float> pixels(static_cast<std::size_t>(in->element_count), 0.5f);
        s->set_input(io.input, pixels.data(), in->bytes);
        arm_failpoint("trt_session.enqueue_poison", 1);
        DFINE_EXPECT_THROW(s->infer(), "unusable");
        DFINE_CHECK(s->shape_state() == ShapeState::kPoisoned);
        DFINE_EXPECT_THROW(s->infer(), "recreate");
        DFINE_EXPECT_THROW((void)s->context(), "recreate");
    }

    return dfine::testing::finish("test_shape_transitions");
}
