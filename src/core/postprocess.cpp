#include "dfine/core/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace dfine {

namespace {

inline float sigmoid(float x) noexcept {
    // Numerically stable across the full logit range.
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

// A flattened (query × class) candidate. Sorting by raw logit is order-equivalent
// to sorting by sigmoid(logit) since sigmoid is monotonic — so top-k selection
// matches the Python reference, which sorts by probability.
struct Candidate {
    float logit;
    int   query;
    int   cls;
};

}  // namespace

Detections decode_detections(const float* logits, const float* boxes, int img_w, int img_h,
                             const PostprocessParams& p) {
    Detections out;
    if (!logits || !boxes || p.num_queries <= 0 || p.num_classes <= 0) {
        return out;
    }

    const int N = p.num_queries;
    const int C = p.num_classes;
    const int total = N * C;
    const int K = std::min(p.topk > 0 ? p.topk : N, total);

    // Reuse the scratch across calls (per-thread): decode runs once per image, so
    // this avoids a ~N*C*sizeof(Candidate) malloc/free on every frame. Candidate is
    // trivially destructible, so clear() keeps capacity and reserve() then no-ops.
    thread_local std::vector<Candidate> cand;
    cand.clear();
    cand.reserve(static_cast<std::size_t>(total));
    for (int q = 0; q < N; ++q) {
        const float* lq = logits + static_cast<std::size_t>(q) * C;
        for (int c = 0; c < C; ++c) {
            // A NaN logit breaks partial_sort's strict weak ordering (UB) and
            // compares as neither above nor below the threshold. Map it to -inf:
            // it sorts past the top-k and sigmoid(-inf) = 0 drops it — mirroring
            // the GPU decode's k_pack (decode_gpu.cu).
            const float lg = lq[c];
            cand.push_back({std::isnan(lg) ? -std::numeric_limits<float>::infinity() : lg, q, c});
        }
    }

    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                      [](const Candidate& a, const Candidate& b) noexcept {
                          return a.logit > b.logit;
                      });

    const float W = static_cast<float>(img_w);
    const float H = static_cast<float>(img_h);
    out.reserve(static_cast<std::size_t>(K));

    for (int i = 0; i < K; ++i) {
        const Candidate& cn = cand[i];
        const float score = sigmoid(cn.logit);
        if (score < p.threshold) continue;

        const float* db = boxes + static_cast<std::size_t>(cn.query) * 4;
        const float cx = db[0], cy = db[1], w = db[2], h = db[3];

        Detection d;
        d.box.x1   = (cx - 0.5f * w) * W;
        d.box.y1   = (cy - 0.5f * h) * H;
        d.box.x2   = (cx + 0.5f * w) * W;
        d.box.y2   = (cy + 0.5f * h) * H;
        d.class_id = cn.cls;
        d.score    = score;
        out.push_back(d);
    }
    return out;
}

}  // namespace dfine
