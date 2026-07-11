#include "testing.hpp"

#include "dfine/core/postprocess.hpp"

#include <cstddef>
#include <vector>

int main() {
    constexpr int kQueries = 100;
    constexpr int kClasses = 4;
    DFINE_CHECK(dfine::detection_limit(kQueries, kClasses) == 300);
    DFINE_CHECK(dfine::detection_limit(kQueries, 2) == 200);
    DFINE_CHECK(dfine::detection_limit(0, kClasses) == 0);

    std::vector<float> logits(static_cast<std::size_t>(kQueries) * kClasses, 0.0f);
    std::vector<float> boxes(static_cast<std::size_t>(kQueries) * 4);
    for (int query = 0; query < kQueries; ++query) {
        float* box = boxes.data() + static_cast<std::size_t>(query) * 4;
        box[0] = 0.5f;
        box[1] = 0.5f;
        box[2] = 0.25f;
        box[3] = 0.25f;
    }

    dfine::PostprocessParams params;
    params.num_queries = kQueries;
    params.num_classes = kClasses;
    params.topk = dfine::detection_limit(kQueries, kClasses);
    params.threshold = 0.0f;

    const dfine::Detections detections =
        dfine::decode_detections(logits.data(), boxes.data(), 640, 480, params);
    DFINE_CHECK(detections.size() == 300);
    for (const auto& detection : detections) {
        DFINE_CHECK(detection.class_id >= 0 && detection.class_id < kClasses);
        DFINE_CHECK(detection.score == 0.5f);
    }

    return dfine::testing::finish("test_postprocess");
}
