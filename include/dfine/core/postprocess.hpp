#pragma once

#include "dfine/core/types.hpp"

namespace dfine {

// Parameters for D-FINE detection decoding. The raw engine emits a focal-style
// class head with NO background slot: `logits [N, num_queries, num_classes]` and
// `boxes [N, num_queries, 4]` (cxcywh, normalized to [0,1]). Decode has no NMS.
struct PostprocessParams {
    int   num_queries{300};   // logits/boxes dim 1
    int   num_classes{80};    // logits dim 2 (contiguous class ids, no background)
    int   topk{300};          // top-k over (query × class); default = num_queries
    float threshold{0.5f};    // score threshold applied after top-k selection
};

// Decode one image's raw D-FINE outputs into pixel-space detections.
//
//   logits : (num_queries, num_classes) float32, raw logits (sigmoid activation)
//   boxes  : (num_queries, 4) float32, cxcywh normalized to [0,1]
//   img_w/h: original image size for rescaling
//
// Reproduces trt-files/scripts/coco_eval.py `decode()` exactly:
//   sigmoid -> top-k over (query × class) -> label = idx % C, query = idx // C
//   -> cxcywh -> xyxy scaled by (img_w, img_h). Boxes are NOT clamped to the
//   image (matching the reference); class_id is the contiguous index 0..C-1.
[[nodiscard]] Detections decode_detections(const float* logits, const float* boxes, int img_w,
                                           int img_h, const PostprocessParams& params);

}  // namespace dfine
