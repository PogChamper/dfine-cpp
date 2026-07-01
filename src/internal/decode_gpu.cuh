#pragma once

// GPU-side D-FINE decode (Zero-D2H): replaces the CPU postprocess with a device
// pipeline so only the survivors cross PCIe. Bit-faithful to postprocess.cpp:
//   sort candidates by RAW logit (descending) -> take top-k over (query x class)
//   -> score = sigmoid(logit), keep score >= threshold -> cxcywh->xyxy scaled by
//   the ORIGINAL image size. No NMS, no clamp, no degenerate-box drop.
//
// Sorting by logit (not sigmoid) matches the reference's sort key exactly and
// avoids float-saturation ties, so the RANKING is bit-identical to the CPU path;
// only the sigmoid score differs by <=1 ULP (GPU expf vs libm). Validate by mAP.
//
// This TU is allocation-free: the caller owns all buffers (a DevPtr set for M1,
// the arena in P2) and passes raw pointers, so it composes with CUDA-graph
// capture and the frozen-memory arena.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dfine {

// One decoded detection, device-side. Layout mirrors Detection's fields so the
// host copy-out is a trivial field assignment. 24 bytes.
struct DetectionGPU {
    float   x1, y1, x2, y2, score;
    int32_t class_id;  // 0..C-1 for a survivor; -1 for a padded (sub-threshold) slot
};

// Device scratch for the decode. All pointers are device memory owned by the
// caller and sized for the MAX batch. n_cand = num_queries * num_classes.
//   keys/vals          [max_batch * n_cand]  logits (sort keys) / packed idx (q*C+c)
//   keys_out/vals_out  [max_batch * n_cand]  sorted double-buffer
//   seg_off            [max_batch + 1]       segment offsets {0, n_cand, 2*n_cand, ...}
//   cub_temp           cub_temp_bytes        radix-sort temp storage
//   out                [max_batch * topk]    survivors (descending score), padded
//   counts             [max_batch]           #survivors per image (a prefix length)
//   scale_wh           [max_batch]           (origW, origH) per image, filled per call
struct GpuDecodeScratch {
    float*        keys           = nullptr;
    uint32_t*     vals           = nullptr;
    float*        keys_out       = nullptr;
    uint32_t*     vals_out       = nullptr;
    int*          seg_off        = nullptr;
    void*         cub_temp       = nullptr;
    std::size_t   cub_temp_bytes = 0;
    DetectionGPU* out            = nullptr;
    uint32_t*     counts         = nullptr;
    float2*       scale_wh       = nullptr;
};

// Bytes of cub temp storage needed for a segmented radix sort of
// (max_batch * n_cand) items across max_batch segments. Query once at setup.
[[nodiscard]] std::size_t gpu_decode_temp_bytes(int max_batch, int n_cand);

// Fill seg_off[0..max_batch] = j * n_cand. One-time (values are batch-invariant,
// so a scratch sized for max_batch serves any B <= max_batch). Synchronous-safe
// to call on the given stream.
void gpu_decode_fill_segoff(int* seg_off, int max_batch, int n_cand, cudaStream_t stream);

// Enqueue the decode on `stream` (no host sync): pack -> segmented radix sort
// (descending logit) -> top-k + threshold + box transform. Fills s.out[B*topk]
// (first s.counts[b] entries are the survivors for image b, descending score) and
// s.counts[B]. `s.scale_wh` must already hold the per-image (origW, origH).
//   logits : [B, Q, C] device, raw logits
//   boxes  : [B, Q, 4] device, cxcywh normalized to [0,1]
void gpu_decode_enqueue(const float* logits, const float* boxes, int B, int Q, int C, int topk,
                        float threshold, const GpuDecodeScratch& s, cudaStream_t stream);

}  // namespace dfine
