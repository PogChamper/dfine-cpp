#include "internal/decode_gpu.cuh"

#include "internal/cuda_check.hpp"

#include <cmath>

#include <cub/device/device_segmented_radix_sort.cuh>

namespace dfine {

namespace {

// Numerically stable sigmoid, matching postprocess.cpp's branch structure exactly.
// `expf` (not the fast __expf intrinsic) keeps this within ~1 ULP of the host libm
// used by the CPU decode; the ranking itself never depends on this (we sort by raw
// logit), so the only cross-path difference is this <=1 ULP score.
__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float z = expf(x);
    return z / (1.0f + z);
}

// Pack (score-key, index) pairs. Key = RAW logit (monotone with sigmoid, so the
// sort order equals the reference's; sorting raw logits avoids float-saturation
// ties that sorting sigmoid probabilities would create). Value = within-image
// flat index (query*C + class), recovered after the sort.
__global__ void k_pack(const float* __restrict__ logits, float* __restrict__ keys,
                       uint32_t* __restrict__ vals, int total, int qc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // A NaN logit would otherwise sort by bit pattern (a positive NaN ranks above
    // +Inf) into the top-k prefix and, being neither >= nor < any threshold, corrupt
    // the survivors-are-a-prefix invariant. Map NaN to -inf so it sorts PAST top-k
    // and is cleanly dropped. (postprocess.cpp's partial_sort is itself UB on NaN.)
    const float lg = logits[i];
    keys[i] = isnan(lg) ? -INFINITY : lg;
    vals[i] = static_cast<uint32_t>(i % qc);  // i = b*qc + (q*C + c)  ->  q*C + c
}

// seg_off[j] = j * n_cand  (batch-invariant segment boundaries; a scratch sized for
// max_batch serves any B <= max_batch by using the first B+1 offsets).
__global__ void k_fill_segoff(int* __restrict__ seg, int n, int n_cand) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) seg[j] = j * n_cand;
}

// One block per image. Walk the top-`topk` sorted candidates: score = sigmoid(logit),
// keep score >= threshold (survivors are a prefix since score is non-increasing along
// the descending-logit sort), decode cxcywh(norm) -> xyxy scaled by the original size.
// Writes out[b*topk + k] for every k (sub-threshold slots get class_id = -1) and
// counts[b] = #survivors. No clamp / no NMS — matches postprocess.cpp.
__global__ void k_decode_topk(const float* __restrict__ keys_sorted,
                              const uint32_t* __restrict__ vals_sorted,
                              const float* __restrict__ boxes, const float2* __restrict__ scale_wh,
                              DetectionGPU* __restrict__ out, uint32_t* __restrict__ counts, int B,
                              int Q, int C, int qc, int topk, float threshold,
                              const float* __restrict__ threshold_dev) {
    const int b = blockIdx.x;
    if (b >= B) return;

    // threshold_dev (mapped pinned) is read at EXECUTION time — one zero-copy load
    // per block, broadcast through shared memory — so a captured graph replays with
    // the caller's current threshold instead of the value baked at capture (P3).
    __shared__ unsigned int s_count;
    __shared__ float s_thr;
    if (threadIdx.x == 0) {
        s_count = 0u;
        s_thr = threshold_dev ? *threshold_dev : threshold;
    }
    __syncthreads();
    threshold = s_thr;

    const float2 sw = scale_wh[b];
    const int base = b * qc;

    for (int k = threadIdx.x; k < topk; k += blockDim.x) {
        const int g = base + k;
        const float logit = keys_sorted[g];
        const float score = sigmoid_stable(logit);

        DetectionGPU d;
        // keep == !(score < threshold): the exact complement of postprocess.cpp's
        // `if (score < threshold) continue`, so it matches the CPU reference even for
        // a NaN threshold (where `score >= threshold` would wrongly reject everything).
        if (!(score < threshold)) {
            atomicAdd(&s_count, 1u);
            const uint32_t idx = vals_sorted[g];
            const int q = static_cast<int>(idx) / C;
            const int c = static_cast<int>(idx) % C;
            const float* db = boxes + static_cast<std::size_t>(b * Q + q) * 4;
            const float cx = db[0], cy = db[1], w = db[2], h = db[3];
            d.x1 = (cx - 0.5f * w) * sw.x;
            d.y1 = (cy - 0.5f * h) * sw.y;
            d.x2 = (cx + 0.5f * w) * sw.x;
            d.y2 = (cy + 0.5f * h) * sw.y;
            d.score = score;
            d.class_id = c;
        } else {
            d.x1 = d.y1 = d.x2 = d.y2 = 0.0f;
            d.score = score;
            d.class_id = -1;
        }
        out[b * topk + k] = d;
    }
    __syncthreads();
    if (threadIdx.x == 0) counts[b] = s_count;
}

}  // namespace

std::size_t gpu_decode_temp_bytes(int max_batch, int n_cand) {
    std::size_t bytes = 0;
    const int total = max_batch * n_cand;
    // Sizing query: CUB computes temp storage from item/segment counts only (the
    // key/value/offset pointers are not dereferenced when d_temp_storage == nullptr).
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr), total, max_batch,
        static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), 0,
        static_cast<int>(sizeof(float) * 8), cudaStream_t{0});
    return bytes;
}

void gpu_decode_fill_segoff(int* seg_off, int max_batch, int n_cand, cudaStream_t stream) {
    const int n = max_batch + 1;
    k_fill_segoff<<<(n + 63) / 64, 64, 0, stream>>>(seg_off, n, n_cand);
    DFINE_CUDA_CHECK(cudaGetLastError());
}

void gpu_decode_enqueue(const float* logits, const float* boxes, int B, int Q, int C, int topk,
                        float threshold, const float* threshold_dev, const GpuDecodeScratch& s,
                        cudaStream_t stream) {
    const int qc = Q * C;
    const int total = B * qc;

    k_pack<<<(total + 255) / 256, 256, 0, stream>>>(logits, s.keys, s.vals, total, qc);
    DFINE_CUDA_CHECK(cudaGetLastError());

    std::size_t temp_bytes = s.cub_temp_bytes;  // sized for max_batch (>= this call's need)
    DFINE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        s.cub_temp, temp_bytes, s.keys, s.keys_out, s.vals, s.vals_out, total, B, s.seg_off,
        s.seg_off + 1, 0, static_cast<int>(sizeof(float) * 8), stream));

    k_decode_topk<<<B, 256, 0, stream>>>(s.keys_out, s.vals_out, boxes, s.scale_wh, s.out, s.counts,
                                         B, Q, C, qc, topk, threshold, threshold_dev);
    DFINE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dfine
