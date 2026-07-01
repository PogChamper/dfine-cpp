#include "internal/cuda_preprocess.cuh"

#include "internal/cuda_check.hpp"
#include "internal/cuda_raii.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace dfine {

namespace {

// Fused preprocess kernel — one pass does bilinear stretch-resize, optional
// BGR->RGB swap, /255 + per-channel (x/std - mean/std) normalize, and HWC->CHW
// transpose. Loop-invariants (scale factors, folded constants) are precomputed
// on the host so the kernel body is pure per-pixel work.
//
// The output->source coordinate map matches OpenCV INTER_LINEAR semantics
// (`src = (dst + 0.5) * scale - 0.5`), so results track the Python cv2.resize
// reference used by trt-files/scripts/coco_eval.py.
__global__ void stretchResizeNormalizeKernel(const std::uint8_t* __restrict__ src, int src_h,
                                             int src_w, int src_pitch, float* __restrict__ dst,
                                             int dst_h, int dst_w, bool src_is_bgr, float scale_x,
                                             float scale_y, float3 pre_mul, float3 pre_sub) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const float sx_f = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    const float sy_f = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

    const int sx0 = max(0, min(src_w - 1, static_cast<int>(floorf(sx_f))));
    const int sy0 = max(0, min(src_h - 1, static_cast<int>(floorf(sy_f))));
    const int sx1 = min(src_w - 1, sx0 + 1);
    const int sy1 = min(src_h - 1, sy0 + 1);
    const float fx = fmaxf(0.0f, fminf(1.0f, sx_f - sx0));
    const float fy = fmaxf(0.0f, fminf(1.0f, sy_f - sy0));
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const std::uint8_t* p00 = src + sy0 * src_pitch + sx0 * 3;
    const std::uint8_t* p01 = src + sy0 * src_pitch + sx1 * 3;
    const std::uint8_t* p10 = src + sy1 * src_pitch + sx0 * 3;
    const std::uint8_t* p11 = src + sy1 * src_pitch + sx1 * 3;

    // __ldg routes through the read-only data cache.
    const float c0 =
        __ldg(p00 + 0) * w00 + __ldg(p01 + 0) * w01 + __ldg(p10 + 0) * w10 + __ldg(p11 + 0) * w11;
    const float c1 =
        __ldg(p00 + 1) * w00 + __ldg(p01 + 1) * w01 + __ldg(p10 + 1) * w10 + __ldg(p11 + 1) * w11;
    const float c2 =
        __ldg(p00 + 2) * w00 + __ldg(p01 + 2) * w01 + __ldg(p10 + 2) * w10 + __ldg(p11 + 2) * w11;

    const float r = src_is_bgr ? c2 : c0;
    const float g = c1;
    const float b = src_is_bgr ? c0 : c2;

    const float nr = r * pre_mul.x - pre_sub.x;
    const float ng = g * pre_mul.y - pre_sub.y;
    const float nb = b * pre_mul.z - pre_sub.z;

    const int hw = dst_h * dst_w;
    const int idx = y * dst_w + x;
    dst[0 * hw + idx] = nr;
    dst[1 * hw + idx] = ng;
    dst[2 * hw + idx] = nb;
}

}  // namespace

void launch_stretch_resize_normalize(cudaStream_t stream, const std::uint8_t* d_src, int src_h,
                                     int src_w, int src_pitch, float* d_dst, int dst_h, int dst_w,
                                     bool src_is_bgr, const float mean[3], const float std[3]) {
    if (dst_h <= 0 || dst_w <= 0 || src_h <= 0 || src_w <= 0) {
        throw std::runtime_error("dfine: launch_stretch_resize_normalize bad dims");
    }

    constexpr float kInv255 = 1.0f / 255.0f;
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float3 pre_mul{kInv255 / std[0], kInv255 / std[1], kInv255 / std[2]};
    const float3 pre_sub{mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]};

    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    stretchResizeNormalizeKernel<<<grid, block, 0, stream>>>(d_src, src_h, src_w, src_pitch, d_dst,
                                                             dst_h, dst_w, src_is_bgr, scale_x,
                                                             scale_y, pre_mul, pre_sub);
    DFINE_CUDA_CHECK(cudaGetLastError());
}

ImagePreprocessor::ImagePreprocessor(int dst_h, int dst_w) : dst_h_(dst_h), dst_w_(dst_w) {
    cudaEvent_t raw_event = nullptr;
    DFINE_CUDA_CHECK(cudaEventCreateWithFlags(&raw_event, cudaEventDisableTiming));
    upload_done_ = CudaEvent(raw_event);
}

// DevPtr/HostPtr/CudaEvent destructors release their resources automatically.
ImagePreprocessor::~ImagePreprocessor() = default;

void ImagePreprocessor::set_mean(float r, float g, float b) noexcept {
    mean_[0] = r;
    mean_[1] = g;
    mean_[2] = b;
}

void ImagePreprocessor::set_std(float r, float g, float b) noexcept {
    std_[0] = r;
    std_[1] = g;
    std_[2] = b;
}

void ImagePreprocessor::ensure_capacity_(std::size_t bytes) {
    if (bytes <= capacity_) return;
    if (frozen_) {
        throw std::runtime_error(
            "dfine: ImagePreprocessor is frozen but the source frame needs " +
            std::to_string(bytes) + " staging bytes (capacity " + std::to_string(capacity_) +
            "); freeze() with src_w/src_h covering the largest steady-state frame");
    }
    d_src_.reset();
    h_pinned_.reset();
    capacity_ = 0;
    void* dp = nullptr;
    DFINE_CUDA_CHECK(cudaMalloc(&dp, bytes));
    d_src_.reset(dp);  // own it before the next throwing call
    void* hp = nullptr;
    DFINE_CUDA_CHECK(cudaMallocHost(&hp, bytes));
    h_pinned_.reset(hp);
    capacity_ = bytes;
}

void ImagePreprocessor::process(const ImageU8& image, float* d_dst, cudaStream_t stream) {
    if (!image.data || image.height <= 0 || image.width <= 0) {
        throw std::runtime_error("dfine: ImagePreprocessor::process given empty image");
    }
    if (image.channels != 3) {
        throw std::runtime_error("dfine: ImagePreprocessor expects a 3-channel HWC image");
    }

    const int rows = image.height;
    const int cols = image.width;
    const int src_row = image.row_bytes();
    const std::size_t packed_row = static_cast<std::size_t>(cols) * 3;
    const std::size_t total_bytes = packed_row * static_cast<std::size_t>(rows);

    // Wait until any prior async H2D has consumed the pinned/source buffers before
    // we (re)allocate or overwrite them — the copy is async, so the host must not
    // clobber or free them mid-flight. No-op on the first call / after a stream sync.
    DFINE_CUDA_CHECK(cudaEventSynchronize(upload_done_.get()));
    ensure_capacity_(total_bytes);

    // Pack into a contiguous pinned buffer (source may have stride > width*3).
    auto* dst_pinned = static_cast<std::uint8_t*>(h_pinned_.get());
    auto* d_src_raw = static_cast<std::uint8_t*>(d_src_.get());
    if (static_cast<std::size_t>(src_row) == packed_row) {
        std::memcpy(dst_pinned, image.data, total_bytes);
    } else {
        for (int r = 0; r < rows; ++r) {
            std::memcpy(dst_pinned + r * packed_row,
                        image.data + static_cast<std::size_t>(r) * src_row, packed_row);
        }
    }
    DFINE_CUDA_CHECK(
        cudaMemcpyAsync(d_src_raw, dst_pinned, total_bytes, cudaMemcpyHostToDevice, stream));
    DFINE_CUDA_CHECK(cudaEventRecord(upload_done_.get(), stream));

    launch_stretch_resize_normalize(stream, d_src_raw, rows, cols, static_cast<int>(packed_row),
                                    d_dst, dst_h_, dst_w_, image.is_bgr, mean_, std_);
}

}  // namespace dfine
