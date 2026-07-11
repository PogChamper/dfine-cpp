#pragma once

#include "cuda_raii.hpp"

#include "dfine/core/types.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace dfine {

// Launch the fused stretch-resize + (optional) BGR->RGB + normalize + HWC->CHW
// kernel. Writes directly into `d_dst` on `stream`.
//
//   d_src       : device pointer to a contiguous HWC uint8 image, src_pitch bytes/row
//   src_h,src_w : source image dims in pixels
//   src_pitch   : bytes per source row (>= src_w * 3)
//   d_dst       : device pointer to (3 * dst_h * dst_w) float32, planar CHW
//   dst_h,dst_w : output dims (stretch resize — no aspect preservation, no letterbox)
//   src_is_bgr  : if true, swaps channels 0/2 to produce RGB
//   mean,std    : per-channel normalize constants in RGB order. D-FINE uses {0,0,0}/
//                 {1,1,1}, i.e. `/255` only — passing ImageNet constants collapses mAP.
//
// The fold is nx = x/255 * (1/std) - mean/std. With mean=0,std=1 this is exactly x/255.
void launch_stretch_resize_normalize(cudaStream_t stream, const std::uint8_t* d_src, int src_h,
                                     int src_w, int src_pitch, float* d_dst, int dst_h, int dst_w,
                                     bool src_is_bgr, const float mean[3], const float std[3]);

// Letterbox placement of a source frame inside the dst canvas: aspect-preserving
// scale `s`, content at [dx, dx+nw) x [dy, dy+nh), the rest is padding. With
// allow_upscale=false a frame that already fits is pasted 1:1 (production
// smart_resize semantics); anchor_topleft=false centers the content.
struct LetterboxMap {
    float s{1.0f};
    int dx{0}, dy{0};
    int nw{0}, nh{0};
};

[[nodiscard]] LetterboxMap compute_letterbox_map(int src_w, int src_h, int dst_w, int dst_h,
                                                 bool anchor_topleft, bool allow_upscale) noexcept;

// Letterbox counterpart of launch_stretch_resize_normalize: same fused
// normalize/BGR-swap/CHW write, but the source maps into the `map` region and
// everything outside it gets `pad_value` (0..255, normalized like a pixel).
// A separate kernel on purpose — the stretch kernel is on the byte-identical
// default path and stays untouched.
void launch_letterbox_resize_normalize(cudaStream_t stream, const std::uint8_t* d_src, int src_h,
                                       int src_w, int src_pitch, float* d_dst, int dst_h, int dst_w,
                                       const LetterboxMap& map, int pad_value, bool src_is_bgr,
                                       const float mean[3], const float std[3]);

// Convenience preprocessor: owns a pinned host staging buffer + a device source
// buffer that grows on demand. Each `process()` call uploads the image to the
// device asynchronously and runs the fused kernel, writing into `d_dst`.
// Designed for reuse across many frames — buffers are not freed between calls.
class ImagePreprocessor {
 public:
    ImagePreprocessor(int dst_h, int dst_w);
    ~ImagePreprocessor();

    ImagePreprocessor(const ImagePreprocessor&) = delete;
    ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;

    void set_mean(float r, float g, float b) noexcept;
    void set_std(float r, float g, float b) noexcept;

    // Select letterbox preprocessing (default is stretch). Applies to every
    // subsequent process() call; the detector derives the same LetterboxMap for
    // its box un-mapping via compute_letterbox_map.
    void set_letterbox(bool anchor_topleft, int pad_value, bool allow_upscale) noexcept {
        letterbox_ = true;
        lb_topleft_ = anchor_topleft;
        lb_pad_ = pad_value;
        lb_upscale_ = allow_upscale;
    }
    [[nodiscard]] bool letterbox() const noexcept { return letterbox_; }

    // image: HWC uint8 view (3 channels). d_dst: 3*dst_h*dst_w floats on device.
    void process(const ImageU8& image, float* d_dst, cudaStream_t stream);

    // Part of the frozen-memory contract: after freeze(), a
    // source frame needing a LARGER staging buffer throws instead of silently
    // cudaMalloc-ing on the hot path. Warm with the largest steady-state frame
    // first (DFineDetector::freeze(FreezeSpec) does this).
    void freeze() noexcept { frozen_ = true; }
    [[nodiscard]] bool frozen() const noexcept { return frozen_; }

    int dst_h() const noexcept { return dst_h_; }
    int dst_w() const noexcept { return dst_w_; }

 private:
    void ensure_capacity_(std::size_t bytes);

    bool frozen_{false};
    bool letterbox_{false};
    bool lb_topleft_{false};
    int lb_pad_{114};
    bool lb_upscale_{true};
    int dst_h_{0};
    int dst_w_{0};
    float mean_[3]{0.0f, 0.0f, 0.0f};
    float std_[3]{1.0f, 1.0f, 1.0f};
    DevPtr d_src_;
    HostPtr h_pinned_;
    std::size_t capacity_{0};
    // Signals that the previous async H2D has drained the pinned staging buffer,
    // so a batch loop can safely overwrite it for the next image without a race
    // (the copy is async; the host must not clobber it mid-flight).
    CudaEvent upload_done_;
};

}  // namespace dfine
