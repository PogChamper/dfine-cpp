#pragma once

#include <cstdint>
#include <vector>

namespace dfine {

// Axis-aligned box in pixel coordinates of the original image.
struct Box {
    float x1{0}, y1{0}, x2{0}, y2{0};
    [[nodiscard]] float width() const noexcept { return x2 - x1; }
    [[nodiscard]] float height() const noexcept { return y2 - y1; }
    [[nodiscard]] float area() const noexcept { return width() * height(); }
};

struct Detection {
    Box box;
    int class_id{-1};  // contiguous 0..num_classes-1 (no background slot)
    float score{0.0f};
};

using Detections = std::vector<Detection>;

// Non-owning view of a packed HWC uint8 image (the detector's input container).
// Kept OpenCV-free so public headers pull in no third-party image types; apps
// decode JPEGs with any loader (stb_image, libjpeg, …) and hand over the buffer.
struct ImageU8 {
    const std::uint8_t* data{nullptr};  // row-major, `channels` bytes per pixel
    int height{0};
    int width{0};
    int channels{3};     // only 3-channel is supported
    int stride{0};       // bytes per row; 0 => width * channels
    bool is_bgr{false};  // true if channel order is BGR (swapped to RGB)

    int row_bytes() const noexcept { return stride > 0 ? stride : width * channels; }
};

}  // namespace dfine
