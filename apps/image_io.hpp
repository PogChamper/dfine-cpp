#pragma once

#include "dfine/core/types.hpp"

#include <cstdint>
#include <string>
#include <utility>

namespace dfine_app {

// Owns pixels decoded by stb_image as packed 3-channel RGB HWC uint8.
class LoadedImage {
 public:
    LoadedImage() = default;
    LoadedImage(std::uint8_t* px, int w, int h) noexcept : pixels_(px), w_(w), h_(h) {}
    ~LoadedImage();

    LoadedImage(const LoadedImage&) = delete;
    LoadedImage& operator=(const LoadedImage&) = delete;
    LoadedImage(LoadedImage&& o) noexcept
        : pixels_(std::exchange(o.pixels_, nullptr)), w_(o.w_), h_(o.h_) {}
    LoadedImage& operator=(LoadedImage&& o) noexcept;

    explicit operator bool() const noexcept { return pixels_ != nullptr; }
    int width() const noexcept { return w_; }
    int height() const noexcept { return h_; }

    // View for the detector: RGB order (is_bgr = false), tightly packed.
    dfine::ImageU8 view() const noexcept {
        return dfine::ImageU8{pixels_, h_, w_, 3, w_ * 3, /*is_bgr=*/false};
    }

 private:
    std::uint8_t* pixels_{nullptr};
    int w_{0};
    int h_{0};
};

// Decode an image file to 3-channel RGB. Returns a falsy LoadedImage on failure.
LoadedImage load_image_rgb(const std::string& path);

}  // namespace dfine_app
