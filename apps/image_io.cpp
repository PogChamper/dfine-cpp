#include "image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

namespace dfine_app {

LoadedImage::~LoadedImage() {
    if (pixels_) stbi_image_free(pixels_);
}

LoadedImage& LoadedImage::operator=(LoadedImage&& o) noexcept {
    if (this != &o) {
        if (pixels_) stbi_image_free(pixels_);
        pixels_ = o.pixels_;
        w_ = o.w_;
        h_ = o.h_;
        o.pixels_ = nullptr;
    }
    return *this;
}

LoadedImage load_image_rgb(const std::string& path) {
    int w = 0, h = 0, comp = 0;
    // Force 3 channels (RGB). stb decodes JPEG/PNG in RGB order.
    std::uint8_t* px = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!px) return LoadedImage{};
    return LoadedImage{px, w, h};
}

}  // namespace dfine_app
