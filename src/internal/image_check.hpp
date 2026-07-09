#pragma once

#include "dfine/core/types.hpp"

#include <climits>
#include <stdexcept>
#include <string>

namespace dfine {

// Validate an ImageU8 view before any path consumes it. Every consumer — the
// preprocessor staging copy, the frozen full-graph pack loop, and the C ABI —
// must agree on what a well-formed view is, so the check lives in one place.
// Geometry only: a non-owning view carries no buffer size to check against.
inline void validate_image_layout(const ImageU8& image) {
    if (!image.data) throw std::invalid_argument("dfine: image data is NULL");
    if (image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("dfine: invalid image size (width=" +
                                    std::to_string(image.width) +
                                    " height=" + std::to_string(image.height) + ")");
    }
    if (image.channels != 3) {
        throw std::invalid_argument("dfine: channels must be 3 (got " +
                                    std::to_string(image.channels) + ")");
    }
    // 64-bit to avoid signed-int overflow (UB) for pathological widths.
    const long long packed_row = static_cast<long long>(image.width) * image.channels;
    if (image.stride < 0) {
        throw std::invalid_argument("dfine: negative stride (" + std::to_string(image.stride) +
                                    "); pass 0 for tightly packed rows");
    }
    if (image.stride > 0 && image.stride < packed_row) {
        // Copying width*channels bytes per row would overrun every source row
        // (and the buffer on the last one) and feed the network sheared pixels.
        throw std::invalid_argument("dfine: stride (" + std::to_string(image.stride) +
                                    ") is smaller than width*channels (" +
                                    std::to_string(packed_row) + ")");
    }
    // The CUDA kernels index the packed source with int arithmetic; bound the
    // packed image size so row offsets cannot overflow.
    if (static_cast<long long>(image.height) * packed_row > INT_MAX) {
        throw std::invalid_argument("dfine: image too large (" + std::to_string(image.width) +
                                    "x" + std::to_string(image.height) +
                                    "); packed size must stay under 2 GiB");
    }
}

}  // namespace dfine
