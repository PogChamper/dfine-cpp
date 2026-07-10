// validate_image_layout: the single geometry gate for every image entry point
// (C++ detect/detect_batch, preprocessor staging, C ABI). CPU-only.

#include "testing.hpp"

#include "internal/image_check.hpp"

#include <climits>
#include <cstdint>
#include <vector>

using dfine::ImageU8;
using dfine::validate_image_layout;

namespace {

ImageU8 view(const std::uint8_t* data, int w, int h, int stride = 0, int channels = 3) {
    ImageU8 im;
    im.data = data;
    im.width = w;
    im.height = h;
    im.stride = stride;
    im.channels = channels;
    return im;
}

}  // namespace

int main() {
    std::vector<std::uint8_t> buf(64 * 64 * 4, 0);
    const std::uint8_t* p = buf.data();

    // Well-formed views pass.
    validate_image_layout(view(p, 64, 64));          // tightly packed (stride 0)
    validate_image_layout(view(p, 64, 64, 64 * 3));  // explicit packed stride
    validate_image_layout(view(p, 64, 64, 64 * 4));  // padded rows
    validate_image_layout(view(p, 1, 1));            // minimal image

    // Rejected views.
    DFINE_EXPECT_THROW(validate_image_layout(view(nullptr, 64, 64)), "data is NULL");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 0, 64)), "invalid image size");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 0)), "invalid image size");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, -64, 64)), "invalid image size");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 64, 0, 1)), "channels must be 3");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 64, 0, 4)), "channels must be 3");

    // The bug class this gate exists for: a stride below width*channels (the
    // classic pixels-vs-bytes mixup) must be an error, not sheared detections.
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 64, 64)), "smaller than width*channels");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 64, 64 * 3 - 1)),
                       "smaller than width*channels");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 64, 64, -192)), "negative stride");

    // int-overflow guards: width*3 and height*packed_row evaluated in 64 bits.
    DFINE_EXPECT_THROW(validate_image_layout(view(p, INT_MAX, 2)), "too large");
    DFINE_EXPECT_THROW(validate_image_layout(view(p, 40000, 40000)), "too large");
    validate_image_layout(view(p, 8192, 8192));  // large-but-fine stays accepted

    return dfine::testing::finish("test_image_layout");
}
