// cpp_detector_example.cpp — minimal use of the D-FINE C++ detector API.
//
// Not wired into CMake; this is a documented snippet. Build it by hand against
// an already-built libdfine.so (see build.sh), linking dfine::dfine. The public
// headers hide TensorRT/CUDA behind a PIMPL, so no TensorRT include path is
// needed; if your TensorRT libs are not on a default linker path, add
// -Wl,-rpath-link,/path/to/tensorrt/lib so ld can resolve libdfine's deps.
//
//   c++ -std=c++17 -I include \
//       examples/cpp_detector_example.cpp -Lbuild -ldfine \
//       -Wl,-rpath,build -o cpp_detector_example
//
//   ./cpp_detector_example path/to/dfine_m_fp32.engine
//
// The library is OpenCV-free: it never decodes images itself. The caller
// decodes pixels by whatever means (stb_image, libjpeg, a camera driver, ...)
// and hands the detector a raw HWC uint8 buffer via `ImageU8`.

#include "dfine/tasks/detector.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <engine_path> [meta_path]\n", argv[0]);
        return 2;
    }

    // 1. Construct from an engine path and optional explicit sidecar. Without
    //    one, discovery checks `<engine>.json`, then the same-stem JSON.
    dfine::DetectorOptions opts;
    opts.threshold = 0.5f;
    dfine::DFineDetector detector = argc > 2 ? dfine::DFineDetector(argv[1], argv[2], opts)
                                             : dfine::DFineDetector(argv[1], opts);

    // 2. Build an ImageU8 view over a raw HWC uint8 buffer. ImageU8 is a
    //    non-owning view: fill/obtain `pixels` however you like, then point
    //    the view at it — no image type from this library owns the memory.
    const int width = 640, height = 480, channels = 3;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * channels, 128);
    // ... fill `pixels` with real RGB (or BGR) image data here ...

    dfine::ImageU8 image;
    image.data = pixels.data();
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.stride = 0;      // 0 => tightly packed (width * channels bytes/row)
    image.is_bgr = false;  // true if `pixels` is BGR (e.g. straight from OpenCV)

    // 3. Run inference + decode (sigmoid -> top-k -> cxcywh->xyxy -> threshold).
    dfine::Detections detections = detector.detect(image);

    // 4. Consume the results.
    std::printf("%zu detection(s)\n", detections.size());
    for (const auto& det : detections) {
        std::printf("  class_id=%d score=%.3f box=[%.1f, %.1f, %.1f, %.1f]\n", det.class_id,
                    det.score, det.box.x1, det.box.y1, det.box.x2, det.box.y2);
    }

    return 0;
}
