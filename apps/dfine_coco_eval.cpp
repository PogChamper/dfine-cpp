// dfine_coco_eval — run the C++ detector over a COCO image list and stream the
// detections as a JSON array for pycocotools scoring (see cpp_coco_eval.py).
//
// The `--filelist` is `<image_id> <file_name>` per line (the Python driver writes
// it from the annotation, honoring the same sort/limit as coco_eval.py). Class ids
// stay contiguous (0..79); the driver maps them to COCO category_id.
//
// usage: dfine_coco_eval --engine E.engine --images-dir DIR --filelist L.txt
//                        --out dets.json [--meta E.json] [--threshold 0.001]

#include "cli_helpers.hpp"
#include "image_io.hpp"

#include "dfine/tasks/detector.hpp"
#include "dfine/version.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    std::filesystem::path engine, meta, images_dir, filelist, out;
    float threshold = 0.001f;  // low threshold: emit for mAP (matches coco_eval.py)
    int batch = 1;
    bool cuda_graph = false;
    bool gpu_decode = false;
    bool own_dev_mem = false;
    bool do_freeze = false;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf("usage: %s --engine E --images-dir DIR --filelist L --out J "
                            "[--meta M] [--threshold 0.001] [--batch 1] [--cuda-graph]\n  dfine v%s\n",
                            argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine"))     engine = next_value(argc, argv, i, "--engine");
            else if (starts_with(a, "--images-dir"))    images_dir = next_value(argc, argv, i, "--images-dir");
            else if (starts_with(a, "--filelist"))      filelist = next_value(argc, argv, i, "--filelist");
            else if (starts_with(a, "--meta"))          meta = next_value(argc, argv, i, "--meta");
            else if (starts_with(a, "--out"))           out = next_value(argc, argv, i, "--out");
            else if (starts_with(a, "--threshold"))     threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            else if (starts_with(a, "--batch"))         batch = parse_int(next_value(argc, argv, i, "--batch"), "--batch");
            else if (a == "--cuda-graph")               cuda_graph = true;
            else if (a == "--gpu-decode")               gpu_decode = true;
            else if (a == "--own-device-memory")        own_dev_mem = true;
            else if (a == "--freeze")                   do_freeze = true;
            else throw std::runtime_error("unknown arg: " + std::string(a));
        }
        if (engine.empty() || images_dir.empty() || filelist.empty() || out.empty()) {
            std::fprintf(stderr, "error: --engine, --images-dir, --filelist, --out are required\n");
            return 2;
        }
        if (batch < 1) batch = 1;

        dfine::DetectorOptions opts;
        opts.threshold = threshold;
        opts.use_cuda_graph = cuda_graph;  // validates the graph path produces == mAP
        opts.gpu_decode = gpu_decode;      // validates the GPU-decode path == CPU-decode mAP
        opts.own_device_memory = own_dev_mem;  // validates the kUSER_MANAGED activation path
        dfine::DFineDetector det = meta.empty() ? dfine::DFineDetector(engine, opts)
                                                : dfine::DFineDetector(engine, meta, opts);

        std::size_t free_after_freeze = 0, total_vram = 0;
        if (do_freeze) {
            det.freeze(batch);  // warm to peak + lock; steady state must not allocate
            cudaMemGetInfo(&free_after_freeze, &total_vram);
        }

        std::ifstream fl(filelist);
        if (!fl) throw std::runtime_error("cannot open filelist: " + filelist.string());
        std::ofstream os(out);
        if (!os) throw std::runtime_error("cannot open output: " + out.string());

        os << "[";
        bool first = true;
        long long n_imgs = 0, n_dets = 0, n_missing = 0;

        struct Item { long long id; dfine_app::LoadedImage img; };
        std::vector<Item> chunk;
        chunk.reserve(static_cast<std::size_t>(batch));

        auto flush = [&]() {
            if (chunk.empty()) return;
            std::vector<dfine::ImageU8> views;
            views.reserve(chunk.size());
            for (const auto& it : chunk) views.push_back(it.img.view());
            std::vector<dfine::Detections> res = det.detect_batch(views, threshold);
            for (std::size_t k = 0; k < res.size(); ++k) {
                for (const auto& d : res[k]) {
                    const float w = d.box.x2 - d.box.x1;
                    const float h = d.box.y2 - d.box.y1;
                    if (!first) os << ',';
                    first = false;
                    os << "{\"image_id\":" << chunk[k].id
                       << ",\"category_contig\":" << d.class_id
                       << ",\"bbox\":[" << d.box.x1 << ',' << d.box.y1 << ',' << w << ',' << h << ']'
                       << ",\"score\":" << d.score << '}';
                    ++n_dets;
                }
            }
            n_imgs += static_cast<long long>(chunk.size());
            if (n_imgs % 500 < static_cast<long long>(chunk.size()))
                std::fprintf(stderr, "[dfine_coco_eval] %lld images...\n", n_imgs);
            chunk.clear();
        };

        std::string line;
        while (std::getline(fl, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            long long image_id = 0;
            std::string fname;
            if (!(ss >> image_id >> fname)) continue;

            dfine_app::LoadedImage img = dfine_app::load_image_rgb((images_dir / fname).string());
            if (!img) { ++n_missing; continue; }
            chunk.push_back({image_id, std::move(img)});
            if (static_cast<int>(chunk.size()) >= batch) flush();
        }
        flush();

        os << "]\n";
        std::fprintf(stderr, "[dfine_coco_eval] done: %lld images, %lld detections, %lld missing "
                     "(batch=%d)\n", n_imgs, n_dets, n_missing, batch);
        if (do_freeze) {
            std::size_t free_end = 0, tot = 0;
            cudaMemGetInfo(&free_end, &tot);
            const long long delta = static_cast<long long>(free_after_freeze) -
                                    static_cast<long long>(free_end);
            std::fprintf(stderr, "[dfine_coco_eval] frozen: free VRAM delta over %lld images = "
                         "%+lld bytes (0 == no steady-state allocation)\n", n_imgs, delta);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
