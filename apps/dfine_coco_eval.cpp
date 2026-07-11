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
    bool full_graph = false;
    int filter_w = 0, filter_h = 0;  // --filter-res: only eval images of exactly WxH
    bool letterbox = false;          // validates the letterbox preprocessing path
    bool lb_topleft = false;
    int lb_pad = 114;
    bool lb_upscale = true;
    try {
        auto parse_wh = [](std::string_view v, int& w, int& h, const char* flag) {
            const std::string s(v);
            const auto x = s.find('x');
            if (x == std::string::npos)
                throw std::runtime_error(std::string(flag) + " expects WxH");
            w = std::stoi(s.substr(0, x));
            h = std::stoi(s.substr(x + 1));
        };
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "-h" || a == "--help") {
                std::printf(
                    "usage: %s --engine E --images-dir DIR --filelist L --out J "
                    "[--meta M] [--threshold 0.001] [--batch 1] [--cuda-graph] "
                    "[--gpu-decode] [--own-device-memory] [--freeze] "
                    "[--full-graph] [--filter-res WxH]\n"
                    "  --full-graph  freeze + capture the full-pipeline CUDA graph; "
                    "requires a fixed source\n                resolution — pair with "
                    "--filter-res so every image matches the frozen size\n"
                    "  --filter-res  skip images whose size differs from WxH (also the "
                    "freeze source size)\n  dfine v%s\n",
                    argv[0], dfine::version());
                return 0;
            } else if (starts_with(a, "--engine"))
                engine = next_value(argc, argv, i, "--engine");
            else if (starts_with(a, "--images-dir"))
                images_dir = next_value(argc, argv, i, "--images-dir");
            else if (starts_with(a, "--filelist"))
                filelist = next_value(argc, argv, i, "--filelist");
            else if (starts_with(a, "--meta"))
                meta = next_value(argc, argv, i, "--meta");
            else if (starts_with(a, "--out"))
                out = next_value(argc, argv, i, "--out");
            else if (starts_with(a, "--threshold"))
                threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
            else if (starts_with(a, "--batch"))
                batch = parse_int(next_value(argc, argv, i, "--batch"), "--batch");
            else if (a == "--cuda-graph")
                cuda_graph = true;
            else if (a == "--gpu-decode")
                gpu_decode = true;
            else if (a == "--own-device-memory")
                own_dev_mem = true;
            else if (a == "--freeze")
                do_freeze = true;
            else if (a == "--full-graph")
                full_graph = true;
            else if (starts_with(a, "--filter-res"))
                parse_wh(next_value(argc, argv, i, "--filter-res"), filter_w, filter_h,
                         "--filter-res");
            else if (a == "--letterbox")
                letterbox = true;
            else if (a == "--letterbox-topleft") {
                letterbox = true;
                lb_topleft = true;
            } else if (starts_with(a, "--letterbox-pad")) {
                letterbox = true;
                lb_pad = parse_int(next_value(argc, argv, i, "--letterbox-pad"), "--letterbox-pad");
            } else if (a == "--no-upscale") {
                letterbox = true;
                lb_upscale = false;
            } else
                throw std::runtime_error("unknown arg: " + std::string(a));
        }
        if (engine.empty() || images_dir.empty() || filelist.empty() || out.empty()) {
            std::fprintf(stderr, "error: --engine, --images-dir, --filelist, --out are required\n");
            return 2;
        }
        if (batch < 1) batch = 1;

        dfine::DetectorOptions opts;
        opts.threshold = threshold;
        opts.use_cuda_graph = cuda_graph;      // validates the graph path produces == mAP
        opts.gpu_decode = gpu_decode;          // validates the GPU-decode path == CPU-decode mAP
        opts.own_device_memory = own_dev_mem;  // validates the kUSER_MANAGED activation path
        opts.full_pipeline_graph = full_graph;
        if (letterbox) {
            opts.preprocess.resize = dfine::PreprocessSpec::Resize::kLetterbox;
            opts.preprocess.anchor_topleft = lb_topleft;
            opts.preprocess.pad_value = lb_pad;
            opts.preprocess.allow_upscale = lb_upscale;
        }
        dfine::DFineDetector det = meta.empty() ? dfine::DFineDetector(engine, opts)
                                                : dfine::DFineDetector(engine, meta, opts);

        if (full_graph && filter_w <= 0) {
            std::fprintf(stderr,
                         "[dfine_coco_eval] warning: --full-graph without --filter-res — "
                         "images not matching the frozen source size fall back to the "
                         "split path\n");
        }

        std::size_t free_after_freeze = 0, total_vram = 0;
        if (do_freeze || full_graph) {
            // Full-graph capture happens inside freeze; the frozen source size must
            // match the frames we will feed (the --filter-res size when given).
            dfine::FreezeSpec fs;
            fs.batch = batch;
            fs.src_w = filter_w;
            fs.src_h = filter_h;
            det.freeze(fs);  // warm to peak + capture + lock; steady state must not allocate
            cudaMemGetInfo(&free_after_freeze, &total_vram);
            if (full_graph) {
                std::fprintf(stderr, "[dfine_coco_eval] full-pipeline graph: %s\n",
                             det.full_pipeline_graph_active()
                                 ? "captured"
                                 : "NOT active (engine needs FP32 outputs + --max-aux-streams 0)");
            }
        }

        std::ifstream fl(filelist);
        if (!fl) throw std::runtime_error("cannot open filelist: " + filelist.string());
        std::ofstream os(out);
        if (!os) throw std::runtime_error("cannot open output: " + out.string());

        os << "[";
        bool first = true;
        long long n_imgs = 0, n_dets = 0, n_missing = 0;

        struct Item {
            long long id;
            dfine_app::LoadedImage img;
        };
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
                    os << "{\"image_id\":" << chunk[k].id << ",\"category_contig\":" << d.class_id
                       << ",\"bbox\":[" << d.box.x1 << ',' << d.box.y1 << ',' << w << ',' << h
                       << ']' << ",\"score\":" << d.score << '}';
                    ++n_dets;
                }
            }
            n_imgs += static_cast<long long>(chunk.size());
            if (n_imgs % 500 < static_cast<long long>(chunk.size()))
                std::fprintf(stderr, "[dfine_coco_eval] %lld images...\n", n_imgs);
            chunk.clear();
        };

        long long n_filtered = 0;
        std::string line;
        while (std::getline(fl, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            long long image_id = 0;
            std::string fname;
            if (!(ss >> image_id >> fname)) continue;

            dfine_app::LoadedImage img = dfine_app::load_image_rgb((images_dir / fname).string());
            if (!img) {
                ++n_missing;
                continue;
            }
            if (filter_w > 0 && (img.view().width != filter_w || img.view().height != filter_h)) {
                ++n_filtered;
                continue;
            }
            chunk.push_back({image_id, std::move(img)});
            if (static_cast<int>(chunk.size()) >= batch) flush();
        }
        flush();

        os << "]\n";
        std::fprintf(stderr,
                     "[dfine_coco_eval] done: %lld images, %lld detections, %lld missing "
                     "(batch=%d)\n",
                     n_imgs, n_dets, n_missing, batch);
        if (filter_w > 0) {
            std::fprintf(stderr, "[dfine_coco_eval] filter-res %dx%d: %lld images skipped\n",
                         filter_w, filter_h, n_filtered);
        }
        if (full_graph) {
            std::fprintf(stderr,
                         "[dfine_coco_eval] full-graph replays: %llu calls over %lld "
                         "images (non-replayed calls used the split path)\n",
                         static_cast<unsigned long long>(det.full_graph_replays()), n_imgs);
        }
        if (do_freeze || full_graph) {
            std::size_t free_end = 0, tot = 0;
            cudaMemGetInfo(&free_end, &tot);
            const long long delta =
                static_cast<long long>(free_after_freeze) - static_cast<long long>(free_end);
            std::fprintf(stderr,
                         "[dfine_coco_eval] frozen: free VRAM delta over %lld images = "
                         "%+lld bytes (0 == no steady-state allocation)\n",
                         n_imgs, delta);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
