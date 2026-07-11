// dfine_build — build an FP32 reference TensorRT engine entirely in C++.
//
// This utility does not convert graph precision. Production surgical-FP16 engines
// are built from the typed ONNX artifact with build_engine.py. The input sidecar is
// copied to <engine>.json and extended with the engine-build facts.
//
// usage:
//   dfine_build --onnx model.onnx [--engine model.engine] [--meta model.json]
//               [--workspace-mib 4096] [--input-name images]
//               [--min-batch 1 --opt-batch 1 --max-batch 8] [--verbose]

#include "cli_helpers.hpp"
#include "dfine/version.hpp"
#include "internal/trt_logger.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path onnx;
    std::filesystem::path engine;    // default: <onnx-stem>_<precision>.engine
    std::filesystem::path meta_in;   // default: <onnx>.json
    std::filesystem::path meta_out;  // default: <engine>.json
    std::string precision{"fp32"};
    int workspace_mib{4096};
    std::string input_name{"images"};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph{false};  // build with zero auxiliary streams for graph capture
    bool verbose{false};
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --onnx PATH [options]\n"
                 "  --engine PATH        Output .engine  (default <onnx-stem>_<precision>.engine)\n"
                 "  --meta PATH          Input meta JSON (default <onnx>.json)\n"
                 "  --meta-out PATH      Output meta JSON (default <engine>.json)\n"
                 "  --precision fp32     Accept an FP32 ONNX artifact (the only supported value)\n"
                 "  --workspace-mib N    Workspace pool MiB (default 4096)\n"
                 "  --input-name NAME    Input tensor (default images)\n"
                 "  --min-batch N --opt-batch N --max-batch N   Dynamic profile (default 1/1/1)\n"
                 "  --cuda-graph         Set max auxiliary streams to 0 for CUDA Graph capture\n"
                 "  --verbose            TRT logger at VERBOSE\n"
                 "\n  dfine v%s\n",
                 argv0, dfine::version());
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (starts_with(arg, "--onnx"))
            a.onnx = next_value(argc, argv, i, "--onnx");
        else if (starts_with(arg, "--engine"))
            a.engine = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--meta-out"))
            a.meta_out = next_value(argc, argv, i, "--meta-out");
        else if (starts_with(arg, "--meta"))
            a.meta_in = next_value(argc, argv, i, "--meta");
        else if (starts_with(arg, "--precision"))
            a.precision = next_value(argc, argv, i, "--precision");
        else if (starts_with(arg, "--workspace-mib"))
            a.workspace_mib =
                parse_int(next_value(argc, argv, i, "--workspace-mib"), "--workspace-mib");
        else if (starts_with(arg, "--input-name"))
            a.input_name = next_value(argc, argv, i, "--input-name");
        else if (starts_with(arg, "--min-batch"))
            a.min_batch = parse_int(next_value(argc, argv, i, "--min-batch"), "--min-batch");
        else if (starts_with(arg, "--opt-batch"))
            a.opt_batch = parse_int(next_value(argc, argv, i, "--opt-batch"), "--opt-batch");
        else if (starts_with(arg, "--max-batch"))
            a.max_batch = parse_int(next_value(argc, argv, i, "--max-batch"), "--max-batch");
        else if (arg == "--cuda-graph")
            a.cuda_graph = true;
        else if (arg == "--verbose")
            a.verbose = true;
        else
            throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.onnx.empty()) {
        usage(argv[0]);
        std::exit(2);
    }
    if (a.precision != "fp32") {
        throw std::runtime_error(
            "dfine_build accepts only FP32 ONNX artifacts; build typed FP16/INT8 artifacts "
            "with build_engine.py");
    }
    if (a.engine.empty()) {
        // Keep the precision visible in development-tree engine names.
        a.engine = a.onnx.parent_path() / (a.onnx.stem().string() + "_" + a.precision + ".engine");
    }
    if (a.meta_in.empty()) {
        a.meta_in = a.onnx;
        a.meta_in.replace_extension(".json");
    }
    if (a.meta_out.empty()) a.meta_out = a.engine.string() + ".json";
    if (a.min_batch <= 0 || a.opt_batch <= 0 || a.max_batch <= 0) {
        throw std::runtime_error("--*-batch values must be positive");
    }
    if (a.min_batch > a.opt_batch || a.opt_batch > a.max_batch) {
        throw std::runtime_error("require min-batch <= opt-batch <= max-batch");
    }
    return a;
}

void print_dims(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) std::fputc(',', stdout);
        std::printf("%ld", static_cast<long>(d.d[i]));
    }
    std::fputc(']', stdout);
}

std::filesystem::path normalized_path(const std::filesystem::path& path) {
    return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n", e.what());
        usage(argv[0]);
        return 2;
    }
    if (!std::filesystem::is_regular_file(args.onnx)) {
        std::fprintf(stderr, "error: onnx not found: %s\n", args.onnx.c_str());
        return 1;
    }
    try {
        const std::filesystem::path onnx = normalized_path(args.onnx);
        const std::filesystem::path engine = normalized_path(args.engine);
        const std::filesystem::path meta_in = normalized_path(args.meta_in);
        const std::filesystem::path meta_out = normalized_path(args.meta_out);
        const std::filesystem::path timing_cache =
            normalized_path(args.engine.string() + ".timing.cache");
        std::filesystem::path onnx_sidecar = args.onnx;
        onnx_sidecar.replace_extension(".json");
        onnx_sidecar = normalized_path(onnx_sidecar);

        if (engine == onnx || engine == meta_in || engine == onnx_sidecar) {
            throw std::runtime_error("engine output would overwrite an input artifact: " +
                                     args.engine.string());
        }
        if (meta_out == onnx || meta_out == engine || meta_out == meta_in ||
            meta_out == onnx_sidecar) {
            throw std::runtime_error("metadata output would overwrite an input or engine: " +
                                     args.meta_out.string());
        }
        if (timing_cache == onnx || timing_cache == engine || timing_cache == meta_in ||
            timing_cache == meta_out || timing_cache == onnx_sidecar) {
            throw std::runtime_error("timing-cache output would overwrite an artifact: " +
                                     timing_cache.string());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    nlohmann::json source_meta;
    bool have_source_meta = false;
    // Do not relabel an explicitly typed artifact as FP32.
    if (std::filesystem::is_regular_file(args.meta_in)) {
        try {
            std::ifstream mi(args.meta_in);
            mi >> source_meta;
            have_source_meta = true;
            const std::string declared = source_meta.value("precision", "fp32");
            if (declared != "fp32") {
                std::fprintf(stderr,
                             "error: %s is a %s export per its sidecar %s; dfine_build builds "
                             "fp32 only — use the Python pipeline (build_engine.py)\n",
                             args.onnx.c_str(), declared.c_str(), args.meta_in.c_str());
                return 1;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: cannot parse sidecar %s (%s); fix or remove it\n",
                         args.meta_in.c_str(), e.what());
            return 1;
        }
    }

    dfine::TrtLogger logger{args.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                         : nvinfer1::ILogger::Severity::kWARNING};

    std::printf("[dfine_build] onnx      : %s\n", args.onnx.c_str());
    std::printf("[dfine_build] engine    : %s\n", args.engine.c_str());
    std::printf("[dfine_build] precision : %s\n", args.precision.c_str());
    std::printf("[dfine_build] batch     : min=%d opt=%d max=%d\n", args.min_batch, args.opt_batch,
                args.max_batch);

    try {
        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        if (!builder) throw std::runtime_error("createInferBuilder failed");
        std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
        if (!network) throw std::runtime_error("createNetworkV2 failed");
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
        if (!parser) throw std::runtime_error("createParser failed");

        const int severity = static_cast<int>(args.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                                           : nvinfer1::ILogger::Severity::kWARNING);
        if (!parser->parseFromFile(args.onnx.c_str(), severity)) {
            for (int i = 0; i < parser->getNbErrors(); ++i)
                std::fprintf(stderr, "  ONNX error: %s\n", parser->getError(i)->desc());
            throw std::runtime_error("ONNX parsing failed");
        }
        std::printf("[dfine_build] parsed ONNX: %d input(s), %d output(s)\n",
                    network->getNbInputs(), network->getNbOutputs());

        if (network->getNbInputs() != 1) {
            throw std::runtime_error("D-FINE runtime requires exactly one input");
        }
        if (network->getNbOutputs() != 2) {
            throw std::runtime_error("D-FINE runtime requires exactly two outputs");
        }
        nvinfer1::ITensor* input = network->getInput(0);
        if (args.input_name != input->getName()) {
            args.input_name = input->getName();
        }

        const auto find_output = [&](const std::string& name) -> nvinfer1::ITensor* {
            for (int i = 0; i < network->getNbOutputs(); ++i) {
                if (name == network->getOutput(i)->getName()) return network->getOutput(i);
            }
            return nullptr;
        };
        nvinfer1::ITensor* logits = nullptr;
        nvinfer1::ITensor* boxes = nullptr;
        if (have_source_meta && source_meta.contains("output_names")) {
            const auto& names = source_meta["output_names"];
            if (!names.is_array() || names.size() != 2 || !names[0].is_string() ||
                !names[1].is_string() || names[0] == names[1]) {
                throw std::runtime_error(
                    "ONNX sidecar output_names must identify distinct logits and boxes outputs");
            }
            logits = find_output(names[0].get<std::string>());
            boxes = find_output(names[1].get<std::string>());
            if (!logits || !boxes) {
                throw std::runtime_error("ONNX sidecar output_names do not match the graph");
            }
        } else {
            logits = find_output("logits");
            boxes = find_output("boxes");
            if (logits && !boxes)
                boxes =
                    network->getOutput(0) == logits ? network->getOutput(1) : network->getOutput(0);
            if (boxes && !logits)
                logits =
                    network->getOutput(0) == boxes ? network->getOutput(1) : network->getOutput(0);
            if (!logits && !boxes) {
                nvinfer1::ITensor* first = network->getOutput(0);
                nvinfer1::ITensor* second = network->getOutput(1);
                const auto last_dim = [](const nvinfer1::ITensor& tensor) {
                    const nvinfer1::Dims dims = tensor.getDimensions();
                    return dims.nbDims > 0 ? dims.d[dims.nbDims - 1] : -1;
                };
                if (last_dim(*first) == 4 && last_dim(*second) != 4) {
                    boxes = first;
                    logits = second;
                } else if (last_dim(*second) == 4 && last_dim(*first) != 4) {
                    boxes = second;
                    logits = first;
                }
            }
        }
        if (!logits || !boxes || logits == boxes) {
            throw std::runtime_error(
                "cannot identify logits and boxes outputs; name them in the ONNX graph or "
                "provide output_names in its sidecar");
        }

        nvinfer1::Dims in_dims = input->getDimensions();
        const nvinfer1::Dims logits_dims = logits->getDimensions();
        const nvinfer1::Dims boxes_dims = boxes->getDimensions();
        if (input->getType() != nvinfer1::DataType::kFLOAT || in_dims.nbDims != 4 ||
            in_dims.d[1] != 3 || in_dims.d[2] <= 0 || in_dims.d[3] <= 0 ||
            (in_dims.d[0] != -1 && in_dims.d[0] != 1)) {
            throw std::runtime_error(
                "D-FINE input must be FP32 [B,3,H,W] with fixed H/W and B equal to 1 or dynamic");
        }
        if (logits->getType() != nvinfer1::DataType::kFLOAT ||
            boxes->getType() != nvinfer1::DataType::kFLOAT || logits_dims.nbDims != 3 ||
            boxes_dims.nbDims != 3 || logits_dims.d[0] != in_dims.d[0] ||
            boxes_dims.d[0] != in_dims.d[0] || logits_dims.d[1] <= 0 || logits_dims.d[2] <= 0 ||
            boxes_dims.d[1] != logits_dims.d[1] || boxes_dims.d[2] != 4) {
            throw std::runtime_error(
                "FP32 D-FINE outputs must be logits [B,Q,C] and boxes [B,Q,4] with fixed Q/C "
                "and the input batch axis");
        }
        if (in_dims.d[0] == 1 &&
            (args.min_batch != 1 || args.opt_batch != 1 || args.max_batch != 1)) {
            throw std::runtime_error("a static-batch ONNX graph requires min/opt/max batch 1");
        }
        std::printf("[dfine_build] input '%s' shape=", input->getName());
        print_dims(in_dims);
        std::putchar('\n');

        bool has_dynamic_dim = false;
        for (int i = 0; i < in_dims.nbDims; ++i)
            if (in_dims.d[i] < 0) has_dynamic_dim = true;
        const bool need_profile = has_dynamic_dim || args.max_batch > 1;

        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config) throw std::runtime_error("createBuilderConfig failed");
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   static_cast<std::size_t>(args.workspace_mib) << 20);
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        if (args.cuda_graph) config->setMaxAuxStreams(0);

        // Timing cache: persist tactic timings so rebuilds of the same architecture
        // skip re-benchmarking (the dominant share of build time). It is arch/TRT-
        // version specific; a mismatched cache is ignored by TensorRT, not fatal.
        const std::filesystem::path cache_path = args.engine.string() + ".timing.cache";
        std::vector<char> cache_blob;
        {
            std::ifstream cf(cache_path, std::ios::binary | std::ios::ate);
            if (cf) {
                const auto sz = cf.tellg();
                cf.seekg(0);
                cache_blob.resize(static_cast<std::size_t>(sz));
                cf.read(cache_blob.data(), sz);
            }
        }
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache{
            config->createTimingCache(cache_blob.data(), cache_blob.size())};
        if (timing_cache) {
            config->setTimingCache(*timing_cache, /*ignoreMismatch=*/false);
        }

        int in_h = (in_dims.nbDims == 4 && in_dims.d[2] > 0) ? static_cast<int>(in_dims.d[2]) : 640;
        int in_w = (in_dims.nbDims == 4 && in_dims.d[3] > 0) ? static_cast<int>(in_dims.d[3]) : 640;
        if (need_profile) {
            const int C =
                (in_dims.nbDims == 4 && in_dims.d[1] > 0) ? static_cast<int>(in_dims.d[1]) : 3;
            auto* profile = builder->createOptimizationProfile();
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4{args.min_batch, C, in_h, in_w});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4{args.opt_batch, C, in_h, in_w});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4{args.max_batch, C, in_h, in_w});
            config->addOptimizationProfile(profile);
            std::printf("[dfine_build] profile   : min=%d opt=%d max=%d @ %dx%d\n", args.min_batch,
                        args.opt_batch, args.max_batch, in_h, in_w);
        }

        std::printf("[dfine_build] building...\n");
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        const auto t1 = std::chrono::steady_clock::now();
        if (!plan) throw std::runtime_error("buildSerializedNetwork returned null");
        std::printf("[dfine_build] built in %.1fs, plan = %.1f MiB\n",
                    std::chrono::duration<double>(t1 - t0).count(),
                    static_cast<double>(plan->size()) / (1024.0 * 1024.0));

        std::ofstream out(args.engine, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open engine output: " + args.engine.string());
        out.write(static_cast<const char*>(plan->data()),
                  static_cast<std::streamsize>(plan->size()));
        out.close();
        std::printf("[dfine_build] wrote %s\n", args.engine.c_str());

        // Persist the (now-populated) timing cache for faster subsequent rebuilds.
        if (timing_cache) {
            std::unique_ptr<nvinfer1::IHostMemory> cache_out{timing_cache->serialize()};
            if (cache_out) {
                std::ofstream cf(cache_path, std::ios::binary | std::ios::trunc);
                cf.write(static_cast<const char*>(cache_out->data()),
                         static_cast<std::streamsize>(cache_out->size()));
                std::printf("[dfine_build] wrote timing cache %s\n", cache_path.c_str());
            }
        }

        // Sidecar: copy the descriptive ONNX sidecar through, then patch build fields.
        nlohmann::json j = have_source_meta ? source_meta : nlohmann::json::object();
        if (!have_source_meta) {
            j["input_names"] = {input->getName()};
            j["output_names"] = {logits->getName(), boxes->getName()};
            j["num_queries"] = logits_dims.d[1];
            j["num_classes"] = logits_dims.d[2];
        }
        j["input_h"] = in_h;
        j["input_w"] = in_w;
        j["schema_version"] = 1;
        j["precision"] = args.precision;
        j["network_typing"] = "weak";
        j["tf32"] = false;
        // This minimal builder does not compute onnx_sha256; cache resolution reports
        // the resulting engine as provenance-unverified.
        j["dynamic_batch"] = need_profile;
        j["min_batch"] = args.min_batch;
        j["opt_batch"] = args.opt_batch;
        j["max_batch"] = args.max_batch;
        j["cuda_graph_compat"] = args.cuda_graph;
        j["max_aux_streams"] = args.cuda_graph ? nlohmann::json(0) : nlohmann::json(nullptr);
        j["trt_version"] = std::to_string(NV_TENSORRT_MAJOR) + "." +
                           std::to_string(NV_TENSORRT_MINOR) + "." +
                           std::to_string(NV_TENSORRT_PATCH);
        const std::filesystem::path meta_out = normalized_path(args.meta_out);
        const std::filesystem::path meta_in = normalized_path(args.meta_in);
        std::filesystem::path onnx_sidecar = args.onnx;
        onnx_sidecar.replace_extension(".json");
        onnx_sidecar = normalized_path(onnx_sidecar);
        if ((std::filesystem::is_regular_file(args.meta_in) && meta_out == meta_in) ||
            (std::filesystem::is_regular_file(onnx_sidecar) && meta_out == onnx_sidecar)) {
            throw std::runtime_error("refusing to overwrite the ONNX sidecar: " +
                                     args.meta_out.string());
        }
        {
            std::ofstream mo(args.meta_out);
            if (!mo) {
                throw std::runtime_error("cannot write meta sidecar: " + args.meta_out.string());
            }
            mo << j.dump(2) << '\n';
            if (!mo) {
                throw std::runtime_error("cannot finish meta sidecar: " + args.meta_out.string());
            }
        }

        const std::filesystem::path appended = normalized_path(args.engine.string() + ".json");
        std::filesystem::path same_stem = args.engine;
        same_stem.replace_extension(".json");
        same_stem = normalized_path(same_stem);
        if (meta_out == appended || meta_out == same_stem) {
            const std::filesystem::path twin = meta_out == appended ? same_stem : appended;
            if (twin != meta_in && twin != onnx_sidecar && std::filesystem::is_regular_file(twin)) {
                std::filesystem::remove(twin);
                std::printf("[dfine_build] removed stale sidecar %s\n", twin.c_str());
            }
        }
        std::printf("[dfine_build] wrote %s\n", args.meta_out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
