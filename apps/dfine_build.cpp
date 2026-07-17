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

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

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
                 "  --min-batch N --opt-batch N --max-batch N   Batch profile (default 1/1/1)\n"
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
    if (a.min_batch == a.max_batch && a.min_batch != 1) {
        throw std::runtime_error(
            "static batch profiles above 1 are not supported by the native runtime; "
            "use min-batch 1 with the target opt/max batch");
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

struct BatchFacts {
    bool dynamic{false};
    int min{1};
    int opt{1};
    int max{1};
};

bool same_dims(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
    if (lhs.nbDims != rhs.nbDims) return false;
    for (int i = 0; i < lhs.nbDims; ++i)
        if (lhs.d[i] != rhs.d[i]) return false;
    return true;
}

BatchFacts inspect_batch_facts(const nvinfer1::IHostMemory& plan, nvinfer1::ILogger& logger,
                               const std::string& input_name, int channels, int height, int width,
                               const Args& args) {
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(plan.data(), plan.size())};
    if (!engine) throw std::runtime_error("failed to deserialize the engine built in this process");
    if (engine->getNbOptimizationProfiles() != 1) {
        throw std::runtime_error("built engine must contain exactly one optimization profile");
    }

    const nvinfer1::Dims shape = engine->getTensorShape(input_name.c_str());
    if (shape.nbDims != 4 || shape.d[1] != channels || shape.d[2] != height ||
        shape.d[3] != width || (shape.d[0] != -1 && shape.d[0] < 1)) {
        throw std::runtime_error("built engine has an invalid input shape");
    }

    const std::array<nvinfer1::OptProfileSelector, 3> selectors{
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::OptProfileSelector::kMAX,
    };
    const std::array<int, 3> requested{args.min_batch, args.opt_batch, args.max_batch};
    std::array<nvinfer1::Dims, 3> actual{};
    for (std::size_t i = 0; i < selectors.size(); ++i) {
        actual[i] = engine->getProfileShape(input_name.c_str(), 0, selectors[i]);
        const nvinfer1::Dims expected = nvinfer1::Dims4{requested[i], channels, height, width};
        if (!same_dims(actual[i], expected)) {
            throw std::runtime_error(
                "built engine batch profile differs from the requested profile");
        }
    }

    const bool dynamic = shape.d[0] == -1;
    if (!dynamic && shape.d[0] != 1) {
        throw std::runtime_error("built engine has static batch " + std::to_string(shape.d[0]) +
                                 "; the native runtime supports static batch 1 or a dynamic "
                                 "batch profile");
    }
    return BatchFacts{dynamic, static_cast<int>(actual[0].d[0]), static_cast<int>(actual[1].d[0]),
                      static_cast<int>(actual[2].d[0])};
}

std::filesystem::path normalized_path(const std::filesystem::path& path) {
    return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
}

std::pair<std::filesystem::path, std::filesystem::path> adjacent_temp_paths(
    const std::filesystem::path& engine, const std::filesystem::path& meta) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string suffix = ".tmp." + std::to_string(::getpid()) + "." + std::to_string(nonce);
    return {engine.string() + suffix, meta.string() + suffix};
}

void write_file_checked(const std::filesystem::path& path, const char* data, std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("staged artifact is too large: " + path.string());
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open staged artifact: " + path.string());

    out.write(data, static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error("cannot write staged artifact: " + path.string());

    out.flush();
    if (!out) throw std::runtime_error("cannot flush staged artifact: " + path.string());

    out.close();
    if (!out) throw std::runtime_error("cannot close staged artifact: " + path.string());
}

void rename_checked(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code error;
    std::filesystem::rename(from, to, error);
    if (error) {
        throw std::runtime_error("cannot publish " + to.string() + ": " + error.message());
    }
}

void validate_output_file_path(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory) return;
    if (error) {
        throw std::runtime_error("cannot inspect output " + path.string() + ": " + error.message());
    }
    if (std::filesystem::exists(status) && !std::filesystem::is_regular_file(status) &&
        !std::filesystem::is_symlink(status)) {
        throw std::runtime_error("output is not a regular file: " + path.string());
    }
}

void publish_engine_pair(const std::filesystem::path& engine, const void* plan,
                         std::size_t plan_size, const std::filesystem::path& meta,
                         const std::string& meta_text) {
    validate_output_file_path(engine);
    validate_output_file_path(meta);

    const auto [engine_tmp, meta_tmp] = adjacent_temp_paths(engine, meta);
    const std::filesystem::path engine_backup = engine_tmp.string() + ".previous";
    const auto cleanup = [&] {
        std::error_code ignored;
        std::filesystem::remove(engine_tmp, ignored);
        ignored.clear();
        std::filesystem::remove(meta_tmp, ignored);
    };

    bool engine_backed_up = false;
    bool engine_published = false;
    try {
        write_file_checked(engine_tmp, static_cast<const char*>(plan), plan_size);
        write_file_checked(meta_tmp, meta_text.data(), meta_text.size());
        std::error_code status_error;
        const auto engine_status = std::filesystem::symlink_status(engine, status_error);
        if (status_error != std::errc::no_such_file_or_directory && status_error) {
            throw std::runtime_error("cannot inspect output " + engine.string() + ": " +
                                     status_error.message());
        }
        if (!status_error && std::filesystem::exists(engine_status)) {
            std::error_code backup_error;
            if (std::filesystem::is_symlink(engine_status)) {
                std::filesystem::copy_symlink(engine, engine_backup, backup_error);
            } else {
                std::filesystem::create_hard_link(engine, engine_backup, backup_error);
            }
            if (backup_error) {
                throw std::runtime_error("cannot preserve previous engine " + engine.string() +
                                         ": " + backup_error.message());
            }
            engine_backed_up = true;
        }
        rename_checked(engine_tmp, engine);
        engine_published = true;
        rename_checked(meta_tmp, meta);
    } catch (...) {
        const std::exception_ptr original = std::current_exception();
        std::error_code rollback_error;
        if (engine_backed_up) {
            std::filesystem::rename(engine_backup, engine, rollback_error);
        } else if (engine_published) {
            std::filesystem::remove(engine, rollback_error);
        }
        cleanup();
        if (rollback_error) {
            throw std::runtime_error("engine publication failed and rollback also failed: " +
                                     rollback_error.message());
        }
        std::rethrow_exception(original);
    }
    if (engine_backed_up) {
        std::error_code remove_error;
        std::filesystem::remove(engine_backup, remove_error);
        if (remove_error) {
            std::fprintf(stderr, "warning: cannot remove previous engine %s: %s\n",
                         engine_backup.c_str(), remove_error.message().c_str());
        }
    }
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
        validate_output_file_path(args.engine);
        validate_output_file_path(args.meta_out);
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
            const std::string artifact_kind = source_meta.value("artifact_kind", "onnx");
            if (artifact_kind != "onnx") {
                std::fprintf(stderr,
                             "error: source sidecar %s declares artifact_kind='%s'; expected "
                             "'onnx'\n",
                             args.meta_in.c_str(), artifact_kind.c_str());
                return 1;
            }
            const std::string declared = source_meta.value("precision", "fp32");
            if (declared != "fp32") {
                std::fprintf(stderr,
                             "error: %s is a %s export per its sidecar %s; dfine_build builds "
                             "fp32 only — use the Python pipeline (build_engine.py)\n",
                             args.onnx.c_str(), declared.c_str(), args.meta_in.c_str());
                return 1;
            }
            const std::string color_order = source_meta.value("color_order", "RGB");
            if (color_order != "RGB") {
                std::fprintf(stderr,
                             "error: %s declares %s model input in %s; D-FINE engines require "
                             "RGB model input\n",
                             args.onnx.c_str(), color_order.c_str(), args.meta_in.c_str());
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
        const int output_count = network->getNbOutputs();
        if (output_count < 2) {
            throw std::runtime_error("D-FINE runtime requires logits and boxes outputs");
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
            if (output_count == 2 && logits && !boxes) {
                boxes =
                    network->getOutput(0) == logits ? network->getOutput(1) : network->getOutput(0);
            }
            if (output_count == 2 && boxes && !logits) {
                logits =
                    network->getOutput(0) == boxes ? network->getOutput(1) : network->getOutput(0);
            }
            if (output_count == 2 && !logits && !boxes) {
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
            const std::string prefix = output_count > 2 ? "extra outputs require " : "";
            throw std::runtime_error(prefix +
                                     "explicit logits and boxes tensor names in the ONNX graph "
                                     "or output_names in its sidecar");
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
        const BatchFacts batch_facts =
            inspect_batch_facts(*plan, logger, input->getName(), in_dims.d[1], in_h, in_w, args);
        std::printf("[dfine_build] verified %s engine profile: min=%d opt=%d max=%d\n",
                    batch_facts.dynamic ? "dynamic" : "static", batch_facts.min, batch_facts.opt,
                    batch_facts.max);

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
        j["artifact_kind"] = "engine";
        j["precision"] = args.precision;
        j["network_typing"] = "weak";
        j["tf32"] = false;
        // This minimal builder does not compute onnx_sha256; cache resolution reports
        // the resulting engine as provenance-unverified.
        j["dynamic_batch"] = batch_facts.dynamic;
        j["min_batch"] = batch_facts.min;
        j["opt_batch"] = batch_facts.opt;
        j["max_batch"] = batch_facts.max;
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
        const std::filesystem::path appended_path = args.engine.string() + ".json";
        const std::filesystem::path appended = normalized_path(appended_path);
        std::filesystem::path same_stem_path = args.engine;
        same_stem_path.replace_extension(".json");
        const std::filesystem::path same_stem = normalized_path(same_stem_path);
        const std::string meta_text = j.dump(2) + '\n';
        publish_engine_pair(args.engine, plan->data(), plan->size(), args.meta_out, meta_text);
        std::printf("[dfine_build] wrote %s\n", args.engine.c_str());
        std::printf("[dfine_build] wrote %s\n", args.meta_out.c_str());

        // An appended sidecar wins discovery without touching a same-stem JSON,
        // which may belong to an ONNX artifact. A same-stem publication must
        // remove an appended engine sidecar because that stale twin would shadow it.
        if (meta_out == same_stem && appended != meta_in && appended != onnx_sidecar &&
            std::filesystem::is_regular_file(appended_path)) {
            std::filesystem::remove(appended_path);
            std::printf("[dfine_build] removed stale sidecar %s\n", appended_path.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
