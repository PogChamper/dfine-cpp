// dfine_inspect — deserialize a D-FINE TensorRT engine and print its I/O contract.
// TensorRT-only (no dfine lib yet); the first compile-verified piece of the C++ port.
// Usage: dfine_inspect <engine.plan>

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if NV_TENSORRT_MAJOR < 10
#  error "D-FINE-cpp requires TensorRT >= 10.0 (tensor-name I/O API)."
#endif

namespace {

class StderrLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << '\n';
    }
};

const char* dtype_name(nvinfer1::DataType d) {
    switch (d) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF:  return "float16";
        case nvinfer1::DataType::kINT8:  return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kINT64: return "int64";
        case nvinfer1::DataType::kBOOL:  return "bool";
        default:                         return "other";
    }
}

std::string dims_str(const nvinfer1::Dims& d) {
    std::string s = "[";
    for (int i = 0; i < d.nbDims; ++i) s += (i ? ", " : "") + std::to_string(d.d[i]);
    return s + "]";
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(n));
    f.read(buf.data(), n);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: dfine_inspect <engine.plan>\n";
        return 2;
    }
    try {
        StderrLogger logger;
        std::cout << "engine: " << argv[1] << "\n"
                  << "TensorRT runtime: " << getInferLibVersion() << " (header "
                  << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << ")\n";

        const std::vector<char> blob = read_file(argv[1]);
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(blob.data(), blob.size())};
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");

        const int n = engine->getNbIOTensors();
        int inputs = 0, outputs = 0;
        std::cout << "bindings (" << n << "):\n";
        for (int i = 0; i < n; ++i) {
            const char* name = engine->getIOTensorName(i);
            const bool is_in = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            (is_in ? inputs : outputs)++;
            std::cout << "  [" << i << "] " << (is_in ? "IN  " : "OUT ") << name << "  "
                      << dtype_name(engine->getTensorDataType(name)) << "  "
                      << dims_str(engine->getTensorShape(name)) << "\n";
        }
        std::cout << "summary: " << inputs << " input(s), " << outputs << " output(s)\n";
        // D-FINE raw contract sanity: 1 input (images), 2 outputs (logits, boxes).
        if (inputs != 1 || outputs != 2)
            std::cout << "note: expected 1 input + 2 outputs for a raw D-FINE detection engine\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
