// dfine_smoke — load an engine, resolve the dynamic batch shape, fill inputs with
// zeros, run infer, and print per-output stats. Proves the TrtSession plumbing.
//
// usage: dfine_smoke <engine.plan> [batch] [iters]

#include "cli_helpers.hpp"
#include "dfine/version.hpp"
#include "internal/trt_session.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

void print_dims(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) { if (i) std::fputc(',', stdout); std::printf("%ld", static_cast<long>(d.d[i])); }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s <engine.plan> [batch] [iters]\n  dfine v%s\n",
                     argv[0], dfine::version());
        return 2;
    }
    try {
        const std::filesystem::path engine_path{argv[1]};
        const int batch = (argc >= 3) ? std::max(1, parse_int(argv[2], "batch")) : 1;
        const int iters = (argc >= 4) ? std::max(1, parse_int(argv[3], "iters")) : 1;

        dfine::TrtSession sess(engine_path);
        std::printf("loaded: %s\n  inputs: %zu  outputs: %zu\n", engine_path.c_str(),
                    sess.input_indices().size(), sess.output_indices().size());

        // Resolve any dynamic input to [batch, C, H, W] from its static tail dims.
        for (int i : sess.input_indices()) {
            const auto& b = sess.bindings()[i];
            if (b.bytes != 0) continue;
            if (b.shape.nbDims != 4)
                throw std::runtime_error("dynamic input '" + b.name + "' is not 4-D; cannot auto-resolve");
            const int C = b.shape.d[1] > 0 ? static_cast<int>(b.shape.d[1]) : 3;
            const int H = b.shape.d[2] > 0 ? static_cast<int>(b.shape.d[2]) : 640;
            const int W = b.shape.d[3] > 0 ? static_cast<int>(b.shape.d[3]) : 640;
            sess.set_input_shape(b.name, nvinfer1::Dims4{batch, C, H, W});
            std::printf("  set %s -> [%d,%d,%d,%d]\n", b.name.c_str(), batch, C, H, W);
        }

        for (const auto& b : sess.bindings()) {
            std::printf("  %s %-8s dtype=%-5s bytes=%-10zu shape=", b.is_input ? "IN " : "OUT",
                        b.name.c_str(), dfine::TrtSession::dtype_name(b.dtype), b.bytes);
            print_dims(b.shape);
            std::putchar('\n');
        }

        std::vector<std::vector<std::uint8_t>> input_zeros;
        for (int i : sess.input_indices()) {
            const auto& b = sess.bindings()[i];
            if (b.bytes == 0) throw std::runtime_error("input '" + b.name + "' still unresolved");
            input_zeros.emplace_back(b.bytes, 0);
            sess.set_input(b.name, input_zeros.back().data(), input_zeros.back().size());
        }

        std::printf("running infer x%d (batch=%d)...\n", iters, batch);
        sess.infer();  // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; ++k) {
            for (std::size_t ii = 0; ii < sess.input_indices().size(); ++ii) {
                const auto& b = sess.bindings()[sess.input_indices()[ii]];
                sess.set_input(b.name, input_zeros[ii].data(), input_zeros[ii].size());
            }
            sess.infer();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("infer OK  total=%.3f ms  mean=%.3f ms/iter\n", ms, ms / iters);

        for (int i : sess.output_indices()) {
            const auto& b = sess.bindings()[i];
            std::vector<std::uint8_t> buf(b.bytes);
            sess.get_output(b.name, buf.data(), buf.size());
            if (b.dtype == nvinfer1::DataType::kFLOAT) {
                const std::size_t n = b.bytes / 4;
                std::vector<float> f(n);
                std::memcpy(f.data(), buf.data(), b.bytes);  // avoid aliasing UB
                double sum = 0.0, max_abs = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    sum += f[k];
                    const double a = std::fabs(static_cast<double>(f[k]));
                    if (a > max_abs) max_abs = a;
                }
                std::printf("  out %-8s fp32 n=%zu mean=%+.6f max|x|=%.6f\n",
                            b.name.c_str(), n, n ? sum / n : 0.0, max_abs);
            } else {
                std::uint64_t s = 0;
                for (auto v : buf) s += v;
                std::printf("  out %-8s %s bytes=%zu byte_sum=%llu\n", b.name.c_str(),
                            dfine::TrtSession::dtype_name(b.dtype), b.bytes,
                            static_cast<unsigned long long>(s));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
