// Derived from rf-detr-cpp (github.com/infracv/rf-detr-cpp, Apache-2.0); modified for D-FINE-cpp.
#include "dfine/core/log.hpp"

#include <atomic>
#include <cstdio>

namespace dfine {

namespace {

// Atomics for lock-free reads on the hot path. A custom callback is rare and
// configured during init, so the cost of writing through atomic stores is
// trivial while reads remain branch-predictable.
std::atomic<LogCallback> g_callback{nullptr};
std::atomic<LogSeverity> g_min_severity{LogSeverity::kWarning};

const char* severity_tag(LogSeverity s) noexcept {
    switch (s) {
        case LogSeverity::kFatal:
            return "FATAL";
        case LogSeverity::kError:
            return "ERROR";
        case LogSeverity::kWarning:
            return "WARN ";
        case LogSeverity::kInfo:
            return "INFO ";
        case LogSeverity::kVerbose:
            return "VERB ";
    }
    return "?";
}

}  // namespace

void set_log_callback(LogCallback callback) noexcept {
    g_callback.store(callback, std::memory_order_release);
}

void set_log_severity(LogSeverity min) noexcept {
    g_min_severity.store(min, std::memory_order_release);
}

void log_message(LogSeverity severity, const char* msg) noexcept {
    if (static_cast<int>(severity) >
        static_cast<int>(g_min_severity.load(std::memory_order_acquire))) {
        return;
    }
    if (auto cb = g_callback.load(std::memory_order_acquire)) {
        cb(severity, msg ? msg : "");
        return;
    }
    std::fprintf(stderr, "[dfine][%s] %s\n", severity_tag(severity), msg ? msg : "");
}

}  // namespace dfine
