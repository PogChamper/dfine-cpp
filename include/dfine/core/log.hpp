#pragma once

#include <string>

namespace dfine {

// Severity ordering matches TensorRT's nvinfer1::ILogger::Severity:
//   FATAL > ERROR > WARN > INFO > VERBOSE
enum class LogSeverity : int {
    kFatal = 0,
    kError = 1,
    kWarning = 2,
    kInfo = 3,
    kVerbose = 4,
};

using LogCallback = void (*)(LogSeverity, const char*) noexcept;

void set_log_callback(LogCallback callback) noexcept;
void set_log_severity(LogSeverity min) noexcept;
void log_message(LogSeverity severity, const char* msg) noexcept;

inline void log_message(LogSeverity severity, const std::string& msg) noexcept {
    log_message(severity, msg.c_str());
}

}  // namespace dfine
