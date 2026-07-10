// Derived from rf-detr-cpp (github.com/infracv/rf-detr-cpp, Apache-2.0); modified for D-FINE-cpp.
#pragma once

#include "dfine/core/log.hpp"

#include <NvInferRuntime.h>

#include <string>

namespace dfine {

// nvinfer1::ILogger adapter — translates TRT severities into dfine's log channel
// so user-installed callbacks see TRT messages too. Severity threshold defaults to
// WARNING; pass kVERBOSE to debug engine builds.
class TrtLogger : public nvinfer1::ILogger {
 public:
    using Severity = nvinfer1::ILogger::Severity;

    explicit TrtLogger(Severity min = Severity::kWARNING) noexcept : min_severity_(min) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > min_severity_) return;
        LogSeverity mapped = LogSeverity::kInfo;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                mapped = LogSeverity::kFatal;
                break;
            case Severity::kERROR:
                mapped = LogSeverity::kError;
                break;
            case Severity::kWARNING:
                mapped = LogSeverity::kWarning;
                break;
            case Severity::kINFO:
                mapped = LogSeverity::kInfo;
                break;
            case Severity::kVERBOSE:
                mapped = LogSeverity::kVerbose;
                break;
        }
        const std::string prefixed = std::string("[TRT] ") + (msg ? msg : "");
        log_message(mapped, prefixed.c_str());
    }

    void set_min_severity(Severity s) noexcept { min_severity_ = s; }
    Severity min_severity() const noexcept { return min_severity_; }

 private:
    Severity min_severity_;
};

}  // namespace dfine
