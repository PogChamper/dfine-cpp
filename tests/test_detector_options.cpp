#include "testing.hpp"

#include "dfine/tasks/detector.hpp"

#include <limits>

namespace {

bool reports_pad_error(int pad_value) {
    dfine::DetectorOptions options;
    options.preprocess.resize = dfine::PreprocessSpec::Resize::kLetterbox;
    options.preprocess.pad_value = pad_value;
    try {
        (void)dfine::DFineDetector("/__dfine_missing_engine__.engine", options);
    } catch (const std::exception& e) {
        return std::string(e.what()).find("pad_value") != std::string::npos;
    }
    return false;
}

bool reports_threshold_error(float threshold) {
    dfine::DetectorOptions options;
    options.threshold = threshold;
    try {
        (void)dfine::DFineDetector("/__dfine_missing_engine__.engine", options);
    } catch (const std::exception& e) {
        return std::string(e.what()).find("threshold") != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    DFINE_CHECK(reports_pad_error(-1));
    DFINE_CHECK(!reports_pad_error(0));
    DFINE_CHECK(!reports_pad_error(255));
    DFINE_CHECK(reports_pad_error(256));
    DFINE_CHECK(reports_threshold_error(-0.1F));
    DFINE_CHECK(!reports_threshold_error(0.0F));
    DFINE_CHECK(!reports_threshold_error(1.0F));
    DFINE_CHECK(reports_threshold_error(1.1F));
    DFINE_CHECK(reports_threshold_error(std::numeric_limits<float>::quiet_NaN()));

    return dfine::testing::finish("test_detector_options");
}
