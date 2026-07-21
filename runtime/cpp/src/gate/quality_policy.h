#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

#include "asr/asr.h"
#include "gate/quality_model.h"

namespace pipeline {

enum class QualityPolicy { RULE, LEARNED, HYBRID };

const char* quality_policy_str(QualityPolicy policy) noexcept;
QualityPolicy parse_quality_policy(std::string_view value);
double parse_quality_threshold(const std::string& value);

struct QualityAdmission {
    bool rule_should_transcribe = false;
    bool learned_should_transcribe = false;
    bool final_should_transcribe = false;
    std::string rejection_reason;
};

QualityAdmission decide_quality_admission(
    QualityPolicy policy,
    bool rule_should_transcribe,
    const QualityPrediction* learned_prediction);

using FileAsrCallback =
    std::function<AsrResult(const float* samples, int sample_count)>;

struct FileAsrExecution {
    bool ran = false;
    AsrResult result;
};

// The callback is invoked zero or one time. The caller owns the complete,
// resampled file buffer for the duration of the call.
FileAsrExecution transcribe_admitted_file_once(
    bool final_admission,
    bool asr_enabled,
    const float* samples,
    std::size_t sample_count,
    const FileAsrCallback& callback);

}  // namespace pipeline
