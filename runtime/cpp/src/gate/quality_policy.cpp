#include "gate/quality_policy.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pipeline {

const char* quality_policy_str(QualityPolicy policy) noexcept {
    switch (policy) {
        case QualityPolicy::RULE:    return "rule";
        case QualityPolicy::LEARNED: return "learned";
        case QualityPolicy::HYBRID:  return "hybrid";
    }
    return "unknown";
}

QualityPolicy parse_quality_policy(std::string_view value) {
    if (value == "rule") return QualityPolicy::RULE;
    if (value == "learned") return QualityPolicy::LEARNED;
    if (value == "hybrid") return QualityPolicy::HYBRID;
    throw std::invalid_argument(
        "quality policy must be one of: rule, learned, hybrid");
}

double parse_quality_threshold(const std::string& value) {
    std::size_t parsed = 0;
    double threshold = 0.0;
    try {
        threshold = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("quality threshold must be a number in [0, 1]");
    }
    if (parsed != value.size() || !std::isfinite(threshold) ||
        threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("quality threshold must be a number in [0, 1]");
    }
    return threshold;
}

QualityAdmission decide_quality_admission(
    QualityPolicy policy,
    bool rule_should_transcribe,
    const QualityPrediction* learned_prediction) {
    QualityAdmission result;
    result.rule_should_transcribe = rule_should_transcribe;

    if (policy == QualityPolicy::RULE) {
        result.final_should_transcribe = rule_should_transcribe;
        if (!result.final_should_transcribe) {
            result.rejection_reason = "rule_rejected";
        }
        return result;
    }
    if (learned_prediction == nullptr) {
        throw std::invalid_argument(
            "learned and hybrid policies require a learned prediction");
    }

    result.learned_should_transcribe =
        learned_prediction->should_transcribe;
    if (policy == QualityPolicy::LEARNED) {
        result.final_should_transcribe = result.learned_should_transcribe;
        if (!result.final_should_transcribe) {
            result.rejection_reason = "learned_below_threshold";
        }
        return result;
    }

    result.final_should_transcribe =
        rule_should_transcribe && result.learned_should_transcribe;
    if (!rule_should_transcribe) {
        result.rejection_reason = "hybrid_rule_rejected";
    } else if (!result.learned_should_transcribe) {
        result.rejection_reason = "hybrid_learned_rejected";
    }
    return result;
}

FileAsrExecution transcribe_admitted_file_once(
    bool final_admission,
    bool asr_enabled,
    const float* samples,
    std::size_t sample_count,
    const FileAsrCallback& callback) {
    FileAsrExecution execution;
    if (!final_admission || !asr_enabled) {
        return execution;
    }
    if (samples == nullptr || sample_count == 0) {
        throw std::invalid_argument("admitted quality file has no audio samples");
    }
    if (!callback) {
        throw std::invalid_argument("file ASR callback is missing");
    }
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("admitted quality file is too large for ASR");
    }

    execution.ran = true;
    execution.result = callback(samples, static_cast<int>(sample_count));
    return execution;
}

}  // namespace pipeline
